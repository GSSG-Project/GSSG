"""ATLAS dashboard backend: a FastAPI server for the scene viewer, CLIP query, and LLM agent.

Run from the repo root (after `pip install -e .`):
    python -m uvicorn visualization.server:app --reload --port 8001
or:
    python visualization/server.py --port 8001
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# Put the repo root on sys.path so `python visualization/server.py` works without
# `pip install -e .` (running a script by path only adds visualization/ to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rtree import index

from gssg.scene_graph.semantic_encoder import SemanticEncoder
from gssg.scene_graph.vector_db import FaissHNSWVectorDB
from visualization.llm.agent import Agent, ToolEnv
from visualization.llm.interface import (
    SceneGraphInterface,
    SpatialIndexInterface,
    VectorDBInterface,
)
from visualization.llm.providers import detect_provider
from visualization.nav import RobotBridge, TrajectoryGraph

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("atlas")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
WEB_DIR = Path(__file__).resolve().parent / "web"
BLACKLIST_FILE = Path(__file__).resolve().parent / "blacklist.json"


def _load_blacklist(scene: str) -> set[int]:
    """Per-scene object-id blacklist from visualization/blacklist.json (empty set if absent)."""
    if not BLACKLIST_FILE.exists():
        return set()
    try:
        with open(BLACKLIST_FILE) as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"blacklist load failed ({e}); ignoring")
        return set()
    raw = data.get(scene, [])
    if not isinstance(raw, list):
        return set()
    return {int(x) for x in raw if isinstance(x, (int, float))}


# scene state


@dataclass
class SceneState:
    name: str
    save_path: Path
    sg_path: Path
    sg_raw: dict
    sg: SceneGraphInterface
    vec_db: VectorDBInterface
    spatial: SpatialIndexInterface
    ply_path: Path
    ply_iter: int
    blacklist: set[int]
    cells_json: Path | None = None  # per-cell viewer manifest (cells.json), if present
    up_axis: int = 2  # scene vertical axis: 1 = Y-up (HM3D), 2 = Z-up (Replica/ROS)
    trajectory: TrajectoryGraph | None = None  # PRM over trajectory.csv, if the run has one
    title: str = ""  # display name (manifest "title"), falls back to the dir name

    def to_meta(self) -> dict:
        return {
            "scene": self.name,
            "title": self.title or self.name,
            "save_path": str(self.save_path),
            "scene_graph_path": str(self.sg_path),
            "ply_path": str(self.ply_path),
            "ply_iter": self.ply_iter,
            "object_count": len(self.sg_raw.get("objects", [])),
            "room_count": len(self.sg_raw.get("rooms", [])),
            "blacklist": sorted(self.blacklist),
            "has_cells": self.cells_json is not None and self.cells_json.exists(),
            "up_axis": self.up_axis,
            "has_trajectory": self.trajectory is not None,
        }


def _read_manifest(scene: str) -> dict | None:
    """Load the run manifest (manifest.json), the source of truth for canonical paths and dim,
    or None when absent."""
    mpath = OUTPUT_DIR / scene / "manifest.json"
    if not mpath.exists():
        return None
    try:
        with open(mpath) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"manifest read failed for {scene}: {e}")
        return None


def _scene_title(scene: str) -> str:
    m = _read_manifest(scene) or {}
    return str(m.get("title") or scene)


def _list_scenes() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    out = []
    for entry in OUTPUT_DIR.iterdir():
        if not entry.is_dir():
            continue
        # Servable if it has a manifest or a scene-graph json.
        has_manifest = (entry / "manifest.json").exists()
        sg_dir = entry / "scene_graph"
        has_sg = sg_dir.exists() and any(sg_dir.glob("objects_*.json"))
        if not (has_manifest or has_sg):
            continue
        out.append(entry.name)
    out.sort(key=lambda s: (OUTPUT_DIR / s).stat().st_mtime, reverse=True)
    return out


def _latest_scene_graph_json(scene: str) -> Path | None:
    sg_dir = OUTPUT_DIR / scene / "scene_graph"
    files = sorted(
        sg_dir.glob("objects_*.json"),
        key=lambda p: int(re.search(r"objects_(\d+)\.json", p.name).group(1)),
        reverse=True,
    )
    return files[0] if files else None


def _latest_ply(scene: str) -> tuple[Path | None, int]:
    """Find the highest-iteration `iter_*_stable.ply` anywhere under output/<scene>/."""
    base = OUTPUT_DIR / scene
    if not base.exists():
        return None, -1
    pattern = re.compile(r"iter_(\d+)_stable\.ply$")
    best: tuple[Path, int] | None = None
    for p in base.rglob("iter_*_stable.ply"):
        m = pattern.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best[1]:
            best = (p, it)
    if best is None:
        return None, -1
    return best


def _load_scene(name: str) -> SceneState:
    save_path = OUTPUT_DIR / name
    if not save_path.exists():
        raise FileNotFoundError(f"scene '{name}' not found under {OUTPUT_DIR}")
    manifest = _read_manifest(name)
    sg_path = None
    if manifest and manifest.get("scene_graph_json"):
        cand = save_path / manifest["scene_graph_json"]
        sg_path = cand if cand.exists() else None
    if sg_path is None:
        sg_path = _latest_scene_graph_json(name)
    if sg_path is None:
        raise FileNotFoundError(f"no objects_*.json in {save_path}/scene_graph/")
    with open(sg_path) as f:
        raw = json.load(f)

    # Strip blacklisted object ids here, the single filter point: downstream
    # the SceneGraphInterface, R-tree, /api/scene/scene_graph, and every LLM
    # agent tool only ever see the filtered objects.
    blacklist = _load_blacklist(name)
    if blacklist:
        before = len(raw.get("objects", []))
        raw["objects"] = [o for o in raw.get("objects", []) if o and o.get("id") not in blacklist]
        log.info(
            f"blacklist for '{name}': dropped {before - len(raw['objects'])} "
            f"objects (ids: {sorted(blacklist)})"
        )

    sg = SceneGraphInterface(raw)

    vec_db_dir = save_path / "scene_graph" / "vector_db"
    if not vec_db_dir.exists():
        raise FileNotFoundError(f"missing vector_db at {vec_db_dir}")
    # Resolve the vector_db prefix: manifest, then non-dotfile index.*, then
    # empty-prefix dotfiles. load() corrects self.dim from the .pkl, so dim here
    # is a placeholder; the manifest value is preferred so non-1024 CLIP backbones
    # aren't mis-defaulted to 1024 by the .index peek, which can't see dotfiles.
    if manifest and manifest.get("vector_db_prefix"):
        vdb_prefix = str(save_path / manifest["vector_db_prefix"])
    elif (vec_db_dir / "index.index").exists():
        vdb_prefix = str(vec_db_dir / "index")
    else:
        vdb_prefix = str(vec_db_dir) + "/"
    dim = (
        int(manifest["vector_db_dim"])
        if (manifest and manifest.get("vector_db_dim"))
        else _peek_vector_db_dim(vec_db_dir)
    )
    raw_db = FaissHNSWVectorDB(dim=dim, device="cpu")
    raw_db.load(vdb_prefix)
    vec_db = VectorDBInterface(raw_db)

    rtree = index.Index(properties=index.Property(dimension=3))
    for obj in sg.objects:
        if obj.aabb is not None:
            rtree.insert(obj.id, obj.aabb)
    spatial = SpatialIndexInterface(rtree, sg)

    # Manifest records the final global-optimized stable PLY; the glob fallback picks the
    # highest iteration number, which can be a mid-run checkpoint rather than the final model.
    ply_path, ply_iter = None, -1
    if manifest and manifest.get("stable_ply"):
        cand = save_path / manifest["stable_ply"]
        if cand.exists():
            ply_path, ply_iter = cand, int(manifest.get("iteration", -1))
    if ply_path is None:
        ply_path, ply_iter = _latest_ply(name)
    if ply_path is None:
        raise FileNotFoundError(f"no iter_*_stable.ply under {save_path}")

    # Per-cell viewer manifest (cells.json): manifest path, then sibling of the
    # stable PLY, then rglob fallback. Optional; when absent the viewer falls back
    # to the single-PLY path.
    cells_json: Path | None = None
    if manifest and manifest.get("cells_manifest"):
        cand = save_path / manifest["cells_manifest"]
        if cand.exists():
            cells_json = cand
    if cells_json is None:
        sib = ply_path.parent / "cells.json"
        if sib.exists():
            cells_json = sib
    if cells_json is None:
        found = sorted(save_path.rglob("cells.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        cells_json = found[0] if found else None

    up_axis = int(manifest["up_axis"]) if (manifest and manifest.get("up_axis") is not None) else 2
    return SceneState(
        name=name,
        save_path=save_path,
        sg_path=sg_path,
        sg_raw=raw,
        sg=sg,
        vec_db=vec_db,
        spatial=spatial,
        ply_path=ply_path,
        ply_iter=ply_iter,
        blacklist=blacklist,
        cells_json=cells_json,
        up_axis=up_axis,
        trajectory=TrajectoryGraph.load(save_path, up_axis),
        title=str((manifest or {}).get("title") or ""),
    )


def _peek_vector_db_dim(vec_db_dir: Path) -> int:
    """Inspect the FAISS index file to recover the vector dim without loading."""
    try:
        import faiss

        idx_files = list(vec_db_dir.glob("*.index"))
        if not idx_files:
            raise FileNotFoundError("no .index file")
        idx = faiss.read_index(str(idx_files[0]))
        return idx.d
    except Exception as e:
        log.warning(f"peek vector_db dim failed ({e}); defaulting to 1024 (ViT-H-14)")
        return 1024


# runtime state


class _AppState:
    def __init__(self):
        self.encoder: SemanticEncoder | None = None
        self.scene: SceneState | None = None
        self.tool_env: ToolEnv | None = None
        self.lock = threading.Lock()

    def set_scene(self, name: str):
        with self.lock:
            scene = _load_scene(name)
            self.scene = scene
            assert self.encoder is not None
            self.tool_env = ToolEnv(
                sg=scene.sg,
                vec_db=scene.vec_db,
                spatial=scene.spatial,
                encoder=self.encoder,
            )
            log.info(
                f"loaded scene '{name}' ({len(scene.sg.objects)} objs, "
                f"{len(scene.sg.rooms)} rooms, ply iter {scene.ply_iter})"
            )


state = _AppState()


# Request bodies must be defined at module scope: FastAPI cannot reliably
# classify closure-scoped Pydantic models as request bodies and falls back to
# treating their fields as query parameters.


class SelectBody(BaseModel):
    scene: str


class ClipQuery(BaseModel):
    text: str
    k: int = 10


class AgentQuery(BaseModel):
    text: str


class RobotConfigBody(BaseModel):
    host: str | None = None
    port: int | None = None
    enabled: bool | None = None


class PlanBody(BaseModel):
    # Goal in the run's data frame, ground plane (up_axis 2 → map x/y).
    x: float
    y: float
    thresh: float = 0.2
    send: bool = False
    dry_run: bool = True
    standoff: float = 0.8  # final leg stops this far from the goal point
    # Optional explicit start (testing / preview without a robot). When absent the
    # robot's live base_pose is fetched from the nav service.
    start_x: float | None = None
    start_y: float | None = None


class _EncoderArgs:
    """Shim matching the constructor signature SemanticEncoder expects."""

    def __init__(self, clip_model: str):
        self.clip_model = clip_model


def create_app(default_scene: str | None = None, clip_model: str = "ViT-H-14") -> FastAPI:
    app = FastAPI(title="ATLAS · Robotic Perception", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dev server: never let the browser cache the frontend, or edits to the JS
    # modules silently don't load (the cause of "my fixes don't show up").
    @app.middleware("http")
    async def _no_store_frontend(request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/src/") or path.endswith((".js", ".css", ".html")):
            resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.on_event("startup")
    def _startup():
        log.info(f"loading CLIP encoder (clip_only=True, model={clip_model}) …")
        state.encoder = SemanticEncoder(_EncoderArgs(clip_model), clip_only=True)
        scenes = _list_scenes()
        picked = default_scene or os.environ.get("ATLAS_SCENE") or (scenes[0] if scenes else None)
        if picked:
            try:
                state.set_scene(picked)
            except Exception as e:
                log.error(f"failed to load default scene '{picked}': {e}")
        else:
            log.warning(f"no scenes found under {OUTPUT_DIR}")

        try:
            provider = detect_provider()
            log.info(f"LLM provider: {provider}")
        except Exception as e:
            log.warning(f"no LLM provider configured: {e}")

    # system / scene management

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "scene_loaded": state.scene is not None,
            "scene": state.scene.name if state.scene else None,
            "llm_provider": _try_provider(),
        }

    @app.get("/api/scenes")
    def list_scenes():
        scenes = _list_scenes()
        return {
            "scenes": scenes,
            "titles": {n: _scene_title(n) for n in scenes},
            "active": state.scene.name if state.scene else None,
        }

    @app.post("/api/scenes/select")
    def select_scene(body: SelectBody):
        try:
            state.set_scene(body.scene)
        except FileNotFoundError as e:
            raise HTTPException(404, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(500, detail=str(e)) from e
        return state.scene.to_meta()

    @app.get("/api/scene")
    def scene_meta():
        _require_scene()
        return state.scene.to_meta()

    @app.get("/api/blacklist")
    def get_blacklist():
        _require_scene()
        return {
            "scene": state.scene.name,
            "ids": sorted(state.scene.blacklist),
            "file": str(BLACKLIST_FILE),
        }

    @app.get("/api/scene/scene_graph")
    def scene_graph():
        _require_scene()
        return state.scene.sg_raw

    @app.get("/api/scene/ply")
    def scene_ply():
        _require_scene()
        ply = state.scene.ply_path
        if not ply.exists():
            raise HTTPException(404, detail=f"ply missing: {ply}")
        return FileResponse(str(ply), media_type="application/octet-stream", filename=ply.name)

    @app.get("/api/scene/cells")
    def scene_cells():
        """Per-cell viewer manifest (cells.json); 404 when the run has no per-cell export."""
        _require_scene()
        cj = state.scene.cells_json
        if cj is None or not cj.exists():
            raise HTTPException(404, detail="no per-cell export (cells.json) for this scene")
        return FileResponse(str(cj), media_type="application/json", filename="cells.json")

    @app.get("/api/scene/cell/{cell_id}")
    def scene_cell(cell_id: int):
        """The PLY for one viewer cell (sibling cells/cell_{id}.ply of cells.json)."""
        _require_scene()
        cj = state.scene.cells_json
        if cj is None or not cj.exists():
            raise HTTPException(404, detail="no per-cell export for this scene")
        # cell_id is int-typed so no traversal is possible; still confirm the
        # resolved path stays inside the manifest's sibling cells/ dir.
        cells_dir = (cj.parent / "cells").resolve()
        ply = (cells_dir / f"cell_{cell_id}.ply").resolve()
        if cells_dir not in ply.parents or not ply.exists():
            raise HTTPException(404, detail=f"cell {cell_id} not found")
        return FileResponse(str(ply), media_type="application/octet-stream", filename=ply.name)

    # nav: trajectory PRM + robot bridge

    robot = RobotBridge()

    def _require_trajectory() -> TrajectoryGraph:
        _require_scene()
        traj = state.scene.trajectory
        if traj is None:
            raise HTTPException(404, detail="this scene has no trajectory.csv")
        return traj

    @app.get("/api/nav/trajectory")
    def nav_trajectory():
        traj = _require_trajectory()
        return {
            "count": traj.n,
            "up_axis": traj.up_axis,
            "frame_ids": traj.frame_ids.tolist(),
            "nodes": [[round(float(v), 4) for v in p] for p in traj.xyz],
        }

    @app.get("/api/nav/prm")
    def nav_prm(thresh: float = 0.2):
        """Proximity (shortcut) edges for the given threshold. Sequential edges are
        implicit — node i connects to i+1 by construction."""
        traj = _require_trajectory()
        thresh = max(0.01, min(float(thresh), 10.0))
        pairs = traj.prox_edges(thresh)
        return {"thresh": thresh, "count": len(pairs), "edges": [[i, j] for i, j in pairs]}

    @app.post("/api/nav/plan")
    def nav_plan(body: PlanBody):
        """Plan over the PRM to the goal; optionally submit to the robot nav service.
        Start = explicit (start_x/start_y) or the robot's live base_pose."""
        traj = _require_trajectory()
        pose_info = None
        if body.start_x is not None and body.start_y is not None:
            start = (body.start_x, body.start_y)
            robot_pose_source = "explicit"
        else:
            if not robot.enabled or not robot.host:
                raise HTTPException(
                    409, detail="robot not connected; connect it (top bar) or pass start_x/start_y"
                )
            try:
                start, pose_info = robot.base_pose_xy()
            except Exception as e:
                raise HTTPException(
                    502,
                    detail=f"robot pose unavailable from {robot.base_url}: {e}",
                ) from e
            robot_pose_source = "robot"

        goal = (body.x, body.y)
        path = traj.plan(start, goal, body.thresh)
        if not path:
            raise HTTPException(422, detail="no path found on the PRM (graph disconnected?)")
        a, b = traj.plane_axes
        standoff = max(0.0, min(float(body.standoff), 3.0))
        request_body = traj.path_to_request(path, goal, body.dry_run, standoff)
        tgt = request_body["target"]
        path_nodes = [[round(float(v), 4) for v in traj.xyz[i]] for i in path]
        approach = list(path_nodes[-1])
        approach[a], approach[b] = tgt["x"], tgt["y"]
        path_nodes.append(approach)  # final approach leg, drawn like the rest of the route

        sent, robot_reply = False, None
        if body.send:
            if not robot.enabled or not robot.host:
                raise HTTPException(409, detail="robot not connected; cannot send trajectory")
            try:
                robot_reply = robot.post("/api/trajectory", request_body)
                sent = True
            except Exception as e:
                # The robot nav service answers 409 when a run is already active.
                code = 409 if "409" in str(e) else 502
                hint = " (a run is active — cancel it first)" if code == 409 else ""
                raise HTTPException(code, detail=f"trajectory submit failed: {e}{hint}") from e

        return {
            "start": {"x": start[0], "y": start[1], "source": robot_pose_source},
            "localized": (pose_info or {}).get("localized"),
            "spatial": (pose_info or {}).get("spatial"),
            "path_indices": path,
            "path_nodes": path_nodes,
            "standoff": standoff,
            "request": request_body,
            "sent": sent,
            "robot_reply": robot_reply,
        }

    @app.get("/api/robot/config")
    def robot_config_get():
        return robot.to_dict()

    @app.post("/api/robot/config")
    def robot_config_set(body: RobotConfigBody):
        robot.configure(body.host, body.port, body.enabled)
        return robot.to_dict()

    @app.get("/api/robot/status")
    def robot_status():
        """Proxied /api/status from the nav service. Never 500s: unreachable robots
        report {reachable: false} so the panel can poll safely."""
        if not robot.host:
            return {"reachable": False, "error": "no robot host configured"}
        return robot.status()

    @app.get("/api/robot/pose")
    def robot_pose():
        """Proxied /api/pose (robot base pose + localization quality)."""
        if not robot.host:
            return {"available": False, "error": "no robot host configured"}
        try:
            return robot.pose()
        except Exception as e:
            return {"available": False, "error": str(e)}

    @app.post("/api/robot/cancel")
    def robot_cancel():
        try:
            return robot.post("/api/cancel")
        except Exception as e:
            raise HTTPException(502, detail=f"cancel failed: {e}") from e

    @app.post("/api/robot/estop")
    def robot_estop():
        try:
            return robot.post("/api/estop")
        except Exception as e:
            raise HTTPException(502, detail=f"estop failed: {e}") from e

    # query: simple CLIP top-k

    @app.post("/api/clip_query")
    def clip_query(body: ClipQuery):
        _require_scene()
        env = state.tool_env
        assert env is not None
        try:
            raw = env.search_scene(body.text, room_id=None, k=body.k)
        except ValueError as e:
            # Almost always a CLIP-model mismatch: the query embedding dim differs
            # from the scene's vector-DB dim (e.g. MobileCLIP2-S0=512 vs ViT-H-14=1024).
            want = (_read_manifest(state.scene.name) or {}).get("clip_model") or \
                "the model its vector DB was built with"
            raise HTTPException(
                400,
                detail=(f"CLIP query failed ({e}). The running CLIP model does not match "
                        f"this scene's vector DB. Restart the server with --clip-model {want}."),
            ) from e
        return {"query": body.text, "matches": json.loads(raw)}

    # query: LLM agent (sync)

    @app.post("/api/llm_query")
    def llm_query(body: AgentQuery):
        _require_scene()
        try:
            agent = Agent(state.tool_env)
        except Exception as e:
            raise HTTPException(500, detail=f"LLM init failed: {e}") from e
        try:
            result = agent.invoke(body.text)
        except Exception as e:
            raise HTTPException(500, detail=f"agent error: {e}") from e
        return {
            "query": body.text,
            "messages": result.messages,
            "terminal": result.terminal,
            "iterations": result.iterations,
            "provider": agent.backend.name,
            "model": getattr(agent.backend, "model", None),
        }

    # query: LLM agent (SSE stream)

    @app.get("/api/llm_query_stream")
    async def llm_query_stream(request: Request, q: str):
        _require_scene()

        async def _gen():
            try:
                agent = Agent(state.tool_env)
            except Exception as e:
                payload = json.dumps({"type": "error", "where": "init", "message": str(e)})
                yield f"event: error\ndata: {payload}\n\n".encode()
                return
            try:
                for event in agent.stream(q):
                    if await request.is_disconnected():
                        return
                    name = event.get("type", "message")
                    yield f"event: {name}\ndata: {json.dumps(event)}\n\n".encode()
            except Exception as e:
                payload = json.dumps({"type": "error", "where": "stream", "message": str(e)})
                yield f"event: error\ndata: {payload}\n\n".encode()

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # static frontend

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:
        log.warning(f"web/ not found at {WEB_DIR}; frontend disabled")

    return app


def _require_scene():
    if state.scene is None or state.tool_env is None:
        raise HTTPException(503, detail="no scene loaded; POST /api/scenes/select")


def _try_provider() -> str | None:
    try:
        return detect_provider()
    except Exception:
        return None


# Module-level app for `uvicorn visualization.server:app`
app = create_app()


def main():
    parser = argparse.ArgumentParser(description="ATLAS · Robotic Perception dashboard")
    parser.add_argument("--scene", default=None, help="scene name under output/")
    parser.add_argument(
        "--clip-model",
        default="ViT-H-14",
        help="CLIP backbone: ViT-H-14 (1024D), ViT-L-14 (768D), MobileCLIP2-S0 (512D)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.scene:
        os.environ["ATLAS_SCENE"] = args.scene

    import uvicorn

    global app
    app = create_app(default_scene=args.scene, clip_model=args.clip_model)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
