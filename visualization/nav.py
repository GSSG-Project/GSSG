"""Trajectory PRM + robot bridge for the ATLAS viewer.

The recorded camera trajectory (`<save_path>/trajectory.csv`, or the keyframe poses in
`cameras.json` when a single-process run did not write one)
doubles as a probabilistic-roadmap: every pose is a node, consecutive poses are
edges (the camera demonstrably traversed them), and any two nodes closer than a
user-tunable threshold in the ground plane get an extra shortcut edge. Paths are
planned with Dijkstra over that graph, then shipped to the robot nav service
as {waypoints, target} in the map ground plane.
"""

from __future__ import annotations

import csv
import heapq
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger("atlas.nav")


class TrajectoryGraph:
    """PRM over the recorded trajectory. Node coords stay in the run's data frame;
    planning happens on the ground-plane projection given by up_axis."""

    def __init__(self, xyz: np.ndarray, frame_ids: np.ndarray, up_axis: int = 2):
        self.xyz = xyz.astype(np.float64)  # [N,3] data frame
        self.frame_ids = frame_ids.astype(np.int64)
        self.up_axis = 2 if up_axis != 1 else 1
        a, b = (0, 1) if self.up_axis == 2 else (0, 2)
        self.plane_axes = (a, b)
        self.pts2d = self.xyz[:, [a, b]]  # [N,2] ground plane
        self._prox_cache: dict[float, list[tuple[int, int]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, save_path: Path, up_axis: int = 2) -> TrajectoryGraph | None:
        """trajectory.csv (every mapped frame, MP runs) or, failing that, cameras.json
        (keyframe c2w poses, written by every run) — same data frame either way."""
        save_path = Path(save_path)
        ids, xyz = [], []
        csv_path = save_path / "trajectory.csv"
        cams_path = save_path / "cameras.json"
        try:
            if csv_path.exists():
                with open(csv_path) as f:
                    for row in csv.DictReader(f):
                        ids.append(int(float(row["frame_id"])))
                        xyz.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            elif cams_path.exists():
                with open(cams_path) as f:
                    for i, cam in enumerate(json.load(f)):
                        ids.append(int(cam.get("id", i)))
                        xyz.append([float(v) for v in cam["position"][:3]])
                log.info(f"no trajectory.csv; PRM built from cameras.json ({len(xyz)} keyframes)")
            else:
                return None
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            log.warning(f"trajectory parse failed ({e}); nav disabled")
            return None
        if len(xyz) < 2:
            return None
        return cls(np.asarray(xyz), np.asarray(ids), up_axis)

    @property
    def n(self) -> int:
        return len(self.xyz)

    def prox_edges(self, thresh: float) -> list[tuple[int, int]]:
        """Non-sequential node pairs closer than `thresh` in the ground plane."""
        thresh = round(float(thresh), 4)
        with self._lock:
            cached = self._prox_cache.get(thresh)
            if cached is not None:
                return cached
        p = self.pts2d
        # Blocked pairwise distances: fine for trajectory-sized N (thousands).
        pairs: list[tuple[int, int]] = []
        block = 2048
        for i0 in range(0, self.n, block):
            pi = p[i0 : i0 + block]
            for j0 in range(i0, self.n, block):
                pj = p[j0 : j0 + block]
                d = np.linalg.norm(pi[:, None, :] - pj[None, :, :], axis=-1)
                ii, jj = np.nonzero(d < thresh)
                for a, b in zip(ii + i0, jj + j0, strict=True):
                    if b - a > 1:  # skip self and already-sequential neighbors
                        pairs.append((int(a), int(b)))
        with self._lock:
            self._prox_cache[thresh] = pairs
            if len(self._prox_cache) > 16:
                self._prox_cache.pop(next(iter(self._prox_cache)))
        return pairs

    def _adjacency(self, thresh: float) -> list[list[tuple[int, float]]]:
        adj: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]

        def link(i: int, j: int):
            w = float(np.linalg.norm(self.pts2d[i] - self.pts2d[j]))
            adj[i].append((j, w))
            adj[j].append((i, w))

        for i in range(self.n - 1):
            link(i, i + 1)
        for i, j in self.prox_edges(thresh):
            link(i, j)
        return adj

    def nearest_node(self, xy) -> int:
        d = np.linalg.norm(self.pts2d - np.asarray(xy, dtype=np.float64), axis=1)
        return int(np.argmin(d))

    def plan(self, start_xy, goal_xy, thresh: float) -> list[int]:
        """Dijkstra from the node nearest `start_xy` to the node nearest `goal_xy`.
        Returns node indices along the path (start → goal)."""
        src, dst = self.nearest_node(start_xy), self.nearest_node(goal_xy)
        if src == dst:
            return [src]
        adj = self._adjacency(thresh)
        dist = np.full(self.n, np.inf)
        prev = np.full(self.n, -1, dtype=np.int64)
        dist[src] = 0.0
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if not np.isfinite(dist[dst]):
            return []
        path = [dst]
        while path[-1] != src:
            path.append(int(prev[path[-1]]))
        return path[::-1]

    @staticmethod
    def approach_point(from_xy, goal_xy, standoff: float = 0.8):
        """Point on the segment from_xy → goal_xy that is `standoff` metres short of
        the goal; from_xy itself if it is already that close."""
        a = np.asarray(from_xy, dtype=np.float64)
        g = np.asarray(goal_xy, dtype=np.float64)
        d = float(np.linalg.norm(g - a))
        if d <= standoff or d < 1e-6:
            return a
        return a + (g - a) * ((d - standoff) / d)

    def path_to_request(self, path: list[int], goal_xy, dry_run: bool,
                        standoff: float = 0.8) -> dict:
        """Robot nav service /api/trajectory body: the PRM nodes as waypoints, then a final
        straight leg from the last node toward the object centre, stopping `standoff`
        metres from it (target), yaw facing the object."""
        pts = [self.pts2d[i] for i in path]
        last = pts[-1]
        # The follower accepts the target within ~0.07 m; aim short so the robot
        # ends inside `standoff`, not on its edge.
        tgt = self.approach_point(last, goal_xy, max(0.0, standoff - 0.08))
        dx, dy = float(goal_xy[0] - tgt[0]), float(goal_xy[1] - tgt[1])
        yaw_deg = math.degrees(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-6 else 0.0
        leg = pts if np.linalg.norm(tgt - last) > 1e-3 else pts[:-1]
        wps = [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4)} for p in leg]
        return {
            "waypoints": wps,
            "target": {
                "x": round(float(tgt[0]), 4),
                "y": round(float(tgt[1]), 4),
                "yaw_deg": round(yaw_deg, 2),
            },
            "dry_run": bool(dry_run),
        }


# ── robot bridge ──────────────────────────────────────────────────────────────


@dataclass
class RobotBridge:
    """Config + thin HTTP client for the robot nav service. The browser can't
    reach the robot cross-origin, so the viewer backend proxies every call."""

    host: str = ""
    port: int = 8100  # robot nav service default
    enabled: bool = False
    timeout: float = 3.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def configure(self, host: str | None, port: int | None, enabled: bool | None):
        with self.lock:
            if host is not None:
                self.host = host.strip()
            if port is not None:
                self.port = int(port)
            if enabled is not None:
                self.enabled = bool(enabled)

    def to_dict(self) -> dict:
        return {"host": self.host, "port": self.port, "enabled": self.enabled}

    @staticmethod
    def _detail(r: requests.Response) -> str:
        """Best-effort error detail from a robot nav service response body."""
        try:
            j = r.json()
            return str(j.get("detail") or j.get("reason") or j)
        except Exception:  # noqa: BLE001
            return r.text[:200] or f"HTTP {r.status_code}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        try:
            r = requests.request(method, url, json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"no service at {self.base_url} (connection refused — the robot nav service "
                f"listens on port 8100)"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"timeout reaching {self.base_url}") from e
        if not r.ok:
            raise RuntimeError(f"{path}: {r.status_code} {self._detail(r)}")
        return r.json()

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    def post(self, path: str, body: dict | None = None) -> dict:
        return self._request("POST", path, body or {})

    def status(self) -> dict:
        try:
            return {"reachable": True, "status": self.get("/api/status")}
        except Exception as e:  # noqa: BLE001 — surface any transport error to the UI
            return {"reachable": False, "error": str(e)}

    def pose(self) -> dict:
        """Robot pose from GET /api/pose: {available, localized, spatial, base_pose:
        {x, y, yaw_deg}, ...}. Falls back to /api/status base_pose for older builds."""
        try:
            return self.get("/api/pose")
        except RuntimeError as e:
            if "404" not in str(e):
                raise
        status = self.get("/api/status")
        bp = status.get("base_pose")
        tracking = status.get("tracking") or {}
        return {
            "available": bp is not None,
            "localized": tracking.get("spatial") in ("KNOWN_MAP", "LOOP_CLOSED"),
            "spatial": tracking.get("spatial"),
            "base_pose": bp,
        }

    def base_pose_xy(self) -> tuple[tuple[float, float], dict]:
        """Ground-plane robot position + the full /api/pose payload (for localization
        state). Raises with an actionable message when the pose can't be used."""
        p = self.pose()
        bp = p.get("base_pose")
        if not p.get("available", bp is not None) or bp is None:
            raise ValueError(f"robot pose not available (spatial: {p.get('spatial')})")
        return (float(bp["x"]), float(bp["y"])), p
