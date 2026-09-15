"""ATLAS robot perception agent: a provider-agnostic, tool-use query loop over the scene graph."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, ValidationError

from visualization.llm.interface import (
    SceneGraphInterface,
    SpatialIndexInterface,
    VectorDBInterface,
)
from visualization.llm.prompts import SYSTEM_PROMPT
from visualization.llm.providers import get_backend

# Pydantic args


class _ListRoomsArgs(BaseModel):
    pass


class _ListFloorsArgs(BaseModel):
    pass


class _GetRoomIdArgs(BaseModel):
    room_name: str = Field(..., min_length=1)


class _SearchSceneArgs(BaseModel):
    description: str = Field(..., min_length=1)
    room_id: int | None = None
    k: int = Field(10, ge=1, le=50)


class _CheckObjectTypeArgs(BaseModel):
    description: str = Field(..., min_length=1)
    candidate_ids: list[int] = Field(..., min_length=1)
    reference_object_ids: list[int] | None = None


class _SpatialCheckArgs(BaseModel):
    reference_obj_ids: list[int] = Field(..., min_length=1)
    relation: str
    radius: float = Field(1.5, gt=0)


class _CheckDistancesArgs(BaseModel):
    reference_object_id: int
    candidate_object_ids: list[int] = Field(..., min_length=1)


class _GetObjectInfoArgs(BaseModel):
    object_id: int


class _ReportFindingArgs(BaseModel):
    object_id: int | None = None
    reason: str = Field(..., min_length=1)
    # Optional audit trail, e.g. {"tools": ["search_scene"], "winning_score": 0.41}.
    evidence: dict | None = None


class _NavigateArgs(BaseModel):
    object_id: int


# Tool schemas (JSON)

_SCHEMAS: list[dict] = [
    {
        "name": "list_rooms",
        "description": "List every room in this scene (id, floor_id, whether it has a CLIP embedding). Use when the user mentions a room and you want to ground the name.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_floors",
        "description": "List the floors (storeys) of this scene with their floor heights and the room ids on each. Use for multi-storey scenes or when the user mentions a floor (e.g. 'upstairs', 'ground floor').",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_room_id",
        "description": "Encode a room name with CLIP and return the closest room id, or an error if no room is similar enough.",
        "parameters": {
            "type": "object",
            "properties": {"room_name": {"type": "string"}},
            "required": ["room_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_scene",
        "description": "CLIP semantic search over scene objects. Returns objects sorted by similarity (top score is the best match). Optional room_id restricts the search.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Plain English description of the object.",
                },
                "room_id": {"type": "integer", "description": "Optional: restrict to this room."},
                "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_object_type",
        "description": "Re-rank a candidate list of object ids by CLIP similarity to a description. Use to filter spatial results. Pass reference_object_ids to also reward candidates physically close to those references (fused semantic + spatial re-rank).",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "candidate_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                "reference_object_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional: object ids to measure spatial proximity against.",
                },
            },
            "required": ["description", "candidate_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "spatial_check",
        "description": "Find objects with a spatial relation to one or more references. Relation: NEAR | ON | INSIDE | ABOVE | BELOW. radius only used for NEAR.",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_obj_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                "relation": {"type": "string", "enum": ["NEAR", "ON", "INSIDE", "ABOVE", "BELOW"]},
                "radius": {"type": "number", "exclusiveMinimum": 0, "default": 1.5},
            },
            "required": ["reference_obj_ids", "relation"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_distances",
        "description": "Compute box-to-box distances from one reference to a list of candidates (sorted closest first).",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_object_id": {"type": "integer"},
                "candidate_object_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                },
            },
            "required": ["reference_object_id", "candidate_object_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_object_info",
        "description": "Return the full record (id, room_id, center, aabb) for one object id.",
        "parameters": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "report_finding",
        "description": "Terminate a pure query with one object id (or null) and a short reason that will be shown to the user. Optionally attach `evidence` (tools consulted, winning score) so the answer is auditable.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": ["integer", "null"]},
                "reason": {"type": "string", "minLength": 1},
                "evidence": {
                    "type": "object",
                    "description": 'Optional provenance: e.g. {"tools": ["search_scene"], "winning_score": 0.41}.',
                    "properties": {
                        "tools": {"type": "array", "items": {"type": "string"}},
                        "winning_score": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["object_id", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "robot_navigate_to",
        "description": "Physically move the robot to an object. Only use when the user clearly asked for movement.",
        "parameters": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
            "additionalProperties": False,
        },
    },
]

_ARG_MODELS: dict[str, type[BaseModel]] = {
    "list_rooms": _ListRoomsArgs,
    "list_floors": _ListFloorsArgs,
    "get_room_id": _GetRoomIdArgs,
    "search_scene": _SearchSceneArgs,
    "check_object_type": _CheckObjectTypeArgs,
    "spatial_check": _SpatialCheckArgs,
    "check_distances": _CheckDistancesArgs,
    "get_object_info": _GetObjectInfoArgs,
    "report_finding": _ReportFindingArgs,
    "robot_navigate_to": _NavigateArgs,
}

TERMINAL_TOOLS = {"report_finding", "robot_navigate_to"}


# Embedding cache


class _LRUCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        return self.store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)


# Tool environment


@dataclass
class ToolEnv:
    sg: SceneGraphInterface
    vec_db: VectorDBInterface
    spatial: SpatialIndexInterface
    encoder: Any
    emb_cache: _LRUCache = field(default_factory=lambda: _LRUCache(256))
    confidence_threshold: float = 0.20  # minimum winning score to trust a finding
    search_min_score: float = 0.10  # CLIP cosine floor for search_scene results
    rerank_min_score: float = 0.10  # CLIP cosine floor for check_object_type re-rank
    overfetch_factor: int = 5  # over-fetch multiplier before a room filter
    overfetch_cap: int = 200  # absolute cap on the over-fetched candidate pool

    def _encode_text(self, text: str):
        key = text.strip().lower()
        v = self.emb_cache.get(key)
        if v is not None:
            return v
        v = self.encoder.encode_text(text)
        if hasattr(v, "cpu"):
            v = v.cpu()
        self.emb_cache.put(key, v)
        return v

    def _obj_dict(self, obj) -> dict:
        return {
            "id": obj.id,
            "room_id": obj.room_id,
            "center": [round(c, 3) for c in obj.center],
            "aabb": [round(c, 3) for c in obj.aabb],
        }

    # tool implementations

    def list_rooms(self) -> str:
        return json.dumps(
            [
                {"id": r.id, "floor_id": r.floor_id, "has_embedding": r.embedding is not None}
                for r in self.sg.rooms
            ]
        )

    def list_floors(self) -> str:
        if not self.sg.floors:
            return json.dumps({"error": "no floor information in this scene"})
        return json.dumps(
            [
                {"id": f.id, "floor_height": f.floor_height, "room_ids": f.room_ids}
                for f in self.sg.floors
            ]
        )

    def get_room_id(self, room_name: str) -> str:
        emb = self._encode_text(room_name)
        rid = self.sg.get_closest_room_id(emb)
        if rid is None:
            return json.dumps({"error": f"no room similar enough to '{room_name}'"})
        return json.dumps({"room_id": rid})

    def search_scene(self, description: str, room_id: int | None = None, k: int = 10) -> str:
        q = self._encode_text(description)
        if hasattr(q, "numpy"):
            q_np = q.numpy()
        else:
            q_np = q
        # When filtering by room, over-fetch first so the room filter doesn't
        # drop recall (a top-k cut before the filter can leave the room with
        # zero of its true matches).
        fetch_k = k if room_id is None else min(k * self.overfetch_factor, self.overfetch_cap)
        results = self.vec_db.get_similar_vector_with_object_id_list(q_np, k=fetch_k)
        out = []
        for _, score, obj_id in results:
            s = float(score)
            if s < self.search_min_score:
                continue
            obj = self.sg.get_object_by_id(obj_id)
            if not obj:
                continue
            if room_id is not None and obj.room_id != room_id:
                continue
            out.append({"id": obj.id, "score": round(s, 3), "room_id": obj.room_id})
            if len(out) >= k:
                break
        return json.dumps(out)

    def check_object_type(
        self,
        description: str,
        candidate_ids: list[int],
        reference_object_ids: list[int] | None = None,
    ) -> str:
        q = self._encode_text(description)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q)
        q = q.reshape(-1).unsqueeze(0)

        ref_aabbs = []
        if reference_object_ids:
            for rid in reference_object_ids:
                ref = self.sg.get_object_by_id(rid)
                if ref and ref.aabb:
                    ref_aabbs.append(ref.aabb)

        out = []
        for cid in candidate_ids:
            obj = self.sg.get_object_by_id(cid)
            if not obj or obj.vector_id is None:
                continue
            tv, _ = self.vec_db.get_vector_and_object_id_by_vector_id(obj.vector_id)
            if not isinstance(tv, torch.Tensor):
                tv = torch.tensor(tv) if hasattr(tv, "__iter__") else torch.from_numpy(tv)
            tv = tv.reshape(-1).unsqueeze(0).to(q.device)
            sem = F.cosine_similarity(q, tv, dim=1).item()
            if sem < self.rerank_min_score:
                continue
            entry = {"id": cid, "semantic": round(sem, 3)}
            score = sem
            # Fuse with a normalized spatial-proximity term when references given.
            if ref_aabbs and obj.aabb:
                dist = min(self.spatial.get_min_distance_aabb(obj.aabb, ra) for ra in ref_aabbs)
                # Proximity in [0, 1]: 1 when touching, decaying with distance (~2m scale).
                proximity = 1.0 / (1.0 + dist / 2.0)
                entry["distance_m"] = round(dist, 3)
                entry["proximity"] = round(proximity, 3)
                score = 0.5 * sem + 0.5 * proximity
            entry["score"] = round(score, 3)
            out.append(entry)
        out.sort(key=lambda x: x["score"], reverse=True)
        return json.dumps(out)

    def spatial_check(
        self, reference_obj_ids: list[int], relation: str, radius: float = 1.5
    ) -> str:
        rel = relation.upper()
        if rel not in {"NEAR", "ON", "INSIDE", "ABOVE", "BELOW"}:
            return json.dumps({"error": f"unknown relation '{relation}'"})
        found: set[int] = set()
        for ref in reference_obj_ids:
            ref_obj = self.sg.get_object_by_id(ref)
            if not ref_obj:
                continue
            if rel == "NEAR":
                pairs = self.spatial.get_nearby_objects(ref, radius=radius)
            elif rel == "ON":
                pairs = self.spatial.get_objects_on(ref)
            elif rel == "INSIDE":
                pairs = self.spatial.get_objects_inside(ref)
            elif rel == "ABOVE":
                pairs = self.spatial.get_objects_above(ref)
            else:
                pairs = self.spatial.get_objects_below(ref)
            for cid, _ in pairs:
                found.add(cid)
        return json.dumps(sorted(found))

    def check_distances(self, reference_object_id: int, candidate_object_ids: list[int]) -> str:
        if self.sg.get_object_by_id(reference_object_id) is None:
            return json.dumps({"error": f"object {reference_object_id} not found"})
        pairs = self.spatial.get_distances_to_list(
            reference_object_id, candidate_object_ids, limit=20
        )
        return json.dumps([{"id": i, "distance_m": round(d, 3)} for i, d in pairs])

    def get_object_info(self, object_id: int) -> str:
        obj = self.sg.get_object_by_id(object_id)
        if not obj:
            return json.dumps({"error": f"object {object_id} not found"})
        return json.dumps(self._obj_dict(obj))

    def report_finding(
        self, object_id: int | None, reason: str, evidence: dict | None = None
    ) -> dict:
        if object_id is not None and self.sg.get_object_by_id(object_id) is None:
            return {"ok": False, "error": f"object {object_id} does not exist in this scene"}
        payload = {"ok": True, "kind": "finding", "object_id": object_id, "reason": reason}
        if evidence:
            payload["evidence"] = evidence
        return payload

    def robot_navigate_to(self, object_id: int) -> dict:
        obj = self.sg.get_object_by_id(object_id)
        if not obj:
            return {"ok": False, "error": f"object {object_id} not found"}
        return {"ok": True, "kind": "navigate", "object_id": object_id, "center": list(obj.center)}


# Agent


@dataclass
class AgentResult:
    messages: list[dict]
    terminal: dict | None  # {kind: 'finding'|'navigate', object_id, reason?}
    iterations: int


def _validate_args(name: str, raw: dict) -> tuple[dict | None, str | None]:
    model = _ARG_MODELS.get(name)
    if model is None:
        return None, f"unknown tool '{name}'"
    try:
        return model.model_validate(raw).model_dump(), None
    except ValidationError as e:
        return None, e.json(indent=None)


def _dispatch(env: ToolEnv, name: str, args: dict) -> Any:
    fn = getattr(env, name, None)
    if fn is None:
        return json.dumps({"error": f"tool '{name}' not implemented"})
    return fn(**args)


class Agent:
    def __init__(self, env: ToolEnv, backend=None, max_iterations: int = 8):
        self.env = env
        self.backend = backend or get_backend()
        self.max_iterations = max_iterations

    def _make_terminal_msg(self, tool_call_id: str, result: Any) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "tool_name": "terminal",
            "content": json.dumps(result) if not isinstance(result, str) else result,
        }

    def stream(self, query: str) -> Iterator[dict]:
        """Yield events: {type, ...} where type is one of:
        'meta', 'assistant_text', 'tool_call', 'tool_result',
        'terminal', 'error', 'done'.
        """
        yield {
            "type": "meta",
            "provider": self.backend.name,
            "model": getattr(self.backend, "model", None),
        }

        messages: list[dict] = [{"role": "user", "content": query}]
        terminal: dict | None = None
        seen_calls: list[str] = []  # call signatures, for cycle detection

        for it in range(self.max_iterations):
            try:
                resp = self.backend.chat(SYSTEM_PROMPT, messages, _SCHEMAS)
            except Exception as e:
                yield {"type": "error", "where": "backend.chat", "message": str(e)}
                return

            assistant_msg = {
                "role": "assistant",
                "content": resp.get("content", ""),
                "tool_calls": resp.get("tool_calls", []),
            }
            if resp.get("raw_blocks"):
                assistant_msg["raw_blocks"] = resp["raw_blocks"]
            messages.append(assistant_msg)
            if assistant_msg["content"]:
                yield {"type": "assistant_text", "text": assistant_msg["content"]}

            # No tool calls means the model considers itself done.
            if not assistant_msg["tool_calls"]:
                yield {"type": "done", "reason": "model_finished_no_tool", "iterations": it + 1}
                return

            for tc in assistant_msg["tool_calls"]:
                name = tc["name"]
                raw_args = tc["args"]
                sig = json.dumps({"n": name, "a": raw_args}, sort_keys=True)
                if seen_calls.count(sig) >= 2:
                    err = json.dumps(
                        {
                            "error": f"refusing to repeat the same {name} call a third time; try a different approach"
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "tool_name": name,
                            "content": err,
                            "is_error": True,
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "id": tc["id"],
                        "name": name,
                        "content": err,
                        "is_error": True,
                    }
                    continue
                seen_calls.append(sig)

                yield {"type": "tool_call", "id": tc["id"], "name": name, "args": raw_args}

                validated, err = _validate_args(name, raw_args)
                if err:
                    content = json.dumps({"error": "invalid arguments", "detail": err})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "tool_name": name,
                            "content": content,
                            "is_error": True,
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "id": tc["id"],
                        "name": name,
                        "content": content,
                        "is_error": True,
                    }
                    continue

                try:
                    result = _dispatch(self.env, name, validated)
                except Exception as e:
                    content = json.dumps({"error": "tool crashed", "detail": str(e)})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "tool_name": name,
                            "content": content,
                            "is_error": True,
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "id": tc["id"],
                        "name": name,
                        "content": content,
                        "is_error": True,
                    }
                    continue

                if name in TERMINAL_TOOLS and isinstance(result, dict) and result.get("ok"):
                    terminal = {k: v for k, v in result.items() if k != "ok"}
                    msg_content = json.dumps(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "tool_name": name,
                            "content": msg_content,
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "id": tc["id"],
                        "name": name,
                        "content": msg_content,
                    }
                    yield {"type": "terminal", **terminal}
                    yield {"type": "done", "reason": "terminal_tool", "iterations": it + 1}
                    return

                msg_content = result if isinstance(result, str) else json.dumps(result)
                is_error = isinstance(result, dict) and result.get("ok") is False
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "tool_name": name,
                        "content": msg_content,
                        "is_error": is_error,
                    }
                )
                yield {
                    "type": "tool_result",
                    "id": tc["id"],
                    "name": name,
                    "content": msg_content,
                    "is_error": is_error,
                }

        yield {
            "type": "done",
            "reason": "max_iterations_reached",
            "iterations": self.max_iterations,
        }

    def invoke(self, query: str) -> AgentResult:
        """Non-streaming wrapper returning the full message log and terminal payload."""
        events = list(self.stream(query))
        terminal = next((e for e in events if e["type"] == "terminal"), None)
        if terminal:
            terminal = {k: v for k, v in terminal.items() if k != "type"}
        iters = next((e["iterations"] for e in events if e["type"] == "done"), self.max_iterations)
        transcript: list[dict] = []
        for e in events:
            t = e["type"]
            if t == "assistant_text":
                transcript.append({"role": "assistant", "content": e["text"]})
            elif t == "tool_call":
                transcript.append({"role": "tool_call", "name": e["name"], "args": e["args"]})
            elif t == "tool_result":
                transcript.append(
                    {
                        "role": "tool_result",
                        "name": e["name"],
                        "content": e["content"],
                        "is_error": e.get("is_error", False),
                    }
                )
        return AgentResult(messages=transcript, terminal=terminal, iterations=iters)
