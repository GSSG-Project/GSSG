"""Provider-agnostic LLM chat interface with tool-use support.

Auto-detection order (when ATLAS_LLM_PROVIDER is unset): ANTHROPIC_API_KEY,
GOOGLE_API_KEY, OPENAI_API_KEY. ATLAS_LLM_PROVIDER=anthropic|google|openai
forces a choice.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

# JSON-schema keywords that strict tool use does not support; they must be
# stripped from the schema sent under strict mode or the request fails.
# Pydantic still enforces them in agent._validate_args.
_STRICT_UNSUPPORTED_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
)


def _strip_strict_unsupported(schema: Any) -> Any:
    """Recursively drop strict-mode-incompatible keywords from a JSON schema."""
    if isinstance(schema, dict):
        return {
            k: _strip_strict_unsupported(v)
            for k, v in schema.items()
            if k not in _STRICT_UNSUPPORTED_KEYS
        }
    if isinstance(schema, list):
        return [_strip_strict_unsupported(v) for v in schema]
    return schema


def detect_provider() -> str:
    forced = os.environ.get("ATLAS_LLM_PROVIDER", "").lower().strip()
    if forced in ("anthropic", "claude"):
        return "anthropic"
    if forced in ("google", "gemini"):
        return "google"
    if forced in ("openai", "gpt"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM provider configured. Set ANTHROPIC_API_KEY, GOOGLE_API_KEY, "
        "or OPENAI_API_KEY (or ATLAS_LLM_PROVIDER + the matching key)."
    )


DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "google": "gemini-2.5-flash",
    "openai": "gpt-4o",
}


def get_backend(provider: str | None = None, model: str | None = None):
    p = provider or detect_provider()
    m = model or os.environ.get("ATLAS_LLM_MODEL") or DEFAULT_MODELS[p]
    if p == "anthropic":
        return AnthropicBackend(m)
    if p == "google":
        return GoogleBackend(m)
    if p == "openai":
        return OpenAIBackend(m)
    raise ValueError(f"unknown provider: {p}")


# Anthropic


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                if m.get("raw_blocks"):
                    # Verbatim replay keeps thinking blocks, which the API requires
                    # when continuing a tool-use turn on models that think by default.
                    out.append({"role": "assistant", "content": m["raw_blocks"]})
                    continue
                content_blocks = []
                if m.get("content"):
                    content_blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["args"],
                        }
                    )
                out.append({"role": "assistant", "content": content_blocks})
            elif role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m["content"],
                                "is_error": bool(m.get("is_error")),
                            }
                        ],
                    }
                )
        return out

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        # strict=True guarantees tool-arg JSON validates against the schema, but
        # strict mode rejects some schema keywords (minItems, minimum, ...), so
        # strip those from the copy we send (Pydantic still enforces them).
        converted = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": _strip_strict_unsupported(copy.deepcopy(t["parameters"])),
                "strict": True,
            }
            for t in tools
        ]
        # Cache the stable tools+system prefix: a breakpoint on the last tool
        # caches everything rendered before it on every loop iteration.
        if converted:
            converted[-1]["cache_control"] = {"type": "ephemeral"}
        return converted

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            # System as a cached text block, stable across the tool-use loop.
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=self._convert_messages(messages),
            tools=self._convert_tools(tools),
        )
        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "args": dict(block.input)})
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": resp.stop_reason,
            "raw_blocks": [b.model_dump(exclude_none=True) for b in resp.content],
        }


# Google Gemini


class GoogleBackend:
    name = "google"

    def __init__(self, model: str):
        from google import genai

        self.genai = genai
        self.client = genai.Client()
        self.model = model

    def _convert_messages(self, messages: list[dict]):
        from google.genai import types as gt

        out = []
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append(gt.Content(role="user", parts=[gt.Part(text=m["content"])]))
            elif role == "assistant":
                parts = []
                if m.get("content"):
                    parts.append(gt.Part(text=m["content"]))
                for tc in m.get("tool_calls", []):
                    parts.append(
                        gt.Part(function_call=gt.FunctionCall(name=tc["name"], args=tc["args"]))
                    )
                if parts:
                    out.append(gt.Content(role="model", parts=parts))
            elif role == "tool":
                # Gemini expects tool results as a function_response inside a user turn.
                payload = m["content"]
                try:
                    parsed = json.loads(payload) if isinstance(payload, str) else payload
                    if not isinstance(parsed, dict):
                        parsed = {"result": parsed}
                except Exception:
                    parsed = {"result": payload}
                out.append(
                    gt.Content(
                        role="user",
                        parts=[
                            gt.Part(
                                function_response=gt.FunctionResponse(
                                    name=m.get("tool_name", "tool"), response=parsed
                                )
                            )
                        ],
                    )
                )
        return out

    def _convert_tools(self, tools: list[dict]):
        from google.genai import types as gt

        decls = [
            gt.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters=t["parameters"]
            )
            for t in tools
        ]
        return [gt.Tool(function_declarations=decls)]

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        from google.genai import types as gt

        cfg = gt.GenerateContentConfig(
            system_instruction=system,
            tools=self._convert_tools(tools),
            temperature=0.0,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=self._convert_messages(messages),
            config=cfg,
        )
        content = ""
        tool_calls = []
        cand = resp.candidates[0] if resp.candidates else None
        if cand and cand.content:
            for i, p in enumerate(cand.content.parts):
                if getattr(p, "text", None):
                    content += p.text
                fc = getattr(p, "function_call", None)
                if fc:
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append({"id": f"call_{i}_{fc.name}", "name": fc.name, "args": args})
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": "end_turn" if not tool_calls else "tool_use",
        }


# OpenAI


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def _convert_messages(self, system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m["content"]
                        if isinstance(m["content"], str)
                        else json.dumps(m["content"]),
                    }
                )
        return out

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        # OpenAI strict mode ("strict": True) additionally requires every property
        # to appear in "required", but several schemas have optional params
        # (room_id, k, ...), so enabling it would fail at request time.
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(system, messages),
            tools=self._convert_tools(tools),
            temperature=0.0,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "stop_reason": resp.choices[0].finish_reason,
        }
