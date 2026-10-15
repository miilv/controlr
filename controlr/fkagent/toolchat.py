"""Minimal OpenAI-compatible tool-calling chat client (non-streaming) + a scripted fake.

Same wire format as inspect-robots' agent plugin through the same router: ``tools`` in the
request, ``tool_calls`` in the assistant message, ``{"role": "tool", "tool_call_id", "content"}``
replies. The key is never logged (``__repr__``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def args(self) -> dict:
        try:
            a = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {"_bad_json": self.arguments}
        return a if isinstance(a, dict) else {"_bad_json": self.arguments}


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)
    t_wall: float = 0.0
    error: str | None = None

    def raw(self) -> dict:
        m: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            m["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                               for c in self.tool_calls]
        return m


class ToolChat:
    def __init__(self, base_url: str, api_key: str, *, timeout_s: float = 120.0, max_retries: int = 2,
                 transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=15.0),
                                  headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                  transport=transport)
        self.max_retries = max_retries

    def __repr__(self) -> str:
        return "ToolChat(...)"

    def close(self) -> None:
        self._http.close()

    def complete(self, model: str, messages: list[dict], tools: list[dict], *, max_tokens: int = 1500,
                 extra_body: dict | None = None) -> Reply:
        body: dict[str, Any] = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
                                "max_tokens": max_tokens}
        body.update(extra_body or {})
        t0 = time.monotonic()
        err = "unknown"
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                err = f"transport: {e}"
                time.sleep(1.0 + attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(1.0 + 2 * attempt)
                continue
            if r.status_code != 200:
                return Reply(None, [], t_wall=time.monotonic() - t0, error=f"HTTP {r.status_code}: {r.text[:500]}")
            p = r.json()
            msg = p["choices"][0]["message"]
            calls = [ToolCall(str(c.get("id") or f"call_{i}"), str(c["function"]["name"]),
                              str(c["function"].get("arguments") or "{}"))
                     for i, c in enumerate(msg.get("tool_calls") or [])]
            return Reply(msg.get("content"), calls, p.get("usage") or {}, time.monotonic() - t0)
        return Reply(None, [], t_wall=time.monotonic() - t0, error=err)


class FakeToolChat:
    """``script(messages) -> (content, [(name, args_dict), ...])``; every call is recorded."""

    is_fake = True

    def __init__(self, script: Callable[[list[dict]], tuple[str, list[tuple[str, dict]]]]) -> None:
        self.script = script
        self.calls: list[dict] = []
        self._n = 0

    def close(self) -> None:
        pass

    def complete(self, model: str, messages: list[dict], tools: list[dict], *, max_tokens: int = 1500,
                 extra_body: dict | None = None) -> Reply:
        self.calls.append({"model": model, "messages": json.loads(json.dumps(messages)), "tools": tools})
        content, calls = self.script(messages)
        out = []
        for name, args in calls:
            self._n += 1
            out.append(ToolCall(f"call_{self._n}", name, json.dumps(args)))
        return Reply(content, out, {"prompt_tokens": 0, "completion_tokens": 0}, 0.001)
