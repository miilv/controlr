"""Chat calls for the agentic perceiver: omniroute (OpenAI-compatible) through ``controlr.llm.client.LLMClient``.

Keys: ``OMNIROUTE_BASE_URL`` / ``OMNIROUTE_API_KEY`` from the environment or from the ``.env`` file named by
``AGENTIC_ENV_FILE`` (never logged). ``FakeChat`` replays scripted replies for tests (no network).
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChatReply:
    text: str
    error: str | None = None
    wall_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    request_bytes: int = 0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


def _read_env_file(path: str) -> dict:
    out = {}
    p = Path(path).expanduser()
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def router_credentials() -> tuple[str, str]:
    env = dict(os.environ)
    if os.environ.get("AGENTIC_ENV_FILE"):
        env = {**_read_env_file(os.environ["AGENTIC_ENV_FILE"]), **{k: v for k, v in os.environ.items() if k.startswith("OMNIROUTE_")}}
    url, key = env.get("OMNIROUTE_BASE_URL"), env.get("OMNIROUTE_API_KEY")
    if not url or not key:
        raise RuntimeError("OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY not set (environment or AGENTIC_ENV_FILE)")
    return url, key


def degenerate(text: str) -> bool:
    """Stop a reply that has turned into repeated call markup (seen on claude/ routes: dozens of empty <invoke> blocks)."""
    return text.count("<invoke") >= 4 or text.count("</invoke>") >= 3 or text.count("<parameter") >= 6


class OmniChat:
    """One reusable HTTP client; ``__call__(model, messages)`` -> ChatReply."""

    def __init__(self, timeout_s: float = 240.0, max_retries: int = 2):
        from controlr.llm.client import LLMClient

        url, key = router_credentials()
        self._c = LLMClient(url, key, timeout_s, max_retries, usage_grace_s=1.0)

    def __call__(self, model: str, messages: list[dict], max_tokens: int = 8000, extra_body: dict | None = None) -> ChatReply:
        t0 = time.time()
        r = self._c.complete(model, messages, max_tokens=max_tokens, extra_body=extra_body, stop_when=degenerate)
        u = r.usage
        return ChatReply(text=r.text or "", error=r.error, wall_s=time.time() - t0,
                         prompt_tokens=u.prompt_tokens if u else 0, completion_tokens=u.completion_tokens if u else 0,
                         reasoning_tokens=u.reasoning_tokens if u else 0, cache_read_tokens=u.cache_read_tokens if u else 0,
                         request_bytes=r.request_bytes, finish_reason=r.finish_reason)

    def close(self):
        self._c.close()


class FakeChat:
    """Scripted replies (str or callable(messages) -> str), in order; records the messages it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def __call__(self, model, messages, max_tokens=8000, extra_body=None) -> ChatReply:
        self.sent.append(list(messages))  # snapshot: the runner keeps appending to its list
        if not self.replies:
            return ChatReply(text="", error="FakeChat: no more replies")
        r = self.replies.pop(0)
        return ChatReply(text=r(messages) if callable(r) else r)

    def close(self):
        pass


def image_part(png_bytes: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png_bytes).decode()}}
