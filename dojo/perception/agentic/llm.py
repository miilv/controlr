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


class UrllibChat:
    """Stdlib-only streaming client (for interpreters without httpx, e.g. the J1 venv on isr-lab): POST
    /chat/completions with stream=true, SSE deltas accumulated, the usage chunk read; retries on 429/5xx/network."""

    def __init__(self, timeout_s: float = 240.0, max_retries: int = 2):
        self.url, self._key = router_credentials()
        self.timeout_s, self.max_retries = timeout_s, max_retries

    def __call__(self, model: str, messages: list[dict], max_tokens: int = 8000, extra_body: dict | None = None) -> ChatReply:
        import json
        import urllib.error
        import urllib.request

        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, **(extra_body or {})}
        data = json.dumps(body).encode()
        t0 = time.time()
        err = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", data=data, method="POST", headers={
                "Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Accept": "text/event-stream"})
            text, usage, finish = [], {}, None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    seen_sse, other = False, []
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            if not seen_sse and line:
                                other.append(line)
                            continue
                        seen_sse = True
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            ch = json.loads(payload)
                        except Exception:
                            continue
                        usage = ch.get("usage") or usage
                        for c in ch.get("choices") or []:
                            d = c.get("delta") or {}
                            if isinstance(d.get("content"), str):
                                text.append(d["content"])
                            finish = c.get("finish_reason") or finish
                        if degenerate("".join(text)):
                            break
                    if not seen_sse and other:  # a server that answered without SSE: one JSON body
                        try:
                            body = json.loads("\n".join(other))
                            ch = (body.get("choices") or [{}])[0]
                            text = [((ch.get("message") or {}).get("content")) or ""]
                            usage, finish = body.get("usage") or {}, ch.get("finish_reason")
                        except Exception:
                            pass
                u = usage or {}
                det = u.get("completion_tokens_details") or {}
                pdet = u.get("prompt_tokens_details") or {}
                return ChatReply(text="".join(text), wall_s=time.time() - t0, prompt_tokens=int(u.get("prompt_tokens") or 0),
                                 completion_tokens=int(u.get("completion_tokens") or 0),
                                 reasoning_tokens=int(det.get("reasoning_tokens") or 0),
                                 cache_read_tokens=int(pdet.get("cached_tokens") or 0), request_bytes=len(data), finish_reason=finish)
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
                if e.code not in (408, 429, 500, 502, 503, 504, 529):  # 529: Anthropic "overloaded"
                    break
            except Exception as e:  # network / timeout
                err = f"{type(e).__name__}: {e}"
            time.sleep(min(20.0, 2.0 * 2 ** attempt))
        return ChatReply(text="", error=err, wall_s=time.time() - t0, request_bytes=len(data))

    def close(self):
        pass


def make_chat():
    """controlr's LLMClient (httpx) when available, else the stdlib client; AGENTIC_HTTP=urllib forces the latter."""
    if os.environ.get("AGENTIC_HTTP") != "urllib":
        try:
            import httpx  # noqa: F401

            return OmniChat()
        except ImportError:
            pass
    return UrllibChat()


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
