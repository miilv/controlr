"""omniroute access for the agent layer: one client factory and per-call accounting.

Keys: only ``OMNIROUTE_BASE_URL`` / ``OMNIROUTE_API_KEY``, from ONE source: the dotenv file named by ``dotenv`` /
``DOJO_AGENT_DOTENV`` (both values from it; nothing else is read from that file, so other credentials in it never
reach this process), or the environment when no file is named. Both at once is refused, and there is no implicit
fallback to a home-dir ``.env`` (``controlr.keyguard.router_credentials``). Keys are never printed.
"""

from __future__ import annotations

import os

from controlr.keyguard import ROUTER_KEYS, router_credentials
from controlr.llm import LLMClient, LLMResult

KEYS = ROUTER_KEYS


def omniroute_env(dotenv: str | None = None) -> dict[str, str]:
    url, key = router_credentials(dotenv or os.environ.get("DOJO_AGENT_DOTENV"), "DOJO_AGENT_DOTENV")
    return {"OMNIROUTE_BASE_URL": url, "OMNIROUTE_API_KEY": key}


def make_client(timeout_s: float = 600.0, max_retries: int = 3, dotenv: str | None = None) -> LLMClient:
    env = omniroute_env(dotenv)
    return LLMClient(env["OMNIROUTE_BASE_URL"], env["OMNIROUTE_API_KEY"], timeout_s=timeout_s,
                     max_retries=max_retries, usage_grace_s=1.0)


def call_record(res: LLMResult, model: str) -> dict:
    """What every run log keeps per call: usage (normalised), timings, status."""
    u = res.usage
    return {"model": model, "ok": res.error is None and bool(res.text), "error": res.error,
            "http_status": res.http_status, "attempts": res.attempts, "finish_reason": res.finish_reason,
            "prompt_tokens": u.prompt_tokens if u else None, "completion_tokens": u.completion_tokens if u else None,
            "cached_tokens": u.cache_read_tokens if u else None, "cache_write_tokens": u.cache_write_tokens if u else None,
            "reasoning_tokens": u.reasoning_tokens if u else None,
            "ttft": res.timings.ttft, "t_end": res.timings.t_end, "t_wall": res.timings.t_wall,
            "request_bytes": res.request_bytes}


__all__ = ["omniroute_env", "make_client", "call_record"]
