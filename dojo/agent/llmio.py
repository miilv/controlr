"""omniroute access for the agent layer: one client factory and per-call accounting.

Keys: only ``OMNIROUTE_BASE_URL`` / ``OMNIROUTE_API_KEY``, from the environment or from a dotenv file
(``DOJO_AGENT_DOTENV``, default the main checkout's ``.env``). Nothing else is read from that file, so
other credentials in it never reach this process's environment. Keys are never printed.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from controlr.llm import LLMClient, LLMResult

KEYS = ("OMNIROUTE_BASE_URL", "OMNIROUTE_API_KEY")
DEFAULT_DOTENVS = ("~/skoltech/controlr/.env", "~/controlr/.env")


def omniroute_env(dotenv: str | None = None) -> dict[str, str]:
    out = {k: os.environ[k] for k in KEYS if os.environ.get(k)}
    if len(out) == len(KEYS):
        return out
    paths = [dotenv or os.environ.get("DOJO_AGENT_DOTENV") or ""] + list(DEFAULT_DOTENVS)
    for p in paths:
        if not p:
            continue
        f = Path(os.path.expanduser(p))
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k in KEYS and k not in out:
                out[k] = v.strip().strip('"').strip("'")
        if len(out) == len(KEYS):
            return out
    missing = [k for k in KEYS if k not in out]
    raise RuntimeError(f"omniroute settings missing: {missing} (env or DOJO_AGENT_DOTENV)")


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
