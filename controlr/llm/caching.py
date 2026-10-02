"""Prompt-cache marker placement, applied at serialisation time only.

WHY markers are never stored in the transcript: Anthropic caches the longest
prefix ending at a ``cache_control`` breakpoint, and the prefix must be
byte-identical between calls. If markers lived in the stored messages, the
"moving" marker on the newest user turn would be left behind on old turns and
the old turns would differ from what a fresh serialisation produces. Adding
them on a deep copy keeps the stored transcript canonical; the only bytes that
change between turn N and N+1 are the marker positions themselves, and those
don't break prefix matching (verified through omniroute: cache reads cover all
but the newest turn).

Placement (``anthropic`` style), at most 4 breakpoints total (API limit):
  1. last content part of the system message — the operating manual is the
     most reused prefix (also shared with the planner call's system prompt);
  2. last content part of the LAST user message — writes the cache for the
     next turn;
  3. last content part of the SECOND-TO-LAST user message — the read point
     for this turn. Anthropic only looks back 20 blocks from a breakpoint for
     an earlier cache entry; a turn with several images + text can add many
     blocks, so we keep an explicit breakpoint where last turn's write ended.

Other routes (OpenAI, Gemini, ...) cache prefixes automatically -> style
``none``: no markers, but prefix stability still matters.
"""

from __future__ import annotations

import copy

MAX_MARKERS = 4

# Route prefixes that go to Anthropic models through omniroute and honour
# cache_control (verified: claude/*, cc/*, no-think/claude/*).
_ANTHROPIC_ROUTES = ("claude/", "cc/", "anthropic/", "no-think/claude", "no-think/cc/")


def cache_style_for(model: str, configured: str) -> str:
    """``configured`` is LLMConfig.cache: ``auto`` | ``anthropic`` | ``none``."""
    c = (configured or "auto").lower()
    if c in ("anthropic", "none"):
        return c
    if c != "auto":
        raise ValueError(f"unknown cache style {configured!r} (auto|anthropic|none)")
    m = model.lower()
    if m.startswith(_ANTHROPIC_ROUTES) or m.startswith("claude-"):
        return "anthropic"
    return "none"


def _marker(ttl: str) -> dict:
    if ttl == "5m":
        return {"type": "ephemeral"}
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    raise ValueError(f"unknown cache ttl {ttl!r} (5m|1h)")


def _count_markers(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and "cache_control" in p)
    return n


def _mark_last_part(msg: dict, marker: dict) -> bool:
    """Put ``marker`` on the last content part of ``msg``; string content is
    converted to a single text part. Returns False when there is nothing
    markable (empty text cannot carry cache_control on Anthropic)."""
    c = msg.get("content")
    if isinstance(c, str):
        if not c:
            return False
        msg["content"] = [{"type": "text", "text": c, "cache_control": dict(marker)}]
        return True
    if isinstance(c, list) and c:
        last = c[-1]
        if not isinstance(last, dict):
            return False
        if last.get("type") == "text" and not last.get("text"):
            return False
        if "cache_control" in last:
            return False          # already marked; don't double count
        last["cache_control"] = dict(marker)
        return True
    return False


def apply_cache_markers(messages: list[dict], style: str, ttl: str = "5m") -> list[dict]:
    """Return a deep copy of ``messages`` with cache markers for ``style``.
    The input list is never modified."""
    out = copy.deepcopy(messages)
    if style == "none":
        return out
    if style != "anthropic":
        raise ValueError(f"unknown cache style {style!r}")
    marker = _marker(ttl)

    targets: list[int] = []
    sys_idx = [i for i, m in enumerate(out) if m.get("role") == "system"]
    if sys_idx:
        targets.append(sys_idx[-1])
    user_idx = [i for i, m in enumerate(out) if m.get("role") == "user"]
    targets += list(reversed(user_idx[-2:]))   # last user first, then second-to-last

    budget = MAX_MARKERS - _count_markers(out)
    for i in targets:
        if budget <= 0:
            break
        if _mark_last_part(out[i], marker):
            budget -= 1
    return out
