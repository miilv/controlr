"""Decision head: the control model on OpenRouter's Decisions API (``llm.backend=decisions``).

WHY an adapter: a decision model (e.g. ``openai/gpt-6-luna-decisions``) writes no text. It reads a
``state`` and returns probabilities for named, typed questions we define — ``score`` (an ordered
scale), ``choice`` and ``noul`` (yes/no) — in one short request (~0.3 s P50 vs 2-8 s for a chat
turn). ``DecisionsClient.complete`` has the ``LLMClient`` signature: it turns the chat transcript
into a state + questions, asks, and renders the answers as an ordinary reply in the grammar
(``MOVE ee_delta dx dy dz [dyaw]`` / ``GRIP`` / ``HOLD`` + ``STATUS``). Parsing, the safety
envelope, execution, feedback and the run log stay the same as for chat models, and the model still
chooses low-level numeric steps — no skills.

Per turn: one ``score`` question per motion axis over ``decisions.levels`` (LLM units), a ``grip``
choice (keep/open/close) and a ``status`` choice (CONTINUE/DONE/FAIL). ``reduce=expected`` takes
the probability-weighted level, so an unsure model takes a smaller step; ``argmax`` the likeliest.

The API keeps no history and the request is rebuilt each turn: ``state`` = the operating manual
(``include_manual``), turn 0's text (task, plan, first STATE), the last ``history`` (action,
feedback) pairs and the newest frame(s). Prompt caching therefore plays no part here; the
transcript stays append-only for the run log. How images go into ``state`` is not documented
(``image_mode``: content parts or a data-URL field) — confirm on a live call.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from .client import (KEEPALIVE_S, RETRY_STATUS, LLMResult, Timings, Usage, _error_message,
                     _parse_retry_after, _Retryable)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
AXES = ("x", "y", "z")
STATUS_OF = {"CONTINUE": "OK", "DONE": "DONE", "FAIL": "FAIL"}


def load_questions(name: str) -> dict:
    path = PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"decision questions {name!r} not found in {PROMPT_DIR}")
    return yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------------
# request: transcript -> state + questions
# ---------------------------------------------------------------------------

def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def _image_urls(content: Any) -> list[str]:
    if isinstance(content, str):
        return []
    return [p["image_url"]["url"] for p in content or [] if p.get("type") == "image_url"]


def _level_text(tpl: dict, v: float, axis: str, unit: str) -> str:
    return tpl["zero"].format(axis=axis) if v == 0 else tpl["move"].format(v=f"{v:+g}", unit=unit, axis=axis)


def build_questions(dcfg, acfg, tpl: dict) -> dict:
    """Question set for one turn; key order is fixed (byte-stable requests)."""
    q: dict[str, dict] = {}
    axes = [(f"d{a}", a, dcfg.levels, acfg.pos_unit, tpl["axis"]) for a in AXES]
    if acfg.rotation == "yaw":
        axes.append(("dyaw", "yaw", dcfg.yaw_levels, acfg.ang_unit, tpl["yaw"]))
    for key, axis, levels, unit, t in axes:
        q[key] = {"type": "score", "instructions": t["instructions"].format(axis=axis),
                  "criteria": [_level_text(t, float(v), axis, unit) for v in levels]}
    for key in ("grip", "status"):
        q[key] = {"type": "choice", "instructions": tpl[key]["instructions"],
                  "criteria": dict(tpl[key]["criteria"])}
    return q


def build_state(messages: list[dict], dcfg) -> Any:
    """``state`` from the chat transcript (system, user0, then assistant/user pairs)."""
    system = next((_text(m["content"]) for m in messages if m["role"] == "system"), "")
    users = [m for m in messages if m["role"] == "user"]
    replies = [_text(m["content"]) for m in messages if m["role"] == "assistant"]
    steps = [{"turn": i, "action": a, "feedback": _text(u["content"])}
             for i, (a, u) in enumerate(zip(replies, users[1:]))]
    obj: dict[str, Any] = {}
    if dcfg.include_manual and system:
        obj["manual"] = system
    obj["task"] = _text(users[0]["content"]) if users else ""
    obj["turn"] = len(replies)
    obj["recent_steps"] = steps[-dcfg.history:] if dcfg.history else []
    images = _image_urls(users[-1]["content"]) if users else []
    if dcfg.image_mode == "field":
        return {**obj, "images": images}
    return [{"type": "text", "text": json.dumps(obj, ensure_ascii=False)},
            *({"type": "image_url", "image_url": {"url": u}} for u in images)]


# ---------------------------------------------------------------------------
# answers -> reply text
# ---------------------------------------------------------------------------

def _probs(ans: dict, n: int) -> list[float] | None:
    p = ans.get("probabilities")
    if not isinstance(p, dict):
        return None
    vals = [float(p.get(str(i), 0.0) or 0.0) for i in range(n)]
    return vals if sum(vals) > 0 else None


def reduce_score(ans: dict, levels: list[float], how: str, deadband: float) -> float:
    """A score answer -> a step in LLM units (rounded to 0.1, -0.0 -> 0.0)."""
    n = len(levels)
    p = _probs(ans, n)
    if p is not None:
        if how == "argmax":
            v = levels[max(range(n), key=lambda i: p[i])]
        else:
            v = sum(pi * lv for pi, lv in zip(p, levels)) / sum(p)
    elif "score" not in ans:     # no answer for this axis: no motion along it
        return 0.0
    else:   # only the expected index: interpolate between levels
        s = min(max(float(ans.get("score", 0.0)), 0.0), n - 1.0)
        if how == "argmax":
            s = round(s)
        i = min(int(s), n - 2)
        v = levels[i] + (s - i) * (levels[i + 1] - levels[i])
    if abs(v) < deadband:
        v = 0.0
    return round(v, 1) + 0.0


def reduce_choice(ans: dict, default: str) -> str:
    if isinstance(ans.get("choice"), str):
        return ans["choice"]
    p = ans.get("probabilities")
    if isinstance(p, dict) and p:
        return max(p, key=lambda k: p[k])
    return default


def render_reply(answers: dict, dcfg, acfg) -> tuple[str, dict]:
    """(reply in the grammar, compact record for the run log)."""
    keys = [f"d{a}" for a in AXES] + (["dyaw"] if acfg.rotation == "yaw" else [])
    steps = {k: reduce_score(answers.get(k, {}), [float(v) for v in
                             (dcfg.yaw_levels if k == "dyaw" else dcfg.levels)], dcfg.reduce, dcfg.deadband)
             for k in keys}
    grip = reduce_choice(answers.get("grip", {}), "keep")
    status = STATUS_OF.get(reduce_choice(answers.get("status", {}), "CONTINUE"), "OK")
    if status != "OK":
        text = f"STATUS {status}"            # no motion on a terminal turn
    elif grip in ("open", "close"):
        text = f"GRIP {grip}\nSTATUS OK"     # gripper changes happen in place
    elif not any(steps.values()):
        text = "HOLD\nSTATUS OK"
    else:
        text = f"MOVE ee_delta {' '.join(f'{steps[k]:.1f}' for k in keys)}\nSTATUS OK"
    rec = {"steps": steps, "grip": grip, "status": status,
           "answers": {k: {f: v for f, v in a.items() if f in ("probabilities", "score", "choice", "confidence")}
                       for k, a in answers.items() if isinstance(a, dict)}}
    return text, rec


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class DecisionsClient:
    """``LLMClient``-compatible control client for a decision model. ``cfg`` is the full
    ``Config`` (``decisions`` + ``action``). The planner never goes through here — it needs text.
    Retries: connection errors, 429, 5xx (exponential backoff with jitter, Retry-After first)."""

    def __init__(self, base_url: str, api_key: str, cfg, timeout_s: float, max_retries: int, *,
                 backoff_base: float = 0.5, backoff_max: float = 20.0, headers: dict | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.cfg = cfg
        self.tpl = load_questions(cfg.decisions.questions)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base, self.backoff_max = backoff_base, backoff_max
        self._sleep, self._rng = sleep, rng or random.Random()
        hdrs = {str(k): str(v) for k, v in (headers or {}).items()}
        hdrs.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=min(15.0, timeout_s)),
                                  headers=hdrs, transport=transport,
                                  limits=httpx.Limits(max_connections=4, max_keepalive_connections=2,
                                                      keepalive_expiry=KEEPALIVE_S))

    def close(self) -> None:
        self._http.close()

    def __repr__(self) -> str:   # never leak the key
        return f"DecisionsClient(base_url={self.base_url!r}, max_retries={self.max_retries})"

    def request_body(self, model: str, messages: list[dict], extra_body: dict | None = None) -> dict:
        body: dict[str, Any] = {"model": model, "state": build_state(messages, self.cfg.decisions),
                                "questions": build_questions(self.cfg.decisions, self.cfg.action, self.tpl)}
        if extra_body:
            body.update(extra_body)
        return body

    def complete(self, model: str, messages: list[dict], *, max_tokens: int = 0,
                 temperature: float | None = None, extra_body: dict | None = None,
                 stop_when: Callable[[str], bool] | None = None,
                 timeout_s: float | None = None, max_retries: int | None = None) -> LLMResult:
        """``max_tokens`` / ``temperature`` / ``stop_when`` do not apply (no text is generated)."""
        payload = json.dumps(self.request_body(model, messages, extra_body)).encode()
        retries = self.max_retries if max_retries is None else max(0, int(max_retries))
        kw = {"timeout": httpx.Timeout(timeout_s, connect=min(15.0, timeout_s))} if timeout_s else {}
        t_wall0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                res = self._attempt(payload, kw)
            except _Retryable as e:
                if attempt > retries:
                    t = time.perf_counter() - t_wall0
                    return LLMResult("", None, Timings(None, None, t, t_wall=t), False, None,
                                     f"{e} (gave up after {attempt} attempts)", e.status, attempt,
                                     request_bytes=len(payload))
                d = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1)) * (0.5 + 0.5 * self._rng.random())
                self._sleep(min(e.retry_after, 60.0) if e.retry_after is not None else d)
                continue
            res.attempts = attempt
            res.timings.t_wall = time.perf_counter() - t_wall0
            res.request_bytes = len(payload)
            return res

    def _attempt(self, payload: bytes, kw: dict) -> LLMResult:
        t0 = time.perf_counter()
        try:
            r = self._http.post(f"{self.base_url}/decisions", content=payload, **kw)
        except httpx.TransportError as e:
            raise _Retryable(f"{type(e).__name__}: {e}") from None
        t_end = time.perf_counter() - t0
        headers = {k: v for k, v in r.headers.items() if k.lower() not in ("set-cookie", "authorization")}
        if r.status_code >= 400:
            msg = _error_message(r.status_code, r.content)
            if r.status_code in RETRY_STATUS or r.status_code >= 500:
                raise _Retryable(msg, r.status_code, _parse_retry_after(r.headers.get("retry-after")))
            return LLMResult("", None, Timings(None, None, t_end), False, None, msg, r.status_code, 1,
                             headers=headers)
        try:
            data = r.json()
            answers = data["answers"]
        except (ValueError, KeyError, TypeError) as e:
            return LLMResult("", None, Timings(None, None, t_end), False, None,
                             f"bad decisions response: {type(e).__name__}: {e}", r.status_code, 1,
                             headers=headers)
        text, rec = render_reply(answers, self.cfg.decisions, self.cfg.action)
        u = data.get("usage") or {}
        usage = Usage(int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0), 0, 0, 0, dict(u))
        rec.update(id=data.get("id"), model=data.get("model"), provider=data.get("provider"))
        return LLMResult(text, usage, Timings(t_end, t_end, t_end, ttft_any=t_end, t_headers=t_end),
                         False, "decisions", None, r.status_code, 1, headers=headers, decisions=rec)
