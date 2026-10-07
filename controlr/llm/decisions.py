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
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from .client import (KEEPALIVE_S, RETRY_STATUS, LLMResult, Timings, Usage, _error_message,
                     _parse_retry_after, _Retryable)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
# seen live (502): 'OpenAI refused to answer question "dir_x"' — the whole request fails, the same
# payload is refused again; the client drops that question (its axis does not move) and asks again
_REFUSED = re.compile(r'refused to answer question "([^"]+)"')
MAX_DROPS = 3
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


def axis_hints(cameras: dict, names: list[str], origin=None, long_edge: int | None = None) -> dict[str, str]:
    """Per base axis, how it appears in each listed camera image, from the calibration — e.g.
    ``{"x": "in the `scene` image +x points right (44 px per 100 mm)", ...}``. Fills ``{hint}``."""
    import numpy as np

    from controlr.observation.renderers import describe_axes

    out: dict[str, list[str]] = {a: [] for a in AXES}
    for n in names:
        if n not in cameras:
            continue
        for part in describe_axes(cameras[n], np.zeros(3) if origin is None else origin, long_edge).split(", "):
            part = part.split(" in the ")[0]
            a = part[1:2]
            px = re.search(r"\((\d+) px", part)
            if px and int(px.group(1)) < 10:      # perspective drift along the viewing ray, not a cue
                continue
            if a in out:
                out[a].append(f"in the `{n}` image {part}")
    return {a: "; ".join(v) for a, v in out.items()}


# "higher/lower" is read physically in a top view (the gripper is always above the object there):
# vertical image relations are named by the image edges (Isaac perception probe 2026-10-07)
_REL = {"right": "to the right of", "left": "to the left of", "up": "nearer the top edge of the image than",
        "down": "nearer the bottom edge of the image than",
        "up-right": "nearer the top-right corner of the image than", "up-left": "nearer the top-left corner of the image than",
        "down-right": "nearer the bottom-right corner of the image than",
        "down-left": "nearer the bottom-left corner of the image than"}
_OPP = {"right": "left", "left": "right", "up": "down", "down": "up", "up-right": "down-left",
        "down-left": "up-right", "up-left": "down-right", "down-right": "up-left"}


def axis_views(cameras: dict, names: list[str], origin=None, min_sep_deg: float = 35.0) -> dict[str, dict | None]:
    """Per base axis, the image in which it reads unambiguously: its projected direction is at least
    ``min_sep_deg`` from every other axis's (on an angled camera +y and +z both point "up") and it
    is not along the viewing ray (an axis projecting < 30 % as long only drifts by perspective and
    does not count); among those the longest projection wins.
    ``{axis: {"image": name, "pos": "to the right of", "neg": "to the left of"} | None}``."""
    import math

    import numpy as np

    from controlr.observation.renderers import _direction_word, _project_segment

    o = np.zeros(3) if origin is None else np.asarray(origin, float)
    best: dict[str, tuple[float, dict] | None] = {a: None for a in AXES}
    for n in names:
        if n not in cameras:
            continue
        info = cameras[n]
        vec = {}
        for a, d in zip(AXES, np.eye(3)):
            seg = _project_segment(info, o, o + 0.1 * d, info.width, info.height)
            if seg is not None:
                du, dv = seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]
                if math.hypot(du, dv) >= 2.0:
                    vec[a] = (du, dv)
        for a, (du, dv) in vec.items():
            ang = math.atan2(-dv, du)
            length = math.hypot(du, dv)
            # an axis along the viewing ray only drifts by perspective: it does not make others ambiguous
            if any(abs((math.degrees(ang - math.atan2(-v2, u2)) + 180) % 360 - 180) < min_sep_deg
                   for b, (u2, v2) in vec.items() if b != a and math.hypot(u2, v2) >= 0.3 * length):
                continue
            w = _direction_word(du, dv)
            if best[a] is None or length > best[a][0]:
                best[a] = (length, {"image": n, "pos": _REL[w], "neg": _REL[_OPP[w]]})
    return {a: (b[1] if b else None) for a, b in best.items()}


def build_questions(dcfg, acfg, tpl: dict, hints: dict[str, str] | None = None,
                    views: dict[str, dict | None] | None = None, task: str = "") -> dict:
    """Question set for one turn; key order is fixed (byte-stable requests). ``hints``: per-axis
    image directions (``axis_hints``) for templates with a ``{hint}`` placeholder."""
    q: dict[str, dict] = {}
    hints = hints or {}
    target = getattr(dcfg, "target", "") .replace("{task}", task.strip())
    if dcfg.head == "split":
        views = views or {}
        for a in AXES:
            h, view = hints.get(a, ""), views.get(a)
            # v3+: an image relation where the axis reads unambiguously, else its own wording
            if view and "direction_image" in tpl:
                fovea = view.get("fovea", False)
                subj = tpl.get("subject_fovea" if fovea else "subject", "the fingertips")
                img = f"{view['image']} fovea" if fovea else view["image"]
                t, fill = tpl["direction_image"], dict(axis=a, hint=h, image=img, pos=view["pos"], neg=view["neg"],
                                                       subject=subj, target=target)
            elif f"direction_{a}" in tpl:
                t, fill = tpl[f"direction_{a}"], dict(axis=a, hint=h, target=target)
            else:
                t, fill = tpl["direction"], dict(axis=a, hint=h, target=target)
            q[f"dir_{a}"] = {"type": "choice", "instructions": t["instructions"].format(**fill),
                             "criteria": {k: v.format(**fill) for k, v in t["criteria"].items()}}
            if view and view.get("fovea") and "direction_image" in tpl:
                # coarse view of the same camera: the target may lie outside the zoomed crop
                wide = dict(fill, image=view["image"], subject=tpl.get("subject", "the fingertips"))
                t = tpl["direction_image"]
                q[f"wide_{a}"] = {"type": "choice", "instructions": t["instructions"].format(**wide),
                                  "criteria": {k: v.format(**wide) for k, v in t["criteria"].items()}}
            q[f"far_{a}"] = {"type": "score", "instructions": tpl["distance"]["instructions"].format(axis=a, hint=h, target=target),
                             "criteria": [tpl["distance"]["level"].format(v=f"{float(m):g}", unit=acfg.pos_unit, axis=a)
                                          for m in dcfg.magnitudes]}
        for key in ("grip", "status"):
            q[key] = {"type": "choice", "instructions": tpl[key]["instructions"],
                      "criteria": dict(tpl[key]["criteria"])}
        return q
    axes = [(f"d{a}", a, dcfg.levels, acfg.pos_unit, tpl["axis"]) for a in AXES]
    if acfg.rotation == "yaw":
        axes.append(("dyaw", "yaw", dcfg.yaw_levels, acfg.ang_unit, tpl["yaw"]))
    for key, axis, levels, unit, t in axes:
        q[key] = {"type": "score", "instructions": t["instructions"].format(axis=axis, hint=hints.get(axis, "")),
                  "criteria": [_level_text(t, float(v), axis, unit) for v in levels]}
    for key in ("grip", "status"):
        q[key] = {"type": "choice", "instructions": tpl[key]["instructions"],
                  "criteria": dict(tpl[key]["criteria"])}
    return q


def build_state(messages: list[dict], dcfg, labels: list[str] | None = None) -> Any:
    """``state`` from the chat transcript (system, user0, then assistant/user pairs). ``labels``:
    image names in send order; when they match the newest turn's images one-to-one, each image part
    is preceded by a text part "image `<name>`:" (questions refer to images by name; an unlabeled
    list left the model guessing which was which — probes 2026-10-07)."""
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
    parts: list[dict] = [{"type": "text", "text": json.dumps(obj, ensure_ascii=False)}]
    named = labels if labels and len(labels) == len(images) else [None] * len(images)
    for name, u in zip(named, images):
        if name:
            parts.append({"type": "text", "text": f"image `{name}`:"})
        parts.append({"type": "image_url", "image_url": {"url": u}})
    return parts


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


def _signed(d: dict, how: str) -> tuple[float, float]:
    """(signed direction, P(zero)) of a neg|zero|pos choice."""
    p = d.get("probabilities") if isinstance(d.get("probabilities"), dict) else None
    if how == "argmax" or p is None:
        c = reduce_choice(d, "zero")
        return {"pos": 1.0, "neg": -1.0}.get(c, 0.0), float(c == "zero")
    g = lambda k: float(p.get(k, 0.0) or 0.0)   # noqa: E731
    return g("pos") - g("neg"), g("zero")


def reduce_split(d: dict, far: dict, mags: list[float], how: str, deadband: float,
                 wide: dict | None = None) -> float:
    """direction choice (neg|zero|pos) x distance score over unsigned ``mags`` -> step.
    expected: (P(pos) - P(neg)) * expected distance; argmax: sign of the likeliest direction x the
    likeliest distance. With ``wide`` (the same question on the full frame when ``d`` was asked in a
    zoomed crop): the crop's "level" mass defers to the wide view — a target outside the crop reads
    as "level" there. A missing direction answer -> 0."""
    if not isinstance(d.get("probabilities"), dict) and not isinstance(d.get("choice"), str):
        return 0.0
    sign, p0 = _signed(d, how)
    if wide and (isinstance(wide.get("probabilities"), dict) or isinstance(wide.get("choice"), str)):
        ws, _ = _signed(wide, how)
        sign = sign + p0 * ws if how != "argmax" else (sign if sign else ws)
    if "score" not in far and _probs(far, len(mags)) is None:
        size = mags[0]                       # no distance answer: the smallest step
    else:
        size = reduce_score(far, mags, how, 0.0) if len(mags) > 1 else mags[0]
    v = sign * size
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
    if getattr(dcfg, "head", "signed") == "split":
        mags = [float(m) for m in dcfg.magnitudes]
        steps = {f"d{a}": reduce_split(answers.get(f"dir_{a}", {}), answers.get(f"far_{a}", {}), mags,
                                       dcfg.reduce, dcfg.deadband, answers.get(f"wide_{a}")) for a in AXES}
    else:
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
        self.hints: dict[str, str] = {}
        self.views: dict[str, dict | None] = {}
        self.task = ""
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

    def set_scene(self, cameras: dict, origin=None, task: str = "") -> None:
        """Called by the loop after reset: per-axis image directions for ``{hint}`` (the
        observation's cameras, as sent: ``observation.cameras`` at ``observation.size``)."""
        self.task = task
        names = list(self.cfg.observation.cameras)
        self.hints = axis_hints(cameras, names, origin, self.cfg.observation.size)
        self.views = axis_views(cameras, names, origin)
        if "fovea" in self.cfg.observation.renderers:     # ask in the zoomed crop of that camera
            self.views = {a: (dict(v, fovea=True) if v else None) for a, v in self.views.items()}

    def request_body(self, model: str, messages: list[dict], extra_body: dict | None = None) -> dict:
        from controlr.observation.renderers import image_labels
        labels = image_labels(self.cfg.observation) if self.views else None   # set_scene ran
        body: dict[str, Any] = {"model": model, "state": build_state(messages, self.cfg.decisions, labels),
                                "questions": build_questions(self.cfg.decisions, self.cfg.action, self.tpl, self.hints, self.views, self.task)}
        if extra_body:
            body.update(extra_body)
        return body

    def complete(self, model: str, messages: list[dict], *, max_tokens: int = 0,
                 temperature: float | None = None, extra_body: dict | None = None,
                 stop_when: Callable[[str], bool] | None = None,
                 timeout_s: float | None = None, max_retries: int | None = None) -> LLMResult:
        """``max_tokens`` / ``temperature`` / ``stop_when`` do not apply (no text is generated).
        A refused question is dropped and the request repeated (at most ``MAX_DROPS``)."""
        body = self.request_body(model, messages, extra_body)
        dropped: list[str] = []
        while True:
            res = self._complete(body, max_retries, timeout_s)
            m = _REFUSED.search(res.error or "")
            if not m or m.group(1) not in body["questions"] or len(dropped) >= MAX_DROPS:
                break
            dropped.append(m.group(1))
            body = {**body, "questions": {k: v for k, v in body["questions"].items() if k != m.group(1)}}
        if dropped and res.decisions is not None:
            res.decisions["refused"] = dropped
        return res

    def _complete(self, body: dict, max_retries: int | None, timeout_s: float | None) -> LLMResult:
        payload = json.dumps(body).encode()
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
            if (r.status_code in RETRY_STATUS or r.status_code >= 500) and not _REFUSED.search(msg):
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
