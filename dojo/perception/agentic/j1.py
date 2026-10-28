"""J1 interface of the agentic perceiver (robot-harness issue #3 contract, D-C6):

    perceive(frames, calib, flange, grip_cmd, instruction, want=None, joints=None, prior=None, **kw) -> dict
    held_look(frames, calib, flange, grip_cmd, arm=None, stage=None, **kw) -> dict
    close()

Same caller contract as ``dojo.perception.j1_adapter`` (stdlib + numpy on the caller's side; a persistent worker in our
venv does the work): ``J1_PERCEIVER=dojo.perception.agentic.j1:perceive``. Every call starts a FRESH agent conversation
(no memory across calls; ``prior`` is the episode's previous perceive result, passed by the driver, shown as a hint).

Worker environment (on top of j1_adapter.install_env):
  AGENTIC_MODEL       omniroute model id (default claude/claude-sonnet-5-5)
  AGENTIC_EXTRA_BODY  JSON merged into the request (e.g. {"reasoning_effort": "low"}); default none
  AGENTIC_MAX_TURNS   agent turns per perceive call (default 8)
  AGENTIC_B1          1 (default): run B1 (j1_adapter) first and pass its output as the hint; 0: no hint
  AGENTIC_FALLBACK    b1 (default): if the agent gives no answer, return B1's output; none: return no objects
  AGENTIC_LOG_DIR     per-call workspaces + transcripts + calls.jsonl (default <J1_ILIA_HOME>/run/agentic)
  AGENTIC_GATE        always (default: the agent on every call) | budget (the agent on the episode's first call, i.e.
                      prior is None, and at most AGENTIC_MAX_PER_EPISODE - 1 more times when B1 misses a wanted object or
                      a B1 object's calibrated risk >= AGENTIC_GATE_THR (flag_model.json); else B1 with the prior; with
                      the strict server's ``episode`` the count is per episode id, seeded from the server's, and a spent
                      budget without B1 is a fail-closed reply: ``episode_begin``, ``gate_decision``)
  AGENTIC_KEEP_UNMENTIONED  1 (default): B1 objects the agent did not mention (none of its objects within 3 cm) stay,
                      at half confidence
  AGENTIC_SNAP        1 (default): agent objects within 2 cm of a confident B1 object (risk < AGENTIC_GATE_THR) of the
                      same kind take B1's pose (agent: identity / labels / missing objects; B1: metric pose)
  AGENTIC_REFINE      1 (default): B1's render-and-compare refines the agent's new / moved objects (seed_refine.py)
  AGENTIC_ARBITRATE   1 (default): an agent pose that moves or turns one of B1's objects stands only when its silhouette
                      explains this frame + the extra observations (the scan) better than B1's pose by
                      AGENTIC_ARB_MARGIN (default 0.0; AGENTIC_ARB_PINNED_MARGIN, default 0.15, for objects B1 placed
                      at their config slot); else B1's pose and asset stay (seed_refine.arbitrate)
  AGENTIC_RISKY_B1    unset (default: off) | a risk, e.g. 0.2: in the arbitration, a pair whose B1 object has a calibrated
                      risk (``risk``) at or above it keeps the agent's pose without the comparison (seed_refine.arbitrate's
                      ``risky_b1``; perceiver-models bake-off: +7..+12 fixed objects on every model, 94 frame-0 scenes)
  AGENTIC_TRACK       auto (default: on with gate=budget) | 1 | 0: the episode tracker (dojo.perception.tracker) over the
                      B1-gated calls, initialised from the agent's answer
  AGENTIC_HELD        held_look source: b1_then_agent (default: B1's answer unless its confidence < 0.5 or dark_risk,
                      then one agent call), agent (always the agent), b1 (never the agent)
  AGENTIC_ENV_FILE    .env with OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY (else the environment)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from .. import j1_adapter as _b1

VERSION = "agentic-j1-1"
ROOT = _b1.ROOT
_LOCK = threading.Lock()
_PROC = None
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


# ------------------------------------------------------------------------------------------- client side

def _worker():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    import subprocess

    py = os.environ.get("J1_ILIA_PYTHON", str(ROOT / ".venv" / "bin" / "python"))
    home = Path(os.environ.get("J1_ILIA_HOME", ROOT.parent))
    log = Path(os.environ.get("AGENTIC_WORKER_LOG", str(home / "run" / "agentic_worker.log")))
    log.parent.mkdir(parents=True, exist_ok=True)
    _PROC = subprocess.Popen(["nice", "-n", "5", py, "-m", "dojo.perception.agentic.j1", "--serve"], cwd=str(ROOT),
                             env=_b1.install_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "ab"))
    return _PROC


def _call(fn: str, kwargs: dict) -> dict:
    with _LOCK:
        p = _worker()
        _b1._send(p.stdin, {"fn": fn, "kwargs": _b1._enc(kwargs)})
        out = _b1._dec(_b1._recv(p.stdout))
    if isinstance(out, dict) and out.get("_error"):
        raise RuntimeError(f"agentic worker: {out['_error']}")
    return out


def perceive(frames, calib, flange, grip_cmd, instruction, want=None, episode=None, **kw) -> dict:
    """``episode``: the strict server's opaque ``{id, perceive_n, agent_calls}`` (P1S; robot-harness's host passes it
    only to a callable that names it): the agent budget is counted per episode id, seeded from the server's count."""
    frames = _b1._abs_frames(frames)
    if episode is not None:
        kw["episode"] = episode
    if os.environ.get("J1_INPROC") == "1":
        return perceive_impl(frames, calib, flange, grip_cmd, instruction, want, **kw)
    return _call("perceive", dict(frames=frames, calib=calib, flange=flange, grip_cmd=grip_cmd, instruction=instruction,
                                  want=want, **kw))


def held_look(frames, calib, flange, grip_cmd, **kw) -> dict:
    frames = _b1._abs_frames(frames)
    if os.environ.get("J1_INPROC") == "1":
        return held_look_impl(frames, calib, flange, grip_cmd, **kw)
    return _call("held_look", dict(frames=frames, calib=calib, flange=flange, grip_cmd=grip_cmd, **kw))


def close():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _b1._send(_PROC.stdin, {"fn": "exit"})
            _PROC.wait(timeout=10)
        except Exception:
            _PROC.kill()
    _PROC = None
    _b1.close()


# ------------------------------------------------------------------------------------------- worker side

_CHAT = None
_N = [0]


def _chat():
    global _CHAT
    if _CHAT is None:
        from .llm import make_chat

        _CHAT = make_chat()
    return _CHAT


def _opt_float(v) -> float | None:
    """A float, or None for unset / empty / off / none."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "off", "none")):
        return None
    return float(v)


def arb_kwargs(cfg: dict) -> dict:
    """``seed_refine.arbitrate``'s keyword arguments from ``_cfg``'s dict."""
    return {"margin": cfg.get("arb_margin", 0.0), "pinned_margin": cfg.get("arb_pinned_margin", 0.15),
            "risky_b1": cfg.get("risky_b1")}


def _cfg(kw: dict) -> dict:
    import json

    home = Path(os.environ.get("J1_ILIA_HOME", ROOT.parent))
    eb = kw.get("extra_body", os.environ.get("AGENTIC_EXTRA_BODY"))
    return {"model": kw.get("model") or os.environ.get("AGENTIC_MODEL", "claude/claude-sonnet-5-5"),
            "extra_body": json.loads(eb) if isinstance(eb, str) and eb else (eb or None),
            "max_turns": int(kw.get("max_turns") or os.environ.get("AGENTIC_MAX_TURNS", 8)),
            "b1": str(kw.get("use_b1", os.environ.get("AGENTIC_B1", "1"))) not in ("0", "false", "False"),
            "fallback": kw.get("fallback") or os.environ.get("AGENTIC_FALLBACK", "b1"),
            "gate": kw.get("gate") or os.environ.get("AGENTIC_GATE", "always"),
            "refine": str(kw.get("refine", os.environ.get("AGENTIC_REFINE", "1"))) not in ("0", "false", "False"),
            "keep_unmentioned": str(kw.get("keep_unmentioned", os.environ.get("AGENTIC_KEEP_UNMENTIONED", "1"))) not in ("0", "false", "False"),
            "snap": str(kw.get("snap", os.environ.get("AGENTIC_SNAP", "1"))) not in ("0", "false", "False"),
            "max_per_episode": int(kw.get("max_per_episode") or os.environ.get("AGENTIC_MAX_PER_EPISODE", 2)),
            "track": str(kw.get("track", os.environ.get("AGENTIC_TRACK", "auto"))),
            "gate_thr": float(kw.get("gate_thr") or os.environ.get("AGENTIC_GATE_THR", 0.2)),
            "arbitrate": str(kw.get("arbitrate", os.environ.get("AGENTIC_ARBITRATE", "1"))) not in ("0", "false", "False"),
            "arb_margin": float(kw.get("arb_margin") or os.environ.get("AGENTIC_ARB_MARGIN", 0.0)),
            "arb_pinned_margin": float(kw.get("arb_pinned_margin") or os.environ.get("AGENTIC_ARB_PINNED_MARGIN", 0.15)),
            "risky_b1": _opt_float(kw["risky_b1"] if "risky_b1" in kw else os.environ.get("AGENTIC_RISKY_B1")),
            "log_dir": Path(kw.get("log_dir") or os.environ.get("AGENTIC_LOG_DIR", str(home / "run" / "agentic")))}


def _save_frames(ws: Path, frames: dict) -> list[str]:
    """Frames (RGB uint8 arrays or image paths) -> <cam>.png in the workspace (PIL only)."""
    import numpy as np
    from PIL import Image

    have = []
    for c in CAMS:
        v = frames.get(c)
        if v is None:
            continue
        im = Image.open(v).convert("RGB") if isinstance(v, (str, Path)) else Image.fromarray(np.ascontiguousarray(np.asarray(v, dtype=np.uint8)[..., :3]))
        im.save(ws / f"{c}.png")
        have.append(c)
    return have


def _yaw_deg(q) -> float | None:
    import math

    if not q:
        return None
    w, x, y, z = [float(v) for v in q]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, nd) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def hint_text(b1: dict | None, calib: dict, title: str) -> str:
    """The hint as a numbered table: pose, yaw, size, confidence, flags, and where pos projects in each camera."""
    from .tools_rd import project_points

    if not b1:
        return ""
    objs = b1.get("objects") or []
    lines = [f"{title}: {len(objs)} objects"]
    pxs = project_points(calib, [o["pos"] for o in objs]) if objs else {}
    for i, o in enumerate(objs):
        fl = o.get("flags") or {}
        flags = ",".join(k for k, v in fl.items() if v and k in ("occluded", "ambiguous", "near_home_grippers"))
        if fl.get("hidden_frac"):
            flags += f",hidden {fl['hidden_frac']:.2f}"
        px = "; ".join(f"{c}: {_fmt(pxs[c][i], 0)}" for c in pxs if pxs[c][i] is not None) or "in no camera"
        extra = ""
        if o.get("value") is not None or (o.get("attributes") or {}).get("digit") is not None:
            extra += f" value={o.get('value', (o.get('attributes') or {}).get('digit'))}"
        if o.get("alternative"):
            extra += f" alternative={o['alternative'].get('asset')} (margin {_fmt(o['alternative'].get('margin'), 3)})"
        lines.append(f"  [{i}] label={o.get('label')!r} category={o.get('category')} pos={_fmt(o.get('pos'))} "
                     f"yaw_deg={_fmt(_yaw_deg(o.get('quat')), 1)} size={_fmt(o.get('size'), 3)} top_z={_fmt(o.get('top_z'))} "
                     f"confidence={_fmt(o.get('confidence'), 2)} source={o.get('source')} flags=[{flags.strip(',')}]"
                     f"{extra} | pixels: {px}")
    for i, n in enumerate(b1.get("needs_look") or []):
        lines.append(f"  look [{i}] cam={n.get('cam')} reason={n.get('reason')} label={n.get('label')} target={_fmt(n.get('target'), 3)}")
    for o in b1.get("last_seen") or []:
        lines.append(f"  last_seen: label={o.get('label')!r} pos={_fmt(o.get('pos'))} (not seen in this call)")
    return "\n".join(lines)


def _save_extra_obs(ws: Path, extra_obs) -> list[tuple[str, list[str], dict]]:
    """Additional observations of the same scene: frames -> <cam>_<tag>.png, calibration -> calib_<tag>.json."""
    from PIL import Image
    import numpy as np

    from .tools_rd import write_calib

    out = []
    for ob in extra_obs or []:
        tag = str(ob["tag"])
        have = []
        for c in CAMS:
            v = (ob.get("frames") or {}).get(c)
            if v is None:
                continue
            im = Image.open(v).convert("RGB") if isinstance(v, (str, Path)) else Image.fromarray(np.ascontiguousarray(np.asarray(v, dtype=np.uint8)[..., :3]))
            im.save(ws / f"{c}_{tag}.png")
            have.append(c)
        write_calib(ws / f"calib_{tag}.json", ob["calib"], ob.get("flange"))
        out.append((tag, have, ob))
    return out


def _first_message(ws: Path, have: list[str], calib, flange, grip_cmd, instruction, want, b1, prior, extras=()) -> list[dict]:
    from .llm import image_part

    g = ", ".join(f"{a} {grip_cmd[a]}" for a in sorted(grip_cmd or {}))
    fl = "; ".join(f"{a} {_fmt([float(x) for x in flange[a]])}" for a in sorted(flange or {}))
    text = [f"Perceive call. Instruction: {instruction!r}", f"Wanted labels: {list(want) if want else 'not given'}",
            f"Commanded grippers (1 open, 0 closed): {g or 'unknown'}", f"Measured flange poses (world xyz, quat wxyz): {fl or 'unknown'}",
            f"Frames attached below in this order: {', '.join(c + '.png' for c in have)}. Calibration: calib0.json."]
    if b1:
        text.append(hint_text(b1, calib, "Hint (B1, an automatic perceiver, this observation)"))
    if prior:
        text.append(hint_text(prior, calib, "Previous perceive result of this episode (an earlier observation; objects may have moved since)"))
    parts = [{"type": "text", "text": "\n".join(text)}]
    for c in have:
        parts.append({"type": "text", "text": f"{c}.png:"})
        parts.append(image_part((ws / f"{c}.png").read_bytes()))
    for tag, hv, ob in extras:
        fl = "; ".join(f"{a} {_fmt([float(x) for x in ob['flange'][a]])}" for a in sorted(ob.get("flange") or {}))
        parts.append({"type": "text", "text": f"Observation '{tag}' of the same scene ({ob.get('note', 'the arms moved; the objects did not')}). "
                                              f"Measured flange poses: {fl or 'unknown'}. Calibration: calib_{tag}.json. Frames: "
                                              + ", ".join(f"{c}_{tag}.png" for c in hv)})
        for c in hv:
            parts.append({"type": "text", "text": f"{c}_{tag}.png:"})
            parts.append(image_part((ws / f"{c}_{tag}.png").read_bytes()))
    parts.append({"type": "text", "text": "Reply with ONE JSON object, starting with '{' (plain text, no markup)."})
    return parts


def _quat_from_yaw(yaw_deg: float) -> list[float]:
    import math

    h = math.radians(float(yaw_deg)) / 2
    return [round(math.cos(h), 6), 0.0, 0.0, round(math.sin(h), 6)]


OBJ_FIELDS = ("label", "category", "asset", "pos", "quat", "size", "top_z", "value", "confidence", "colour")


def answer_to_output(ans: dict, b1: dict | None) -> tuple[list[dict], list[dict], list[str]]:
    """Agent answer -> (objects, needs_look, problems) in the J1 schema; `keep` takes B1's object / look i."""
    objs, looks, probs = [], [], []
    b1o = (b1 or {}).get("objects") or []
    b1l = (b1 or {}).get("needs_look") or []
    for i, a in enumerate(ans.get("objects") or []):
        if not isinstance(a, dict):
            probs.append(f"objects[{i}] not an object")
            continue
        o = {}
        if a.get("keep") is not None:
            try:
                o = {k: v for k, v in b1o[int(a["keep"])].items() if k in OBJ_FIELDS + ("source",)}
                o["source"] = o.get("source") or "head"
            except (ValueError, IndexError, TypeError):
                probs.append(f"objects[{i}]: keep {a.get('keep')!r} is not a hint index")
                continue
        for k in OBJ_FIELDS:
            if a.get(k) is not None:
                o[k] = a[k]
        if a.get("yaw_deg") is not None:
            try:
                o["quat"] = _quat_from_yaw(a["yaw_deg"])
            except (TypeError, ValueError):
                probs.append(f"objects[{i}]: bad yaw_deg")
        if a.get("keep") is None:
            o["source"] = "head"
            o["_agent_point"] = True  # the agent's own point (a head-plane hit), not one of B1's poses
        if "label" not in o or "pos" not in o:
            probs.append(f"objects[{i}]: needs label and pos")
            continue
        known = {b.get("category") for b in b1o if b.get("category")}
        if o.get("category") and known and o["category"] not in known and not o.get("asset"):
            o["agent_category"] = o.pop("category")  # free text ("board"): the hint's vocabulary decides the kind
        if not o.get("category") and not o.get("asset") and b1o:  # the hint object it names (or sits on) gives its kind
            import math

            same = [b for b in b1o if b.get("label") == o["label"]]
            near = sorted(b1o, key=lambda b: math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1]))
            src = same[0] if same else (near[0] if math.hypot(near[0]["pos"][0] - o["pos"][0], near[0]["pos"][1] - o["pos"][1]) < 0.05 else None)
            if src is not None:
                o["category"] = src.get("category")
                if src.get("asset"):
                    o["asset"] = src["asset"]
            elif len(known) == 1:  # one kind in the scene's vocabulary: that kind, identity left to H2
                o["category"] = next(iter(known))
        o.setdefault("confidence", 0.8)
        objs.append(o)
    for i, n in enumerate(ans.get("needs_look") or []):
        if isinstance(n, dict) and n.get("keep") is not None:
            try:
                looks.append(dict(b1l[int(n["keep"])]))
            except (ValueError, IndexError, TypeError):
                probs.append(f"needs_look[{i}]: keep {n.get('keep')!r} is not a hint index")
        elif isinstance(n, dict):
            looks.append({k: n[k] for k in ("cam", "reason", "label", "target") if k in n})
    return objs, looks, probs


def _cat_of(o: dict) -> str | None:
    a = str(o.get("asset") or "").split("/")
    return a[1] if len(a) == 3 else o.get("category")


def snap_to_b1(objects: list[dict], b1: dict | None, thr: float = 0.2, radius_m: float = 0.02,
               trust_b1_identity: bool = True, agree_m: float = 0.01) -> tuple[list[dict], int]:
    """The agent decides WHAT is where; B1 keeps the metric pose where it is confident. An agent object within
    ``radius_m`` of a B1 object of the same category whose calibrated risk is below ``thr`` takes B1's pose and asset
    (``trust_b1_identity``; else only when the agent does not name another asset); the agent's label stays. Within
    ``agree_m`` the agent agrees with B1 whatever B1's risk: B1's fitted pose is the better metric estimate (the
    agent's own points are head-plane hits; tower boards are right but read at confidence 0.36). Agent objects farther away
    (a correction), on risky B1 objects, or with another asset keep the agent's reading (H2 refines them).
    Bridge-isr's first 5.1 episode: the merged agent points were 2.1 mm median vs B1's 0.4 mm on a clean scene."""
    import math

    b1o = (b1 or {}).get("objects") or []
    out, n, used = [], 0, set()
    for o in objects:
        best = None
        for j, b in enumerate(b1o):
            if j in used or _cat_of(b) != _cat_of(o):
                continue
            if o.get("asset") and b.get("asset") and o["asset"] != b["asset"] and not trust_b1_identity:
                continue
            d = math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1])
            if d <= radius_m and (best is None or d < best[0]):
                best = (d, j, b)
        if best is not None and (best[0] <= agree_m or risk(best[2]) < thr):
            b = best[2]
            used.add(best[1])
            if best[0] > 1e-6 or o.get("quat") != b.get("quat"):
                n += 1
            q = dict(o)
            q.update({k: b[k] for k in ("pos", "quat", "asset", "size", "top_z", "category") if b.get(k) is not None})
            q["source"] = b.get("source", "head")
            q.pop("_agent_point", None)
            out.append(q)
        else:
            out.append(o)
    return out, n


def add_unmentioned(objects: list[dict], b1: dict | None, radius_m: float = 0.03) -> tuple[list[dict], int]:
    """B1 objects the agent neither kept nor replaced stay in the output at half confidence: on our D0 hard set the
    agents left out objects they could not confirm, and most of those were right. A B1 object counts as mentioned when
    an agent object carries its label or lies within ``radius_m``; and none is re-added once the agent already reports
    as many objects of that category as B1 did (a moved object must not come back as a duplicate)."""
    import math

    out, n = list(objects), 0
    labels = {str(x.get("label")) for x in objects}
    b1o = (b1 or {}).get("objects") or []
    want = {}
    for o in b1o:
        want[_cat_of(o)] = want.get(_cat_of(o), 0) + 1
    have = {}
    for x in objects:  # the agent's objects per category (free-text ones: the nearest B1 object's category)
        c = _cat_of(x)
        if c not in want and b1o:
            c = _cat_of(min(b1o, key=lambda o: math.hypot(o["pos"][0] - x["pos"][0], o["pos"][1] - x["pos"][1])))
        have[c] = have.get(c, 0) + 1
    for o in b1o:
        c = _cat_of(o)
        if str(o.get("label")) in labels or have.get(c, 0) >= want.get(c, 0):
            continue
        if all(math.hypot(o["pos"][0] - x["pos"][0], o["pos"][1] - x["pos"][1]) > radius_m for x in out):
            out.append(dict({k: v for k, v in o.items() if k in OBJ_FIELDS + ("source",)},
                            confidence=round(0.5 * float(o.get("confidence", 0.5)), 3)))
            have[c] = have.get(c, 0) + 1
            n += 1
    return out, n


def _validate(out: dict) -> tuple[dict, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    try:
        import j1_schema  # vendored, byte-identical
    finally:
        sys.path.pop(0)
    clean, rep = j1_schema.validate_perceive(out)
    return clean, rep


def _log_call(log_dir: Path, row: dict) -> None:
    import json

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "calls.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


_EPISODE = {"agent_calls": 0, "id": None}


def episode_begin(episode=None, prior=None) -> None:
    """The worker's episode state (agent count, tracker). With the server's opaque ``episode`` (P1S I1-1): a new
    ``id`` starts an episode with the agent count seeded from the server's ``agent_calls`` (the server outlives a
    worker restart after a timeout); the same id keeps the larger of the two counts. Without one (older servers,
    in-process callers): a call without ``prior`` starts the episode, as before."""
    eid = episode.get("id") if isinstance(episode, dict) else None
    if eid is not None:
        try:
            n = max(0, int(episode.get("agent_calls") or 0))
        except (TypeError, ValueError):
            n = 0
        if _EPISODE.get("id") != eid:
            _EPISODE.update(id=eid, agent_calls=n, tracker=None)
        else:
            _EPISODE["agent_calls"] = max(int(_EPISODE.get("agent_calls") or 0), n)
    elif prior is None:
        _EPISODE.update(id=None, agent_calls=0)


def risk(o: dict, model: dict | None = None) -> float:
    """Calibrated probability that a B1 object's pose is outside the task tolerance (flag_model.json, fitted on our D0)."""
    import json
    import math

    if model is None:
        model = json.loads((Path(__file__).resolve().parent / "flag_model.json").read_text())
    fl = o.get("flags") or {}
    conf = o.get("confidence")
    alt = o.get("alternative") or {}
    x = [1.0, 1 - (conf if conf is not None else 0.0), o.get("residual") if o.get("residual") is not None else 1.0,
         fl.get("hidden_frac") or 0.0, fl.get("fg_resid_near") or 0.0, float(bool(fl.get("occluded"))),
         float(bool(fl.get("ambiguous"))), float(o.get("source") in ("fused", "refined")),
         float(alt.get("margin") is not None and alt["margin"] < 0.05)]
    z = sum(a * b for a, b in zip(model["weights"], x))
    r = 1 / (1 + math.exp(-z))
    floor = _opt_float(os.environ.get("AGENTIC_DOUBT_RISK"))
    if floor is not None and fl.get("b1_doubt"):  # perception.doubt flagged it (appearance / height): the gate fires,
        r = max(r, floor)                         # and with risky_b1 <= floor the agent's reading of it stands
    return r


def gate_decision(b1: dict | None, want, prior, cfg: dict, episode=None) -> tuple[bool, str]:
    """AGENTIC_GATE=budget: the agent at the episode's first call and at most ``max_per_episode`` times in the episode,
    when B1 misses a wanted object or a B1 object's calibrated risk >= the threshold; else B1 with the prior (it tracks
    the agent's scene). The episode and its count: ``episode_begin`` (the server's ``episode``: its first call is
    ``perceive_n`` 1; without one, a call without ``prior``). A spent budget is spent whatever else holds."""
    if cfg["gate"] == "always":
        return True, "always"
    episode_begin(episode, prior)
    if _EPISODE["agent_calls"] >= cfg["max_per_episode"]:
        return False, "episode_budget_spent"
    pn = episode.get("perceive_n") if isinstance(episode, dict) and episode.get("id") is not None else None
    if (int(pn) <= 1) if pn is not None else prior is None:
        return True, "first_call"
    if b1 is None:
        return True, "no_b1"
    wm = b1.get("want_match") or {}
    missing = [w for w, lab in wm.items() if lab is None]
    if missing:
        return True, f"missing:{missing}"
    hi = [(o.get("label"), round(risk(o), 2)) for o in b1.get("objects") or [] if risk(o) >= cfg["gate_thr"]]
    if hi:
        return True, f"risk:{hi[:4]}"
    return False, "b1_confident"


def _extra_views(extra_obs, grip_cmd) -> list[tuple[dict, dict, dict]]:
    """``extra_obs`` (other observations of the same scene, e.g. the scan) -> [(images, state, cams)]; the robot mask
    from IK on each observation's measured flange."""
    out = []
    for ob in extra_obs or []:
        try:
            st_, _ = _b1._state(ob.get("flange") or {}, grip_cmd or {}, ob.get("joints"), None)
            out.append((_b1._load_frames(ob["frames"]), st_, _b1._cameras(ob["calib"])))
        except Exception:  # noqa: BLE001 (an unreadable extra view only weakens the check)
            continue
    return out


def executable_looks(looks: list[dict], b1_looks: list[dict], objects: list[dict]) -> tuple[list[dict], int]:
    """The agent's look requests made executable: each needs a wrist camera and a ``flange_pose`` the J1 driver can
    move to (Mikhail's D-C7 / D-C8: 29 agent looks without one were never executed; push L2 planned from a 140 deg
    heading error B1-only's own look fixed). A look for a label B1 also wants to look at takes B1's pose; else one is
    computed for its target (given, or the labelled object's position) like B1's own (``look.default_look``); a look
    with neither falls back to B1's first look. Returns (looks, how many got a pose)."""
    out, n = [], 0
    by_label = {lk.get("label"): lk for lk in b1_looks or [] if lk.get("flange_pose") and lk.get("label")}
    pos_of = {o.get("label"): o.get("pos") for o in objects or [] if o.get("label") and o.get("pos")}
    for lk in looks or []:
        if lk.get("flange_pose"):
            out.append(lk)
            continue
        new = dict(lk)
        src = by_label.get(lk.get("label"))
        if src is not None:
            new.update({k: src[k] for k in ("cam", "target", "flange_pose") if k in src})
        else:
            tgt = lk.get("target") or pos_of.get(lk.get("label"))
            if tgt is not None and len(tgt) >= 3:
                import numpy as np

                from ..look import default_look, flange_for_camera

                cam = str(lk.get("cam") or "")
                arm = "left" if "left" in cam else "right" if "right" in cam else ("left" if float(tgt[0]) < 0 else "right")
                T = default_look(np.asarray(tgt[:3], float), arm)
                new.update({"cam": f"cam_{arm}_wrist", "target": [round(float(v), 4) for v in tgt[:3]],
                            "flange_pose": np.round(flange_for_camera(T), 5).tolist()})
            elif b1_looks and any(b.get("flange_pose") for b in b1_looks):
                src = next(b for b in b1_looks if b.get("flange_pose"))
                new.update({k: src[k] for k in ("cam", "target", "flange_pose", "label") if k in src})
                new["reason"] = f"{lk.get('reason') or 'agent'} (B1's look)"
        if new.get("flange_pose"):
            n += 1
        out.append(new)
    return out, n


def finish_answer(answer: dict, b1: dict | None, frames, calib, flange, grip_cmd, cfg: dict, joints=None,
                  instruction: str = "", **kw) -> tuple[list[dict], list[dict], str, list[str], dict | None]:
    """The agent's answer -> (objects, needs_look, source, problems, refine info): snap to B1, keep B1's unmentioned
    objects, H2 (B1's render-and-compare from the agent's points), then the arbitration against B1 (``cfg["arbitrate"]``,
    ``seed_refine.arbitrate``): an agent pose that MOVES or TURNS one of B1's objects stands only when it explains the
    views (this frame + ``kw["extra_obs"]``, e.g. the scan) better than B1's pose. No LLM here: replays use it as is."""
    objects, looks, probs = answer_to_output(answer, b1)
    src, ref_info = "agent", None
    if cfg["snap"] and b1 is not None:
        objects, n_snap = snap_to_b1(objects, b1, thr=cfg["gate_thr"])
        if n_snap:
            probs.append(f"snapped_to_b1={n_snap}")
    if cfg["keep_unmentioned"] and b1 is not None:
        objects, n_kept = add_unmentioned(objects, b1)
        if n_kept:
            probs.append(f"b1_unmentioned_kept={n_kept}")
    if cfg["refine"] and b1 is not None and objects:
        try:  # H2: B1's render-and-compare from the agent's points (new / moved / orientation-less objects)
            from ..segment import load_backgrounds
            from .seed_refine import arbitrate, build_views, seeded_refine

            bgs = kw.get("bgs") or _b1._BG.get("bg")
            if bgs is None:
                bgs = _b1._BG["bg"] = load_backgrounds()
            state, _ = _b1._state(flange, grip_cmd or {}, joints, kw.get("gripper"))
            imgs, cams = _b1._load_frames(frames), _b1._cameras(calib)
            task = kw.get("task") or b1.get("task") or _b1.infer_task(instruction)
            objects, ref_info = seeded_refine(imgs, state, cams, task, objects, b1.get("objects") or [], bgs)
            src = "agent+refine"
            if cfg.get("arbitrate"):
                views = build_views(imgs, state, cams, bgs, _extra_views(kw.get("extra_obs"), grip_cmd))
                objects, arb = arbitrate(views, objects, b1.get("objects") or [], **arb_kwargs(cfg))
                ref_info = list(ref_info or []) + [dict(a, arbitration=True) for a in arb]
                n_kept = sum(1 for a in arb if a.get("kept") == "b1")
                if n_kept:
                    probs.append(f"arbitration_kept_b1={n_kept}")
        except Exception as e:
            probs.append(f"refine: {type(e).__name__}: {e}")
    if b1 is not None:
        from .seed_refine import keep_read_yaws

        n_yaw = keep_read_yaws(objects, b1.get("objects") or [])
        if n_yaw:
            probs.append(f"b1_read_yaw_kept={n_yaw}")
    task_ = kw.get("task") or (b1 or {}).get("task") or _b1.infer_task(instruction)
    if task_:
        try:
            from .seed_refine import settle_slot_children

            n_slot = settle_slot_children(objects, task_)
            if n_slot:
                probs.append(f"slot_children_settled={n_slot}")
        except Exception as e:  # noqa: BLE001 (catalogue gaps: the agent's poses stand)
            probs.append(f"slot_settle: {type(e).__name__}: {e}")
    if not kw.get("keep_markers"):  # replays arbitrate the same H2 output themselves
        for o in objects:
            o.pop("_agent_point", None)
    try:
        looks, n_pose = executable_looks(looks, (b1 or {}).get("needs_look") or [], objects)
        if n_pose:
            probs.append(f"look_poses_added={n_pose}")
    except Exception as e:  # noqa: BLE001 (a look suggestion must never cost the scene)
        probs.append(f"look_poses: {type(e).__name__}: {e}")
    return objects, looks, src, probs, ref_info


def perceive_impl(frames, calib, flange, grip_cmd, instruction, want=None, joints=None, prior=None, b1=None, chat=None,
                  episode=None, **kw) -> dict:
    """Worker side. ``b1``: a precomputed B1 output (dev harness); else B1 runs here when AGENTIC_B1=1.
    ``chat``: a chat callable (tests: FakeChat). ``episode``: the server's ``{id, perceive_n, agent_calls}``
    (``episode_begin``). Every output says ``agent_call`` true / false (the server counts by it, P1S I1-1); a spent
    budget without a B1 output is a fail-closed reply (no objects, ``error``, ``result=budget_spent_no_b1``), never an
    agent call."""
    from .prompt import pin, system_prompt
    from .runner import run_agent
    from .tools_rd import RoboDojoTools, write_calib

    _b1.require_data()  # J1_REQUIRE_DATA=1: refuse here, not as a B1 failure the agent would run without
    cfg = _cfg(kw)
    t0 = time.time()
    _N[0] += 1
    ws = cfg["log_dir"] / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{_N[0]:04d}"
    ws.mkdir(parents=True, exist_ok=True)
    have = _save_frames(ws, frames)
    write_calib(ws / "calib0.json", calib, flange)
    b1_err = None
    if b1 is None and cfg["b1"]:
        try:
            b1 = _b1._perceive_impl(frames, calib, flange, grip_cmd, instruction, want, joints=joints, prior=prior,
                                    episode=episode, **{k: v for k, v in kw.items() if k in ("gripper", "task")})
        except Exception as e:  # the agent still runs without its hint
            b1, b1_err = None, f"{type(e).__name__}: {e}"
    if b1 is not None:
        import json

        (ws / "b1.json").write_text(json.dumps(b1, default=str))
    t_b1 = time.time()
    use_agent, why = gate_decision(b1, want, prior, cfg, episode)
    track = cfg["track"] == "1" or (cfg["track"] == "auto" and cfg["gate"] == "budget")
    if prior is None:
        _EPISODE["tracker"] = None
    if not use_agent and b1 is None:  # budget spent and no B1 output: fail closed, never an agent call (P1S I1-1)
        t_ms = round(1000 * (time.time() - t0), 1)
        _log_call(cfg["log_dir"], {"t": time.time(), "fn": "perceive", "ws": str(ws), "model": cfg["model"],
                                   "result": "budget_spent_no_b1", "gate": why, "calls": 0, "b1_error": b1_err})
        return {"objects": [], "regions": [], "needs_look": [], "last_seen": [], "t_ms": t_ms, "version": VERSION,
                "agent_call": False, "error": f"agent budget spent and no B1 output ({b1_err or 'B1 off'})"[:300],
                "notes": (f"{VERSION} result=budget_spent_no_b1 gate={why} agent_calls_this_episode="
                          f"{_EPISODE['agent_calls']}" + (f" b1_error={b1_err}" if b1_err else ""))[:1500]}
    if not use_agent and b1 is not None:
        out = dict(b1)
        out.pop("task", None)
        tev = ""
        if track and _EPISODE.get("tracker") is not None:  # B1 tracks the agent's scene: only touched objects move
            objs, last, ev = _EPISODE["tracker"].update(list(b1.get("objects") or []), flange, grip_cmd)
            out["objects"], out["last_seen"] = objs, list(b1.get("last_seen") or []) + last
            tev = " track=" + ",".join(sorted({e["event"] for e in ev if e["event"] not in ("carried",)}))[:200]
        elif track and _EPISODE.get("id") is not None and _EPISODE["agent_calls"] > 0:
            tev = " tracker=lost"  # the episode had an agent scene this worker never saw (a restart): B1 untracked (G2-9)
        out["notes"] = f"{VERSION} result=b1_gated gate={why} agent_calls_this_episode={_EPISODE['agent_calls']}{tev}"
        out["agent_call"] = False
        _log_call(cfg["log_dir"], {"t": time.time(), "fn": "perceive", "ws": str(ws), "model": cfg["model"], "result": "b1_gated",
                                   "gate": why, "calls": 0, "b1_s": round(t_b1 - t0, 2)})
        return out
    _EPISODE["agent_calls"] += 1
    extras = _save_extra_obs(ws, kw.get("extra_obs"))
    kind = "robodojo_multi" if extras else "robodojo"
    first = _first_message(ws, have, calib, flange, grip_cmd or {}, instruction, want, b1, prior, extras)
    st = run_agent(chat or _chat(), cfg["model"], system_prompt(kind), first, RoboDojoTools(ws),
                   max_turns=cfg["max_turns"], extra_body=cfg["extra_body"], log_dir=ws)
    probs = []
    ref_info = None
    if st["answer"] is not None:
        objects, looks, src, probs, ref_info = finish_answer(st["answer"], b1, frames, calib, flange, grip_cmd, cfg,
                                                             joints=joints, instruction=instruction, **kw)
    elif cfg["fallback"] == "b1" and b1 is not None:
        objects, looks, src = list(b1.get("objects") or []), list(b1.get("needs_look") or []), "b1_fallback"
    else:
        objects, looks, src = [], [], "none"
    if track:  # the agent's answer is the episode's state from here on (B1 tracks it)
        from ..tracker import Tracker

        _EPISODE["tracker"] = Tracker()
        _EPISODE["tracker"].update(objects, flange, grip_cmd)
    t_ms = round(1000 * (time.time() - t0), 1)
    for o in objects:
        o.setdefault("t_ms", t_ms)
    p = pin(kind, cfg["model"])
    notes = (f"{VERSION} model={cfg['model']} system_md5={p['system_md5']} protocol_md5={p['protocol_md5']} result={src} "
             f"gate={why} turns={st['turns']} calls={st['calls']} commands={st['commands']} stop={st['stop']} "
             f"agent_s={st['wall_s']} b1_s={round(t_b1 - t0, 1)}" + (f" b1_error={b1_err}" if b1_err else "")
             + (f" problems={probs}" if probs else ""))
    out = {"objects": objects, "regions": list((b1 or {}).get("regions") or []), "needs_look": looks,
           "last_seen": list((b1 or {}).get("last_seen") or []), "t_ms": t_ms, "version": VERSION, "notes": notes[:1500],
           "agent_call": True, "data": (b1 or {}).get("data") or _b1.data_identity()}
    if b1 is not None and b1.get("gl"):
        out["gl"] = b1["gl"]
    if kw.get("return_stats"):  # in-process callers (record.agentic_estimate); never sent over J1
        out["_agent"] = {"result": src, "gate": why, "turns": st["turns"], "calls": st["calls"], "stop": st["stop"],
                         "agent_s": st["wall_s"], "tokens": st["tokens"], "call_s": st["call_s"], "commands": st["commands"],
                         "errors": st["errors"], "refine": ref_info, "problems": probs,
                         "system_md5": p["system_md5"], "model": cfg["model"], "ws": str(ws)}
    _log_call(cfg["log_dir"], {"t": time.time(), "fn": "perceive", "ws": str(ws), "model": cfg["model"], "result": src,
                               "turns": st["turns"], "calls": st["calls"], "commands": st["commands"], "stop": st["stop"],
                               "tokens": st["tokens"], "call_s": st["call_s"], "agent_s": st["wall_s"], "b1_s": round(t_b1 - t0, 2),
                               "t_ms": t_ms, "problems": probs, "errors": st["errors"], "refine": ref_info, "pin": {k: v for k, v in p.items() if k != "vendor"}})
    return out


def held_look_impl(frames, calib, flange, grip_cmd, arm=None, stage=None, chat=None, **kw) -> dict:
    """One call, no tools: the protocol's held_look question on the asking arm's wrist frame + the head frame."""
    from .llm import image_part
    from .prompt import system_prompt
    from .runner import parse_reply

    cfg = _cfg(kw)
    t0 = time.time()
    mode = kw.get("held_mode") or os.environ.get("AGENTIC_HELD", "b1_then_agent")
    hb: dict = {}
    if mode in ("b1", "b1_then_agent"):
        try:
            hb = _b1._held_look_impl(frames, calib, flange, grip_cmd, arm=arm, stage=stage,
                                     **{k: v for k, v in kw.items() if k in ("joints", "gripper", "candidates")})
        except Exception as e:
            hb = {"_error": f"{type(e).__name__}: {e}"}
        a_ = arm or next((a for a in ("left", "right") if a in hb), None)
        sub = hb.get(a_) if a_ else None
        conf = (sub or hb).get("confidence", 0.0) if isinstance(sub or hb, dict) else 0.0
        dark = bool((sub or {}).get("dark_risk")) if isinstance(sub, dict) else False
        if mode == "b1" or ("_error" not in hb and conf >= 0.5 and not dark):
            hb = dict(hb, note=(hb.get("note") or "") + " source=B1")
            _log_call(cfg["log_dir"], {"t": time.time(), "fn": "held_look", "source": "b1", "arm": a_, "stage": stage,
                                       "held": hb.get("held"), "confidence": conf, "calls": 0})
            return hb
    if arm is None:  # the arm whose gripper is commanded closed
        closed = [a for a in ("left", "right") if a in (grip_cmd or {}) and _b1._grip01(grip_cmd[a]) < _b1.HELD_OPEN_CMD]
        arm = closed[0] if closed else "left"
    _N[0] += 1
    ws = cfg["log_dir"] / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{_N[0]:04d}-held"
    ws.mkdir(parents=True, exist_ok=True)
    have = _save_frames(ws, frames)
    q = f"Is an object held between the {arm} gripper's fingers (wrist image)? Answer yes or no."
    parts = [{"type": "text", "text": f"held_look question{f' (stage {stage})' if stage else ''}: {q}"}]
    for c in (f"cam_{arm}_wrist", "cam_head"):
        if c in have:
            parts += [{"type": "text", "text": f"s_{c}.png:"}, image_part((ws / f"{c}.png").read_bytes())]
    msgs = [{"role": "system", "content": system_prompt("held")}, {"role": "user", "content": parts}]
    ch = chat or _chat()
    held, note, calls = None, "", 0
    for _ in range(2):
        r = ch(cfg["model"], msgs, max_tokens=4000, extra_body=cfg["extra_body"])
        calls += 1
        d = parse_reply(r.text or "", keys=("held",))
        if isinstance(d, dict) and str(d.get("held", "")).lower() in ("yes", "no", "true", "false"):
            held = str(d["held"]).lower() in ("yes", "true")
            note = str(d.get("note", ""))[:300]
            break
        msgs = msgs + [{"role": "assistant", "content": (r.text or "").strip() or "(empty reply)"},
                       {"role": "user", "content": [{"type": "text", "text": 'Reply with exactly {"held": "yes" | "no", "note": "..."}.'}]}]
    t_ms = round(1000 * (time.time() - t0), 1)
    if held is None:
        out = {"held": False, "confidence": 0.0, "note": "agent gave no yes/no answer", "t_ms": t_ms, "version": VERSION}
    else:
        out = {"held": held, "confidence": 0.9, "note": note, "t_ms": t_ms, "version": VERSION}
    out[arm] = {"held": out["held"], "confidence": out["confidence"]}
    b1_arm = hb.get(arm) if isinstance(hb.get(arm), dict) else {}
    if out["held"] and b1_arm.get("what") is not None:  # identity of what is there: B1's colour ranking
        out[arm].update({k: b1_arm[k] for k in ("what", "scored", "what_region") if k in b1_arm})
        asked = (kw.get("candidates") or [{}])[0]
        from dojo.perception import identity
        out[arm]["asked_held"] = identity.asked_verdict(b1_arm["what"], asked, True) if asked.get("key") else None
    elif kw.get("candidates"):
        out[arm].update({"what": None, "asked_held": False if held is False else None, "scored": []})
    _log_call(cfg["log_dir"], {"t": time.time(), "fn": "held_look", "ws": str(ws), "model": cfg["model"], "arm": arm,
                               "stage": stage, "held": out["held"], "calls": calls, "t_ms": t_ms})
    return out


def _serve():
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_in = sys.stdin.buffer
    fns = {"perceive": perceive_impl, "held_look": held_look_impl}
    while True:
        try:
            req = _b1._recv(proto_in)
        except EOFError:
            break
        if req.get("fn") == "exit":
            break
        try:
            res = fns[req["fn"]](**_b1._dec(req["kwargs"]))
        except Exception as e:
            import traceback

            traceback.print_exc()
            res = {"_error": f"{type(e).__name__}: {e}"}
        _b1._send(proto_out, _b1._enc(res))


if __name__ == "__main__":
    if "--serve" in sys.argv:
        _serve()
