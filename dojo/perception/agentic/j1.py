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
                      a B1 object's calibrated risk >= AGENTIC_GATE_THR (flag_model.json); else B1 with the prior)
  AGENTIC_KEEP_UNMENTIONED  1 (default): B1 objects the agent did not mention (none of its objects within 3 cm) stay,
                      at half confidence
  AGENTIC_SNAP        1 (default): agent objects within 2 cm of a confident B1 object (risk < AGENTIC_GATE_THR) of the
                      same kind take B1's pose (agent: identity / labels / missing objects; B1: metric pose)
  AGENTIC_REFINE      1 (default): B1's render-and-compare refines the agent's new / moved objects (seed_refine.py)
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


def perceive(frames, calib, flange, grip_cmd, instruction, want=None, **kw) -> dict:
    frames = _b1._abs_frames(frames)
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
        if "label" not in o or "pos" not in o:
            probs.append(f"objects[{i}]: needs label and pos")
            continue
        if not o.get("category") and not o.get("asset") and b1o:  # the hint object it names (or sits on) gives its kind
            import math

            same = [b for b in b1o if b.get("label") == o["label"]]
            near = sorted(b1o, key=lambda b: math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1]))
            src = same[0] if same else (near[0] if math.hypot(near[0]["pos"][0] - o["pos"][0], near[0]["pos"][1] - o["pos"][1]) < 0.05 else None)
            if src is not None:
                o["category"] = src.get("category")
                if src.get("asset"):
                    o["asset"] = src["asset"]
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


def snap_to_b1(objects: list[dict], b1: dict | None, thr: float = 0.2, radius_m: float = 0.02) -> tuple[list[dict], int]:
    """The agent decides WHAT is where; B1 keeps the metric pose where it is confident. An agent object within
    ``radius_m`` of a B1 object of the same category, whose calibrated risk is below ``thr`` and whose asset the agent
    does not contradict, takes B1's pose (and asset, size, top_z); the agent's label stays. Agent objects farther away
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
            if o.get("asset") and b.get("asset") and o["asset"] != b["asset"]:
                continue
            d = math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1])
            if d <= radius_m and (best is None or d < best[0]):
                best = (d, j, b)
        if best is not None and risk(best[2]) < thr:
            b = best[2]
            used.add(best[1])
            if best[0] > 1e-6 or o.get("quat") != b.get("quat"):
                n += 1
            q = dict(o)
            q.update({k: b[k] for k in ("pos", "quat", "asset", "size", "top_z", "category") if b.get(k) is not None})
            q["source"] = b.get("source", "head")
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


_EPISODE = {"agent_calls": 0}


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
    return 1 / (1 + math.exp(-z))


def gate_decision(b1: dict | None, want, prior, cfg: dict) -> tuple[bool, str]:
    """AGENTIC_GATE=budget: the agent at the episode's first call (prior is None) and at most once more, when B1 misses a
    wanted object or a B1 object's calibrated risk >= the threshold; else B1 with the prior (it tracks the agent's scene)."""
    if cfg["gate"] == "always":
        return True, "always"
    if prior is None:
        _EPISODE["agent_calls"] = 0
        return True, "first_call"
    if _EPISODE["agent_calls"] >= cfg["max_per_episode"]:
        return False, "episode_budget_spent"
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


def perceive_impl(frames, calib, flange, grip_cmd, instruction, want=None, joints=None, prior=None, b1=None, chat=None,
                  **kw) -> dict:
    """Worker side. ``b1``: a precomputed B1 output (dev harness); else B1 runs here when AGENTIC_B1=1.
    ``chat``: a chat callable (tests: FakeChat)."""
    from .prompt import pin, system_prompt
    from .runner import run_agent
    from .tools_rd import RoboDojoTools, write_calib

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
                                    **{k: v for k, v in kw.items() if k in ("gripper", "task")})
        except Exception as e:  # the agent still runs without its hint
            b1, b1_err = None, f"{type(e).__name__}: {e}"
    if b1 is not None:
        import json

        (ws / "b1.json").write_text(json.dumps(b1, default=str))
    t_b1 = time.time()
    use_agent, why = gate_decision(b1, want, prior, cfg)
    track = cfg["track"] == "1" or (cfg["track"] == "auto" and cfg["gate"] == "budget")
    if prior is None:
        _EPISODE["tracker"] = None
    if not use_agent and b1 is not None:
        out = dict(b1)
        out.pop("task", None)
        tev = ""
        if track and _EPISODE.get("tracker") is not None:  # B1 tracks the agent's scene: only touched objects move
            objs, last, ev = _EPISODE["tracker"].update(list(b1.get("objects") or []), flange, grip_cmd)
            out["objects"], out["last_seen"] = objs, list(b1.get("last_seen") or []) + last
            tev = " track=" + ",".join(sorted({e["event"] for e in ev if e["event"] not in ("carried",)}))[:200]
        out["notes"] = f"{VERSION} result=b1_gated gate={why} agent_calls_this_episode={_EPISODE['agent_calls']}{tev}"
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
        objects, looks, probs = answer_to_output(st["answer"], b1)
        src = "agent"
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
                from .seed_refine import seeded_refine

                bgs = kw.get("bgs") or _b1._BG.get("bg")
                if bgs is None:
                    bgs = _b1._BG["bg"] = load_backgrounds()
                state, _ = _b1._state(flange, grip_cmd or {}, joints, kw.get("gripper"))
                objects, ref_info = seeded_refine(_b1._load_frames(frames), state, _b1._cameras(calib),
                                                  kw.get("task") or b1.get("task") or _b1.infer_task(instruction), objects,
                                                  b1.get("objects") or [], bgs)
                src = "agent+refine"
            except Exception as e:
                probs.append(f"refine: {type(e).__name__}: {e}")
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
           "last_seen": list((b1 or {}).get("last_seen") or []), "t_ms": t_ms, "version": VERSION, "notes": notes[:1500]}
    if b1 is not None and b1.get("gl"):
        out["gl"] = b1["gl"]
    if kw.get("return_stats"):  # in-process callers (record.agentic_estimate); never sent over J1
        out["_agent"] = {"result": src, "gate": why, "turns": st["turns"], "calls": st["calls"], "stop": st["stop"],
                         "agent_s": st["wall_s"], "tokens": st["tokens"], "refine": ref_info, "problems": probs,
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
    if mode in ("b1", "b1_then_agent"):
        try:
            hb = _b1._held_look_impl(frames, calib, flange, grip_cmd, arm=arm, stage=stage,
                                     **{k: v for k, v in kw.items() if k in ("joints", "gripper")})
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
        closed = [a for a in ("left", "right") if a in (grip_cmd or {}) and _b1._grip01(grip_cmd[a]) < 0.5]
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
