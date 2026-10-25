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
        from .llm import OmniChat

        _CHAT = OmniChat()
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


def _first_message(ws: Path, have: list[str], calib, flange, grip_cmd, instruction, want, b1, prior) -> list[dict]:
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
    first = _first_message(ws, have, calib, flange, grip_cmd or {}, instruction, want, b1, prior)
    st = run_agent(chat or _chat(), cfg["model"], system_prompt("robodojo"), first, RoboDojoTools(ws),
                   max_turns=cfg["max_turns"], extra_body=cfg["extra_body"], log_dir=ws)
    probs = []
    if st["answer"] is not None:
        objects, looks, probs = answer_to_output(st["answer"], b1)
        src = "agent"
    elif cfg["fallback"] == "b1" and b1 is not None:
        objects, looks, src = list(b1.get("objects") or []), list(b1.get("needs_look") or []), "b1_fallback"
    else:
        objects, looks, src = [], [], "none"
    t_ms = round(1000 * (time.time() - t0), 1)
    for o in objects:
        o.setdefault("t_ms", t_ms)
    p = pin("robodojo", cfg["model"])
    notes = (f"{VERSION} model={cfg['model']} system_md5={p['system_md5']} protocol_md5={p['protocol_md5']} result={src} "
             f"turns={st['turns']} calls={st['calls']} commands={st['commands']} stop={st['stop']} "
             f"agent_s={st['wall_s']} b1_s={round(t_b1 - t0, 1)}" + (f" b1_error={b1_err}" if b1_err else "")
             + (f" problems={probs}" if probs else ""))
    out = {"objects": objects, "regions": list((b1 or {}).get("regions") or []), "needs_look": looks,
           "last_seen": list((b1 or {}).get("last_seen") or []), "t_ms": t_ms, "version": VERSION, "notes": notes[:1500]}
    if b1 is not None and b1.get("gl"):
        out["gl"] = b1["gl"]
    _log_call(cfg["log_dir"], {"t": time.time(), "fn": "perceive", "ws": str(ws), "model": cfg["model"], "result": src,
                               "turns": st["turns"], "calls": st["calls"], "commands": st["commands"], "stop": st["stop"],
                               "tokens": st["tokens"], "call_s": st["call_s"], "agent_s": st["wall_s"], "b1_s": round(t_b1 - t0, 2),
                               "t_ms": t_ms, "problems": probs, "errors": st["errors"], "pin": {k: v for k, v in p.items() if k != "vendor"}})
    return out


def held_look_impl(frames, calib, flange, grip_cmd, arm=None, stage=None, chat=None, **kw) -> dict:
    """One call, no tools: the protocol's held_look question on the asking arm's wrist frame + the head frame."""
    from .llm import image_part
    from .prompt import system_prompt
    from .runner import parse_reply

    cfg = _cfg(kw)
    t0 = time.time()
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
        msgs = msgs + [{"role": "assistant", "content": r.text or ""},
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
