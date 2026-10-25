"""j1_schema.py: output schema of the J1 `perceive` adapter (issue #3) and its validator. Stdlib only, so the strict
J1 server (policy_venv) and the perceiver host's --once mode (the adapter's own python) import the same rules.

validate_perceive(out) -> (clean, report)
  clean   {"objects": [...], "regions": [...], "needs_look": [...], "t_ms": float|None, + optional metadata}
  report  {"ok": bool, "rejected": [key paths], "stripped": [key paths], "dropped": [why], "error": str|None}
A forbidden key or value anywhere in the output (eval state, truth, instance names, layout or trajectory files) rejects
the WHOLE output: ok False, clean None, and only the key paths are reported, never the values. Unknown keys are stripped
and listed; an object missing a required field, or with an out-of-range number, is dropped and listed.

Object (one per detected thing):
  label       str, required (from the adapter's vocabulary; e.g. "block", "T", "T pad", "button red", "number card")
  pos         [x, y, z] world m, required (the asset root, the same point RoboDojo's checker compares)
  confidence  0..1, required
  category, asset            str, optional (catalogue names; never a file path or a simulator instance name)
  quat        [w, x, y, z] world, optional (normalised here; |q| must be 0.9..1.1)
  size        [l, w, h] m, or radius + height m, optional
  top_z       m, optional (top face height; J1 addition, see coord/J1-RUNBOOK.md open questions)
  value       int | str, optional (a digit read from a number card; J1 addition). Ilia's adapter sends the digit as
              attributes.digit (int 1-9); the validator copies it into value and strips the rest of attributes
  source      "head" | "fused" | "refined" | "prior", optional ("prior" = an unseen static fixture placed from its
              earlier reading or config; a normal reading for the driver)
  t_ms        number, optional
last_seen (optional, Ilia's adapter; shape to be confirmed): a list of objects in the object schema above (label and
  pos required here) for movable objects NOT seen in this call, at their last good pose (possibly picked; never carried).
  The driver uses them for planning only and never as evidence of a hold.
Region (plate / pad / basket footprint, optional): label, category, pos, quat, size, corners, lo, hi, confidence, source.
needs_look (optional): cam (cam_head | cam_left_wrist | cam_right_wrist), reason, label, target [x,y,z],
  tcp [x,y,z] + quat [w,x,y,z] | yaw_deg, flange [x,y,z,qw,qx,qy,qz]. The J1 driver executes only entries that carry
  a pose (tcp or flange); the rest are logged.
held_look reply: {"held": bool | "yes" | "no", "confidence": 0..1 (optional), "note": str (optional), "t_ms" (optional)},
  or per arm (Ilia's adapter): {"left" | "right": {held, confidence, fg_frac, dark_risk}, "held", "confidence"}; the
  asking arm's entry wins over the top-level fields.
Adapter metadata (Ilia's j1_adapter, commit 868fc4a6): top-level "task" (the adapter's own instruction -> task-config
routing, infer_task) is NOT treated as eval state: it is removed from the clean output and reported as
report["adapter_task"] for the log only. "want_match", "gl", "error", "t_estimate_ms" pass through as metadata;
needs_look "flange_pose" is read as "flange"; object "colour" is kept (button colour); regions keep polygon_xy / z_top.
"""
import math
import re

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SOURCES = ("head", "fused", "refined", "prior")   # "prior": the adapter's unseen static fixtures (Ilia, 25 Oct)
FORBIDDEN_KEY = re.compile(
    r"(^|_)(gt|truth|true|oracle|inst|instance|instances|inst_name|prim|prim_path|usd|layout|layout_id|layouts|eval|"
    r"traj|trajectory|seed|task|task_name|privileged|objects_gt|dev)($|_)", re.I)
FORBIDDEN_VALUE = re.compile(r"Eval_Layout|Assets/Traj|/World/|_result\.json|dev\.jsonl", re.I)
TOP_ALLOWED = ("objects", "regions", "needs_look", "t_ms", "version", "deterministic", "notes", "want_match", "gl", "error",
               "t_estimate_ms", "last_seen")
OBJ_ALLOWED = ("label", "category", "asset", "pos", "quat", "size", "radius", "height", "top_z", "value", "confidence",
               "source", "t_ms", "colour")
REG_ALLOWED = ("label", "category", "pos", "quat", "size", "corners", "lo", "hi", "confidence", "source", "polygon_xy", "z_top")
LOOK_ALLOWED = ("cam", "reason", "label", "target", "tcp", "quat", "yaw_deg", "flange", "flange_pose")
HELD_ALLOWED = ("held", "confidence", "note", "t_ms", "fg_frac", "dark_risk", "version", "left", "right")
WORKSPACE = ((-1.2, 1.2), (-1.2, 1.2), (0.40, 1.80))
MAX_OBJECTS, MAX_REGIONS, MAX_LOOKS = 64, 32, 8


def forbidden_paths(x, path=""):
    """Key paths whose key looks like eval state, plus paths of string values naming eval files or prims. The one
    exemption is a top-level "task" holding a string or None: the adapter's instruction routing (see the docstring)."""
    hits = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}" if path else str(k)
            if not path and k == "task" and (v is None or isinstance(v, str)):
                continue
            if FORBIDDEN_KEY.search(str(k)):
                hits.append(p)
            else:
                hits += forbidden_paths(v, p)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            hits += forbidden_paths(v, f"{path}[{i}]")
    elif isinstance(x, str) and FORBIDDEN_VALUE.search(x):
        hits.append(path + " (value)")
    return hits


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("not a number")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("not finite")
    return v


def _vec(v, n):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ValueError(f"needs {n} numbers")
    return [_num(e) for e in v]


def _str(v, n=64):
    if not isinstance(v, str) or not v.strip() or len(v) > n:
        raise ValueError(f"needs a non-empty string <= {n} chars")
    return v.strip()


def _pos(v):
    p = _vec(v, 3)
    for c, (lo, hi) in zip(p, WORKSPACE):
        if not lo <= c <= hi:
            raise ValueError(f"outside the workspace box {WORKSPACE}")
    return p


def _quat(v):
    q = _vec(v, 4)
    nq = math.sqrt(sum(c * c for c in q))
    if not 0.9 <= nq <= 1.1:
        raise ValueError(f"|quat| = {nq:.3f}, not a unit quaternion (wxyz)")
    return [c / nq for c in q]


def _conf(v):
    c = _num(v)
    if not -0.5 <= c <= 1.5:
        raise ValueError("confidence far outside 0..1")
    return min(1.0, max(0.0, c))


def _pos_m(v):
    c = _num(v)
    if not 0.0 <= c < 2.0:
        raise ValueError("size outside [0, 2) m")
    return c


def _clean_item(d, allowed, required, path, stripped):
    if not isinstance(d, dict):
        raise ValueError("not an object")
    if isinstance(d.get("attributes"), dict) and d["attributes"].get("digit") is not None:
        dg = d["attributes"]["digit"]
        if isinstance(dg, bool) or not isinstance(dg, int) or not 0 <= dg <= 9:
            raise ValueError("attributes.digit must be an int 0-9")
        d = dict(d, value=dg)            # Ilia's card digit -> value (takes precedence over a value key)
    out = {}
    for k in d:
        if k not in allowed:
            stripped.append(f"{path}.{k}")
    for k in required:
        if k not in d or d[k] is None:
            raise ValueError(f"missing {k}")
    for k in allowed:
        if k not in d or d[k] is None:
            continue
        v = d[k]
        if k in ("label", "category", "reason", "colour"):
            out[k] = _str(v, 200 if k == "reason" else 64)
        elif k == "asset":
            out[k] = _str(v, 128)
        elif k in ("pos", "target", "tcp"):
            out[k] = _pos(v)
        elif k == "quat":
            out[k] = _quat(v)
        elif k == "size":
            out[k] = [_pos_m(e) for e in _vec(v, 3)]
        elif k in ("radius", "height"):
            out[k] = _pos_m(v)
        elif k == "top_z":
            z = _num(v)
            if not WORKSPACE[2][0] <= z <= WORKSPACE[2][1]:
                raise ValueError(f"top_z outside {WORKSPACE[2]}")
            out[k] = z
        elif k == "value":
            if isinstance(v, bool) or not isinstance(v, (int, str)) or len(str(v)) > 16:
                raise ValueError("value must be an int or a short string")
            out[k] = v
        elif k == "confidence":
            out[k] = _conf(v)
        elif k == "source":
            if v not in SOURCES:
                raise ValueError(f"source must be one of {SOURCES}")
            out[k] = v
        elif k in ("t_ms", "yaw_deg"):
            out[k] = _num(v)
        elif k == "cam":
            if v is not None and v not in CAMS:
                raise ValueError(f"cam must be one of {CAMS} or null")
            out[k] = v
        elif k in ("flange", "flange_pose"):
            f = _vec(v, 7)
            out["flange"] = _pos(f[:3]) + _quat(f[3:])
        elif k == "polygon_xy":
            if not isinstance(v, (list, tuple)) or not 2 <= len(v) <= 64:
                raise ValueError("polygon_xy: 2..64 points")
            out[k] = [_vec(c, 2) for c in v]
        elif k == "z_top":
            out[k] = _num(v)
        elif k == "corners":
            if not isinstance(v, (list, tuple)) or not 2 <= len(v) <= 16:
                raise ValueError("corners: 2..16 points")
            out[k] = [_vec(c, len(c)) for c in v if isinstance(c, (list, tuple)) and len(c) in (2, 3)]
        elif k in ("lo", "hi"):
            out[k] = _vec(v, 3)
    return out


def validate_perceive(out):
    rep = {"ok": False, "rejected": [], "stripped": [], "dropped": [], "error": None, "adapter_task": None}
    if not isinstance(out, dict):
        rep["error"] = f"adapter returned {type(out).__name__}, not a dict"
        return None, rep
    bad = forbidden_paths(out)
    if bad:
        rep["rejected"] = bad[:50]
        rep["error"] = f"forbidden keys or values (eval state / truth / instance names / eval files): {bad[:10]}"
        return None, rep
    rep["adapter_task"] = out.get("task")
    for k in out:
        if k not in TOP_ALLOWED and k != "task":
            rep["stripped"].append(k)
    objs = out.get("objects")
    if not isinstance(objs, list):
        rep["error"] = "output has no 'objects' list"
        return None, rep
    clean = {"objects": [], "regions": [], "needs_look": [], "t_ms": None, "last_seen": []}
    for i, o in enumerate(objs[:MAX_OBJECTS]):
        try:
            clean["objects"].append(_clean_item(o, OBJ_ALLOWED, ("label", "pos", "confidence"), f"objects[{i}]",
                                                rep["stripped"]))
        except ValueError as e:
            rep["dropped"].append(f"objects[{i}]: {e}")
    if len(objs) > MAX_OBJECTS:
        rep["dropped"].append(f"objects[{MAX_OBJECTS}:]: more than {MAX_OBJECTS}")
    ls = out.get("last_seen")
    if ls is not None and not isinstance(ls, list):
        rep["dropped"].append("last_seen: not a list")
    for i, o in enumerate((ls if isinstance(ls, list) else [])[:MAX_OBJECTS]):
        try:
            clean["last_seen"].append(_clean_item(o, OBJ_ALLOWED, ("label", "pos"), f"last_seen[{i}]", rep["stripped"]))
        except ValueError as e:
            rep["dropped"].append(f"last_seen[{i}]: {e}")
    for key, allowed, req, lim in (("regions", REG_ALLOWED, ("label",), MAX_REGIONS),
                                   ("needs_look", LOOK_ALLOWED, ("reason",), MAX_LOOKS)):
        items = out.get(key) or []
        if not isinstance(items, list):
            rep["dropped"].append(f"{key}: not a list")
            continue
        for i, d in enumerate(items[:lim]):
            try:
                clean[key].append(_clean_item(d, allowed, req, f"{key}[{i}]", rep["stripped"]))
            except ValueError as e:
                rep["dropped"].append(f"{key}[{i}]: {e}")
    try:
        clean["t_ms"] = _num(out["t_ms"]) if out.get("t_ms") is not None else None
    except ValueError:
        rep["dropped"].append("t_ms: not a number")
    if out.get("version") is not None:
        clean["version"] = str(out["version"])[:64]
    if out.get("deterministic") is not None:
        clean["deterministic"] = bool(out["deterministic"])
    if out.get("notes") is not None:
        clean["notes"] = str(out["notes"])[:300]
    if out.get("gl") is not None:
        clean["gl"] = str(out["gl"])[:16]
    if out.get("error") is not None:
        clean["error"] = str(out["error"])[:300]
    try:
        if out.get("t_estimate_ms") is not None:
            clean["t_estimate_ms"] = _num(out["t_estimate_ms"])
    except ValueError:
        rep["dropped"].append("t_estimate_ms: not a number")
    wm = out.get("want_match")
    if isinstance(wm, dict):
        clean["want_match"] = {str(k)[:64]: (str(v)[:64] if v is not None else None) for k, v in list(wm.items())[:16]}
    elif wm is not None:
        rep["dropped"].append("want_match: not a dict")
    rep["ok"] = True
    return clean, rep


def validate_held(out, arm=None):
    """-> (clean {"held": bool, "confidence", "note", "t_ms", "fg_frac", "dark_risk"}, report) or (None, report).
    A per-arm entry out[arm] (a dict) is used when present; else the top-level fields."""
    rep = {"ok": False, "rejected": [], "stripped": [], "dropped": [], "error": None}
    if not isinstance(out, dict):
        rep["error"] = f"held_look returned {type(out).__name__}, not a dict"
        return None, rep
    bad = forbidden_paths(out)
    if bad:
        rep["rejected"] = bad[:50]
        rep["error"] = f"forbidden keys or values: {bad[:10]}"
        return None, rep
    rep["stripped"] = [k for k in out if k not in HELD_ALLOWED]
    top_t = out.get("t_ms")
    if arm in ("left", "right") and isinstance(out.get(arm), dict):
        out = dict(out[arm], t_ms=out[arm].get("t_ms", top_t))
        rep["per_arm"] = arm
    elif arm in ("left", "right") and any(isinstance(out.get(a), dict) for a in ("left", "right")):
        rep["error"] = f"held_look answered for {[a for a in ('left', 'right') if isinstance(out.get(a), dict)]}, not {arm}"
        return None, rep
    h = out.get("held")
    if isinstance(h, bool):
        held = h
    elif isinstance(h, str) and h.strip().lower() in ("yes", "no"):
        held = h.strip().lower() == "yes"
    else:
        rep["error"] = "held must be a bool or 'yes'/'no'"
        return None, rep
    clean = {"held": held, "confidence": None, "note": str(out.get("note") or "")[:300], "t_ms": None,
             "fg_frac": None, "dark_risk": bool(out.get("dark_risk")) if out.get("dark_risk") is not None else None}
    try:
        if out.get("confidence") is not None:
            clean["confidence"] = _conf(out["confidence"])
        if out.get("t_ms") is not None:
            clean["t_ms"] = _num(out["t_ms"])
        if out.get("fg_frac") is not None:
            clean["fg_frac"] = _num(out["fg_frac"])
    except ValueError as e:
        rep["error"] = str(e)
        return None, rep
    rep["ok"] = True
    return clean, rep
