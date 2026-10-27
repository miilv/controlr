"""The agentic perceiver in layout-record form, for the runtime's perception service (integration's mode B).

    agentic_estimate(images, state, task, instruction, bgs, cfg=None, b1_record=None, **kw) -> record
        B1 (``inverse_layout.estimate``) unless ``b1_record`` is given -> one fresh agent call with B1 as the hint ->
        H2 refine -> the agent's objects written back into B1's record under the task's labels (same format as
        ``estimate()``: ``Rigid/<category>: [{category, category_idx, label, default_pos, default_ori, scale,
        _perception}]``, ``_not_found``). ``record["_perception"]["agentic"]`` = the call's stats.
    needs_agent(record, thr=0.2) -> (bool, why)
        the budget gate's escalation rule on a B1 record: a label not found, or an object whose calibrated risk
        (``flag_model.json``) is >= thr.

Budget use (D-C6 decision): the caller runs ``agentic_estimate`` on the episode's first estimate, then plain B1 with
the prior, and at most once more when ``needs_agent`` says so. GT is never read here.
"""
from __future__ import annotations

import copy
import math

import numpy as np

TYPES = ("Rigid", "Geometry", "Articulation")


def _yaw(q) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rest_quat(key: str) -> list[float] | None:
    """The asset's first catalogue resting orientation at yaw 0 (wxyz): an object the agent names without a turn rests
    like this, not in its asset frame (identity stood fill_egg_holder's lying egg on its end: the skill pinched 21-37 mm
    above its middle 8 times, rendered 100000)."""
    try:
        from ..cameras import R_to_quat_wxyz
        from ..inverse_layout import shape_models
        return [float(v) for v in R_to_quat_wxyz(shape_models(key)[0].R0)]
    except Exception:  # noqa: BLE001 (an asset outside the catalogue)
        return None


def record_objects(rec: dict) -> list[dict]:
    """B1 record -> J1-style objects (label, category, asset, pos, quat, size, confidence, flags, source, residual)."""
    out = []
    for typ in TYPES:
        for cat, lst in (rec.get(typ) or {}).items():
            for r in lst:
                p = r.get("_perception") or {}
                o = {"label": r["label"], "category": cat, "asset": f"{typ}/{cat}/{int(r['category_idx']):05d}",
                     "pos": [float(v) for v in r["default_pos"]], "quat": [float(v) for v in r["default_ori"]],
                     "confidence": float(p.get("confidence", 0.5)), "residual": p.get("residual"),
                     "source": "fused" if p.get("n_views", 0) > 1 else "head",
                     "flags": {**{k: p[k] for k in ("occluded", "ambiguous", "hidden_frac", "near_home_grippers",
                                                    "fg_resid_near", "slot") if k in p},
                               **({"yaw_read": True} if p.get("yaw_read") else {})}}
                if p.get("alternative"):
                    o["alternative"] = {"asset": p["alternative"].get("key"), "margin": p["alternative"].get("margin")}
                out.append(o)
    return out


def needs_agent(rec: dict, thr: float = 0.2) -> tuple[bool, str]:
    from .j1 import risk

    nf = [x.get("label") for x in rec.get("_not_found") or []]
    if nf:
        return True, f"not_found:{nf}"
    hi = [(o["label"], round(risk(o), 2)) for o in record_objects(rec) if risk(o) >= thr]
    return (True, f"risk:{hi[:4]}") if hi else (False, "b1_confident")


def _calib_from_state(state: dict) -> dict:
    from ..cameras import cameras_from_state

    cams = cameras_from_state({k: np.asarray(v, float) for k, v in state.items()})
    return {c: {"K": cam.K.tolist(), "T_world_cam": cam.T_wc.tolist()} for c, cam in cams.items()}


def merge_into_record(rec: dict, objects: list[dict], unmatched_cost_m: float = 0.15) -> tuple[dict, list[dict]]:
    """Write the agent's objects into a copy of B1's record. Per category, Hungarian on xy between the agent's objects
    and B1's objects + the category's not-found labels (a fixed cost); a matched B1 entry takes the agent's pose (and
    asset, when given); a matched not-found label becomes a new entry. Returns (record, per-object log)."""
    from scipy.optimize import linear_sum_assignment

    from ..inverse_layout import label_groups

    out = copy.deepcopy(rec)
    log = []
    by_cat_b1: dict[str, list[dict]] = {}
    for typ in TYPES:
        for cat, lst in (out.get(typ) or {}).items():
            for r in lst:
                r["_type"] = typ
                by_cat_b1.setdefault(cat, []).append(r)
    nf_by_cat: dict[str, list[dict]] = {}
    task = out.get("task")
    groups = label_groups(task) if task else []
    for nf in out.get("_not_found") or []:
        keys = nf.get("candidates") or next((g["candidates"] for g in groups if nf["label"] in g["labels"]), [])
        cats = {k.split("/")[1] for k in keys}
        for c in cats:
            nf_by_cat.setdefault(c, []).append(dict(nf, _keys=keys))
    agent_by_cat: dict[str, list[dict]] = {}
    for o in objects:
        a = str(o.get("asset") or "").split("/")
        cat = a[1] if len(a) == 3 else o.get("category")
        if cat not in by_cat_b1 and cat not in nf_by_cat:  # free-text category: the nearest B1 object's category
            best = min(((math.hypot(r["default_pos"][0] - o["pos"][0], r["default_pos"][1] - o["pos"][1]), c)
                        for c, rs in by_cat_b1.items() for r in rs), default=(None, None))
            cat = best[1]
        if cat is not None:
            agent_by_cat.setdefault(cat, []).append(o)
    filled_nf = set()
    for cat, A in agent_by_cat.items():
        B = by_cat_b1.get(cat, [])
        N = [n for n in nf_by_cat.get(cat, []) if n["label"] not in filled_nf]
        C = np.full((len(A), len(B) + len(N)), unmatched_cost_m)
        for i, o in enumerate(A):
            for j, r in enumerate(B):
                C[i, j] = math.hypot(r["default_pos"][0] - o["pos"][0], r["default_pos"][1] - o["pos"][1])
        if C.size == 0:
            continue
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            o = A[i]
            if j < len(B):
                r = B[j]
                moved = math.hypot(r["default_pos"][0] - o["pos"][0], r["default_pos"][1] - o["pos"][1])
                if moved < 1e-3 and o.get("source") not in ("refined",):
                    continue  # kept as B1 had it
                old = list(r["default_pos"])
                r["default_pos"] = [float(v) for v in o["pos"]]
                if o.get("quat"):
                    r["default_ori"] = [float(v) for v in o["quat"]]
                a = str(o.get("asset") or "").split("/")
                if len(a) == 3 and a[1] == cat:
                    r["category_idx"] = int(a[2])
                p = r.setdefault("_perception", {})
                p.update(source="agent" if o.get("source") != "refined" else "agent+refine", confidence=float(o.get("confidence", 0.6)),
                         b1_pos=old)
                if o.get("source") == "refined" and len(a) == 3:
                    p.pop("place_tag", None)  # the refined shape's tag is re-derived from the rotation downstream
                log.append({"label": r["label"], "action": "moved", "moved_mm": round(moved * 1000, 1), "source": p["source"]})
            else:
                n = N[j - len(B)]
                a = str(o.get("asset") or "").split("/")
                key = o["asset"] if len(a) == 3 and a[1] == cat else next(k for k in n["_keys"] if k.split("/")[1] == cat)
                typ, _, idx = key.split("/")
                q = o.get("quat") or rest_quat(key) or [1.0, 0.0, 0.0, 0.0]
                entry = {"category": cat, "category_idx": int(idx), "label": n["label"],
                         "default_pos": [float(v) for v in o["pos"]], "default_ori": [float(v) for v in q],
                         "scale": [1.0, 1.0, 1.0],
                         "_perception": {"source": "agent" if o.get("source") != "refined" else "agent+refine",
                                         "confidence": float(o.get("confidence", 0.5)),
                                         "orientation_known": bool(o.get("quat")) and not o.get("orientation_default")}}
                out.setdefault(typ, {}).setdefault(cat, []).append(entry)
                filled_nf.add(n["label"])
                log.append({"label": n["label"], "action": "found", "source": entry["_perception"]["source"]})
    out["_not_found"] = [x for x in out.get("_not_found") or [] if x["label"] not in filled_nf]
    for typ in TYPES:
        for lst in (out.get(typ) or {}).values():
            for r in lst:
                r.pop("_type", None)
    return out, log


def agentic_estimate(images: dict, state: dict, task: str, instruction: str, bgs: dict, cfg=None,
                     b1_record: dict | None = None, model: str | None = None, max_turns: int | None = None,
                     log_dir: str | None = None, chat=None, **kw) -> dict:
    """One agentic perceive on this frame, as a layout record. ``state``: the official observation's state dict."""
    from ..inverse_layout import estimate
    from .j1 import perceive_impl

    rec = b1_record if b1_record is not None else estimate(images, state, task, instruction, bgs=bgs, cfg=cfg)
    b1 = {"objects": record_objects(rec), "regions": [], "needs_look": [], "task": task,
          "want_match": {}, "last_seen": []}
    st = {k: [float(x) for x in np.ravel(v)] for k, v in state.items()}
    flange = {a: st[f"{a}_ee_pose"] for a in ("left", "right") if f"{a}_ee_pose" in st}
    grip = {a: st[f"{a}_ee_joint_state"][0] for a in ("left", "right") if f"{a}_ee_joint_state" in st}
    joints = {a: st[f"{a}_arm_joint_state"][:6] for a in ("left", "right") if f"{a}_arm_joint_state" in st}
    out = perceive_impl(images, _calib_from_state(state), flange, grip, instruction, None, joints=joints, b1=b1,
                        use_b1=True, model=model, max_turns=max_turns, log_dir=log_dir, chat=chat, task=task, bgs=bgs,
                        gate="always", return_stats=True, keep_unmentioned=False, **kw)
    stats = out.get("_agent") or {"result": "b1_fallback"}
    if stats.get("result", "").startswith("agent"):
        new, log = merge_into_record(rec, out["objects"])
    else:
        new, log = copy.deepcopy(rec), []
    new.setdefault("_perception", {})["agentic"] = {**{k: v for k, v in stats.items() if k != "refine"}, "merge": log}
    return new
