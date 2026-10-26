# DEV: run in the perception venv on compute3 (Bridge-isr 5.1 calibration set copied to ~/dojo-percrnd-data/replay51_all).
"""Score B1 held_look and the post-place refine on Bridge-isr's 5.1 calibration set (186 check frames, 20 dev-eval
stack / tower episodes). Code under test = PYTHONPATH (old: main before #44; new: #44 + what). Truth only scores and
builds the refine prior (protocol below); argv: OUT.json MODE(old|new).

held_look truth: ``physical_in_jaws``. what truth: the physical object's asset.
post-place truth ("well placed", from the true poses, per task):
- stack_blocks: level (some axis within 5 deg of vertical) and either resting on the table (the stack's base) or on
  another block: centre height difference within 5 mm of the two half heights, xy offset <= the task's stack threshold
  (by block_0's size: 15 / 17.5 / 20 mm).
- build_tower: the checker's up axis within 5 deg (block4: y, others z); resting (bottom within 5 mm of the table or of
  a top under its footprint); on a support: block7 within 43 mm of block0's centre, block3 within 23 mm of block7's,
  block4 within 23 mm of block3's, a pillar inside block0's footprint. Staged on the table, upright: well placed.
refine prior (same for old and new): well placed -> the true pose with dx, dy ~ U(-3, 3) mm and yaw ~ U(-3, 3) deg
(a belief's typical error); misplaced -> the intended pose: upright, on the support under it (stack: centred on the
block below; tower: same xy) at its height, i.e. where the runtime believed it put the object.
"""
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

R = Path(os.path.expanduser("~/dojo-percrnd-data/replay51_all"))
LAY = Path(os.path.expanduser("~/dojo-data/simlab/layouts"))
OUT, MODE = sys.argv[1], sys.argv[2]
TABLE = 0.765
STACK_XY = {0: 0.015, 1: 0.0175, 2: 0.02}  # by size group ((idx - 5) % 3)
IDX = json.loads((Path(os.environ["DOJO_CATALOG"]) / "index.json").read_text())
rng = np.random.default_rng(0)

from dojo.perception import j1_adapter as J  # noqa: E402
from dojo.perception.cameras import cameras_from_state  # noqa: E402
from dojo.perception.refine import refine  # noqa: E402
from dojo.perception.tracker import tcp_from_flange  # noqa: E402


def assets(task, layout):
    d = json.loads((LAY / task / f"{layout}.settled.json").read_text())
    return {o["label"]: f"Rigid/{o['category']}/{o['category_idx']:05d}" for o in d["Rigid"]["block"]}


def Rm(q):
    return Rotation.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()


def ext(key):
    return np.asarray(IDX[key]["extents"], float)


def world_box(key, p):
    """(bottom z, top z, xy min, xy max) of the asset's mesh at the pose (the prism's origin is not its box centre)."""
    from dojo.perception.render import load_asset
    W = load_asset(key).V @ Rm(p).T + np.asarray(p[:3])
    return W[:, 2].min(), W[:, 2].max(), W[:, :2].min(0), W[:, :2].max(0)


def tilt_deg(p, axis):
    v = Rm(p) @ np.eye(3)[axis]
    return math.degrees(math.acos(min(1.0, abs(v[2]))))


def overlap(a, b, frac=0.3):
    lo, hi = np.maximum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    small = min(float(np.prod(a[3] - a[2])), float(np.prod(b[3] - b[2])))
    return inter >= frac * max(small, 1e-9)


def support_under(lab, objs, keys):
    """(label | 'table', top z) of what the object rests on (bottom within 5 mm), else (None, None)."""
    bA = world_box(keys[lab], objs[lab])
    if abs(bA[0] - TABLE) <= 0.005:
        return "table", TABLE
    best = None
    for l2, p2 in objs.items():
        if l2 == lab or l2 not in keys:
            continue
        b2 = world_box(keys[l2], p2)
        if abs(bA[0] - b2[1]) <= 0.005 and overlap(bA, b2):
            best = (l2, b2[1])
    return best or (None, None)


def well_placed(task, lab, objs, keys):
    p = objs[lab]
    if task == "stack_blocks":
        if min(tilt_deg(p, a) for a in range(3)) > 5:
            return False, "tilted"
        thr = STACK_XY[(int(keys["block_0"][-5:]) - 5) % 3]
        if abs(p[2] - ext(keys[lab]).min() / 2 - TABLE) <= 0.005:
            return True, "table"  # the stack's base, put down at the build spot
        for l2, p2 in objs.items():
            if l2 == lab or l2 not in keys:
                continue
            h = (ext(keys[lab]).min() + ext(keys[l2]).min()) / 2
            if abs(p[2] - p2[2] - h) <= 0.005 and math.hypot(p[0] - p2[0], p[1] - p2[1]) <= thr:
                return True, f"on {l2}"
        return False, "not on a block"
    axis = 1 if lab == "block4" else 2
    if tilt_deg(p, axis) > 5:
        return False, f"tilt {tilt_deg(p, axis):.0f} deg"
    sup, _ = support_under(lab, objs, keys)
    if sup is None:
        return False, "not resting"
    if sup == "table":
        return True, "table"
    xy = lambda l: np.asarray(objs[l][:2])  # noqa: E731
    tol = {"block7": ("block0", 0.043), "block3": ("block7", 0.023), "block4": ("block3", 0.023)}.get(lab)
    if tol and np.linalg.norm(xy(lab) - xy(tol[0])) > tol[1]:
        return False, f"off {tol[0]} by {1000 * np.linalg.norm(xy(lab) - xy(tol[0])):.0f} mm"
    if lab in ("block1", "block2", "block5", "block6") and sup != "table":
        b0 = world_box(keys["block0"], objs["block0"])
        if not (np.all(xy(lab) >= b0[2]) and np.all(xy(lab) <= b0[3])):
            return False, "pillar off block0"
    return True, f"on {sup}"


def upright(p, lab, task):
    """The pose with the tilt removed (yaw kept)."""
    if task == "build_tower" and lab == "block4":
        return list(p[3:7])
    yaw = Rotation.from_matrix(Rm(p)).as_euler("zyx")[0]
    q = Rotation.from_euler("z", yaw).as_quat()
    return [q[3], q[0], q[1], q[2]]


def prior_for(task, lab, objs, keys, ok_truth):
    p = list(objs[lab])
    if ok_truth:
        d = rng.uniform(-0.003, 0.003, 2)
        a = math.radians(rng.uniform(-3, 3))
        q = (Rotation.from_euler("z", a) * Rotation.from_quat([p[4], p[5], p[6], p[3]])).as_quat()
        return [p[0] + d[0], p[1] + d[1], p[2], q[3], q[0], q[1], q[2]]
    q = upright(p, lab, task)
    bU = world_box(keys[lab], list(p[:3]) + list(q))
    hA = p[2] - bU[0]  # centre above the bottom, upright
    if task == "stack_blocks":
        hA = ext(keys[lab]).min() / 2
        others = {l: o for l, o in objs.items() if l != lab and l in keys}
        near = [l for l in others if math.hypot(objs[l][0] - p[0], objs[l][1] - p[1]) < 0.04 and objs[l][2] < p[2]]
        base = max(near, key=lambda l: objs[l][2]) if near else max(others, key=lambda l: objs[l][2])
        top = objs[base][2] + ext(keys[base]).min() / 2
        return [objs[base][0], objs[base][1], top + hA] + q
    sup_top = TABLE
    bA = world_box(keys[lab], objs[lab])
    for l2, p2 in objs.items():
        if l2 != lab and l2 in keys:
            b2 = world_box(keys[l2], p2)
            if b2[1] <= p[2] and overlap(bA, b2):
                sup_top = max(sup_top, b2[1])
    return [p[0], p[1], sup_top + hA] + q


rows = []
for e in json.loads((R / "index.json").read_text()):
    d = R / e["task"] / Path(e["dir"]).name
    st = json.loads((d / "state.json").read_text())
    sta = {k: np.asarray(v, float) for k, v in st.items()}
    keys = assets(e["task"], e["layout"])
    objs = e["objects_true"]
    c = e["check"]
    arm = c["arm"]
    asked_key = "/".join([e["belief_asset"][0], e["belief_asset"][1], f"{int(e['belief_asset'][2]):05d}"])
    row = {"task": e["task"], "layout": e["layout"], "step": e["step"], "kind": c["kind"], "label": c["label"], "arm": arm,
           "old_logged": {k: c.get(k) for k in ("held", "confidence", "ok", "iou", "shift_mm", "why")},
           "physical": e.get("physical_object"), "in_jaws": e.get("physical_in_jaws")}
    if c["kind"] == "held_look":
        cams = cameras_from_state(sta)
        calib = {k: {"K": cm.K.tolist(), "T_world_cam": cm.T_wc.tolist()} for k, cm in cams.items()}
        frames = {k: str(d / f"{k}.png") for k in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
        flange = {a: st[f"{a}_ee_pose"] for a in ("left", "right")}
        grip = {a: st[f"{a}_ee_joint_state"][0] for a in ("left", "right")}
        joints = {a: st[f"{a}_arm_joint_state"] for a in ("left", "right")}
        tcp = tcp_from_flange(flange[arm])
        phys = e.get("physical_object")
        pose = objs[phys] if (e.get("physical_in_jaws") and phys in objs) else list(tcp) + list(flange[arm][3:7])
        cands = [{"label": c["label"], "key": asked_key, "pose7": list(pose)}] + \
                [{"label": l, "key": keys[l], "pose7": list(objs[l])} for l in sorted(keys) if l != c["label"]]
        out = J._held_look_impl(frames, calib, flange, grip, joints=joints, candidates=cands, arm=arm)
        s = out.get(arm, {})
        w = s.get("what") or {}
        row.update({"held": s.get("held"), "conf": s.get("confidence"), "evidence": s.get("evidence"),
                    "opening": s.get("opening"), "what_key": w.get("key"), "what_conf": w.get("conf"),
                    "what_margin": w.get("margin"), "asked_held": s.get("asked_held"),
                    "true_key": keys.get(phys) if e.get("physical_in_jaws") else None, "asked_key": asked_key})
    else:
        phys = e.get("physical_object") or c["label"]
        ok_truth, why_truth = well_placed(e["task"], phys, objs, keys)
        prior = prior_for(e["task"], phys, objs, keys, ok_truth)
        img = cv2.cvtColor(cv2.imread(str(d / f"cam_{arm}_wrist.png")), cv2.COLOR_BGR2RGB)
        others = [(keys[l], list(objs[l])) for l in objs if l != phys and l in keys]
        r = refine(keys[phys], img, sta, prior, arm=arm, others=others)
        pos = np.asarray(r["pose"][:3])
        row.update({"truth_well": ok_truth, "truth_why": why_truth, "ok": r["ok"], "iou": r["iou"], "why": r.get("why"),
                    "err_mm": round(1000 * float(np.linalg.norm(pos[:2] - np.asarray(objs[phys][:2]))), 1),
                    "dz_mm": round(1000 * float(pos[2] - objs[phys][2]), 1),
                    "prior_err_mm": round(1000 * float(np.linalg.norm(np.asarray(prior[:2]) - np.asarray(objs[phys][:2]))), 1),
                    "z_dropped_mm": r.get("z_dropped_mm")})
    rows.append(row)
    print(json.dumps({k: row.get(k) for k in ("task", "layout", "step", "kind", "in_jaws", "held", "conf", "what_key", "true_key",
                                              "truth_well", "truth_why", "ok", "iou", "why")}), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
