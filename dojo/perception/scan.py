"""Scan macro: one fixed pose pair at episode start that parks both arms out of the head camera's view of the table
and aims each wrist camera at its table-front corner, where the home grippers hide objects (D0: 39 % of B1's frame-0
failures are occlusion; ok 61 % at |x| >= 0.33 vs 92 % in the centre).

The pose is the same for every episode (no perception input). It is defined by a wrist-camera eye and target per
arm (mirrored in x), turned into a flange pose (``look.flange_for_camera``) and joints (IK on the public URDF,
nearest to home). ``design()`` is the offline search that picked the constants: candidate eyes x targets scored by the
head-view occlusion of the table's object region (both arms at the pose) and the wrist camera's coverage of its corner.

    python -m dojo.perception.scan design        # print the best candidates (FK + robot masks only; no sim, no GT)
    python -m dojo.perception.scan show          # the frozen pose: flange, joints, steps, occlusion, coverage
"""
from __future__ import annotations

import json
import math

import numpy as np

from .cameras import TABLE_Z

# frozen by design() on 2026-10-25 (left arm; the right arm mirrors x)
SCAN_EYE_LEFT = (-0.44, -0.46, 1.06)
SCAN_TARGET_LEFT = (-0.34, -0.14, TABLE_Z)
STEP_RAD = 0.16           # skills' Mover default (rad per env step, largest joint)
OBJECT_REGION = ((-0.47, 0.47), (-0.27, 0.10))      # where D0 / task configs put objects (x, y)
CORNER = {"left": ((-0.47, -0.25), (-0.30, 0.00)), "right": ((0.25, 0.47), (-0.30, 0.00))}
HOME_Q = np.zeros(6)


def _mirror(p):
    return (-p[0], p[1], p[2])


def eye_target(arm: str, eye_left=SCAN_EYE_LEFT, target_left=SCAN_TARGET_LEFT):
    if arm == "left":
        return np.array(eye_left, float), np.array(target_left, float)
    return np.array(_mirror(eye_left), float), np.array(_mirror(target_left), float)


def pose_for(arm: str, eye, target) -> dict:
    """Wrist camera at ``eye`` looking at ``target`` -> flange pose, joints (IK from home), IK residuals."""
    from .cameras import WRIST, intrinsics
    from .inverse_layout import look_at_camera
    from .j1_adapter import ik_flange
    from .look import flange_for_camera

    K = intrinsics(WRIST["focal_length"], WRIST["horizontal_aperture"])
    cam = look_at_camera(np.asarray(eye, float), np.asarray(target, float), K)
    fl = flange_for_camera(cam.T_wc)
    q, ep, er = ik_flange(arm, fl, HOME_Q)
    return {"arm": arm, "flange_pose": [float(v) for v in fl], "joints": [float(v) for v in q],
            "ik_pos_err_mm": round(ep * 1000, 2), "ik_rot_err_deg": round(er, 3), "T_world_cam": cam.T_wc.tolist()}


def scan_state(poses: dict, gripper: float = 1.0) -> dict:
    st = {}
    for arm in ("left", "right"):
        p = poses.get(arm)
        st[f"{arm}_arm_joint_state"] = np.asarray(p["joints"] if p else HOME_Q, float)
        st[f"{arm}_ee_joint_state"] = np.array([gripper])
    return st


def _region_pts(xr, yr, n=24, z=TABLE_Z + 0.02):
    xs, ys = np.linspace(*xr, n), np.linspace(*yr, n)
    return np.array([[x, y, z] for x in xs for y in ys])


def head_occlusion(state: dict) -> float:
    """Share of the object region's sample points (table + 2 cm) hidden by the arms in the head view."""
    from .cameras import head_camera
    from .render import robot_mask

    cam = head_camera()
    m = robot_mask(cam, state, dilate_px=2)
    uv, z = cam.project(_region_pts(*OBJECT_REGION))
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.w) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.h)
    hid = np.zeros(len(uv), bool)
    hid[ok] = m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] > 0
    return float(hid.mean())


def corner_coverage(arm: str, state: dict) -> tuple[float, float]:
    """(share of the arm's corner zone inside its wrist image and not under its own fingers, mean distance m)."""
    from .cameras import wrist_camera
    from .render import robot_mask

    cam = wrist_camera(arm, state[f"{arm}_arm_joint_state"])
    m = robot_mask(cam, state, dilate_px=2)
    P = _region_pts(*CORNER[arm], n=16)
    uv, z = cam.project(P)
    ok = (z > 0.03) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.w) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.h)
    vis = ok.copy()
    vis[ok] = m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] == 0
    return float(vis.mean()), float(np.linalg.norm(P - cam.center, axis=1).mean())


def steps_between(q0, q1, step_rad: float = STEP_RAD) -> int:
    return int(math.ceil(float(np.max(np.abs(np.asarray(q1) - np.asarray(q0)))) / step_rad))


def frozen() -> dict:
    """The scan pose pair (deterministic: fixed constants + IK from home)."""
    poses = {arm: pose_for(arm, *eye_target(arm)) for arm in ("left", "right")}
    poses["steps_out"] = max(steps_between(HOME_Q, poses[a]["joints"]) for a in ("left", "right"))
    poses["steps_total"] = 2 * poses["steps_out"]
    return poses


def evaluate(eye_left, target_left) -> dict | None:
    poses = {arm: pose_for(arm, *eye_target(arm, eye_left, target_left)) for arm in ("left", "right")}
    if any(poses[a]["ik_pos_err_mm"] > 1.0 or poses[a]["ik_rot_err_deg"] > 0.5 for a in ("left", "right")):
        return None
    st = scan_state(poses)
    cov = [corner_coverage(a, st) for a in ("left", "right")]
    return {"eye": list(eye_left), "target": list(target_left), "occlusion": round(head_occlusion(st), 4),
            "coverage": [round(c[0], 3) for c in cov], "dist_m": [round(c[1], 3) for c in cov],
            "steps_out": max(steps_between(HOME_Q, poses[a]["joints"]) for a in ("left", "right")),
            "joints_left": [round(v, 3) for v in poses["left"]["joints"]]}


def design() -> list[dict]:
    home = scan_state({})
    print(json.dumps({"home_occlusion": round(head_occlusion(home), 4),
                      "home_coverage": [round(corner_coverage(a, home)[0], 3) for a in ("left", "right")]}))
    res = []
    for ex in (-0.44, -0.48, -0.52, -0.56):
        for ey in (-0.46, -0.42, -0.36, -0.30):
            for ez in (0.98, 1.06, 1.14):
                for tx, ty in ((-0.34, -0.14), (-0.36, -0.18), (-0.30, -0.10)):
                    r = evaluate((ex, ey, ez), (tx, ty, TABLE_Z))
                    if r is None:
                        continue
                    r["score"] = round(min(r["coverage"]) - 3.0 * r["occlusion"] - 0.005 * r["steps_out"], 4)
                    res.append(r)
    res.sort(key=lambda r: -r["score"])
    for r in res[:10]:
        print(json.dumps(r))
    return res


def looks() -> list[dict]:
    """The scan as a look request per arm (runtime look macros / J1 needs_look): flange pose (link6, world, wxyz),
    joints, the wrist camera pose it gives, and its target."""
    p = frozen()
    return [{"arm": a, "cam": f"cam_{a}_wrist", "reason": "scan", "purpose": "scan", "label": None,
             "flange_pose": p[a]["flange_pose"], "joints": p[a]["joints"], "T_world_cam": p[a]["T_world_cam"],
             "target": [float(v) for v in eye_target(a)[1]], "steps_out": p["steps_out"]} for a in ("left", "right")]


def _groups_by_label(task: str) -> dict:
    from .inverse_layout import label_groups

    return {lab: gi for gi, g in enumerate(label_groups(task)) for lab in g["labels"]}


def fill_from_frame0(rec: dict, rec0: dict, task: str, same_m: float = 0.03) -> tuple[dict, list[str]]:
    """Labels the scan-primary estimate did not find take frame 0's reading of the same label group, when that
    reading is not already one of the scan estimate's objects (> ``same_m`` from all of them). Flat grey pads were
    the case on D0: found from the home head view, missed from the scan view's farther wrists."""
    import copy

    out = copy.deepcopy(rec)
    g_of = _groups_by_label(task)
    have = []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (out.get(typ) or {}).items():
            for r in lst:
                have.append((g_of.get(r["label"]), r["default_pos"]))
    filled = []
    for nf in list(out.get("_not_found") or []):
        g = g_of.get(nf["label"])
        best = None
        for typ in ("Rigid", "Geometry", "Articulation"):
            for cat, lst in (rec0.get(typ) or {}).items():
                for r in lst:
                    if g_of.get(r["label"]) != g:
                        continue
                    if any(hg == g and math.hypot(hp[0] - r["default_pos"][0], hp[1] - r["default_pos"][1]) <= same_m
                           for hg, hp in have):
                        continue
                    c = float((r.get("_perception") or {}).get("confidence", 0.5))
                    if best is None or c > best[0]:
                        best = (c, typ, cat, r)
        if best is None:
            continue
        _, typ, cat, r = best
        e = copy.deepcopy(r)
        e["label"] = nf["label"]
        e.setdefault("_perception", {})["source"] = "frame0_fill"
        out.setdefault(typ, {}).setdefault(cat, []).append(e)
        have.append((g, e["default_pos"]))
        out["_not_found"] = [x for x in out["_not_found"] if x["label"] != nf["label"]]
        filled.append(nf["label"])
    return out, filled


def estimate_with_scan(scan: dict, frame0: dict, task: str, instruction: str = "", bgs=None, cfg=None,
                       rec0: dict | None = None) -> dict:
    """The scan frame as the primary view (its head view is unoccluded), frame 0 as extra views, then the frame-0
    fill. ``scan`` / ``frame0`` = {"images": {cam: RGB}, "state": {...}}. ``bgs`` must be a scan-aware background
    (clean head background built with scan frames), else the zone under the home grippers stays "unseen"."""
    from .inverse_layout import estimate

    rec = estimate(scan["images"], scan["state"], task, instruction, bgs=bgs, cfg=cfg,
                   extra_frames=[(frame0["images"], frame0["state"])])
    if rec.get("_not_found"):
        if rec0 is None:
            rec0 = estimate(frame0["images"], frame0["state"], task, instruction, bgs=bgs, cfg=cfg)
        rec, filled = fill_from_frame0(rec, rec0, task)
        rec.setdefault("_perception", {})["frame0_fill"] = filled
    rec.setdefault("_perception", {})["scan_primary"] = True
    return rec


def export_looks(root: str, out: str, tasks: list[str]) -> dict:
    """simlab look_render input (look_poses format) with the same scan pose pair for every layout of ``tasks``."""
    from pathlib import Path

    p = frozen()
    looks = [{"arm": a, "flange_pose": p[a]["flange_pose"], "T_world_cam_cv": p[a]["T_world_cam"],
              "look_at": list(eye_target(a)[1]), "purpose": "scan", "label": None} for a in ("left", "right")]
    res = {"version": 1, "source": "dojo.perception.scan (fixed scan pose pair)", "layouts": {}}
    for t in tasks:
        for d in sorted(Path(root, t).iterdir()):
            if d.is_dir() and (d / "layout.json").exists():
                res["layouts"][f"{t}/{d.name}"] = {"parallel": True, "looks": looks}
    Path(out).write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    import sys

    cmd = (argv or sys.argv[1:] or ["show"])[0]
    if cmd == "design":
        design()
    elif cmd == "export":
        a = argv or sys.argv[1:]
        r = export_looks(a[1], a[2], a[3:])
        print(len(r["layouts"]), "layouts")
    else:
        p = frozen()
        st = scan_state(p)
        print(json.dumps({"left": {k: p["left"][k] for k in ("flange_pose", "joints", "ik_pos_err_mm", "ik_rot_err_deg")},
                          "right": {k: p["right"][k] for k in ("flange_pose", "joints", "ik_pos_err_mm", "ik_rot_err_deg")},
                          "steps_out": p["steps_out"], "steps_total": p["steps_total"],
                          "head_occlusion": round(head_occlusion(st), 4), "home_occlusion": round(head_occlusion(scan_state({})), 4),
                          "corner_coverage": [round(corner_coverage(a, st)[0], 3) for a in ("left", "right")]}, indent=1))


if __name__ == "__main__":
    main()
