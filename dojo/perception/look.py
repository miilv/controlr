"""Canonical "look" macro: which wrist looks to take at episode start, and fusing the frames afterwards.

    plan = look_plan(obs, record)     # record = estimate(...) on the first frame
    ... the runtime moves each arm to plan[i]["flange_pose"] (IK on the benchmark URDF), observes,
    ... returns home, observes ...
    record2 = fuse([obs0, obs_after_look1, ...], task)

At most one look per arm (both can move together). Targets, in order: objects whose record carries a
``review`` request (ambiguous flip / identity, or occluded), labels not found (look at the table regions
the home grippers hide from the head camera), then the least confident objects. Each look puts the wrist
camera ~14 cm from the target at 55 degrees elevation, from the arm's side, unless a review request gives
its own camera pose.

Step cost (D0 scripted looks, cuRobo moves): median 25 env steps per look move (p90 ~34), so two looks with
both arms in parallel plus the return home is about 50 steps; sequentially about 75.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .cameras import TABLE_Z, T_to_pose, wrist_mount_in_link
from .inverse_layout import ARM_BASES, estimate, look_at_camera

# table regions the home-pose grippers hide from the head camera (from the D0 failure analysis)
HIDDEN_UNDER_GRIPPERS = {"left": np.array([-0.36, -0.21, TABLE_Z]), "right": np.array([0.36, -0.21, TABLE_Z])}
STEPS_PER_LOOK_MOVE = 25


@lru_cache(maxsize=1)
def camera_in_flange() -> np.ndarray:
    """Fixed transform flange (link6) -> wrist camera (OpenCV axes), from the URDF chain + mount."""
    from .robot_model import X5Model

    fk = X5Model.default().fk(np.zeros(6))
    return np.linalg.inv(fk["link6"]) @ fk["camera"] @ wrist_mount_in_link()


def flange_for_camera(T_world_cam: np.ndarray) -> np.ndarray:
    """Flange pose (x, y, z, qw, qx, qy, qz) that puts the wrist camera at ``T_world_cam``."""
    return T_to_pose(T_world_cam @ np.linalg.inv(camera_in_flange()))


def _clearance(arm: str, eye: np.ndarray, obstacles: list[tuple[np.ndarray, float]]) -> float:
    """Smallest xy gap between the arm (approximated by the segment base -> eye) and other objects'
    footprint circles: an arm hovering over an object hides it from the head camera in the look frame."""
    if not obstacles:
        return 1.0
    b = ARM_BASES[arm][:2]
    pts = b[None, :] + np.linspace(0.3, 1.0, 8)[:, None] * (eye[:2] - b)[None, :]
    return min(float(np.min(np.linalg.norm(pts - c[None, :2], axis=1))) - r for c, r in obstacles)


def default_look(target: np.ndarray, arm: str, dist: float = 0.14, elev_deg: float = 55.0,
                 obstacles: list[tuple[np.ndarray, float]] | None = None) -> np.ndarray:
    """Wrist camera pose looking at ``target`` from the arm's side; the azimuth (within +-60 deg of the
    base direction) keeps the arm clear of other objects, large ones first."""
    from .cameras import WRIST, intrinsics

    back = ARM_BASES[arm][:2] - target[:2]
    base_az = float(np.arctan2(back[1], back[0]))
    e = np.deg2rad(elev_deg)
    best = None
    for off in (0, -30, 30, -60, 60):
        az = base_az + np.deg2rad(off)
        eye = target + dist * np.r_[np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)]
        c = _clearance(arm, eye, obstacles or [])
        score = min(c, 0.05) - 0.0002 * abs(off)  # clear by 5 cm is enough; then prefer the direct azimuth
        if best is None or score > best[0]:
            best = (score, eye)
    return look_at_camera(best[1], target, intrinsics(WRIST["focal_length"], WRIST["horizontal_aperture"])).T_wc


def _records(record: dict):
    for typ in ("Rigid", "Geometry", "Articulation"):
        for lst in (record.get(typ) or {}).values():
            yield from lst


def look_plan(obs: dict, record: dict, max_looks: int = 2) -> list[dict]:
    """Up to one look per arm, most useful first. ``obs`` is unused beyond the arm state (kept for the API)."""
    cands = []  # (priority, arm, purpose, label, target, T_world_cam)
    objs = [(np.asarray(r["default_pos"], dtype=float), r) for r in _records(record)
            if r.get("_perception", {}).get("place_tag") != "fixed"]

    def obstacles(exclude):
        from .render import load_asset

        out = []
        for pos, r in objs:
            if np.hypot(*(pos[:2] - exclude[:2])) < 0.03:
                continue
            key = next((f"{t}/{r['category']}/{int(r['category_idx']):05d}" for t in ("Rigid", "Geometry", "Articulation")
                        if r["category"] in (record.get(t) or {})), None)
            try:
                rad = 0.5 * float(np.linalg.norm(load_asset(key).extents[:2]))
            except Exception:
                rad = 0.04
            out.append((pos, rad))
        return out

    for pos, r in objs:
        p = r.get("_perception", {})
        arm = "left" if pos[0] < 0 else "right"
        if p.get("review"):
            rv = p["review"]
            cands.append((0, rv["arm"], "review", r["label"], np.asarray(rv["look_at"]), np.asarray(rv["T_world_cam_cv"])))
        elif p.get("occluded") or p.get("hidden_frac", 0) > 0.1 or p.get("near_home_grippers"):
            # partly under the robot or in the region the home grippers hide: the visible part fits well, so
            # confidence is overstated (D0 stack_blocks 100006: 22 mm off at confidence 0.77)
            cands.append((0, arm, "partly_hidden", r["label"], pos, default_look(pos, arm, obstacles=obstacles(pos))))
    if record.get("_not_found"):
        for arm, tgt in HIDDEN_UNDER_GRIPPERS.items():
            cands.append((1, arm, "not_found:" + ",".join(x["label"] for x in record["_not_found"]), None, tgt,
                          default_look(tgt, arm, obstacles=obstacles(tgt))))
    for r in sorted(_records(record), key=lambda r: r.get("_perception", {}).get("confidence", 1.0)):
        p = r.get("_perception", {})
        if p.get("place_tag") == "fixed" or p.get("confidence", 1.0) >= 0.8:
            continue
        pos = np.asarray(r["default_pos"], dtype=float)
        arm = "left" if pos[0] < 0 else "right"
        cands.append((2, arm, "low_confidence", r["label"], pos, default_look(pos, arm, obstacles=obstacles(pos))))
    plan, used = [], set()
    for prio, arm, purpose, label, tgt, T in sorted(cands, key=lambda c: c[0]):
        if arm in used:
            continue
        used.add(arm)
        plan.append({"arm": arm, "purpose": purpose, "label": label, "target": np.round(tgt, 4).tolist(),
                     "T_world_cam_cv": np.round(T, 5).tolist(), "flange_pose": np.round(flange_for_camera(T), 5).tolist(),
                     "est_steps": STEPS_PER_LOOK_MOVE})
        if len(plan) >= max_looks:
            break
    return plan


def look_cost_steps(plan: list[dict], parallel: bool = True) -> int:
    """Env steps for the plan: out and back per look; both arms together when ``parallel``."""
    if not plan:
        return 0
    return 2 * STEPS_PER_LOOK_MOVE * (1 if parallel else len(plan))


def fuse(frames: list[dict], task: str, bgs: dict | None = None, cfg=None, prior_record: dict | None = None,
         mode: str = "conservative", accept_drop: float = 0.02, extra_slack: float = 0.01,
         big_move: float = 0.005, good_fit: float = 0.35) -> dict:
    """Fuse later frames (looks; objects assumed static) into a frame-0 estimate.
    ``frames``: [{"images": {cam: RGB}, "state": {...}, "instruction": str}, ...].

    ``mode="conservative"`` (default): keep the frame-0 record's objects, identities and assignment
    (``prior_record``, else estimated from frames[0]); refine each object locally from its frame-0 pose over
    all views and accept the new pose only if the total residual drops by > ``accept_drop`` AND the later
    frames' views alone do not get worse (+ ``extra_slack``). Objects the looks do not see keep frame 0.
    ``mode="full"``: re-run the whole estimate with the extra views (re-selection included; integration saw
    it move well-placed blocks by 12-14 mm)."""
    import copy

    from .cameras import R_to_quat_wxyz, cameras_from_state, pose_to_T
    from .inverse_layout import EstimatorConfig, _views_subset, make_roi_views, refine as nm_refine, roi_loss
    from .refine import _shape_for, _ZShift
    from .segment import load_backgrounds, segment_views

    f0 = frames[0]
    extra = [(f["images"], f["state"]) for f in frames[1:]]
    if mode == "full":
        return estimate(f0["images"], f0["state"], task, f0.get("instruction", ""), bgs=bgs, cfg=cfg, extra_frames=extra)
    cfg = cfg or EstimatorConfig()
    bgs = bgs if bgs is not None else load_backgrounds()
    rec0 = prior_record or estimate(f0["images"], f0["state"], task, f0.get("instruction", ""), bgs=bgs, cfg=cfg)
    rec = copy.deepcopy(rec0)
    if not extra:
        return rec
    views = _views_subset(segment_views(f0["images"], cameras_from_state(f0["state"]), f0["state"], bgs, thr=cfg.fg_thr), cfg)
    for k, (imgs_k, st_k) in enumerate(extra, start=1):
        vk = _views_subset(segment_views(imgs_k, cameras_from_state(st_k), st_k, bgs, thr=cfg.fg_thr), cfg)
        views.update({f"{name}@{k}": v for name, v in vk.items()})
    entries = []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec.get(typ) or {}).items():
            for r in lst:
                if r.get("_perception", {}).get("place_tag") == "fixed":
                    continue
                key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
                T0 = pose_to_T(r["default_pos"], r["default_ori"])
                sm = _shape_for(key, T0[:3, :3], r.get("_perception", {}).get("place_tag"))
                entries.append((r, key, sm, T0))

    def union_others(skip):
        out = {}
        for name, v in views.items():
            m = np.zeros((v.cam.h, v.cam.w), np.uint8)
            for r2, _, sm2, T2 in entries:
                if r2 is not skip:
                    m |= sm2.sil(v.cam, T2)
            out[name] = m
        return out

    for r, key, sm, T0 in entries:
        others = union_others(r)
        Rz = T0[:3, :3] @ sm.R0.T
        yaw0 = float(np.arctan2(Rz[1, 0], Rz[0, 0]))
        z_off = float(T0[2, 3] - (sm.pose(0, 0, 0)[2, 3]))
        zsm = _ZShift(sm, z_off)
        rvs = make_roi_views(zsm, T0, views, others)
        ex_rvs = [rv for rv in rvs if "@" in rv.name]
        info = {"accepted": False, "extra_views": [rv.name for rv in ex_rvs]}
        if ex_rvs:
            l0 = roi_loss(zsm, T0, rvs)[0]
            e0 = roi_loss(zsm, T0, ex_rvs)[0]
            x1, y1, yaw1, _, _ = nm_refine(zsm, T0[0, 3], T0[1, 3], yaw0, views, others, xy_step=0.004, yaw_step=0.06,
                                           maxiter=cfg.refine_iters, fix_yaw=(sm.yaw_period < 0.2))
            T1 = zsm.pose(x1, y1, yaw1)
            l1 = roi_loss(zsm, T1, rvs)[0]
            e1 = roi_loss(zsm, T1, ex_rvs)[0]
            info.update({"loss0": round(l0, 4), "loss1": round(l1, 4), "extra0": round(e0, 4), "extra1": round(e1, 4),
                         "moved_mm": round(float(np.hypot(*(T1[:2, 3] - T0[:2, 3])) * 1000), 1)})
            moved = float(np.hypot(*(T1[:2, 3] - T0[:2, 3])))
            # a large move must be backed by a look view that actually fits well (D0 stack_blocks 100003: a
            # 9 mm move toward a mediocre look-view fit, IoU 0.54, made a correct block wrong)
            if (l1 < l0 - accept_drop and e1 <= e0 + extra_slack and e1 < 0.6
                    and (moved <= big_move or e1 < good_fit)):
                info["accepted"] = True
                r["default_pos"] = T1[:3, 3].tolist()
                r["default_ori"] = R_to_quat_wxyz(T1[:3, :3]).tolist()
                p = r.setdefault("_perception", {})
                p["confidence"] = round(float(max(p.get("confidence", 0.0), 1 - e1)), 3)
        r.setdefault("_perception", {})["fused"] = info
    rec.setdefault("_perception", {})["fused_mode"] = "conservative"
    return rec


def export_look_poses(root, out, tasks=None, commit: str = "") -> dict:
    """Frame-0 estimate + look_plan for every D0 layout -> JSON for simlab to render post-look frames.
    Uses the 2-fold backgrounds/colour map like the evaluation (no GT is read)."""
    import json
    import os
    import zlib
    from pathlib import Path

    from . import d0
    from .eval_g1 import fold_backgrounds
    from .inverse_layout import EstimatorConfig

    root, out = Path(root), Path(out)
    cal = os.environ.get("DOJO_COLOR_CALIB")
    cal = json.loads(Path(cal).read_text()) if cal and Path(cal).exists() else None
    res = {"version": 1, "source": f"dojo.perception.look_plan @ {commit}", "estimated_from": "frame_0", "layouts": {}}
    if out.exists():  # resume: keep layouts already planned (their commit is recorded per layout)
        old = json.loads(out.read_text())
        res["layouts"] = old.get("layouts", {})
        res["source"] = old.get("source", res["source"]) + f"; resumed @ {commit}"
    tnames = tasks or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "logs")
    for t in tnames:
        for ep in d0.episodes(root, t):
            if f"{t}/{ep.name}" in res["layouts"]:
                continue
            obs = d0.load_obs(ep, 0)
            kw = {"color_M": cal[f"fold{zlib.crc32(ep.name.encode()) % 2}"]} if cal else {}
            rec = estimate(obs["images"], obs["state"], t, obs["instruction"], bgs=fold_backgrounds(ep), cfg=EstimatorConfig(**kw))
            plan = look_plan(obs, rec)
            res["layouts"][f"{t}/{ep.name}"] = {"parallel": True, "looks": [
                {"arm": p["arm"], "flange_pose": p["flange_pose"], "T_world_cam_cv": p["T_world_cam_cv"],
                 "look_at": p["target"], "purpose": p["purpose"], "label": p["label"]} for p in plan],
                "est_steps": look_cost_steps(plan), "commit": commit}
            print(t, ep.name, [(p["arm"], p["purpose"], p["label"]) for p in plan], flush=True)
            out.write_text(json.dumps(res, indent=1))  # incremental
    return res


if __name__ == "__main__":
    import argparse
    import os
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    export_look_poses(a.d0, a.out, a.tasks, commit or os.environ.get("DOJO_COMMIT", ""))
