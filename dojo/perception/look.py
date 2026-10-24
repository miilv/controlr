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


def default_look(target: np.ndarray, arm: str, dist: float = 0.14, elev_deg: float = 55.0) -> np.ndarray:
    """Wrist camera pose looking at ``target`` from the arm's side."""
    from .cameras import WRIST, intrinsics

    back = ARM_BASES[arm][:2] - target[:2]
    back = back / (np.linalg.norm(back) + 1e-9)
    e = np.deg2rad(elev_deg)
    eye = target + dist * np.r_[np.cos(e) * back, np.sin(e)]
    return look_at_camera(eye, target, intrinsics(WRIST["focal_length"], WRIST["horizontal_aperture"])).T_wc


def _records(record: dict):
    for typ in ("Rigid", "Geometry", "Articulation"):
        for lst in (record.get(typ) or {}).values():
            yield from lst


def look_plan(obs: dict, record: dict, max_looks: int = 2) -> list[dict]:
    """Up to one look per arm, most useful first. ``obs`` is unused beyond the arm state (kept for the API)."""
    cands = []  # (priority, arm, purpose, label, target, T_world_cam)
    for r in _records(record):
        p = r.get("_perception", {})
        pos = np.asarray(r["default_pos"], dtype=float)
        if p.get("place_tag") == "fixed":
            continue
        arm = "left" if pos[0] < 0 else "right"
        if p.get("review"):
            rv = p["review"]
            cands.append((0, rv["arm"], "review", r["label"], np.asarray(rv["look_at"]), np.asarray(rv["T_world_cam_cv"])))
        elif p.get("occluded"):
            cands.append((0, arm, "occluded", r["label"], pos, default_look(pos, arm)))
    if record.get("_not_found"):
        for arm, tgt in HIDDEN_UNDER_GRIPPERS.items():
            cands.append((1, arm, "not_found:" + ",".join(x["label"] for x in record["_not_found"]), None, tgt,
                          default_look(tgt, arm)))
    for r in sorted(_records(record), key=lambda r: r.get("_perception", {}).get("confidence", 1.0)):
        p = r.get("_perception", {})
        if p.get("place_tag") == "fixed" or p.get("confidence", 1.0) >= 0.8:
            continue
        pos = np.asarray(r["default_pos"], dtype=float)
        arm = "left" if pos[0] < 0 else "right"
        cands.append((2, arm, "low_confidence", r["label"], pos, default_look(pos, arm)))
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


def fuse(frames: list[dict], task: str, bgs: dict | None = None, cfg=None) -> dict:
    """Estimate from the first frame with later frames as extra views (objects assumed static).
    ``frames``: [{"images": {cam: RGB}, "state": {...}, "instruction": str}, ...]."""
    f0 = frames[0]
    extra = [(f["images"], f["state"]) for f in frames[1:]]
    return estimate(f0["images"], f0["state"], task, f0.get("instruction", ""), bgs=bgs, cfg=cfg, extra_frames=extra)
