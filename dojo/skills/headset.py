"""Cradle carry experiment: turn clear of the support, then translate upright."""
from __future__ import annotations

import numpy as np

from dojo.skills.geometry import inv_T


def staged_carry(motion, arm, q0, T_goal, step_rad, name, *, gripper, held, ignore=(), ease=3):
    """Plan both legs before execution, retaining the ordinary collision validators.

    Rotate to the goal orientation above the current object position. Then use
    a Cartesian carry with fixed orientation: a joint-space carry can rotate
    the ear cups through the stand's pole while its bridge is still approaching.
    The caller owns any permitted final cradle contact region.
    """
    now = motion.kin.tcp(arm, q0) @ held.T_tcp_obj
    mid = T_goal @ held.T_tcp_obj
    mid[:2, 3] = now[:2, 3]
    mid[2, 3] = max(mid[2, 3], now[2, 3])
    T_mid = mid @ inv_T(held.T_tcp_obj)
    turn = motion.to_pose(arm, q0, T_mid, step_rad, name + "_rotate", gripper=gripper,
                          held=held, ignore=ignore, ease=ease)
    if not turn.ok:
        return turn
    carry = motion.line(arm, turn.q_end, T_goal, step_rad, name + "_upright", gripper=gripper,
                        held=held, ignore=ignore, check=True)
    return turn.then(carry)


def dense_grasps(sk, obj, arm):
    """Wide cradle pinches with denser opposed-face and approach coverage.

    The fixed RNG in the existing sampler retains reproducibility. This bank
    changes proposals only; native IK, collision and physical pick verification
    still decide what may execute.
    """
    from dojo.skills import grasps as G
    candidates = (G.box_grasps(obj, sk.kin, arm) + G.slice_grasps(obj, sk.kin, arm)
                  + G.antipodal_grasps(obj, sk.kin, arm, n_samples=400, n_approach=13))
    return [g for g in candidates if g.width >= 0.035]


def cup_grasps(sk, obj, arm, bridge_tag="headband_bridge"):
    """Add broad contact patches on narrow cups, away from the curved bridge.

    A small nominal slice width can still enclose a cup. These proposals only
    enter the hang bank after a verified staging transfer; the existing lift
    and loose-hold verifiers still decide whether a grasp may carry.
    """
    from dojo.skills import grasps as G
    from dojo.skills.library import functional_frames

    wide = dense_grasps(sk, obj, arm)
    frames = functional_frames(obj, bridge_tag, active=True) or functional_frames(obj, bridge_tag)
    if not frames or obj.asset.mesh is None:
        return _cup_opening(sk, wide)
    bridge = obj.T @ frames[0]
    narrow = []
    for g in G.antipodal_grasps(obj, sk.kin, arm, n_samples=800, n_approach=13):
        if not 0.018 <= g.width < 0.035:
            continue
        if float((g.T[:3, 3] - bridge[:3, 3]) @ bridge[:3, 2]) > -0.06:
            continue
        patch, _ = G.pad_contact(obj, g.T)
        if patch < 0.8 or getattr(G.pad_contact, "depth", 0.0) < 0.006:
            continue
        g.note = "cup " + g.note
        narrow.append(g)
    return _cup_opening(sk, wide + narrow)


def stage_for_cradle(sk, objects, rack, hook_tag, bridge_tag):
    """Use a checked table rest when pickup and cradle favor different arms.

    Keep the lying orientation; legacy executes at most one transfer per object,
    optional site retries allow four. Re-plan from the measured rest. Failures
    retain the original physical verdict and never manufacture a held state.
    """
    from dojo.skills.library import free_spots, hook_frames, on_hook, rest_with_R
    from dojo.skills.skills import SkillResult

    w = sk.refresh()
    hooks = hook_frames(w.obj(rack), hook_tag)
    staged = set()
    if not hooks:
        return staged
    for label in objects:
        if label not in sk.world.objects:
            continue
        o = sk.world.obj(label)
        if on_hook(o, hooks, bridge_tag) is not None:
            continue
        H = min(hooks, key=lambda h: float(np.linalg.norm(h[1][:2, 3] - o.pos[:2])))[1]
        nearest = lambda xy: min(("left", "right"),
                                 key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - xy)))
        arm = nearest(o.pos[:2])
        if arm == nearest(H[:2, 3]) or float(np.linalg.norm(o.pos[:2] - H[:2, 3])) <= 0.40:
            continue
        sites = GraspSites() if getattr(sk, "headset_retry", "legacy") == "sites" else None
        for _attempt in range(4 if sites is not None else 1):
            o = sk.world.obj(label)
            executed = False
            result = None
            for xy in free_spots(sk, label, near_xy=(0.0, -0.15), clearance=0.03, avoid=(rack,))[:8]:
                target = rest_with_R(sk, label, xy, o.R)
                if getattr(sk, "headset_stage", "off") in ("mesh_center", "mesh_clear"):
                    target = mesh_support_target(o, target)
                def grasps(a):
                    candidates = cup_grasps(sk, o, a, bridge_tag)
                    return sites.filter(o, a, candidates) if sites is not None else candidates
                plans = plan_stage(sk, label, target, arms=[arm], max_full=24, grasps=grasps,
                                   pick_kw=cradle_pick_kwargs(sk, seat=True),
                                   place_kw={"pre_h": 0.03, "approach_rad": 0.03, "approach_line": True})
                if isinstance(plans, SkillResult):
                    sk.log(f"[headset_stage] {label} {xy}: {plans.note}")
                    continue
                executed = True
                result = sk.run_transfer(*plans, tol_xy=0.015, tol_tilt=25.0, tol_z=0.05)
                sk.log(f"[headset_stage] {label} {arm}: {result.status} {result.note}")
                if sites is not None and result.status in ("loose", "grasp_slip", "grasp_unusable"):
                    sites.remember_pick(sk, plans[0])
                sk.bank()
                sk.refresh()
                if result.ok:
                    staged.add(label)
                break
            if not executed or label in staged or result.status not in ("loose", "grasp_slip", "grasp_unusable"):
                break
    return staged


def plan_stage(sk, label, target, **kw):
    """Screen at the release height; verify the actual ground rest after opening.

    The optional clearance leaves room for the attached hand above the table.
    Repairing a refined pickup must retain this distinction, too.
    """
    if getattr(sk, "headset_stage", "off") != "mesh_clear":
        return sk.plan_transfer(label, [target], **kw)
    from dojo.skills.motion import held_box
    from dojo.skills.skills import Held, PlacePlan, SkillResult

    gap = 0.025
    raised = target.copy()
    raised[2, 3] += gap
    actual = {**(kw.get("place_kw") or {}), "release_gap": gap}
    screened = sk.plan_transfer(label, [raised], **{**kw, "place_kw": {**actual, "release_gap": 0.0}})
    if isinstance(screened, SkillResult):
        return screened
    pick, _ = screened
    obj = sk.world.obj(label)
    held = Held(label, pick.T_tcp_obj, held_box(obj, pick.T_tcp_obj), pick.grasp)
    place_kw = {"ignore": tuple(kw.get("ignore", ())), **actual}
    place = sk.plan_place(pick.arm, target, q_start=pick.plan.q_end, held=held, **place_kw)
    if not isinstance(place, PlacePlan):
        return place
    place.transfer_kw = place_kw
    pick.repair = lambda q: plan_stage(sk, label, target, **{**kw, "arms": [pick.arm], "q_start": q})
    return pick, place


class GraspSites:
    """Failed pinch neighborhoods in the object frame, separately for each arm."""

    def __init__(self):
        self.rejected = {}

    def remember_pick(self, sk, planned):
        actual = getattr(sk, "last_pick", None)
        pick = actual if (actual is not None and actual.label == planned.label
                          and actual.arm == planned.arm) else planned
        self.rejected.setdefault((pick.label, pick.arm), []).append(inv_T(pick.T_tcp_obj))

    def filter(self, obj, arm, candidates):
        bad = self.rejected.get((obj.label, arm), ())
        if not bad:
            return candidates
        out = []
        T_inv = inv_T(obj.T)
        cos_close, cos_approach = np.cos(np.deg2rad(20)), np.cos(np.deg2rad(30))
        for g in candidates:
            local = T_inv @ g.T
            duplicate = any(float(np.linalg.norm(local[:3, 3] - T[:3, 3])) < 0.015
                            and abs(float(local[:3, 1] @ T[:3, 1])) > cos_close
                            and float(local[:3, 0] @ T[:3, 0]) > cos_approach for T in bad)
            if not duplicate:
                out.append(g)
        return out


def _cup_opening(sk, candidates):
    # A visual slice can be narrower than the collider swept during descent.
    if getattr(sk, "headset_grasp", "legacy") == "cups":
        from dojo.skills.grasps import MAX_GAP
        for g in candidates:
            g.clear_w = MAX_GAP
    return candidates


def mesh_support_target(obj, target):
    """A planned rest with actual mesh clearance, keeping the original xy/R.

    Empty OBB corners can otherwise leave a sparse headset high enough to tip
    when released. Native hand, object and scene checks still validate this
    target and its full path. Invalid geometry retains the bbox target.
    """
    from dojo.skills.kinematics import TABLE_Z
    mesh = obj.asset.mesh
    if mesh is None:
        return target
    try:
        vertices = np.asarray(mesh[0], float)
    except (TypeError, ValueError):
        return target
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        return target
    out = target.copy()
    out[2, 3] = TABLE_Z + .003 - float((vertices @ target[:3, :3].T)[:, 2].min())
    return out


def cradle_pick_kwargs(sk, *, seat):
    """Slow the checked lift's load onset without changing its hold verifier."""
    kw = {"squeeze": 8} if seat else {}
    if getattr(sk, "headset_lift", "legacy") == "slow":
        kw["lift_rad"] = .03
    return kw
