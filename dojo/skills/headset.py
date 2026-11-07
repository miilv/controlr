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
        return wide
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
    return wide + narrow


def stage_for_cradle(sk, objects, rack, hook_tag, bridge_tag):
    """Use a checked table rest when pickup and cradle favor different arms.

    Keep the lying orientation, execute at most one transfer per object, then
    let the ordinary hang planner re-grasp from the measured rest. Failures
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
        for xy in free_spots(sk, label, near_xy=(0.0, -0.15), clearance=0.03, avoid=(rack,))[:8]:
            target = rest_with_R(sk, label, xy, o.R)
            plans = sk.plan_transfer(label, [target], arms=[arm], max_full=24,
                                     grasps=lambda a: cup_grasps(sk, o, a, bridge_tag),
                                     pick_kw={"squeeze": 8},
                                     place_kw={"pre_h": 0.03, "approach_rad": 0.03, "approach_line": True})
            if isinstance(plans, SkillResult):
                sk.log(f"[headset_stage] {label} {xy}: {plans.note}")
                continue
            result = sk.run_transfer(*plans, tol_xy=0.015, tol_tilt=25.0, tol_z=0.05)
            sk.log(f"[headset_stage] {label} {arm}: {result.status} {result.note}")
            sk.bank()
            sk.refresh()
            if result.ok:
                staged.add(label)
            break
    return staged
