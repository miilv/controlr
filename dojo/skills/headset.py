"""Cradle carry experiment: turn clear of the support, then translate upright."""
from __future__ import annotations

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
