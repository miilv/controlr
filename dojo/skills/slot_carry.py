"""Bounded, measurement-driven slot transit. No oracle reads outside G3's provider."""
from __future__ import annotations

import copy
import math


from dojo.skills.executor import ArmPlan
from dojo.skills.geometry import inv_T, slerp_R
from dojo.skills.motion import held_box


def measured_approach(sk, pp, commands, tol_xy, tol_tilt):
    """Return (handled, failure). Four checked hops; unavailable first read keeps the legacy path.

    G3 reads its measured world. Perception requests only a current-view held read at each
    stop, under the existing sigma gate; no observer/show moves are introduced. A later
    unavailable read keeps the last accepted attachment. An infeasible hop stops motion and retains the grasp.
    """
    from dojo.skills.skills import Held

    arm = pp.arm
    original = sk.held[arm]
    if original is None or not commands:
        return False, None
    if not sk.remeasure and getattr(sk.provider, 'held_pose', None) is None:
        return False, None
    request = copy.copy(pp)
    request.tail_kw = {**pp.tail_kw, 'servo': False, 'held_read': True}
    start = None
    end = sk.kin.tcp(arm, commands[-1][0]) @ original.T_tcp_obj
    stops = sorted({math.ceil(len(commands) * k / 4) - 1 for k in range(1, 5)})
    rate = pp.approach.segments[0].step_rad
    ease = pp.approach.segments[0].ease
    for hop, index in enumerate(stops):
        world = sk.refresh()
        obj = world.obj(pp.label)
        h = sk.held[arm]
        measured = (inv_T(sk.kin.tcp(arm, world.arms[arm].q)) @ obj.T if sk.remeasure else
                    sk._held_protocol(request, h, obj, tol_xy, tol_tilt))
        if measured is None and hop == 0:
            return False, None
        if measured is not None:
            h = Held(h.label, measured, held_box(obj, measured), h.grasp)
            sk.held[arm] = h
        if hop == 0:
            start = sk.kin.tcp(arm, world.arms[arm].q) @ h.T_tcp_obj
        target = sk.kin.tcp(arm, commands[index][0]) @ original.T_tcp_obj
        # The object stays level between the measured start and the intended hover;
        # joint interpolation's intermediate wrist tilt is not an object target.
        target[:3, :3] = slerp_R(start[:3, :3], end[:3, :3], (index + 1) / len(commands))
        tcp = target @ inv_T(h.T_tcp_obj)
        move = sk.motion.line(arm, sk.ex.q[arm], tcp, rate, 'slot_carry', gripper=0.0,
                              held=h.box, ignore=(h.label,))
        if not move.ok:
            return True, f'hop {hop + 1}: {move.note}'
        for segment in move.segments:
            segment.ease = ease
        plan = ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(move).hold(1, 'carry_read')
        sk.ex.run({arm: plan}, name=f'place {pp.label} measured carry {hop + 1}')
    return True, None
