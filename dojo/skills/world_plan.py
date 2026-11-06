"""Bounded free-space routing on the skill's belief, with no simulator dependencies.

The attachment is a HeldBox in TCP coordinates. Container parts/height maps and
per-object allowances are inherited from the caller's Collision, never reconstructed
from simulator poses. This is a sampled collision model, not a safety guarantee.
"""
from __future__ import annotations

import copy
import time

import numpy as np

BUDGET_S = 0.75
MAX_ROUTES = 8


def plan_objects(motion, arm, q0, target, step_rad, name, gripper, ignore, held, ease):
    from dojo.skills.motion import Motion

    if motion.plan_solver == "curobo":
        from dojo.skills.curobo_world import plan_curobo
        return plan_curobo(motion, arm, q0, target, step_rad, name, gripper, ignore, held, ease)
    started = time.perf_counter()
    event = {"arm": arm, "name": name, "fallback": True, "routes": 0}
    # Keep the caller's contact models and margins; include static Geometry too.
    coll = copy.copy(motion.coll)
    coll.include = set(coll.include) | {
        o.label for o in coll.world.objects.values() if o.obj_type == "Geometry"
    }
    planner = Motion(motion.kin, coll)
    target = np.asarray(target, float)
    q0 = np.asarray(q0, float)
    kwargs = dict(step_rad=step_rad, name=name, gripper=gripper, ignore=ignore, held=held)
    try:
        # A Cartesian transit preserves object clearance during orientation changes
        # better than a straight joint interpolation; try it before joint-space vias.
        direct = planner.line(arm, q0, target, **kwargs)
        event["routes"] += 1
        if direct.ok:
            for segment in direct.segments:
                segment.ease = ease
            event.update(fallback=False, route="line")
            return direct
        start = motion.kin.tcp(arm, q0)
        midpoint = (start[:3, 3] + target[:3, 3]) / 2
        # Deterministic workspace detours, both sides of the straight-line corridor.
        delta = target[:2, 3] - start[:2, 3]
        normal = np.array([-delta[1], delta[0]]) / max(np.linalg.norm(delta), 1e-9)
        if np.linalg.norm(normal) < 0.5:
            normal = np.array([1., 0.])
        base_z = max(start[2, 3], target[2, 3])
        routes = [(0., 0.06), (0., 0.10), (0.08, 0.03), (-0.08, 0.03),
                  (0.14, 0.06), (-0.14, 0.06), (0., 0.16)]
        for offset, dz in routes[:MAX_ROUTES - 1]:
            # Cooperative budget: one IK/checked segment can overrun the deadline;
            # no new segment is started after expiry. Never execute a partial plan.
            if time.perf_counter() - started >= BUDGET_S:
                break
            via = start.copy()
            via[:2, 3] = midpoint[:2] + offset * normal
            via[2, 3] = base_z + dz
            event["routes"] += 1
            first = planner.line(arm, q0, via, **kwargs)
            if not first.ok or time.perf_counter() - started >= BUDGET_S:
                continue
            second = planner.line(arm, first.q_end, target, **kwargs)
            if second.ok:
                result = first.then(second)
                for segment in result.segments:
                    segment.ease = ease
                event.update(fallback=False, route=f"via:{offset}:{dz}")
                return result
        return None
    finally:
        event["seconds"] = round(time.perf_counter() - started, 6)
        motion.world_plans.append(event)
