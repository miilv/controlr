"""Opt-in basket hook: reobserve after transit, keep the wall brace until loaded."""

from __future__ import annotations

import numpy as np

from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import T_from
from dojo.skills.motion import CARRY, SLOW
from dojo.skills.skills import OPEN, SkillResult


def hook_pose(o, base):
    """Horizontal fingers across the long top bar, below its underside."""
    if o.asset.mesh is None:
        return None
    v = np.asarray(o.asset.mesh[0], float)
    band = v[v[:, 2] >= v[:, 2].max() - 0.02]
    if len(band) < 8:
        return None
    _, _, vt = np.linalg.svd(band[:, :2] - band[:, :2].mean(0), full_matrices=False)
    d = o.R @ np.r_[vt[0], 0.0]
    d[2] = 0
    d /= max(1e-9, np.linalg.norm(d))
    p = o.pos + o.R @ np.r_[band[:, :2].mean(0), v[:, 2].max() - 0.011]
    a = np.array([-d[1], d[0], 0.0])
    if a @ (p - base) < 0:
        a = -a
    return T_from(p - [0, 0, 0.025], np.stack([a, d, np.cross(a, d)], axis=1))


def braced_hook(sk, container, arm, holder, ign):
    """Transit then refresh the hook aim; load it before releasing the wall brace."""

    def fail(status, note):
        return SkillResult(False, "braced_hook", status, 0, arm, note=note)

    try:
        o = sk.refresh().obj(container)
        t = hook_pose(o, sk.kin.base(arm)[:3, 3])
        if t is None:
            return fail("no_plan", "no handle mesh")
        pre = t.copy()
        pre[:3, 3] -= t[:3, 0] * 0.08
        mv = sk.motion.to_pose(
            arm,
            sk.ex.q[arm],
            pre,
            CARRY,
            "hook_pre",
            gripper=0.0,
            ignore=tuple(ign) + (container,),
        )
        if not mv.ok:
            return fail(mv.status, "hook pre")
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(mv, "hook_pre")},
            name="hook transit",
        )
        # The wall pinch is a brace, not a rigid attachment: use its CURRENT pose.
        t = hook_pose(sk.refresh().obj(container), sk.kin.base(arm)[:3, 3])
        if t is None:
            return fail("no_plan", "handle lost")
        mv = sk.motion.line(
            arm, sk.ex.q[arm], t, SLOW, "hook_in", gripper=0.0, check=False
        )
        if not mv.ok:
            return fail(mv.status, "hook in")
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(mv, "hook_in")},
            name="hook contact",
        )
        z0 = float(sk.refresh().obj(container).bottom_z)
        # Lift both contacts together while the bar seats on the closed fingers.
        plans = {}
        for a in (arm, holder):
            t = sk.kin.tcp(a, sk.ex.q[a]).copy()
            t[2, 3] += 0.04
            mv = sk.motion.line(
                a, sk.ex.q[a], t, SLOW, "hook_load", gripper=0.0, check=False
            )
            if not mv.ok:
                return fail(mv.status, f"brace load {a}")
            plans[a] = ArmPlan(a, sk.ex.q[a].copy(), 0.0).move(mv, "hook_load")
        sk.ex.run(plans, name="load hook with brace")
        sk.ex.run(
            {
                holder: ArmPlan(holder, sk.ex.q[holder].copy(), 0.0).grip(
                    OPEN, 3, "release"
                )
            },
            name="release brace",
        )
        sk.held[holder] = None
        for dz in (0.12, 0.10):
            t = sk.kin.tcp(arm, sk.ex.q[arm]).copy()
            t[2, 3] += dz
            mv = sk.motion.line(
                arm, sk.ex.q[arm], t, SLOW, "hook_lift", gripper=0.0, check=False
            )
            if not mv.ok:
                continue
            sk.ex.run(
                {
                    arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0)
                    .move(mv, "hook_lift")
                    .hold(8, "still")
                },
                name="hook lift",
            )
            rose = float(sk.refresh().obj(container).bottom_z) - z0
            sk.log(f"[basket] braced hook bottom rise {rose * 1000:.1f} mm")
            return sk._record(
                SkillResult(
                    rose > 0.6 * dz,
                    "braced_hook",
                    "ok" if rose > 0.6 * dz else "grasp_slip",
                    0,
                    arm,
                    note=f"bottom rose {rose * 1000:.1f} mm",
                )
            )
        return fail("ik_fail", "hook lift unreachable")
    except (BudgetExceeded, EpisodeEnded) as e:
        return fail("step_budget", str(e))


def route(text, world, routes):
    """Require the pictured conveyor instruction and unambiguous observed roles."""
    import re

    if "basket_image" not in routes or not all(
        re.search(p, text)
        for p in (
            r"\blift\b",
            r"\bbasket\b",
            r"\bconveyor\b",
            r"\bimage\b|\bpicture\b|\bphoto\b",
        )
    ):
        return None
    objs = list(world.task_objects())
    baskets = [
        o for o in objs if o.obj_type == "Rigid" and "basket" in o.category.lower()
    ]
    belts = [
        o for o in objs if o.obj_type == "Dynamic" and "conveyor" in o.category.lower()
    ]
    pictures = [
        o
        for o in objs
        if o.obj_type in ("Geometry", "Rigid")
        and re.search(r"photo|picture|image", o.category.lower())
    ]
    if len(baskets) != 1 or len(belts) != 1 or len(pictures) != 1:
        return None
    excluded = {baskets[0].label, pictures[0].label}
    movers = [
        o.label
        for o in objs
        if o.obj_type == "Rigid" and not o.clutter and o.label not in excluded
    ]
    if not movers:
        return None
    return (
        "pick_into_held",
        dict(
            objects=movers,
            container=baskets[0].label,
            picture=pictures[0].label,
            conveyor=belts[0].label,
        ),
        "pictured object into basket",
    )
