"""Opt-in lift-first conveyor sequence using only the current skill world."""

from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import inv_T, rot_y, rot_z
from dojo.skills.kinematics import ARMS, down_R
from dojo.skills.motion import SLOW, held_box
from dojo.skills.skills import Held, SkillResult


@contextmanager
def profile(sk):
    """Restrict joints for this sequence; restore all experimental state even on exit."""
    old = {k: getattr(sk, k, "off") for k in ("belt_container", "belt_deposit")}
    limits = sk.kin.limits.copy()
    mode = sk.joint_limits
    try:
        sk.joint_limits = "measured"
        # Intersect rather than widen a caller's tighter envelope.
        sk.kin.limits[:, 0] = np.maximum(sk.kin.limits[:, 0], limits[:, 0])
        sk.kin.limits[:, 1] = np.minimum(sk.kin.limits[:, 1], limits[:, 1])
        sk.kin.__dict__.pop("_ik_cache", None)
        sk.belt_container = sk.belt_deposit = "on"
        yield
    finally:
        for k, value in old.items():
            setattr(sk, k, value)
        sk.joint_limits = mode
        sk.kin.limits = limits
        sk.kin.__dict__.pop("_ik_cache", None)


def end_grasps(sk, obj, arm, u, side=1):
    gs = G.box_rim_grasps(
        obj, sk.kin, arm, depth=0.012, fracs=(-0.3, -0.1, 0.0, 0.1, 0.3), tilts=(25.0,)
    )
    gs = [g for g in gs if side * float((g.T[:2, 3] - obj.center_w[:2]) @ u[:2]) > 0.10]
    perp = np.array([-u[1], u[0]])
    return sorted(gs, key=lambda g: abs(float((g.T[:2, 3] - obj.center_w[:2]) @ perp)))


def translate(sk, arm, dz, name, hold=5):
    t = sk.kin.tcp(arm, sk.ex.q[arm]).copy()
    t[2, 3] += dz
    mv = sk.motion.to_pose(arm, sk.ex.q[arm], t, SLOW, name, gripper=0.0, check=False)
    if mv.ok:
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(mv).hold(hold)}, name=name
        )
    return mv.ok


def level_hold(sk, container, arm, initial_R, xy, bottom):
    """Replan from the measured attachment, rather than assuming the rim did not rotate."""
    w = sk.refresh()
    obj = w.obj(container)
    rel = inv_T(sk.kin.tcp(arm, w.arms[arm].q)) @ obj.T
    local_bottom = float((np.asarray(obj.asset.mesh[0]) @ initial_R.T)[:, 2].min())
    desired = obj.T.copy()
    desired[:3, :3] = initial_R
    desired[:3, 3] = [*xy, bottom - local_bottom]
    goal = desired @ inv_T(rel)
    mv = sk.motion.to_pose(
        arm,
        sk.ex.q[arm],
        goal,
        SLOW,
        "basket_level",
        gripper=0.0,
        ignore=(container,),
        check=False,
    )
    if mv.ok:
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(mv).hold(8)},
            name="basket level",
        )
        w = sk.refresh()
        obj = w.obj(container)
        new_rel = inv_T(sk.kin.tcp(arm, w.arms[arm].q)) @ obj.T
        sk.held[arm] = Held(
            container, new_rel, held_box(obj, new_rel), sk.held[arm].grasp
        )
    tilt = float(np.degrees(np.arccos(np.clip(obj.R[2, 2], -1.0, 1.0))))
    sk.log(
        "[basket-first] "
        + json.dumps(dict(status=mv.status, bottom=obj.bottom_z, tilt=tilt))
    )
    return mv.ok, obj.bottom_z, tilt


def drop_center(sk, arm, label, container, ignore):
    """Checked slow transit, then a bounded contact correction before releasing centrally."""
    w = sk.refresh()
    h = sk.held[arm]
    c, o = w.obj(container), w.obj(label)
    if h is None or h.label != label or c.asset.mesh is None or o.asset.mesh is None:
        return SkillResult(
            False, "basket_drop", "logic", 0, arm, note="missing held object or mesh"
        )
    mid = (c.asset.bbox_min + c.asset.bbox_max) / 2
    rim = G.box_rim_z(np.asarray(c.asset.mesh[0]))
    if rim is None:
        return SkillResult(False, "basket_drop", "no_plan", 0, arm, note="no rim")
    mid[2] = rim
    p = c.pos + c.R @ mid
    base = sk.kin.base(arm)[:3, 3]
    ang = float(np.arctan2(p[1] - base[1], p[0] - base[0]))
    rotations = [sk.kin.tcp(arm, sk.ex.q[arm])[:3, :3]] + [
        rot_z(ang) @ rot_y(-np.radians(tilt)) @ rot_z(-ang) @ down_R(np.pi * yaw / 12)
        for tilt in (0.0, 20.0, 40.0, 60.0, 75.0)
        for yaw in range(12)
    ]
    vertices = np.asarray(o.asset.mesh[0])
    for rotation in rotations:
        points = (
            rotation @ (h.T_tcp_obj[:3, :3] @ vertices.T + h.T_tcp_obj[:3, 3:4])
        ).T
        center = (points.max(0) + points.min(0)) / 2
        goal = np.eye(4)
        goal[:3, :3] = rotation
        goal[:3, 3] = [
            p[0] - center[0],
            p[1] - center[1],
            p[2] + 0.02 - points[:, 2].min(),
        ]
        mv = sk.motion.to_pose(
            arm,
            sk.ex.q[arm],
            goal,
            SLOW,
            "basket_drop",
            gripper=0.0,
            held=h.box,
            ignore=tuple(ignore) + (label, container),
        )
        if not mv.ok:
            continue
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(mv).hold(3)},
            name="basket drop",
        )
        w = sk.refresh()
        c, o = w.obj(container), w.obj(label)
        delta = c.center_w[:2] - o.center_w[:2]
        distance = float(np.linalg.norm(delta))
        sk.log(
            "[basket-first] "
            + json.dumps(
                dict(
                    drop_xy_error=distance,
                    target_bottom=o.bottom_z,
                    basket_bottom=c.bottom_z,
                )
            )
        )
        if distance >= 0.08:
            return SkillResult(
                False,
                "basket_drop",
                "grasp_slip",
                0,
                arm,
                note="target lost before release",
            )
        if distance > 0.003:
            goal = sk.kin.tcp(arm, sk.ex.q[arm]).copy()
            goal[:2, 3] += delta
            mv = sk.motion.line(
                arm, sk.ex.q[arm], goal, SLOW, "drop_correct", gripper=0.0, check=False
            )
            if not mv.ok:
                return SkillResult(
                    False,
                    "basket_drop",
                    mv.status,
                    0,
                    arm,
                    note="center correction unreachable",
                )
            sk.ex.run(
                {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(mv).hold(2)},
                name="drop correct",
            )
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).grip(1.0, 3).hold(20)},
            name="release center",
        )
        sk.held[arm] = None
        return SkillResult(True, "basket_drop", "ok", 0, arm)
    return SkillResult(
        False, "basket_drop", "no_plan", 0, arm, note="no central release pose"
    )


def lift_first(sk, container, target, velocity, ignore):
    from dojo.skills import interactive as I

    u = np.r_[velocity / np.linalg.norm(velocity), 0.0]
    holder = max(ARMS, key=lambda a: float(sk.kin.base(a)[:3, 3] @ u))
    picker = next(a for a in ARMS if a != holder)
    try:
        with profile(sk):
            from dojo.skills import basket_transfer as transfer

            recover = getattr(sk, "belt_first_recover", "off") == "on"
            phase = getattr(sk, "belt_first_phase", "off") == "on"
            if phase:
                holder, picker = transfer.choose_arms(sk, target, container, velocity, ignore)
            obj = sk.refresh().obj(container)
            if obj.asset.mesh is None:
                return dict(ok=False, stage="basket", note="no basket mesh")
            initial_R, bottom0 = obj.R.copy(), obj.bottom_z
            holder_side = 1 if float((sk.kin.base(holder)[:3, 3]
                                      - sk.kin.base(picker)[:3, 3]) @ u) >= 0 else -1
            grasps = (
                end_grasps(sk, obj, holder, u, side=holder_side)
                if phase else end_grasps(sk, obj, holder, u)
            )
            rel = {holder: [(g, g.T[:3, 3] - obj.pos, I.UP) for g in grasps]}
            plan = I._intercept_search(
                sk,
                container,
                obj,
                np.r_[velocity, 0.0],
                rel,
                [holder],
                25,
                600,
                8,
                5,
                0.06,
                0.10,
                ignore,
                0.10,
            )
            if plan is None:
                return dict(ok=False, stage="basket intercept", note="no end-wall lift")
            result = I.pick_from_belt(
                sk,
                container,
                velocity,
                lift=0.10,
                pull=0.10,
                min_rise=0.015,
                arms=[holder],
                ignore=ignore,
                planned_intercept=plan,
            )
            if not result.ok:
                return dict(ok=False, stage="basket pick", note=result.note)
            mid = 0.5 * (sk.kin.base("left")[:3, 3] + sk.kin.base("right")[:3, 3])
            side = holder_side if phase else 1
            xy = (mid[0] + side * 0.05 * u[0], mid[1] + 0.27)
            ready = False
            for _ in range(4):
                reached, bottom, tilt = level_hold(
                    sk, container, holder, initial_R, xy, float(mid[2]) + 0.115
                )
                required_rise = 0.08 if recover else 0.068
                ready = reached and bottom > bottom0 + required_rise and tilt < 28.0
                if ready:
                    break
            if not ready:
                return dict(
                    ok=False, stage="basket hold", note="no stable elevated hold"
                )
            if recover:
                result = transfer.pick_target(sk, target, container, velocity, picker, ignore)
            else:
                result = I.pick_from_belt(
                    sk,
                    target,
                    velocity,
                    lift=I.target_lift(sk),
                    arms=[picker],
                    ignore=ignore,
                )
            if not result.ok:
                return dict(ok=False, stage="target pick", note=result.note)
            # Load from a lower airborne pose to bring the central release within reach.
            translate(sk, holder, -0.05, "lower in air")
            result = (transfer.drop_inset(sk, picker, target, container, ignore)
                      if recover else drop_center(sk, picker, target, container, ignore))
            if not result.ok:
                return dict(ok=False, stage="target carry/drop", note=result.note)
            lifted = (transfer.lift_loaded(sk, container, target, holder, picker, ignore)
                      if recover else translate(sk, holder, 0.08, "loaded basket lift", hold=20))
            return dict(
                ok=lifted,
                stage="loaded lift",
                note="" if lifted else "loaded lift unreachable",
            )
    except (BudgetExceeded, EpisodeEnded) as exc:
        return dict(ok=False, stage="ended", note=str(exc))
