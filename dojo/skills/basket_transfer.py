"""Opt-in measured transfers for the lift-first image-container sequence."""

from __future__ import annotations

from collections import Counter

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.executor import ArmPlan
from dojo.skills.geometry import inv_T, rot_z, slerp_R
from dojo.skills.kinematics import ARMS
from dojo.skills.motion import held_box
from dojo.skills.skills import SkillResult


def choose_arms(sk, target, container, velocity, ignore):
    """Reserve the picker from predicted belt phase after roughly 75 basket actions."""
    from dojo.skills import interactive as I

    future = I.plan_intercept(
        sk, target, velocity, t_min=75, t_max=225, lift=I.target_lift(sk),
        ignore=tuple(ignore) + (container,),
    )
    if future is not None:
        picker = future[0]
        return next(a for a in ARMS if a != picker), picker
    u = velocity / np.linalg.norm(velocity)
    holder = max(ARMS, key=lambda a: float(sk.kin.base(a)[:2, 3] @ u))
    return holder, next(a for a in ARMS if a != holder)


def pick_target(sk, target, container, velocity, picker, ignore):
    """Matchbelt lift/retry, with only the empty arm homed between attempts."""
    from dojo.skills import interactive as I

    # The held hollow basket's contact volume is excluded, as during loading.
    # Native physics remains active; all other obstacles and joint bounds remain.
    ignored = tuple(ignore) + (container,)
    obj = sk.refresh().obj(target)
    # Thin riders need time for the actual arm to reach its approach pose.
    timing = {"approach_settle": 24} if obj.top_z - obj.bottom_z < 0.030 else {}
    before = sk.ex.steps_used
    result = I.pick_from_belt(sk, target, velocity, lift=I.target_lift(sk),
                              arms=[picker], ignore=ignored, **timing)
    if result.status == "ik_fail" and sk.ex.steps_used == before:
        # Only a planning-only miss permits a new pull trajectory from this state.
        result = I.pick_from_belt(
            sk, target, velocity, lift=I.target_lift(sk), pull=0.10,
            arms=[picker], ignore=ignored, **timing,
        )
    if result.ok or result.status != "grasp_slip":
        return result
    failed = getattr(sk, "notes_belt_method", None)
    sk.home(arms=[picker], gripper=1.0)
    previous = getattr(sk, "belt_waist_first", None)
    try:
        # This also makes the waist fallback honor the rejected grasp family.
        sk.belt_waist_first = "on"
        return I.pick_from_belt(
            sk, target, velocity, lift=I.target_lift(sk), arms=[picker],
            ignore=ignored, skip_methods=(failed,) if failed else (), **timing,
        )
    finally:
        if previous is None:
            del sk.belt_waist_first
        else:
            sk.belt_waist_first = previous


def _failure(arm, status, note):
    return SkillResult(False, "basket_drop", status, 0, arm, note=note)


def drop_inset(sk, arm, label, container, ignore):
    """Clear the rim with a fixed wrist, then release at a reachable inset site."""
    w = sk.refresh()
    h, c, o = sk.held[arm], w.obj(container), w.obj(label)
    if h is None or h.label != label or c.asset.mesh is None or o.asset.mesh is None:
        return _failure(arm, "logic", "missing held object or mesh")
    tcp = sk.kin.tcp(arm, w.arms[arm].q)
    rel = inv_T(tcp) @ o.T
    vertices = np.asarray(o.asset.mesh[0])
    points = (tcp[:3, :3] @ (rel[:3, :3] @ vertices.T + rel[:3, 3:4])).T
    center = (points.max(0) + points.min(0)) / 2
    half = (points.max(0) - points.min(0)) / 2
    rim = G.box_rim_z(np.asarray(c.asset.mesh[0]))
    if rim is None:
        return _failure(arm, "no_plan", "no rim")
    mid = (c.asset.bbox_min + c.asset.bbox_max) / 2
    mid[2] = rim
    p = c.pos + c.R @ mid
    toward = sk.kin.base(arm)[:2, 3] - p[:2]
    toward /= max(float(np.linalg.norm(toward)), 1e-9)
    ignored = tuple(ignore) + (label, container)
    box = held_box(o, rel)
    candidates = []
    rejected = Counter()
    for shift in (0.08, 0.04, 0.0):
        for extra_y in (0.04, 0.0, -0.04):
            desired = p.copy()
            desired[:2] += shift * toward
            desired[1] += extra_y
            local = c.R.T @ (desired - c.pos)
            extent = np.abs(c.R).T @ half
            if (np.any(local[:2] - extent[:2] < c.asset.bbox_min[:2] + 0.015)
                    or np.any(local[:2] + extent[:2] > c.asset.bbox_max[:2] - 0.015)):
                rejected["footprint"] += 1
                continue
            goal = tcp.copy()
            goal[:3, 3] = [desired[0] - center[0], desired[1] - center[1],
                            desired[2] + 0.015 - points[:, 2].min()]
            waypoint = tcp.copy()
            waypoint[:2, 3] = 0.5 * (tcp[:2, 3] + goal[:2, 3])
            waypoint[2, 3] = max(tcp[2, 3], goal[2, 3]) + 0.005
            first = sk.motion.line(
                arm, sk.ex.q[arm], waypoint, 0.025, "clear_rim",
                gripper=0.0, held=box, ignore=ignored,
            )
            if not first.ok:
                rejected[f"clear:{first.status}"] += 1
                continue
            second = sk.motion.line(
                arm, first.q_end, goal, 0.025, "near_rim_drop",
                gripper=0.0, held=box, ignore=ignored,
            )
            if second.ok:
                candidates.append((first.steps + second.steps, first, second, desired))
            else:
                rejected[f"carry:{second.status}"] += 1
    sk.log(f"[basket-transfer] reachable sites {len(candidates)}/9, rejected {dict(rejected)}")
    if not candidates:
        return _failure(arm, "no_plan", "no fixed-wrist inset drop")
    _, first, second, desired = min(candidates, key=lambda row: row[0])
    sk.log(f"[basket-transfer] inset {desired.tolist()}")
    for move in (first, second):
        for segment in move.segments:
            segment.ease = 4
    sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).hold(6)}, name="squeeze check")
    w = sk.refresh()
    checked = inv_T(sk.kin.tcp(arm, w.arms[arm].q)) @ w.obj(label).T
    if np.linalg.norm(checked[:3, 3] - rel[:3, 3]) > 0.02:
        return _failure(arm, "grasp_slip", "drift during squeeze")
    sk.ex.run(
        {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(first).move(second).hold(3)},
        name="fixed wrist inset carry",
    )
    w = sk.refresh()
    checked = inv_T(sk.kin.tcp(arm, w.arms[arm].q)) @ w.obj(label).T
    if np.linalg.norm(checked[:3, 3] - rel[:3, 3]) > 0.04:
        return _failure(arm, "grasp_slip", "attachment lost during carry")
    delta = desired[:2] - w.obj(label).center_w[:2]
    if np.linalg.norm(delta) > 0.04:
        return _failure(arm, "grasp_slip", "drop alignment drift")
    if np.linalg.norm(delta) > 0.003:
        goal = sk.kin.tcp(arm, sk.ex.q[arm]).copy()
        goal[:2, 3] += delta
        correction = sk.motion.line(
            arm, sk.ex.q[arm], goal, 0.025, "near_correct", gripper=0.0, check=False,
        )
        if not correction.ok:
            return _failure(arm, "no_plan", "correction unreachable")
        sk.ex.run(
            {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(correction).hold(2)},
            name="near correct",
        )
    sk.ex.run(
        {arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).grip(1.0, 3).hold(25)},
        name="release inset",
    )
    sk.held[arm] = None
    return SkillResult(True, "basket_drop", "ok", 0, arm)


def lift_loaded(sk, container, target, holder, picker, ignore):
    """Pinch the other rim after release, then lift both attachments synchronously."""
    c = sk.refresh().obj(container)
    bottom0 = c.bottom_z
    base = sk.kin.base(picker)[:3, 3]
    grasps = G.box_rim_grasps(
        c, sk.kin, picker, depth=0.025, fracs=(-0.3, 0.0, 0.3), tilts=(0.0, 25.0),
    )
    grasps.sort(key=lambda g: np.linalg.norm(g.T[:2, 3] - base[:2]))
    for grasp in grasps:
        pre = grasp.T.copy()
        pre[:3, 3] -= grasp.T[:3, 0] * 0.06
        approach = sk.motion.to_pose(
            picker, sk.ex.q[picker], pre, 0.04, "second_rim_pre", gripper=1.0,
            ignore=tuple(ignore) + (container, target),
        )
        if not approach.ok:
            continue
        contact = sk.motion.line(
            picker, approach.q_end, grasp.T, 0.025, "second_rim", gripper=1.0, check=False,
        )
        if not contact.ok:
            continue
        sk.ex.run(
            {picker: ArmPlan(picker, sk.ex.q[picker].copy(), 1.0)
             .move(approach).move(contact).grip(0.0, 8)}, name="second rim grip",
        )
        for correction in range(3):
            w = sk.refresh()
            c = w.obj(container)
            relative = {a: inv_T(sk.kin.tcp(a, w.arms[a].q)) @ c.T for a in (holder, picker)}
            q = {a: sk.ex.q[a].copy() for a in relative}
            queues = {a: [] for a in relative}
            yaw = float(np.arctan2(c.R[1, 0], c.R[0, 0]))
            count = 40 if correction == 0 else 20
            for step in range(1, count + 1):
                desired = c.T.copy()
                if correction == 0:
                    desired[2, 3] += 0.10 * step / count
                # The rim rolls in the pads: correct half the measured tilt,
                # then remeasure rather than assuming a rigid attachment.
                desired[:3, :3] = slerp_R(c.R, rot_z(yaw), 0.5 * step / count)
                for arm in relative:
                    ik = sk.kin.ik(arm, desired @ inv_T(relative[arm]), q[arm])
                    if not ik.ok or np.max(np.abs(ik.q - q[arm])) > 0.15:
                        return False
                    q[arm] = ik.q
                    queues[arm].append((ik.q, 0.0))
            sk.ex.run_lists(queues, name="synchronous loaded lift")
            sk.ex.hold(8)
            c = sk.refresh().obj(container)
            tilt = float(np.degrees(np.arccos(np.clip(c.R[2, 2], -1.0, 1.0))))
            sk.log(f"[basket-transfer] loaded tilt {tilt:.2f}, bottom {c.bottom_z:.4f}")
            if tilt < 5.0:
                break
        w = sk.refresh()
        before = inv_T(w.obj(container).T) @ w.obj(target).T
        sk.ex.hold(40)
        w = sk.refresh()
        c = w.obj(container)
        after = inv_T(c.T) @ w.obj(target).T
        tilt = float(np.degrees(np.arccos(np.clip(c.R[2, 2], -1.0, 1.0))))
        return bool(c.bottom_z > bottom0 + 0.08 and tilt < 8.0
                    and np.linalg.norm(after[:3, 3] - before[:3, 3]) < 0.015)
    return False
