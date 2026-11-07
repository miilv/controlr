"""Instruction-grounded classification using the native skill executor.

All decisions consume the current WorldState; no task or layout identifiers.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

from dojo.skills.executor import BudgetExceeded, EpisodeEnded
from dojo.skills.kinematics import ARMS
from dojo.skills.skills import PickPlan


def assignment(world, kin):
    """Exact category/basket assignment with a distance lower bound.

    Fixed clauses always win. An incomplete fixed instruction is never made free.
    """
    objects = list(world.task_objects())
    baskets = sorted((o for o in objects if re.search(r"basket|\bbin\b", o.category)),
                     key=lambda o: float(o.center_w[0]))
    basket_labels = {b.label for b in baskets}
    groups = {}
    for o in objects:
        if o.obj_type == "Rigid" and o.label not in basket_labels and o.identity_known:
            groups.setdefault(o.category, []).append(o)
    text = world.instruction.lower()
    clauses = re.findall(r"([a-z_ ]+?) objects? into the (left|middle|center|centre|right) basket", text)
    if clauses:
        out = {}
        for phrase, side in clauses:
            matches = [c for c in groups if re.search(r"\b" + re.escape(c.replace("_", " ")) + r"\b",
                                                                    phrase.replace("_", " "))]
            if len(matches) == 1 and len(baskets) == 3:
                out[matches[0]] = baskets[{"left": 0, "middle": 1, "center": 1, "centre": 1, "right": 2}[side]].label
        if len(set(out.values())) != len(out):
            return {}, groups, None
        return out, groups, None
    if not re.search(r"\b(sort|classify|group)\b.*\bcategor", text) or len(groups) > len(baskets):
        return {}, groups, None
    cats = sorted(groups)
    def cost(perm):
        total = 0.0
        for cat, basket in zip(cats, perm):
            for o in groups[cat]:
                total += min(float(np.linalg.norm(o.center_w[:2] - kin.base(a)[:2, 3])
                                  + np.linalg.norm(basket.center_w[:2] - kin.base(a)[:2, 3])) for a in ARMS)
        return total
    if not cats:
        return {}, groups, None
    best = min(itertools.permutations(baskets, len(cats)), key=cost)
    return dict(zip(cats, (b.label for b in best))), groups, cost(best)


def classify(sk):
    """Sequential central placement, at most one budgeted repair per instance."""
    from dojo.skills.library import _inside
    mapping, groups, cost = assignment(sk.refresh(), sk.kin)
    events = []
    sk.world.notes["classify2"] = {"assignment": mapping, "cost": cost, "events": events}
    todo = [(o.label, mapping[c]) for c, os in groups.items() if c in mapping for o in os]
    if not todo:
        return {"ok": False, "stage": "grounding", "note": "no permitted category assignment"}
    done = set()
    rounds = 2 if sk.classify_repair == "budget" else 1
    for repair in range(rounds):
        for label, basket in todo:
            sk.refresh()
            if label in done or _inside(sk, label, basket):
                done.add(label)
                continue
            # Compute the home reserve from our own executor plans, not another
            # controller's action count. A failed estimate never lowers it.
            home = max(sk.plan_home(a).steps for a in ARMS) + 12
            old_reserve = sk.ex.reserve
            sk.ex.reserve = max(old_reserve, home)
            before = sk.ex.steps_used
            event = {"label": label, "basket": basket, "repair": repair, "step": before}
            events.append(event)
            try:
                if repair and sk.ex.steps_left <= sk.ex.reserve + max(1, max((e.get("steps", 0) for e in events), default=0)):
                    event["status"] = "repair_budget"
                    continue
                o = sk.world.obj(label)
                event["asset"] = o.asset.key
                event["object_before7"] = [*o.pos.tolist(), *o.quat.tolist()]
                method = "top" if o.category in ("watch", "pen", "toy_car", "chocolate") else "auto"
                pp = sk.plan_pick(label, method=method)
                if not isinstance(pp, PickPlan):
                    event["status"] = "pick_plan"
                    continue
                result = sk.run_pick(pp)
                if not result.ok or sk.held[pp.arm] is None:
                    event["status"] = "pick_failed"
                    continue
                sk.refresh()
                result = central_drop(sk, pp.arm, basket)
                sk.refresh()
                placed = sk.world.obj(label)
                event["object_after7"] = [*placed.pos.tolist(), *placed.quat.tolist()]
                event["place_status"] = result.status
                event["place_note"] = result.note
                event["status"] = "placed" if _inside(sk, label, basket) else result.status
                if event["status"] == "placed":
                    done.add(label)
                if sk.held[pp.arm] is not None:
                    event["status"] = "still_held"
                    return {"ok": False, "stage": "holding", "events": events}
            except (BudgetExceeded, EpisodeEnded) as exc:
                event["status"] = type(exc).__name__
                return {"ok": False, "stage": "budget", "events": events}
            finally:
                event["steps"] = sk.ex.steps_used - before
                sk.ex.reserve = old_reserve
    return {"ok": len(done) == len(todo), "done": sorted(done), "events": events}


def central_drop(sk, arm, basket):
    """Plan tilt over the table, then a fixed-wrist dogleg into one basket.

    Both carry legs are checked before the wrist moves. The lateral leg stays
    at the current table-side y, so it cannot pass above a neighboring basket.
    """
    from dojo.skills.executor import ArmPlan
    from dojo.skills.geometry import rot_y
    from dojo.skills.skills import SkillResult

    h = sk.held[arm]
    current = sk.kin.tcp(arm, sk.ex.q[arm])
    c = sk.world.obj(basket)
    mn, mx = c.aabb()
    ignored = (h.label,)
    for angle in (0, 15, -15, 30, -30, 45, -45, 60, -60):
        orient = current.copy()
        orient[:3, :3] = current[:3, :3] @ rot_y(np.radians(angle))
        corners = h.box.corners_w(orient)
        half = (corners.max(0) - corners.min(0)) / 2
        if np.any(2 * half[:2] >= mx[:2] - mn[:2]):
            continue
        goal = orient.copy()
        goal[:2, 3] += c.center_w[:2] - (corners.max(0)[:2] + corners.min(0)[:2]) / 2
        goal[2, 3] += max(0.0, c.top_z - corners[:, 2].min())
        waypoint = orient.copy()
        waypoint[0, 3] = goal[0, 3]
        waypoint[2, 3] = max(orient[2, 3], goal[2, 3])
        fixed = sk.classify_carry == "fixed"
        planner = sk.motion.line if fixed else sk.motion.to_pose
        tilt = sk.motion.line(arm, sk.ex.q[arm], orient, 0.04, "classify_tilt",
                              held=h.box, gripper=0.0, ignore=ignored)
        if not tilt.ok:
            continue
        lateral = planner(arm, tilt.q_end, waypoint, 0.04, "classify_lateral",
                          held=h.box, gripper=0.0, ignore=ignored)
        if not lateral.ok:
            continue
        carry = planner(arm, lateral.q_end, goal, 0.04, "classify_central",
                        held=h.box, gripper=0.0, ignore=ignored + (basket,))
        if not carry.ok:
            continue
        p = ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).move(tilt).move(lateral).move(carry).grip(1.0, 3).hold(8)
        sk.ex.run({arm: p}, name="classify central release")
        sk.held[arm] = None
        return SkillResult(True, "classify_drop", "released", p.steps, arm)
    return SkillResult(False, "classify_drop", "no_plan", 0, arm, note="no checked tilt/central carry fits")
