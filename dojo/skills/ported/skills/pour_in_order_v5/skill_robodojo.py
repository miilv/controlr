# Vendored from msannikov03/robot-harness 6769a6c5b3df8c55f56b0324cfdf196084b9ec94.
# Opt-in original checked composite; pairs supplied from instruction-grounded handles.
"""pour_in_order_v5 (RoboDojo, tier A composite; V3 fixes, 27 Oct 2026): pour_in_order_v4 that stops after any
unsuccessful pair, over pour_bottle_v5.

Why (docs/notes/AUDIT-A3-ROBODOJO-SKILLS-2026-10-27.md finding 5): v4 stopped only for a held bottle or a waypoint
error; any other failed pour (an interrupted profile, a short drain, an exhausted grasp) went on to the next pair, so
a later pour could run out of the instruction's order and the final verified=False could not undo it.
pour_in_order_v4 is unchanged; this file started as its copy.
Changes:
  * a pair that is not verified after its allowed retry stops the sequence (the retry runs only for the two errors
    raised before any liquid moved, as in v4); the reply's log names the pair and the reason;
  * each pour is pour_bottle_v5 (no open over a refused put-down; A3 finding 4); its `holding` is propagated as in v4;
  * pour_bottle_v5's defaults are the fine single-pour profile, so this composite passes v4's lean profile explicitly
    (LEAN: phis 0,75,115,145, drain_from 115, hold 2, untilt 0), overridable through pour_args; the drain stays
    budget-sized.
Everything else as pour_in_order_v4. Its description follows.

pour_in_order_v4 (RoboDojo, tier A composite; C5 review, 27 Oct 2026): pour_in_order_v3 over pour_bottle_v4.

run(robot, pairs=None, ..., put_down=True, oracle_labels=False)

Why (docs/notes/NIGHT2-RESULTS-REVIEW-2026-10-27.md finding 3): after a refused return waypoint pour_bottle_v3
deliberately keeps the bottle held over the target row, but pour_in_order_v3 then broke the sequence and called the
generic return_home(), which opens the hands: the bottle was dropped wherever the arm was. pour_in_order_v3 is unchanged
(md5-frozen in run records); this file started as its copy.
Changes:
  * each pour is pour_bottle_v4 (a pour verifies only with a completed profile, its intended drain and the put-back,
    finding 4), whose reply names the arm and bottle in `holding` when it ends with the bottle in the hand;
  * a held bottle stops the sequence like a refused waypoint does, and is propagated: the reply carries `holding`;
  * with put_down=True the held bottle goes through pour_bottle_v4.put_down (a free spot in its row, checked: standing,
    upright, out of the hand); only after that check passes may the generic home run. Otherwise only the arms that
    hold nothing are homed and `holding` stays set; verified is False either way.
  * `pairs` takes instance names only; a checker label is oracle grounding (finding 12) and is refused unless
    oracle_labels=True, which marks the reply oracle_grounded=True and its grounding "args (oracle: checker labels)".
Everything else as pour_in_order_v3: grounding from the instruction, stop at a refused waypoint, home after the last
pour, drain sized to the budget, verified = every pour verified and arms home.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import colour_order as co  # noqa: E402

CHILD = "pour_bottle_v5"
LEAN = {"phis": [0, 75, 115, 145], "drain_from": 115, "hold_steps": 2, "untilt_phis": [0]}

NO_LIQUID = ("no reachable neck grasp + pour plan", "neck grasp failed (bottle did not rise upright)")


def _resolve(objs, name, oracle=False):
    """A `pairs` member: an instance name. A checker label is oracle grounding (C5 finding 12): refused unless
    oracle_labels=True, and then the reply is marked oracle_grounded."""
    if name in objs:
        return name
    lab = [k for k, v in objs.items() if v.get("label") == name]
    if lab and not oracle:
        raise ValueError(f"{name!r} is a checker label of {lab}: oracle grounding, refused unless oracle_labels=True")
    if len(lab) == 1:
        return lab[0]
    raise ValueError(f"{name!r} matches {lab or 'nothing'}")


def _tilt(q):
    w, x, y, z = (float(v) for v in q)
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def run(robot, pairs=None, retries=1, per_pour=240, base_drain=16, min_drain=4, max_drain=30, final_reserve=40,
        pour_args=None, put_down=True, oracle_labels=False):
    o = robot.obs(cameras=[])
    objs = o.get("objects")
    if objs is None:
        return {"ok": False, "verified": False, "error": "pour_in_order_v5 needs --privileged (tier A)"}
    instr = o.get("instruction") or ""
    try:
        if pairs:
            plan = [(_resolve(objs, c, oracle_labels), _resolve(objs, t, oracle_labels)) for c, t in pairs]
            labels = any(x not in objs for pr in pairs for x in pr)
            src = "args (oracle: checker labels)" if labels else "args (instance names)"
        else:
            parsed = co.parse_pairs(instr)
            if not parsed:
                raise ValueError(f"no 'from the <colour> <container> into the <colour> <target>' pairs in {instr!r}")
            plan, used = [], set()
            for cc, cn, tc, tn in parsed:
                c = co.ground(objs, cc, cn, skip=used)
                t = co.ground(objs, tc, tn, skip=used | {c})
                used |= {c, t}
                plan.append((c, t))
            src = "instruction"
    except ValueError as e:
        return {"ok": False, "verified": False, "error": f"grounding: {e}", "instruction": instr}
    log, good, held = [], [], None
    for i, (c, t) in enumerate(plan):
        o = robot.obs(cameras=[])
        if o.get("ended"):
            break
        after = len(plan) - i - 1
        left = int(o.get("steps_left", 0))
        drain = int(np.clip(left - final_reserve - per_pour * (after + 1) + base_drain, min_drain, max_drain))
        args = dict(LEAN, **(pour_args or {}), container=c, target=t, home=(after == 0), drain_steps=drain)
        r = robot.skill(name=CHILD, args=args)
        tries = [{"error": r.get("error"), "verified": r.get("verified"), "drain": drain, "steps": r.get("skill_steps_used")}]
        n = 0
        while r.get("error") in NO_LIQUID and n < retries and not r.get("ended"):
            ob = robot.obs(cameras=[]).get("objects") or {}
            if c in ob and _tilt(ob[c]["quat"]) > 20:
                tries.append({"skip_retry": f"bottle tilted {round(_tilt(ob[c]['quat']), 1)} deg"})
                break
            n += 1
            r = robot.skill(name=CHILD, args=args)
            tries.append({"error": r.get("error"), "verified": r.get("verified"), "retry": n, "steps": r.get("skill_steps_used")})
        d = r.get("details") or {}
        good.append(bool(r.get("verified")))
        log.append({"pair": i, "container": c, "target": t, "tries": tries, "stop": d.get("pour_stop"),
                    "max_phi": max([x.get("phi", 0) for x in d.get("pour") or []] or [0]),
                    "final": d.get("final"), "via": d.get("via"), "step": r.get("step"), "steps_left": r.get("steps_left")})
        held = r.get("holding") or (r.get("details") or {}).get("holding")
        if r.get("ended"):
            break
        if held or not r.get("verified"):
            why = r.get("error") or f"not verified ({d.get('profile') or d.get('pour_stop')})"
            log.append({"stop": f"pair {i}: {why}; later pairs would pour out of order", "holding": held})
            break
    o = robot.obs(cameras=[])
    put = None
    if held and put_down and not o.get("ended"):
        put = co.load_sibling(CHILD).put_down(robot, held["arm"], held["object"])
        log.append({"put_down": put})
        if put.get("ok"):
            held = None
        elif put.get("released"):                  # opened but the check failed: report it, still no generic home
            held = dict(held, released_unchecked=True)
        o = robot.obs(cameras=[])
    homed = all(e.get("at_home") for e in (o.get("eef") or {}).values()) if o.get("eef") else False
    if not homed and not o.get("ended"):
        if held:                                   # never the generic home (it opens the hand) on the holding arm
            free = [a for a, e in o["eef"].items() if a != held["arm"] and not e.get("at_home")]
            if free:
                h = robot.return_home(arms=free)
                log.append({"home_free_arms": free, "all_home": h.get("all_home")})
            log.append({"home_suppressed": held["arm"], "why": "bottle held, no checked put-down"})
        else:
            h = robot.return_home()
            log.append({"final_home": h.get("all_home"), "steps": h.get("steps")})
        o = robot.obs(cameras=[])
        homed = all(e.get("at_home") for e in (o.get("eef") or {}).values()) if o.get("eef") else False
    verified = bool(len(good) == len(plan) and all(good) and not held and (homed or o.get("ended")))
    return {"ok": True, "verified": verified, "plan": plan, "grounding": src, "oracle_grounded": src.startswith("args (oracle"),
            "homed": homed, "holding": held,
            "put_down": put,
            "details": {"log": log, "instruction": instr, "steps_left": o.get("steps_left"), "ended": o.get("ended"),
                        "note": "liquid amount not measured; verified = profiles ran, bottles upright, arms home"}}
