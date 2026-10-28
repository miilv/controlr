# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/cover_uncover_v2/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""cover_uncover_v2 (RoboDojo, tier A composite; C5 review, 27 Oct 2026): cover_uncover with a final check after home.

run(robot, order=None, cover_dir=None, ...)

Why (docs/notes/NIGHT2-RESULTS-REVIEW-2026-10-27.md finding 5): cover_uncover checked cups and blocks after each move,
all before the final home; the observation after home checked only the arms, so a block knocked during that home still
verified. cover_uncover is unchanged (md5-frozen in run records); this file started as its copy.
Change: after the final home (or the last observation, if the episode ended) every block must still be within 2 cm of
its start, every cup within 10 deg of its start axis (upside down) and back on its own start spot (xy within
cover_tol, root z within 12 mm of rest), and no cup may stand over a block (uncovered). Overall verified = all six
moves verified (the per-move checks, kept for the required sequence) + the final check + both arms home; details
carry `final_check`.
Everything else as cover_uncover: grounding, cup assignment, pick + place_held per move, arm choice.
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import colour_order as co  # noqa: E402

TABLE_Z = 0.765


def zaxis(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def ang(u, v):
    c = float(np.dot(u, v) / max(1e-12, np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def parse_uncover(instr, table=None):
    text = (instr or "").lower()
    for key in ("uncover", "order"):
        i = text.find(key)
        if i >= 0:
            got = co.colours_in(text[i:], table)
            if got:
                return got
    return co.colours_in(text, table)


def cover_reverse(instr):
    return bool(re.search(r"right\s+to\s+(the\s+)?left", (instr or "").lower()))


def run(robot, order=None, cover_dir=None, noun="block", cover_noun="cup", lift_over=0.04, hover_over=0.04,
        release_dz=0.004, clear_dz=0.05, cover_tol=0.015, near_x=0.25, attempts=3, min_left=80, home=True):
    o = robot.obs(cameras=[])
    objs = o.get("objects")
    if objs is None:
        return {"ok": False, "verified": False, "error": "cover_uncover needs --privileged (tier A)"}
    instr = o.get("instruction") or ""
    blocks = sorted(k for k, v in objs.items() if "error" not in v and co.cat_matches(v, k, noun) and co.colour_of(k, v))
    cups = sorted(k for k, v in objs.items() if "error" not in v and co.cat_matches(v, k, cover_noun))
    if not blocks or len(cups) < len(blocks):
        return {"ok": False, "verified": False, "error": f"grounding: blocks {blocks}, cups {cups}"}
    colour = {b: co.colour_of(b, objs[b]) for b in blocks}          # the colour memory, read before any cover
    try:
        words = list(order) if order else parse_uncover(instr)
        unc = [w if w in blocks else co.ground(objs, w, noun) for w in words]
        if sorted(unc) != sorted(blocks):
            raise ValueError(f"uncover order {words} -> {unc} does not name every block once {blocks}")
    except ValueError as e:
        return {"ok": False, "verified": False, "error": f"grounding: {e}", "instruction": instr}
    o = robot.obs(cameras=[], grasps=blocks + cups)
    objs = o["objects"]
    rev = cover_reverse(instr) if cover_dir is None else cover_dir == "right_to_left"
    cover_order = sorted(blocks, key=lambda b: objs[b]["pos"][0], reverse=rev)
    start = {k: np.asarray(objs[k]["pos"], float) for k in blocks + cups}
    axis0 = {c: zaxis(objs[c]["quat"]) for c in cups}
    bh = max(float(objs[b]["aabb"][1][2]) - TABLE_Z for b in blocks)
    chh = {c: (float(objs[c]["aabb"][1][2]) - float(objs[c]["aabb"][0][2])) / 2 for c in cups}
    assign, free = {}, list(cups)
    for b in cover_order:
        c = min(free, key=lambda k: float(np.linalg.norm(start[k][:2] - start[b][:2])))
        assign[b] = c
        free.remove(c)
    moves = [("cover", assign[b], b, start[b][:2]) for b in cover_order] + \
            [("uncover", assign[b], b, start[assign[b]][:2]) for b in unc]
    P = co.load_sibling("place_on")
    lift = round(max(0.06, bh + lift_over), 4)
    hover = round(max(0.05, bh + hover_over), 4)
    log, good = [], []
    for kind, c, b, dest in moves:
        o = robot.obs(cameras=[])
        if o.get("ended"):
            log.append({"stop": "episode ended"})
            break
        if o.get("steps_left", 0) < min_left:
            log.append({"stop": f"budget: {o.get('steps_left')} steps left < {min_left}"})
            break
        x = 0.5 * (float(o["objects"][c]["pos"][0]) + float(dest[0]))
        eef = o["eef"]
        if x < -0.05:
            arms = ["left", "right"]
        elif x > 0.05:
            arms = ["right", "left"]
        else:
            arms = sorted(("left", "right"), key=lambda a: (not eef[a].get("at_home"), a != "right"))
        rec = {"move": kind, "cup": c, "block": b, "colour": colour[b], "dest": [round(float(v), 4) for v in dest]}
        held = None
        for a in arms:
            other = "right" if a == "left" else "left"
            e = robot.obs(cameras=[])["eef"]
            if not e[other].get("at_home") and abs(float(e[other]["pos"][0]) - x) < near_x:
                h = robot.return_home(arms=[other])
                rec.setdefault("homed", []).append({"arm": other, "steps": h.get("steps")})
            p = robot.skill(name="pick", args={"obj": c, "arm": a, "lift": lift, "attempts": attempts})
            rec.setdefault("picks", []).append({"arm": a, "verified": p.get("verified"), "steps": p.get("skill_steps_used"),
                                                "error": p.get("error")})
            if p.get("verified"):
                held = p.get("arm") or a
                break
            if p.get("ended"):
                break
            h = robot.return_home(arms=[a])
        if held is None:
            rec["result"] = "pick failed"
            log.append(rec)
            good.append(False)
            break
        z_rest = float(start[c][2])
        pl = P.place_held(robot, c, held, {"xy": [float(dest[0]), float(dest[1])], "z": z_rest - chh[c]}, chh[c],
                          hover_dz=hover, release_dz=release_dz, clear_dz=clear_dz, yaw_mod=90.0, xy_tol=cover_tol,
                          z_tol=0.012, settle_steps=2)
        rec["place"] = {k: pl.get(k) for k in ("ok", "why", "hover_err_m", "xy_err_m", "z_err_m", "steps", "step")}
        o = robot.obs(cameras=[])
        ob = o["objects"]
        cp = np.asarray(ob[c]["pos"], float)
        xy_err = float(np.linalg.norm(cp[:2] - np.asarray(dest, float)))
        tilt = ang(zaxis(ob[c]["quat"]), axis0[c])
        moved = {k: round(float(np.linalg.norm(np.asarray(ob[k]["pos"], float)[:2] - start[k][:2])), 4) for k in blocks}
        ok = bool(pl.get("ok") and xy_err <= cover_tol and abs(cp[2] - z_rest) <= 0.012 and tilt <= 10.0
                  and max(moved.values()) <= 0.02)
        rec.update(xy_err_m=round(xy_err, 4), z_err_m=round(float(cp[2] - z_rest), 4), axis_dev_deg=round(tilt, 1),
                   blocks_moved_m=moved, verified=ok, arm=held)
        log.append(rec)
        good.append(ok)
        if pl.get("ended") or not pl.get("ok"):
            break
    o = robot.obs(cameras=[])
    if home and not o.get("ended"):
        h = robot.return_home()
        log.append({"final_home": h.get("all_home"), "steps": h.get("steps")})
        o = robot.obs(cameras=[])
    homed = all(e.get("at_home") for e in o["eef"].values())
    ob = o.get("objects") or robot.obs(cameras=[])["objects"]
    pos = {k: np.asarray(ob[k]["pos"], float) for k in blocks + cups}
    fc = {"blocks_moved_m": {b: round(float(np.linalg.norm(pos[b][:2] - start[b][:2])), 4) for b in blocks},
          "cup_axis_dev_deg": {c: round(ang(zaxis(ob[c]["quat"]), axis0[c]), 1) for c in cups},
          "cup_home_xy_m": {c: round(float(np.linalg.norm(pos[c][:2] - start[c][:2])), 4) for c in cups},
          "cup_z_err_m": {c: round(float(pos[c][2] - start[c][2]), 4) for c in cups},
          "covered": sorted({b for b in blocks for c in cups if np.linalg.norm(pos[c][:2] - pos[b][:2]) < cover_tol + 0.02})}
    fc["ok"] = bool(max(fc["blocks_moved_m"].values()) <= 0.02 and max(fc["cup_axis_dev_deg"].values()) <= 10.0
                    and max(fc["cup_home_xy_m"].values()) <= cover_tol
                    and max(abs(v) for v in fc["cup_z_err_m"].values()) <= 0.012 and not fc["covered"])
    verified = bool(len(good) == len(moves) and all(good) and fc["ok"] and homed)
    return {"ok": True, "verified": verified, "cover_order": cover_order, "uncover_order": unc, "cups": assign,
            "colours": colour, "homed": homed,
            "details": {"log": log, "final_check": fc, "lift": lift, "hover_dz": hover, "block_h": round(bh, 4), "instruction": instr,
                        "steps_left": o.get("steps_left"), "ended": o.get("ended")}}
