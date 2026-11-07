# Vendored from msannikov03/robot-harness 6769a6c5b3df8c55f56b0324cfdf196084b9ec94.
# Used behind the original strict RGB wrapper; opt-in checked profile.
"""pour_bottle_v5 (RoboDojo, tier A; V3 fixes, 27 Oct 2026): pour_bottle_v4 that never opens the hand over a refused or
short put-down.

Why (docs/notes/AUDIT-A3-ROBODOJO-SKILLS-2026-10-27.md finding 4, docs/notes/C5-FIXES-REVIEW-2026-10-27.md finding 3):
v4 opened after the put-back hover and descent even when either was refused (then `holding` was null and the generic
home ran), and the outbound-waypoint abort went "straight back down" and opened assuming the arm was still above the
bottle's spot, even after a partial waypoint move or a refused descent. pour_bottle_v4 is unchanged; this file started
as its copy.
Changes:
  * a move counts as arrived only when it moved and its residual_m is known and within tolerance (hover: via_tol,
    descent: set_tol = 0.012 m);
  * normal put-back: hover and descent must arrive before the open; otherwise the bottle stays in the hand and the
    reply's `holding` names arm and bottle (no generic home);
  * outbound abort: the arm first returns over the bottle's spot at its current height (checked), then descends
    (checked); only then does it open, retreat and home; otherwise `holding`, no home;
  * required_moves.put_back uses the same arrived test;
  * defaults = pour_bottle (v1)'s profile: phis 0,40,70,90,105,120,135,150, hold 3 from 120, drain 24, staged untilt
    90,60,0 (v2-v4: 0,75,115,145, hold 2 from 115, drain 16, one untilt to 0). docs/notes/POUR-V3-REGRESSION-2026-10-27.md:
    on five paired dev layouts the original scored 4/5 and v3 2/5; in two v3 losses the coarse 0->75 roll displaced
    the cup and the single 87-step untilt was refused (lease_too_short). A belief, not a demonstrated fix: needs a paired
    Isaac check. Callers that need the lean profile for the step budget pass it explicitly (pour_in_order_v5 does).
Everything else as pour_bottle_v4. Its description follows.

pour_bottle_v4 (RoboDojo, tier A; C5 review, 27 Oct 2026): pour_bottle_v3 that verifies only a completed pour.

run(robot, container=None, target=None, ..., via=True, via_raise=(0.0, 0.05, 0.10), home=True)

Why (docs/notes/NIGHT2-RESULTS-REVIEW-2026-10-27.md finding 4): v1-v3 set `ran` once the profile reached drain_from;
only an in-hand slip cleared it. A move refused after that (115 deg reached, 145 deg refused) skipped the drain, yet
the bottle went upright and home and the skill verified. pour_bottle_v3 is unchanged (md5-frozen in run records); this
file started as its copy.
Changes:
  * the profile is "complete" only when every move of `phis` arrived and no slip, budget or episode stop cut it short;
    anything else is "interrupted: <reason>";
  * the drain hold must run its intended `drain_steps` (the hold reply's steps), not just be started;
  * the put-back hover and descent must arrive;
  * verified additionally requires all three; details carry `profile`, `drain` and `required_moves`;
  * finding 3: the reply's `holding` names the arm and bottle whenever the run ends with the bottle still in the hand
    (return waypoint refused, or no budget for the put-back); no return_home (which opens the hand) runs on that arm.
    `put_down()` is the checked put-down a caller runs before any generic home.
Everything else as pour_bottle_v3 (mandatory waypoints, refusal handling).
"""
import importlib.util
import math
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_Z = 0.765
NOUNS = {"bottle": ("bottle", "wuliangye", "flask", "wine"), "cup": ("cup", "mug", "goblet", "glass", "tumbler")}
FIXTURES = ("camera_stand",)


def load_pour():
    p = os.path.join(HERE, "..", "pour", "skill_robodojo.py")
    spec = importlib.util.spec_from_file_location(f"pour_ro_bottle_{os.getpid()}_{time.time_ns()}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bottle_class(P):
    class Bottle(P.Cup):
        """pour.Cup with the spout at the mouth (top centre, mouth radius) and a body-radius shoulder disc."""

        def __init__(self, ob, mouth_r, shoulder_frac):
            super().__init__(ob)
            lo, hi = np.asarray(ob["aabb"][0], float), np.asarray(ob["aabb"][1], float)
            self.R_body = 0.25 * ((hi[0] - lo[0]) + (hi[1] - lo[1]))
            self.R_rim = float(mouth_r)
            self.r_bot = self.R_body
            self.sh_c = self.bot_c + float(shoulder_frac) * (self.top_c - self.bot_c)

        def lowest(self, tcp, Qf):
            a = self.axis(Qf)
            s = math.sqrt(max(0.0, 1 - a[2] * a[2]))
            m, sh, b = (self.world(tcp, Qf, x) for x in (self.top_c, self.sh_c, self.bot_c))
            return min(m[2] - self.R_rim * s, sh[2] - self.R_body * s, b[2] - self.R_body * s)

        def mouth(self, tcp, Qf):
            return self.world(tcp, Qf, self.top_c)
    return Bottle


def _cat_hit(rec, inst, keys):
    hay = " ".join(str(s or "") for s in (rec.get("category"), inst)).lower()
    return any(k in hay for k in keys)


def ground(objs, instr, container=None, target=None):
    def res(name):
        if name in objs:
            return name
        hit = [k for k, v in objs.items() if v.get("category") == name]
        if len(hit) == 1:
            return hit[0]
        raise ValueError(f"{name!r} matches {hit or 'nothing'}")
    mov = {k: v for k, v in objs.items() if v.get("category") not in FIXTURES and "error" not in v and v.get("aabb")}
    words = set(re.findall(r"[a-z]+", (instr or "").lower()))
    ci = res(container) if container else None
    ti = res(target) if target else None
    if ci is None and "bottle" in words:
        hit = [k for k, v in mov.items() if _cat_hit(v, k, NOUNS["bottle"])]
        ci = hit[0] if len(hit) == 1 else None
    if ti is None and (words & {"cup", "mug", "glass"}):
        hit = [k for k, v in mov.items() if k != ci and _cat_hit(v, k, NOUNS["cup"])]
        ti = hit[0] if len(hit) == 1 else None
    how = "names" if container or target else "instruction nouns"
    if ci is None:
        ci = max(mov, key=lambda k: float(mov[k]["aabb"][1][2] - mov[k]["aabb"][0][2]))
        how = "geometry (tallest = bottle)"
    if ti is None:
        rest = [k for k in mov if k != ci]
        if not rest:
            raise ValueError("no target object")
        c0 = np.asarray(mov[ci]["pos"][:2], float)
        ti = min(rest, key=lambda k: float(np.linalg.norm(np.asarray(mov[k]["pos"][:2], float) - c0)))
        how += " + geometry (nearest = cup)"
    if ci == ti:
        raise ValueError(f"container and target are the same object {ci}")
    return ci, ti, how


def mouth_radius(rec, default):
    gs = [g for g in rec.get("grasps") or [] if g.get("jaw_width_m")]
    if not gs:
        return float(default), "default"
    return float(min(0.03, max(0.006, 0.5 * min(g["jaw_width_m"] for g in gs)))), "top-band mesh grasp"


def run(robot, container=None, target=None, arm="auto", grasp_frac=0.72, back=0.10, lift=0.04, pitches=(0, -10, 10),
        yaws=(0, 15, -15, 30, -30), sides="auto", phis=(0, 40, 70, 90, 105, 120, 135, 150), drain_from=120,
        hold_steps=3, drain_steps=24, untilt_phis=(90, 60, 0), lip_clear=0.012, body_clear=0.012, spout_back=0.0, mouth_r=0.015,
        shoulder_frac=0.55, slip_deg=14.0, slip_m=0.012, set_dz=0.004, min_return_steps=60, home=True, via=True,
        near_x=0.25, via_raise=(0.0, 0.05, 0.10), via_tol=0.03, z_cap=1.10, set_tol=0.012):
    P = load_pour()
    Bottle = bottle_class(P)
    log = []
    o = robot.obs(cameras=[], grasps=["*"])
    objs = o.get("objects")
    if objs is None:
        return {"ok": False, "verified": False, "error": "pour_bottle needs --privileged (tier A)"}
    instr = o.get("instruction") or ""
    try:
        ci, ti, how = ground(objs, instr, container, target)
    except ValueError as e:
        return {"ok": False, "verified": False, "error": f"grounding: {e}", "instruction": instr}
    mr, mr_src = mouth_radius(objs[ci], mouth_r)
    cup = Bottle(objs[ci], mr, shoulder_frac)
    tlo, thi = np.asarray(objs[ti]["aabb"][0], float), np.asarray(objs[ti]["aabb"][1], float)
    top_z, tc = float(thi[2]), 0.5 * (tlo[:2] + thi[:2])
    c0 = cup.start_bottom.copy()
    H = cup.h
    gz = c0[2] + grasp_frac * H
    lift_dz = max(float(lift), top_z + body_clear + 0.01 - float(c0[2]))   # bottle bottom over the cup top before the carry
    details = {"container": ci, "target": ti, "grounding": how, "bottle": {"h": round(H, 4), "R_body": round(cup.R_body, 4),
               "mouth_r": round(mr, 4), "mouth_r_src": mr_src, "grasp_z": round(gz, 4)},
               "target_top": round(top_z, 4), "target_c": P.r4(tc), "log": log}

    use_via = bool(via) and abs(float(tc[1]) - float(c0[1])) > 0.03
    via_fail = []

    def try_via(arm_, p, quat, max_dev, tag):
        """A mandatory waypoint: p, else p raised by via_raise; True only when one is reached (moved, residual <= via_tol)."""
        tries = []
        for dz in via_raise:
            q = np.asarray(p, float) + [0.0, 0.0, float(dz)]
            if q[2] > z_cap + 1e-9:
                break
            mv = robot.move_to(arm=arm_, pos=P.r4(q), quat=P.r4(quat, 6), max_dev=max_dev)
            tries.append(dict(P._short(mv), dz=dz))
            res = mv.get("residual_m")
            if mv.get("moved") and (res is None or float(res) <= via_tol):
                log.append({tag: tries})
                return True
            if mv.get("ended"):
                break
        log.append({tag: tries, "refused": True})
        via_fail.append(tag)
        return False

    def arrived(m, tol):
        """Moved, with a known residual within tol (a refusal, a stop short or a missing residual is not arrived)."""
        res = m.get("residual_m")
        return bool(m.get("moved") and res is not None and float(res) <= float(tol))

    def via_pt(p):                                 # same height and x, at the target row's y
        return np.array([float(p[0]), float(p[1]) + float(tc[1]) - float(c0[1]), float(p[2])])
    details["via"] = use_via

    # ------------------------------------------------------------ plan (free IK): pre, grasp, lift, via, whole profile
    arms = [arm] if arm in ("left", "right") else (["right", "left"] if c0[0] >= 0 else ["left", "right"])
    best, tried = None, 0
    for a_ in arms:
        sd_list = [int(sides)] if sides in (1, -1) else ([-1, 1] if a_ == "right" else [1, -1])
        for pitch in pitches:
            for yaw in yaws:
                Q0 = P.qmul(P.axq(2, yaw), P.qmul(P.axq(0, pitch), P.QH))
                head = P.rotm(Q0)[:, 0]
                head_h = np.array([head[0], head[1], 0.0]) / max(1e-9, math.hypot(head[0], head[1]))
                g = np.array([c0[0], c0[1], gz])
                pre = g - back * head_h
                up = g + np.array([0, 0, lift_dz])
                chain = [(pre, Q0), (g, Q0), (up, Q0)] + ([(via_pt(up), Q0)] if use_via else [])
                ok, trav, ql, _ = P._reach_chain(robot, a_, chain, o["eef"][a_].get("q"))
                tried += 1
                if not ok:
                    continue
                cup.set_hand(g, Q0, objs[ci]["pos"], objs[ci]["quat"])
                for sd in sd_list:
                    cup.set_spout(sd)
                    pdir = P._pour_dir(cup, Q0, sd, "tool", head_h)
                    sxy = tc - spout_back * pdir
                    poses = [P._pour_pose(cup, Q0, sd, ph, sxy, top_z + lip_clear, top_z, body_clear, "tool", head_h)[:2]
                             for ph in phis]
                    ok3, trav3, _, fail = P._reach_chain(robot, a_, poses, ql)
                    tried += 1
                    if ok3:
                        cand = {"arm": a_, "pitch": pitch, "yaw": yaw, "side": sd, "travel": round(trav + trav3, 3)}
                        if best is None or cand["travel"] < best["travel"] - 0.3:
                            best = cand
                if best is not None:
                    break
            if best is not None:
                break
        if best is not None:
            break
    details["plan"] = best or {"none": True, "tried": tried}
    if best is None:
        if home:
            robot.return_home()
        return {"ok": True, "verified": False, "error": "no reachable neck grasp + pour plan", "details": details}
    A, pitch, yaw, sd = best["arm"], best["pitch"], best["yaw"], best["side"]
    Q0 = P.qmul(P.axq(2, yaw), P.qmul(P.axq(0, pitch), P.QH))
    head = P.rotm(Q0)[:, 0]
    head_h = np.array([head[0], head[1], 0.0]) / max(1e-9, math.hypot(head[0], head[1]))
    g = np.array([c0[0], c0[1], gz])
    other = "left" if A == "right" else "right"
    eo = robot.obs(cameras=[])["eef"][other]
    if eo.get("at_home") is False and min(abs(float(eo["pos"][0]) - float(c0[0])), abs(float(eo["pos"][0]) - float(tc[0]))) < near_x:
        robot.return_home(arms=[other])
        details["homed_other"] = other

    # ------------------------------------------------------------ horizontal approach to the neck, close, lift
    m1 = robot.move_to(arm=A, pos=P.r4(g - back * head_h), quat=P.r4(Q0, 6))
    m2 = robot.move_to(arm=A, pos=P.r4(g), quat=P.r4(Q0, 6), speed="slow", max_dev=0.008)
    gr = robot.gripper(arm=A, action="close", wait_steps=3)
    m3 = robot.move_to(arm=A, pos=P.r4(g + [0, 0, lift_dz]), quat=P.r4(Q0, 6), speed="slow")
    o = P._obs(robot, [ci])
    e, cp = o["eef"][A], o["objects"][ci]
    rise = float(cp["pos"][2] - objs[ci]["pos"][2])
    tilt0 = P.ang(P.rotm(cp["quat"])[:, 2], [0, 0, 1])
    log.append({"grasp": {"pre": P._short(m1), "in": P._short(m2), "close_steps": gr.get("steps"),
                          "finger_q": e.get("finger_q"), "lift": P._short(m3), "rise_m": round(rise, 4),
                          "tilt": round(tilt0, 1)}})
    if rise < 0.6 * lift_dz or tilt0 > 20:
        robot.gripper(arm=A, action="open", wait_steps=2)
        if home:
            robot.return_home()
        return {"ok": True, "verified": False, "error": "neck grasp failed (bottle did not rise upright)", "details": details}
    if use_via:
        if not try_via(A, via_pt(e["pos"]), Q0, 0.03, "via_out"):
            # no carry, no pour: back over its own spot (a partial waypoint move may have carried it off), checked
            # descent, and only then release and home; otherwise the bottle stays in the hand
            ab = {}
            cur = np.asarray(robot.obs(cameras=[])["eef"][A]["pos"], float)
            above = np.array([g[0], g[1], cur[2]])
            over_ok = float(np.linalg.norm(cur[:2] - above[:2])) <= set_tol
            if not over_ok:
                mb = robot.move_to(arm=A, pos=P.r4(above), quat=P.r4(Q0, 6), max_dev=0.03)
                ab["back_over"] = P._short(mb)
                over_ok = arrived(mb, set_tol)
            if over_ok:
                md = robot.move_to(arm=A, pos=P.r4(g + [0, 0, 0.004]), quat=P.r4(Q0, 6), speed="slow", max_dev=0.01)
                ab["down"] = P._short(md)
            if over_ok and arrived(md, set_tol):
                robot.gripper(arm=A, action="open", wait_steps=2)
                ab["retreat"] = P._short(robot.move_to(arm=A, pos=P.r4(g - back * head_h), quat=P.r4(Q0, 6), reserve=0))
                if home:
                    ab["home"] = robot.return_home().get("all_home")
                details["aborted"] = ab
                details["holding"] = None
                return {"ok": True, "verified": False, "error": "outbound via refused", "holding": None,
                        "details": details}
            held_ = {"arm": A, "object": ci}
            ab["held"] = "put-down not arrived: bottle kept in the hand, no home"
            details["aborted"], details["holding"] = ab, held_
            return {"ok": True, "verified": False, "error": "outbound via refused; bottle still held", "holding": held_,
                    "details": details}
        o = P._obs(robot, [ci])
        e, cp = o["eef"][A], o["objects"][ci]
    cup.set_hand(e["pos"], e["quat"], cp["pos"], cp["quat"])
    cup.set_spout(sd)
    ref_ax, ref_r = cup.R_hc[:, 2].copy(), cup.r_hc.copy()
    Qg = np.asarray(e["quat"], float)
    pdir = P._pour_dir(cup, Qg, sd, "tool", head_h)
    sxy = tc - spout_back * pdir
    details["in_hand"] = {"r_hc": P.r4(cup.r_hc), "spout_xy": P.r4(sxy), "pour_dir": P.r4(pdir, 3)}

    # ------------------------------------------------------------ liquid profile: mouth over the cup opening
    trace, stop, max_phi = [], None, None
    for i, ph in enumerate(phis):
        o = P._obs(robot, [ci])
        if o.get("ended"):
            stop = "episode ended"
            break
        if o.get("steps_left", 999) < min_return_steps:
            stop = "step budget"
            break
        e, cp = o["eef"][A], o["objects"][ci]
        Rf = P.rotm(e["quat"])
        cup.r_hc, cup.R_hc = Rf.T @ (np.asarray(cp["pos"]) - np.asarray(e["pos"])), Rf.T @ P.rotm(cp["quat"])
        tcp, Qn, tilt = P._pour_pose(cup, Qg, sd, ph, sxy, top_z + lip_clear, top_z, body_clear, "tool", head_h)
        mv = robot.move_to(arm=A, pos=P.r4(tcp), quat=P.r4(Qn, 6), speed=("normal" if i == 0 else "slow"), max_dev=0.03,
                           max_steps=80)
        if ph >= drain_from and mv.get("moved"):
            robot.hold(steps=int(hold_steps))
        o = P._obs(robot, [ci])
        e, cp = o["eef"][A], o["objects"][ci]
        Rf = P.rotm(e["quat"])
        R_now, r_now = Rf.T @ P.rotm(cp["quat"]), Rf.T @ (np.asarray(cp["pos"]) - np.asarray(e["pos"]))
        slip_a, slip_r = P.ang(R_now[:, 2], ref_ax), float(np.linalg.norm(r_now - ref_r))
        cup.r_hc, cup.R_hc = r_now, R_now
        mouth = cup.mouth(e["pos"], e["quat"])
        spout = cup.world(e["pos"], e["quat"], cup.spout_c)
        trace.append({"phi": ph, "tilt_plan": round(tilt, 1), "tilt": round(P.ang(P.rotm(cp["quat"])[:, 2], [0, 0, 1]), 1),
                      "move": P._short(mv), "mouth": P.r4(mouth, 3),
                      "mouth_off_xy": round(float(np.linalg.norm(mouth[:2] - tc)), 4),
                      "spout_off_xy": round(float(np.linalg.norm(spout[:2] - sxy)), 4), "slip_deg": round(slip_a, 1),
                      "slip_m": round(slip_r, 4), "step": o.get("step")})
        if not mv.get("moved"):
            stop = f"move refused: {mv.get('plan_fail_reason')}"
            break
        max_phi = ph
        if slip_a > slip_deg or slip_r > slip_m:
            stop = "in-hand slip"
            break
    complete = stop is None and len(trace) == len(phis)
    drained = 0
    if max_phi is not None and max_phi >= drain_from and complete:
        drained = int(robot.hold(steps=int(drain_steps)).get("steps") or 0)
    details["pour"] = trace
    details["pour_stop"] = stop or "profile end"
    details["profile"] = "complete" if complete else f"interrupted: {stop}"
    details["drain"] = {"intended": int(drain_steps), "held": drained}

    # ------------------------------------------------------------ untilt with the mouth kept over the cup (drips fall in)
    ret = {"untilt": []}
    for ph in [p_ for p_ in untilt_phis if max_phi is not None and p_ < max_phi]:
        o = P._obs(robot, [ci])
        if o.get("ended"):
            break
        e, cp = o["eef"][A], o["objects"][ci]
        Rf = P.rotm(e["quat"])
        cup.r_hc, cup.R_hc = Rf.T @ (np.asarray(cp["pos"]) - np.asarray(e["pos"])), Rf.T @ P.rotm(cp["quat"])
        tcp, Qn, _ = P._pour_pose(cup, Qg, sd, ph, sxy, top_z + lip_clear, top_z, body_clear, "tool", head_h)
        mu = robot.move_to(arm=A, pos=P.r4(tcp), quat=P.r4(Qn, 6), speed="slow", max_dev=0.03, max_steps=80)
        ret["untilt"].append({"phi": ph, **P._short(mu)})
        if not mu.get("moved"):
            mu = robot.move_to(arm=A, pos=P.r4(np.asarray(e["pos"], float) + [0, 0, 0.03]), quat=P.r4(Qg, 6),
                               speed="slow", max_dev=0.05)
            ret["untilt"].append({"phi": "direct upright", **P._short(mu)})
            break
    o = P._obs(robot, [ci])
    released = False
    if not o.get("ended") and o.get("steps_left", 0) >= min_return_steps:
        e, cp = o["eef"][A], o["objects"][ci]
        cup.set_hand(e["pos"], e["quat"], cp["pos"], cp["quat"])
        bot_rel = P.rotm(Qg) @ (cup.r_hc + cup.R_hc @ cup.bot_c)
        tcp_set = np.array([c0[0], c0[1], TABLE_Z + set_dz]) - bot_rel
        hov = tcp_set + np.array([0, 0, max(0.04, top_z + 0.03 - TABLE_Z)])
        hov[2] = max(hov[2], min(float(e["pos"][2]), 1.05))
        back_ok = (not use_via) or try_via(A, via_pt(hov), Qg, 0.05, "via_back")
        if back_ok:
            ret["hover"] = P._short(robot.move_to(arm=A, pos=P.r4(hov), quat=P.r4(Qg, 6), max_dev=0.05))
            if arrived(ret["hover"], via_tol):
                ret["down"] = P._short(robot.move_to(arm=A, pos=P.r4(tcp_set), quat=P.r4(Qg, 6), speed="slow",
                                                     max_dev=0.01))
            if arrived(ret["hover"], via_tol) and arrived(ret["down"], set_tol):
                robot.gripper(arm=A, action="open", wait_steps=2)
                released = True
                back_pos = np.asarray(robot.obs(cameras=[])["eef"][A]["pos"], float) - back * head_h
                ret["retreat"] = P._short(robot.move_to(arm=A, pos=P.r4(back_pos), quat=P.r4(Qg, 6), reserve=0))
            else:
                ret["aborted"] = "put-back hover or descent not arrived: bottle kept in the hand, no home"
        else:
            ret["aborted"] = "return via refused: bottle held upright over the target row, no straight carry"

    details["return"] = ret
    holding = None if released else {"arm": A, "object": ci}
    details["holding"] = holding
    o = P._obs(robot, [ci])
    fin = {"bottle_tilt": round(P.ang(P.rotm(o["objects"][ci]["quat"])[:, 2], [0, 0, 1]), 1), "ended": o.get("ended")}
    if home and not o.get("ended") and holding is None:
        h = robot.return_home()
        fin["all_home"] = h.get("all_home")
        o = P._obs(robot, [ci])
        fin["bottle_tilt"] = round(P.ang(P.rotm(o["objects"][ci]["quat"])[:, 2], [0, 0, 1]), 1)
    details["final"] = fin
    ran = max_phi is not None and max_phi >= drain_from and details["pour_stop"] != "in-hand slip"
    details["via_failed"] = list(via_fail)
    put_back = [ret.get(k) or {} for k in ("hover", "down")]
    details["required_moves"] = {"profile": complete, "drain": drained >= int(drain_steps),
                                 "put_back": bool(released and arrived(put_back[0], via_tol) and arrived(put_back[1], set_tol))}
    verified = bool(ran and all(details["required_moves"].values()) and not via_fail and fin["bottle_tilt"] <= 15.0
                    and (fin.get("all_home") or o.get("ended") or not home))
    out = {"ok": True, "verified": verified, "steps_left": o.get("steps_left"), "holding": holding, "details": details}
    if via_fail:
        out["error"] = "return via refused" if "via_back" in via_fail else "via refused"
    elif holding:
        out["error"] = "bottle still held (" + ("put-back not arrived" if ret.get("aborted") else "no budget for the put-back") + ")"
    return out


def put_down(robot, arm, obj, step=0.05, reach=0.30, margin=0.03, set_dz=0.004, lift_back=0.08, tol=0.03):
    """Checked put-down of a held upright bottle on a free table spot in its current row (tier A, privileged poses).
    Spots: the bottle's x +- k*step out to `reach`, nearest first, each clear of every other object's footprint by
    `margin`, reached along the row at the current height without passing a standing object that reaches above the
    bottle's bottom. Carry, descend, open, lift. ok only when the bottle then stands within `tol` of the spot, upright
    (<= 15 deg), out of the hand. A refused carry tries the next spot; a refused descent gives up still holding."""
    P = load_pour()
    o = robot.obs(cameras=[], grasps=["*"])
    objs, e = o.get("objects") or {}, o["eef"][arm]
    lo, hi = np.asarray(objs[obj]["aabb"][0], float), np.asarray(objs[obj]["aabb"][1], float)
    c, r_b, tcp = 0.5 * (lo[:2] + hi[:2]), 0.5 * float(max(hi[:2] - lo[:2])), np.asarray(e["pos"], float)
    drop = float(lo[2]) - (TABLE_Z + set_dz)
    foot = {k: (0.5 * (np.asarray(v["aabb"][0][:2], float) + np.asarray(v["aabb"][1][:2], float)),
                0.5 * float(max(np.asarray(v["aabb"][1][:2], float) - np.asarray(v["aabb"][0][:2], float))),
                float(v["aabb"][1][2])) for k, v in objs.items() if k != obj and v.get("aabb") and "error" not in v}

    def clear(p, tall_only):
        return all(np.linalg.norm(p - fc) >= r_b + fr + margin for fc, fr, top in foot.values()
                   if not tall_only or top > float(lo[2]) - 0.005)
    n = int(round(reach / step))
    tries, out = [], {"ok": False, "spot": None, "released": False}
    for d in sorted((k * step for k in range(-n, n + 1)), key=abs):
        spot = c + [d, 0.0]
        if not clear(spot, False) or not all(clear(c + [d * s, 0.0], True) for s in np.linspace(0, 1, 11)):
            continue
        q = P.r4(e["quat"], 6)
        hand = np.array([spot[0] + tcp[0] - c[0], spot[1] + tcp[1] - c[1], tcp[2]])
        m1 = robot.move_to(arm=arm, pos=P.r4(hand), quat=q, max_dev=0.05)
        tries.append({"spot": P.r4(spot), "carry": P._short(m1)})
        if not m1.get("moved"):
            continue
        m2 = robot.move_to(arm=arm, pos=P.r4(hand - [0, 0, drop]), quat=q, speed="slow", max_dev=0.01)
        tries[-1]["down"] = P._short(m2)
        if not m2.get("moved"):
            break
        robot.gripper(arm=arm, action="open", wait_steps=2)
        robot.move_to(arm=arm, pos=P.r4(hand - [0, 0, drop - lift_back]), quat=q, reserve=0)
        o = robot.obs(cameras=[], grasps=[obj])
        b = o["objects"][obj]
        bl, bh = np.asarray(b["aabb"][0], float), np.asarray(b["aabb"][1], float)
        chk = {"xy_err_m": round(float(np.linalg.norm(0.5 * (bl[:2] + bh[:2]) - spot)), 4),
               "tilt": round(P.ang(P.rotm(b["quat"])[:, 2], [0, 0, 1]), 1), "bottom_z": round(float(bl[2]), 4),
               "grip_cmd": o["eef"][arm].get("grip_cmd")}
        ok = chk["xy_err_m"] <= tol and chk["tilt"] <= 15.0 and abs(chk["bottom_z"] - TABLE_Z) <= 0.02 \
            and (chk["grip_cmd"] is None or float(chk["grip_cmd"]) > 0.5)
        out.update(ok=bool(ok), spot=P.r4(spot), check=chk, released=True)
        break
    out["tries"] = tries
    return out
