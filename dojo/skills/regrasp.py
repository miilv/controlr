"""In-air handover with the object kept as it rests (``rest_handover``): for an insertion whose grasps from rest cannot
reach the target (a key lying on its face is gripped from that face; standing in its lock, the same grasp points the
hand back at the robot), and whose re-orienting handover lost the object (``library.plan_air_handover`` turned the key to
its final orientation in the giver's hand: held across its round bow, it pivoted about the closing axis and fell, 0/34).

* The giver pinches the object lengthwise, straight down, pads on its two ends (``lengthwise_grasps``; twin, a key: in
  hand 0.1 mm / 0.1 deg after the lift, 0.1 mm / 0.0 deg after the carry; its 5 mm shaft pinched across slipped
  5 mm / 11 deg at the lift and 16 mm / 37 deg more on the carry), lifts it and carries it to the handover pose turned
  only about the vertical.
* The taker's grasp is chosen at the TARGET (``top_grasps_at``): across the thinnest side of the object's top part (a
  key's bow faces), the approach in the plane of the other two sides tilted ``TILTS`` down, so that the hand reaches the
  target and the turn after it and clears the receptacle (a horizontal hand over a lock block 23 mm under the bow hits
  it). At the handover the same grasp in the object frame is horizontal-ish; its hand must clear the giver's.
* The giver lets go and backs off upwards; the taker places through the closed-loop tail (insert, turn) holding on
  (the checker needs no open gripper; a release / retreat after the turn was out of reach).

Planning uses the believed in-hand pose (mode B: the runtime attachment), the execution re-reads the object at the
handover (G3: measured; mode B: the belief).
"""

from __future__ import annotations

import math
import time

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import T_from, inv_T, rot_z
from dojo.skills.kinematics import ARMS, TABLE_Z, TIP_BEYOND_TCP, gap_to_grip
from dojo.skills.motion import held_box
from dojo.skills.skills import CLOSED, OPEN, Held, PlacePlan, SkillResult, Skills

LENGTHWISE_TIP = (0.0028, 0.0038)   # m: the giver's fingertips over the support (twin: 2.8 mm passes the collision check)
REST_CARRY = 0.04                   # rad/step: the giver's carry to the handover pose (eased)
TILTS = (15.0, 30.0)                # deg: the taker's approach under the horizontal at the target (45 / 60: the pads
                                    # held only the bow's top edge)
TOP_BAND = 0.02                     # m: the object's top part (the taker's pads) at the target
TAKE_GAP = 0.006                    # m: the taker opens this much wider than the part it takes
TAKE_PAST = (0.004, 0.006)          # m: its fingertip this far past the part's centre (at the centre, half the bow
                                    # between the pads: the key fell when the giver opened, 3 of 7 in G3)
HAND_CLEAR = 0.003                  # m between the hands at the handover (grasp and pre-grasp)
HANDOVER_XY = ((0.06, -0.22), (0.0, -0.22), (0.06, -0.28), (0.10, -0.22), (0.0, -0.18), (0.06, -0.18), (-0.04, -0.22))
HANDOVER_Z = (0.92, 0.98, 1.04)
PLAN_S = 60.0                       # wall-time cap of the handover search
PLACE_CHECKS = 6                    # handover candidates whose taker place is planned before one is chosen
TURN_FRACTIONS = (1.0, 0.85, 0.7, 0.6)
TURN_LIFT = 0.0009                  # m: the object is turned this far above where it seated
TURN_STEP_DEG = 3.0                 # deg per env step of that turn ...
TURN_FINE = (0.45, 0.85, 1.5)       # ... except between these fractions of it: 1.5 deg steps, each held one step
# (twin, a key spawned in its lock, the turn about its root: the teeth catch at -14 deg, the key tilts 7-11 deg while
# forced, then snaps through at -30 to -36 deg with 5-7 deg of tilt for a step or two, the moment the checker can pass;
# 0.3 / 0.9 / 1.2 mm up passed at that step, 0.6 mm missed by 0.3 deg of tilt; pressed on the floor it never passed)


def lengthwise_grasps(sk: Skills, label: str, arm: str) -> list:
    """Straight-down pinches across the object's longest horizontal side, pads on its two ends, centred on its centre
    of mass across the other side, fingertips ``LENGTHWISE_TIP`` over its support. Gravity has no lever about such a
    closing axis (it runs through the centre of mass), and both pads press flat ends."""
    o = sk.world.obj(label)
    if o.asset.mesh is None:
        return []
    V = o.asset.mesh[0] @ o.R.T + o.pos
    best = None
    for i in range(3):
        d = o.R[:, i].copy()
        if abs(d[2]) > 0.3:
            continue
        d[2] = 0.0
        d /= np.linalg.norm(d)
        s = V @ d
        L = float(s.max() - s.min())
        if best is None or L > best[0]:
            best = (L, d, float(s.min()), float(s.max()))
    if best is None:
        return []
    L, d, s0, s1 = best
    if L + 2 * 0.004 > G.MAX_GAP:
        return []
    t = np.cross([0.0, 0.0, 1.0], d)
    com = o.R @ G.mesh_com(o.asset) + o.pos
    a = np.array([0.0, 0.0, -1.0])
    p = d * (s0 + s1) / 2 + t * float(com @ t)
    z_sup = float(V[:, 2].min())
    out = []
    for dz in LENGTHWISE_TIP:
        for sg in (1.0, -1.0):
            c = sg * d
            R = np.stack([a, c, np.cross(a, c)], 1)
            pos = np.array([p[0], p[1], z_sup + dz + TIP_BEYOND_TCP])
            out.append(G.Grasp("lengthwise", T_from(pos, R), L, note=f"lengthwise {L*1000:.0f} mm, tips {dz*1000:.1f} mm up"))
    return out


def top_grasps_at(o, T_target) -> list[tuple[str, np.ndarray, float]]:
    """(note, TCP in the object frame, width) across the thinnest side of the object's top part at ``T_target``,
    approaching in the plane of the other two sides, tilted ``TILTS`` down, the fingertip at the part's centre or
    4 mm past it, from either side."""
    if o.asset.mesh is None:
        return []
    V = o.asset.mesh[0]
    R_t = np.asarray(T_target, float)[:3, :3]
    up = R_t.T @ np.array([0.0, 0.0, 1.0])                  # the object's up at the target, object frame
    h = V @ up
    top = V[h >= h.max() - TOP_BAND - 1e-9]
    axes = [i for i in range(3) if abs(up[i]) < 0.5]
    if len(axes) != 2 or len(top) < 4:
        return []
    ext = {i: float(top[:, i].max() - top[:, i].min()) for i in axes}
    ic = min(axes, key=lambda i: ext[i])                    # closing: the part's thinnest side
    ih = max(axes, key=lambda i: ext[i])
    width = ext[ic]
    ctr = (top.min(0) + top.max(0)) / 2
    e_c = np.eye(3)[ic]
    e_h = np.eye(3)[ih]
    out = []
    for s in (1.0, -1.0):
        for beta in TILTS:
            b = math.radians(beta)
            a = s * math.cos(b) * e_h - math.sin(b) * up
            for cs in (1.0, -1.0):
                c = cs * e_c
                R = np.stack([a, c, np.cross(a, c)], 1)
                for past in TAKE_PAST:
                    tip = ctr + s * past * e_h - 0.002 * up
                    out.append((f"top s{s:+.0f} tilt {beta:.0f} past {past*1000:.0f}", T_from(tip - a * TIP_BEYOND_TCP, R), width))
    return out


MIN_TURN_FRAC = 0.7                 # of the turn the taker's wrist must reach at the target (the key snapped through its
                                    # lock's narrow zone at -30..-36 deg commanded, probes; -36 = 0.6 left no margin)


def _turn_reach(sk: Skills, arm: str, T_tg, T_ko, q0, turn_rad: float) -> float:
    """The largest of ``TURN_FRACTIONS`` whose stepped turn (``_turn_schedule``) the arm follows from ``q0`` at the
    target with the grasp ``T_ko``; 0 when none."""
    for frac in TURN_FRACTIONS:
        q, ok = q0, True
        sched = _turn_schedule(abs(math.degrees(turn_rad * frac)))
        for f in sched[::2] + sched[-1:]:
            ik = sk.kin.ik(arm, _turned(T_tg, turn_rad * frac * f) @ T_ko, q, n_random=0)
            if not ik.ok:
                ok = False
                break
            q = ik.q
        if ok:
            return frac
    return 0.0


def _turned(T, turn_rad):
    T = np.asarray(T, float).copy()
    T[:3, :3] = rot_z(turn_rad) @ T[:3, :3]
    return T


def plan_rest_handover(sk: Skills, label: str, targets: list, receptacle: str, turn_rad: float = 0.0,
                       place_kw: dict | None = None):
    """The giver, its lengthwise pick, the handover pose (the object as lifted, turned about the vertical) and the
    taker's grasp (object frame) that reaches a target, its turn and clears the receptacle. A dict or a SkillResult."""
    from dojo.skills.library import _grippers_apart, _hand_points, _min_dist
    t0 = time.perf_counter()
    w = sk.refresh()
    o = w.obj(label)
    notes = []
    for g_arm in sorted(ARMS, key=lambda a_: float(np.linalg.norm(sk.kin.base(a_)[:2, 3] - o.pos[:2]))):
        t_arm = "right" if g_arm == "left" else "left"
        # the taker's grasps that work at the target: IK there and at the hover, the hand clear of the receptacle,
        # and some turn that reaches
        fam = []
        for T_tg in targets:
            for note, T_ko, width in top_grasps_at(o, T_tg):
                g_open = gap_to_grip(min(G.MAX_GAP, width + TAKE_GAP))
                ok = True
                q_ins = None
                for up in (0.0, 0.025):
                    Tq = T_tg @ T_ko
                    Tq[2, 3] += up
                    r_ = sk.kin.ik(t_arm, Tq, w.home_q[t_arm], n_random=4)
                    if not r_.ok or not sk.coll.check(t_arm, r_.q, gap_to_grip(width), ignore=(label,),
                                                      grip_only=(receptacle,))[0]:
                        ok = False
                        break
                    q_ins = r_.q if q_ins is None else q_ins
                if not ok:
                    continue
                frac = _turn_reach(sk, t_arm, T_tg, T_ko, q_ins, turn_rad) if turn_rad else 1.0
                if frac < MIN_TURN_FRAC:
                    continue
                fam.append((note, T_ko, width, g_open, T_tg, frac))
        if not fam:
            notes.append(f"{t_arm}: no top grasp reaches the target")
            continue
        picks = sk.plan_pick(label, g_arm, grasps=lengthwise_grasps(sk, label, g_arm), lift=0.10, keep=4, max_eval=4,
                             lift_rad=0.05, return_all=True)
        if not isinstance(picks, list):
            notes.append(f"{g_arm} lengthwise pick: {picks.note[:80]}")
            continue
        best, sols = None, []
        for pp in picks[:2]:
            g_hold = gap_to_grip(pp.grasp.width)
            T_lift = sk.kin.tcp(g_arm, pp.plan.q_end) @ pp.T_tcp_obj
            for (hx, hy) in HANDOVER_XY:
                hx = hx if g_arm == "right" else -hx
                for hz in HANDOVER_Z:
                    for j in range(12):
                        if time.perf_counter() - t0 > PLAN_S:
                            break
                        R_H = rot_z(2 * math.pi * j / 12) @ T_lift[:3, :3]
                        T_H = T_from(np.array([hx, hy, hz]) - R_H @ G.mesh_com(o.asset), R_H)
                        rg = sk.kin.ik(g_arm, T_H @ inv_T(pp.T_tcp_obj), pp.plan.q_end, n_random=0)
                        if not rg.ok:
                            continue
                        P_give = _hand_points(sk, sk.kin.flange(g_arm, rg.q), g_hold)
                        for note, T_ko, width, g_open, T_tg, frac in fam:
                            Tt = T_H @ T_ko
                            pre = Tt.copy()
                            pre[:3, 3] -= Tt[:3, 0] * 0.06
                            if (_min_dist(P_give, _hand_points(sk, sk.kin.flange_from_tcp(Tt), g_open)) < HAND_CLEAR
                                    or _min_dist(P_give, _hand_points(sk, sk.kin.flange_from_tcp(pre), g_open)) < HAND_CLEAR):
                                continue
                            rp = sk.kin.ik(t_arm, pre, w.home_q[t_arm], n_random=4)
                            if not rp.ok:
                                continue
                            rt = sk.kin.ik(t_arm, Tt, rp.q, n_random=0)
                            if not rt.ok or not _grippers_apart(sk, g_arm, rg.q, g_hold, t_arm, rt.q, g_open,
                                                                min_d=HAND_CLEAR):
                                continue
                            cost = (float(np.abs(rg.q - pp.plan.q_end).max() + np.abs(rp.q - w.home_q[t_arm]).max())
                                    + pp.steps / 100.0 + 3.0 * (1.0 - frac))     # the full turn first
                            sols.append((cost, {"giver": g_arm, "taker": t_arm, "pick": pp, "T_H": T_H, "T_ko": T_ko,
                                                "width": width, "g_open": g_open, "target": T_tg, "note": note,
                                                "q_take": rt.q}))
                            if best is None or cost < best:
                                best = cost
                        if best is not None and best < 0.8:
                            break
        # the taker's place from where it takes the object must plan too (the joint move from the handover to the
        # hover met the lock block or the table in G3, 2 of 13); the cheapest few are checked
        for cost, sol in sorted(sols, key=lambda x: x[0])[:PLACE_CHECKS]:
            held = Held(label, inv_T(sol["T_ko"]), held_box(o, inv_T(sol["T_ko"])), None)
            kw = dict(place_kw or {})
            kw.pop("turn", None)
            kw["release"] = False
            pl = sk.plan_place(t_arm, sol["target"], yaw_sym=1, ignore=(receptacle,), q_start=sol["q_take"], held=held,
                               **kw)
            if isinstance(pl, PlacePlan):
                return sol
            notes.append(f"{sol['note']}: place from the handover: {pl.note[:80]}")
        notes.append(f"{g_arm} -> {t_arm}: no handover pose ({len(fam)} taker grasps reach the target, "
                     f"{len(sols)} handovers)")
    return SkillResult(False, "rest_handover", "ik_fail", 0, None, note="; ".join(notes[:4]))


def rest_handover(sk: Skills, label: str, targets: list, receptacle: str, place_kw: dict,
                  tol=(0.004, 7.0, 0.01)) -> SkillResult:
    """Plan (``plan_rest_handover``) and run: the giver's lengthwise pick, the carry to the handover pose, the taker's
    approach and close, the giver's release (open, up 6 cm, home), then the taker's place through ``place_kw`` (held on
    after a turn)."""
    from dojo.skills.motion import MEDIUM
    h = plan_rest_handover(sk, label, targets, receptacle, float(place_kw.get("turn", 0.0)), place_kw)
    if isinstance(h, SkillResult):
        return sk._record(h)
    g_arm, t_arm, pp = h["giver"], h["taker"], h["pick"]
    sk.log(f"[rest_handover] {label}: {g_arm} ({pp.grasp.note}) -> {t_arm} ({h['note']}) at "
           f"{np.round(h['T_H'][:3, 3], 3).tolist()}")
    r = sk._record(sk.run_pick(pp))
    if not r.ok:
        return r
    held = sk.held[g_arm]
    try:
        Tg = h["T_H"] @ inv_T(held.T_tcp_obj)
        mv = sk.motion.line(g_arm, sk.ex.q[g_arm], Tg, REST_CARRY, "rest_handover_carry", gripper=CLOSED, check=False,
                            max_step_deg=3.0)
        if not mv.ok:
            mv = sk.motion.to_pose(g_arm, sk.ex.q[g_arm], Tg, REST_CARRY, "rest_handover_carry", gripper=CLOSED,
                                   check=False)
        if not mv.ok:
            return sk._record(SkillResult(False, "rest_handover", "ik_fail", 0, g_arm, note=f"carry: {mv.note[:120]}"))
        for seg in mv.segments:
            seg.ease = 4
        sk.ex.run({g_arm: ArmPlan(g_arm, sk.ex.q[g_arm].copy(), CLOSED).move(mv).hold(3)}, name="rest handover carry")
        w = sk.refresh()
        o = w.obj(label)
        T_rel_g = inv_T(sk.kin.tcp(g_arm, w.arms[g_arm].q)) @ o.T
        drift_g = float(np.linalg.norm(T_rel_g[:3, 3] - held.T_tcp_obj[:3, 3]))
        if drift_g > 0.01 or o.pos[2] < TABLE_Z + 0.03:
            sk.held[g_arm] = None
            return sk._record(SkillResult(False, "rest_handover", "grasp_slip", 0, g_arm,
                                          note=f"{label} moved {drift_g*1000:.0f} mm in the giver's hand on the carry"))
        # the taker's grasp on the object as it is now
        Tt = o.T @ h["T_ko"]
        pre = Tt.copy()
        pre[:3, 3] -= Tt[:3, 0] * 0.06
        mp = sk.motion.to_pose(t_arm, sk.ex.q[t_arm], pre, 0.16, "take_pre", gripper=h["g_open"], check=False)
        mi = sk.motion.line(t_arm, mp.q_end, Tt, 0.04, "take_in", gripper=h["g_open"], check=False) if mp.ok else mp
        if not mp.ok or not mi.ok:
            return sk._record(SkillResult(False, "rest_handover", "ik_fail", 0, t_arm,
                                          note=f"taker approach on the measured pose: {mi.note[:120]}"))
        p = ArmPlan(t_arm, sk.ex.q[t_arm].copy(), sk.ex.g[t_arm]).grip(h["g_open"], 0).move(mp).move(mi).hold(2)
        p.grip(CLOSED, 4, "take_close").hold(4)
        sk.ex.run({t_arm: p}, name="rest handover take")
        Tq = sk.kin.tcp(g_arm, sk.ex.q[g_arm])
        Tb = Tq.copy()
        Tb[2, 3] += 0.06
        back = sk.motion.line(g_arm, sk.ex.q[g_arm], Tb, MEDIUM, "give_back", gripper=OPEN, check=False)
        rel = ArmPlan(g_arm, sk.ex.q[g_arm].copy(), CLOSED).grip(OPEN, 3).hold(2)
        if back.segments:
            rel.move(back)
        sk.ex.run({g_arm: rel}, name="rest handover release")
    except (BudgetExceeded, EpisodeEnded) as e:
        return sk._record(SkillResult(False, "rest_handover", "step_budget" if isinstance(e, BudgetExceeded) else "ended",
                                      0, t_arm, note=str(e)))
    sk.held[g_arm] = None
    w = sk.refresh()
    o = w.obj(label)
    T_rel = inv_T(sk.kin.tcp(t_arm, w.arms[t_arm].q)) @ o.T
    drift = float(np.linalg.norm(T_rel[:3, 3] - inv_T(h["T_ko"])[:3, 3]))
    if o.pos[2] < TABLE_Z + 0.03 or drift > 0.015:
        return sk._record(SkillResult(False, "rest_handover", "grasp_slip", 0, t_arm,
                                      note=f"{label} not in the taker's hand (drift {drift*1000:.0f} mm)"))
    gt = G.Grasp("rest_take", o.T @ inv_T(T_rel), h["width"], note=h["note"])
    sk.held[t_arm] = Held(label, T_rel, held_box(o, T_rel), gt)
    sk._record(SkillResult(True, "rest_handover", "ok", 0, t_arm,
                           note=f"{label} {g_arm} -> {t_arm} ({h['note']}), carry drift {drift_g*1000:.1f} mm, "
                                f"take drift {drift*1000:.1f} mm"))
    try:
        sk.ex.run({g_arm: sk.plan_home(g_arm)}, name="giver home")
    except (BudgetExceeded, EpisodeEnded) as e:
        return sk._record(SkillResult(False, "rest_handover", "step_budget", 0, g_arm, note=str(e)))
    kw = dict(place_kw)
    turn = float(kw.pop("turn", 0.0) or 0.0)
    if turn:
        kw["release"] = False                 # insert, hold on, then turn (``turn_held``); the checker wants no release
    pl = sk.plan_place(t_arm, h["target"], yaw_sym=1, ignore=(receptacle,), **kw)
    if not isinstance(pl, PlacePlan):
        return sk._record(SkillResult(False, "place", pl.status, 0, t_arm, note=f"after the handover: {pl.note[:150]}"))
    r = sk._record(sk.run_place(pl, tol[0], tol[1], tol_z=tol[2], ref_label=receptacle))
    if not turn or r.status in ("ended", "step_budget"):
        return r
    return sk._record(turn_held(sk, t_arm, label, turn))


def _turn_schedule(deg: float) -> list[float]:
    """Fractions of a turn of ``deg`` degrees, one per env step: ``TURN_STEP_DEG`` steps, ``TURN_FINE[2]`` deg steps
    each repeated once (a held step) between the fractions ``TURN_FINE[:2]``."""
    out, x = [], 0.0
    while x < deg - 1e-9:
        fine = TURN_FINE[0] * deg <= x < TURN_FINE[1] * deg
        x = min(deg, x + (TURN_FINE[2] if fine else TURN_STEP_DEG))
        out += [x / deg] * (2 if fine else 1)
    return out


def turn_held(sk: Skills, arm: str, label: str, turn: float, lift: float = TURN_LIFT) -> SkillResult:
    """Turn the held object about the vertical through its root by ``turn`` (rad), ``lift`` above where it rests now
    (sag-corrected; ``TURN_STEP_DEG`` per step; the largest of ``TURN_FRACTIONS`` that the wrist reaches)."""
    w = sk.refresh()
    o = w.obj(label)
    T_rel = inv_T(sk.kin.tcp(arm, w.arms[arm].q)) @ o.T
    C = sk._sag(arm, w)
    T0 = o.T.copy()
    T0[2, 3] += lift
    mv, frac = None, 0.0
    for frac in TURN_FRACTIONS:
        q, segs, ok = sk.ex.q[arm].copy(), [], True
        for f in _turn_schedule(abs(math.degrees(turn * frac))):
            T_t = T0.copy()
            T_t[:3, :3] = rot_z(turn * frac * f) @ T0[:3, :3]
            ik = sk.kin.ik(arm, C @ T_t @ inv_T(T_rel), q, n_random=0)
            if not ik.ok:
                ok = False
                break
            q = ik.q
            segs.append(q.copy())
        if ok:
            mv = segs
            break
    if mv is None:
        return SkillResult(False, "turn", "ik_fail", 0, arm, note=f"{label}: no turn of {math.degrees(turn):.0f} deg reaches")
    n0 = sk.ex.steps_used
    p = ArmPlan(arm, sk.ex.q[arm].copy(), CLOSED)
    from dojo.skills.motion import Segment
    p.move([Segment(np.vstack([sk.ex.q[arm]] + mv), 1.0)], "turn").hold(3, "turned")
    try:
        sk.ex.run({arm: p}, name=f"turn {label}")
    except EpisodeEnded as e:
        return SkillResult(True, "turn", "ended", sk.ex.steps_used - n0, arm,
                           note=f"{label} turned {math.degrees(turn * frac):.0f} deg: {e}")
    except BudgetExceeded as e:
        return SkillResult(False, "turn", "step_budget", sk.ex.steps_used - n0, arm, note=str(e))
    o = sk.refresh().obj(label)
    d = math.degrees(math.atan2(o.R[1, 0], o.R[0, 0]) - math.atan2(T0[1, 0], T0[0, 0]))
    d = (d + 180.0) % 360.0 - 180.0
    return SkillResult(True, "turn", "ok", sk.ex.steps_used - n0, arm,
                       note=f"{label} turned {d:.0f} deg of {math.degrees(turn * frac):.0f}, tilt "
                            f"{math.degrees(math.acos(min(1.0, float(o.R[2, 2])))):.1f} deg")
