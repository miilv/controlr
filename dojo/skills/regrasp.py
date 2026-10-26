"""Re-grasps against the table (skills-8): an elongated object lying on the table is stood up on its leading end.

``stand_up(sk, label, arm)``: grip the wide end from above, the closing axis level and across the long axis; loosen
the grip; move the hand along a quarter circle about the leading end's contact with the table, the hand's orientation
fixed, so the object turns in the pads about the closing axis until it stands on its leading end under the hand;
grip again and lift. The object then hangs leading end down from a straight-down hand, so any heading is one wrist
turn away: a lying key ends up ready for a keyhole without a second hand (twin, insert_key: the in-air handover lost
the key on the carry 0/31; a loose grip lifted clear of the table let the key swing down and wedge out of the pads,
3.5 mm of bow above the fingertips; turned against the table it never swings).

``stand_up_insert(sk, obj, receptacle, T_target, turn_deg)``: stand it up with the arm that reaches the insertion
(first relaying it as it lies to a spot that arm reaches, by the other arm, when it is out of reach), then the
closed-loop insertion tail (``Skills.plan_place(servo=True)``) and the turn, still holding it.
"""

from __future__ import annotations

import math

import numpy as np

from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import T_from, axis_angle_R, inv_T
from dojo.skills.kinematics import ARMS, TABLE_Z, TIP_BEYOND_TCP
from dojo.skills.skills import CLOSED, Held, SkillResult, Skills

PIVOT_GRIP = 0.27          # normalised gripper command while the object turns in the pads (twin, a key's 19 mm bow:
                           # ~12 N per finger; the full grip, 25 N, turned it 12 deg in 40 steps)
PIVOT_ARC_STEPS = 24       # env steps of the quarter circle
PIVOT_LIFT = 0.07          # m, the lift after the stand-up (a lift of 5 cm is the first rung of insert_key's ladder)
STAND_TOL_DEG = 12.0       # the stand-up counts when the long axis is this close to vertical


def wide_end_point(o, band: float = 0.3) -> np.ndarray | None:
    """Body-frame centre of an elongated object's wide end: the bbox centre of its mesh within ``band`` of its
    length from the end opposite the leading end (``insertion.insertion_axis``)."""
    from dojo.skills.insertion import insertion_axis
    if o.asset.mesh is None:
        return None
    V = np.asarray(o.asset.mesh[0], float)
    k, sg = insertion_axis(V)
    s = sg * V[:, k]                                      # grows towards the leading end
    far = V[s <= s.min() + band * (s.max() - s.min())]
    return (far.min(0) + far.max(0)) / 2


def plan_stand_up(sk: Skills, label: str, arm: str, q_start=None, steps: int = PIVOT_ARC_STEPS) -> dict | SkillResult:
    """The stand-up of ``label`` (lying) by ``arm``: the pick (approach, descent, close; no lift), the arc's joint
    targets, the lift, and the predicted in-hand pose once it stands. Returns a dict or a SkillResult."""
    from dojo.skills import grasps as G
    from dojo.skills.library import _leading_end
    w = sk.world
    o = w.obj(label)
    le = _leading_end(o)
    c_b = wide_end_point(o)
    if le is None or c_b is None:
        return SkillResult(False, "stand_up", "logic", 0, arm, note=f"{label}: no mesh")
    tip, up_obj = le
    if abs(float(up_obj[2])) > 0.5:
        return SkillResult(False, "stand_up", "logic", 0, arm, note=f"{label} is not lying ({up_obj.round(2).tolist()})")
    P0 = o.R @ c_b + o.pos                                # pivot point (the wide end's centre)
    tip_t = np.array([tip[0], tip[1], TABLE_Z])
    d = P0 - tip_t
    d[2] = 0.0
    r_h = float(np.linalg.norm(d))
    d /= max(r_h, 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    closing = np.cross(up, d)
    a = np.array([0.0, 0.0, -1.0])
    tcp0 = P0.copy()
    tcp0[2] = max(float(P0[2]), TABLE_Z + 0.0025 + TIP_BEYOND_TCP)
    off = tcp0 - P0
    r = float(np.linalg.norm(P0 - tip_t))
    th0 = math.atan2(float(P0[2] - tip_t[2]), r_h)
    Vw = np.asarray(o.asset.mesh[0], float) @ o.R.T + o.pos
    near = Vw[np.linalg.norm(Vw - P0, axis=1) < 0.015]
    width = float(np.ptp((near if len(near) else Vw) @ closing))
    notes = []
    for sgn in (1.0, -1.0):
        c = sgn * closing
        Rg = np.stack([a, c, np.cross(a, c)], 1)
        g = G.Grasp("stand_up", T_from(tcp0, Rg), min(width, 0.03), note=f"wide end, closing {'+' if sgn > 0 else '-'}")
        pp = sk.plan_pick(label, arm, grasps=[g], lift=0.0, keep=1, max_eval=1, q_start=q_start)
        if not hasattr(pp, "plan"):
            notes.append(f"pick {g.note}: {pp.note[:80]}")
            continue
        q = pp.plan.q_end.copy()
        arc = []
        for i in range(1, steps + 1):
            th = th0 + (math.pi / 2 - th0) * i / steps
            P = tip_t + r * (math.cos(th) * d + math.sin(th) * up)
            ik = sk.kin.ik(arm, T_from(P + off, Rg), q, n_random=0)
            if not ik.ok or float(np.abs(ik.q - q).max()) > 0.25:
                arc = None
                break
            q = ik.q
            arc.append(q.copy())
        if arc is None:
            notes.append(f"arc {g.note}: ik")
            continue
        P_up = tip_t + r * up
        T_up = T_from(P_up + off + np.array([0, 0, PIVOT_LIFT]), Rg)
        lift = sk.motion.line(arm, q, T_up, 0.04, "stand_up_lift", gripper=CLOSED, check=False)
        if not lift.ok:
            notes.append(f"lift {g.note}: {lift.note[:60]}")
            continue
        # predicted pose once it stands: turned about the closing axis through the leading end's contact
        Rr = axis_angle_R(np.cross(d, up), math.pi / 2 - th0)
        R_f = Rr @ o.R
        p_f = tip_t + Rr @ (o.pos - tip_t)
        T_tcp_obj = inv_T(T_from(P_up + off, Rg)) @ T_from(p_f, R_f)
        return {"arm": arm, "pick": pp, "arc": arc, "lift": lift, "T_tcp_obj": T_tcp_obj, "grasp": g,
                "steps": pp.steps + len(arc) + lift.steps + 14}
    return SkillResult(False, "stand_up", "ik_fail", 0, arm, note="; ".join(notes[:4]))


def run_stand_up(sk: Skills, plan: dict) -> SkillResult:
    """Execute ``plan_stand_up``'s plan; the in-hand pose is re-measured when the provider can (G3), else taken from
    the prediction."""
    from dojo.skills.motion import held_box
    arm, pp, label = plan["arm"], plan["pick"], plan["pick"].label
    n0 = sk.ex.steps_used
    try:
        p = ArmPlan(arm, pp.plan.q0, pp.plan.g0)
        p.ops = [op for op in pp.plan.ops if op[-1] != "lift"]
        sk.ex.run({arm: p}, name=f"stand_up {label} grasp")
        sk.ex.g[arm] = PIVOT_GRIP
        sk.ex.hold(2)
        for q in plan["arc"]:
            sk.ex.q[arm] = np.asarray(q, float).copy()
            sk.ex.hold(1)
        sk.ex.hold(3)
        sk.ex.g[arm] = CLOSED
        sk.ex.hold(6)
        sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), CLOSED).move(plan["lift"], "lift").hold(3)},
                  name=f"stand_up {label} lift")
    except BudgetExceeded as e:
        return SkillResult(False, "stand_up", "step_budget", sk.ex.steps_used - n0, arm, note=str(e))
    except EpisodeEnded as e:
        return SkillResult(False, "stand_up", "ended", sk.ex.steps_used - n0, arm, note=str(e))
    w = sk.refresh()
    o = w.obj(label)
    T_tcp = sk.kin.tcp(arm, w.arms[arm].q)
    T_rel = inv_T(T_tcp) @ o.T if sk.remeasure else plan["T_tcp_obj"]
    from dojo.skills.library import _leading_end
    tip, up_obj = _leading_end(o)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up_obj[2])))))
    rose = float(o.bottom_z - TABLE_Z)
    if sk.remeasure and (tilt > STAND_TOL_DEG or rose < 0.02):
        sk.held[arm] = None
        return SkillResult(False, "stand_up", "grasp_slip", sk.ex.steps_used - n0, arm,
                           note=f"{label}: tilt {tilt:.0f} deg, {rose*1000:.0f} mm over the table")
    sk.held[arm] = Held(label, T_rel, held_box(o, T_rel), plan["grasp"])
    drift = float(np.linalg.norm(T_rel[:3, 3] - plan["T_tcp_obj"][:3, 3]))
    return SkillResult(True, "stand_up", "ok", sk.ex.steps_used - n0, arm,
                       note=f"{label} standing in the hand: tilt {tilt:.1f} deg, {drift*1000:.1f} mm off the prediction")


def _insert_plan(sk: Skills, arm: str, label: str, T_tcp_obj, grasp, T_target, turn: float, q_start, rec: str):
    from dojo.skills.motion import held_box
    o = sk.world.obj(label)
    h = Held(label, T_tcp_obj, held_box(o, T_tcp_obj), grasp)
    return sk.plan_place(arm, T_target, yaw_sym=1, ignore=(rec,), q_start=q_start, held=h, pre_h=0.02,
                         descend_rad=0.02, release_gap=0.0, settle=3, pre_settle=3, turn=turn, release=False,
                         grip_only=(rec,), servo=True, servo_ref=rec, approach_rad=0.09)


def stand_up_insert(sk: Skills, obj: str, rec: str, T_target, turn_deg: float = 0.0,
                    relay_spots=((0.0, -0.20), (0.0, -0.16), (0.05, -0.2), (-0.05, -0.2), (0.0, -0.24))) -> SkillResult:
    """Stand ``obj`` up (``plan_stand_up``) with an arm that can then insert it at ``T_target`` (object pose, upright),
    insert it and turn it ``turn_deg`` about the vertical, still holding it. When the arm that reaches the insertion
    cannot reach the object, the other arm first puts it down as it lies on a spot both reach."""
    from dojo.skills.library import _run_leg, rest_with_R
    turn = math.radians(turn_deg)
    T_target = np.asarray(T_target, float)

    def plan_for(arm):
        st = plan_stand_up(sk, obj, arm)
        if isinstance(st, SkillResult):
            return st, None
        q_end = st["lift"].q_end
        pl = _insert_plan(sk, arm, obj, st["T_tcp_obj"], st["grasp"], T_target, turn, q_end, rec)
        return st, pl
    w = sk.refresh()
    arms = sorted(ARMS, key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - T_target[:2, 3])))
    notes = []
    for arm in arms:
        st, pl = plan_for(arm)
        if pl is not None and not isinstance(pl, SkillResult):
            return _run(sk, st, obj, rec, T_target, turn)
        notes.append(f"{arm}: {(st.note if isinstance(st, SkillResult) else pl.note)[:90]}")
    # relay: the inserting arm (nearest the target) stands it up on a spot it reaches, the other arm puts it there
    arm = arms[0]
    o = w.obj(obj)
    for xy in relay_spots:
        T_spot = rest_with_R(sk, obj, xy, o.R)
        saved = (o.pos.copy(), o.quat.copy())
        from dojo.skills.geometry import R_to_quat
        o.pos, o.quat = T_spot[:3, 3].copy(), R_to_quat(T_spot[:3, :3])
        try:
            st, pl = plan_for(arm)
        finally:
            o.pos, o.quat = saved
        if pl is None or isinstance(pl, SkillResult):
            notes.append(f"relay {xy}: {(st.note if isinstance(st, SkillResult) else pl.note)[:70]}")
            continue
        sk.log(f"[stand_up] {obj}: relay as it lies to {xy}, then {arm} stands it up")
        r = _run_leg(sk, obj, [T_spot], 1, tol=(0.03, 15.0, 0.03))
        sk.bank()
        if r.status in ("step_budget", "ended"):
            return r
        w = sk.refresh()
        for a2 in (arm,) + tuple(x for x in ARMS if x != arm):
            st, pl = plan_for(a2)
            if pl is not None and not isinstance(pl, SkillResult):
                return _run(sk, st, obj, rec, T_target, turn)
        return sk._record(SkillResult(False, "stand_up_insert", "ik_fail", 0, arm,
                                      note=f"after the relay: {(st.note if isinstance(st, SkillResult) else pl.note)[:120]}"))
    return sk._record(SkillResult(False, "stand_up_insert", "ik_fail", 0, None, note="; ".join(notes[:5])))


def _run(sk: Skills, st: dict, obj: str, rec: str, T_target, turn: float) -> SkillResult:
    arm = st["arm"]
    sk.log(f"[stand_up] {obj}: {arm} ({st['grasp'].note}), ~{st['steps']} steps")
    r = sk._record(run_stand_up(sk, st))
    if not r.ok:
        return r
    pl = _insert_plan(sk, arm, obj, sk.held[arm].T_tcp_obj, st["grasp"], T_target, turn, sk.ex.q[arm], rec)
    if isinstance(pl, SkillResult):
        return sk._record(pl)
    return sk._record(sk.run_place(pl, tol_xy=0.004, tol_tilt=7.0, tol_z=0.01, ref_label=rec))
