"""Sweeping small objects into a dustpan with a hand broom (skills, Generalization-New).

The dustpan is a receptacle whose collecting region is its support circle (``passive.support``, e.g.
``broom_shovel/0``: 81 mm about a point 53 mm in from its centre) behind a front lip it lies flush with the table at;
the opening faces the asset frame's -y, the handle sticks out at +y. The broom is a long tool whose head (the
bristle block) is the end with the larger cross-section.

1. the dustpan is dragged flat along the table (never lifted: the checker wants it within 10 mm of its start
   height and within 5 deg of upright) by the arm on its handle's side to a spot beside the objects, its opening
   facing them, and held there;
2. the broom: the other arm picks it by the handle (when it lies out of that arm's reach, the first arm puts it
   down within reach first: a relay through the table);
3. each object, the nearest to the opening first: pushed with the side of the broom's head held level just over
   the table (its long axis across the push, a wide flat pusher), first onto the dustpan's axis in front of the
   lip, then along the axis over the lip into the pan; re-read after every stroke;
4. the broom is put down beside the dustpan, away from it (the checker wants the dustpan left of the broom by
   10 cm), the dustpan let go; both arms home.

Pushes and drags are contact motions: unchecked lines, the world re-read after each.
"""
from __future__ import annotations

import math

import numpy as np

from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import T_from, inv_T, rot_z
from dojo.skills.kinematics import ARMS, TABLE_Z, TIP_BEYOND_TCP
from dojo.skills.motion import FAST, MEDIUM, SLOW
from dojo.skills.skills import CLOSED, OPEN, SkillResult, Skills

PUSH_Z = 0.008                 # m: the broom head's lowest point over the table while pushing (clears the pan's lip)
PUSH_STROKE = 0.14             # m: longest single push
PUSH_TOL = 0.015               # m: a leg is done within this of its target
STAGE_OUT = 0.06               # m: the staging point is this far in front of the lip
INSIDE = 0.07                  # m: objects are pushed this far past the lip (the support circle's centre is 87 mm in)
LANE = 0.035                   # m: lateral spacing of the objects' lanes into the pan
LIFT = 0.05                    # m: the head is lifted this far over the objects' tops between strokes


def _other(arm: str) -> str:
    return "right" if arm == "left" else "left"


def pan_geometry(pan) -> dict:
    """World: the support centre and radius, the entry direction (into the pan, from its lip), the lip's centre, the
    lateral axis, and the handle's point (asset frame +y end)."""
    sup = pan.asset.support or {}
    c_l, r = np.array([0.0, -0.053, -0.015]), 0.081
    for v in sup.values():
        try:
            fr = v["center"][0] if isinstance(v, dict) else v[0][0]
            c_l = np.asarray(fr[:3], float)
            r = float((v.get("radius") or [r])[0]) if isinstance(v, dict) else r
        except (KeyError, IndexError, TypeError):
            pass
        break
    R = pan.R
    e = R @ np.array([0.0, 1.0, 0.0])
    e[2] = 0.0
    e /= np.linalg.norm(e) or 1.0
    lat = np.array([-e[1], e[0], 0.0])
    c = R @ c_l + pan.pos
    lip_l = float(pan.asset.bbox_min[1])
    lip = R @ np.array([c_l[0], lip_l, 0.0]) + pan.pos
    return {"c": c, "r": r, "e": e, "lat": lat, "lip": lip, "lip_in": float(c_l[1] - lip_l)}


def tool_head(tool) -> dict:
    """The broom's head in its own frame: the half of its long axis with the larger cross-section. {"axis": k,
    "sign", "centre", "pts"} (pts: head mesh points, object frame)."""
    V = np.asarray(tool.asset.mesh[0], float) if tool.asset.mesh is not None else None
    ext = tool.asset.extents
    k = int(np.argmax(ext))
    if V is None:
        c = np.zeros(3)
        c[k] = 0.25 * ext[k]
        return {"axis": k, "sign": 1.0, "centre": c, "pts": None}
    mid = 0.5 * (V[:, k].min() + V[:, k].max())
    halves = []
    for sgn in (1.0, -1.0):
        P = V[(V[:, k] - mid) * sgn > 0]
        o = [i for i in range(3) if i != k]
        area = float(np.ptp(P[:, o[0]]) * np.ptp(P[:, o[1]])) if len(P) else 0.0
        halves.append((area, sgn, P))
    area, sgn, P = max(halves, key=lambda h: h[0])
    return {"axis": k, "sign": sgn, "centre": 0.5 * (P.min(0) + P.max(0)), "pts": P[:: max(1, len(P) // 3000)]}


def handle_grasps(sk: Skills, tool, arm: str) -> list:
    """Top-down grasps of the tool on its handle half (away from the head)."""
    from dojo.skills import grasps as G
    hd = tool_head(tool)
    k, sgn = hd["axis"], hd["sign"]
    mid = 0.5 * (tool.asset.bbox_min[k] + tool.asset.bbox_max[k])
    out = []
    for g in G.box_grasps(tool, sk.kin, arm) + G.slice_grasps(tool, sk.kin, arm) + G.antipodal_grasps(tool, sk.kin, arm):
        p = (inv_T(tool.T) @ np.r_[g.T[:3, 3], 1.0])[:3]
        if (p[k] - mid) * sgn < -0.02 and -float(g.T[2, 0]) > 0.8 and float(g.width) >= 0.012:
            # nearest the head first (the least lever on a level carry), straight down first
            g.score = float(g.score) + 2.0 * abs(float(p[k] - mid)) + 0.5 * (1.0 + float(g.T[2, 0]))
            out.append(g)
    return sorted(out, key=lambda g: g.score)


def _pick_tool(sk: Skills, tool: str, arm: str):
    o = sk.world.obj(tool)
    pp = sk.plan_pick(tool, arm, grasps=handle_grasps(sk, o, arm), keep=2, lift=0.06)
    if isinstance(pp, SkillResult):
        return pp
    r = sk._record(sk.run_pick(pp))
    return r


def relay_tool(sk: Skills, tool: str, arm: str, near_xy=(0.08, -0.27)) -> SkillResult:
    """``arm`` cannot pick the tool where it lies: the other arm puts it down, lying as it is, on a free spot near
    ``near_xy`` (in front, between the arms) - the hand-over, through the table."""
    from dojo.skills import library as L
    from dojo.skills.targets import level_R
    o = sk.world.obj(tool)
    giver = _other(arm)
    for sp in L.free_spots(sk, tool, near_xy=near_xy, clearance=0.02)[:4]:
        T_sp = L.rest_with_R(sk, tool, sp, level_R(o.R))
        # any grasp near the centre of mass (default candidates, hold-ranked): a 40 g broom pinched at its handle's
        # end swung 17 cm in the pads on the carry (twin)
        plan = sk.plan_transfer(tool, [T_sp], arms=[giver], yaw_sym=4)
        if isinstance(plan, SkillResult):
            continue
        r = sk.run_transfer(*plan, tol_xy=0.05, tol_tilt=15.0, tol_z=0.03)
        sk.held[giver] = None
        return sk._record(r)
    return sk._record(SkillResult(False, "relay", "ik_fail", 0, giver, note=f"no spot for {tool} near {near_xy}"))


def pan_handle_grasps(o) -> list:
    """Top-down pinches across the dustpan's handle (its asset frame's +y end): slices of the mesh at a few stations
    along the handle, pads on its sides (closing along the asset x), the TCP at the slice's mid height."""
    from dojo.skills import grasps as G
    from dojo.skills.kinematics import down_R
    if o.asset.mesh is None:
        return []
    V = np.asarray(o.asset.mesh[0], float)
    y1 = float(V[:, 1].max())
    out = []
    for dy in (0.03, 0.045, 0.06, 0.075):
        y = y1 - dy
        S = V[np.abs(V[:, 1] - y) < 0.006]
        if len(S) < 6:
            continue
        w = float(np.ptp(S[:, 0]))
        if w > 0.075:
            continue
        p_l = np.array([0.5 * (S[:, 0].min() + S[:, 0].max()), y, 0.5 * (S[:, 2].min() + S[:, 2].max())])
        p = o.R @ p_l + o.pos
        ax = o.R[:, 0]
        yaw = math.atan2(ax[1], ax[0])
        for flip in (0.0, math.pi):
            T = T_from(p, down_R(yaw + flip))
            out.append(G.Grasp("handle", T, w, score=dy, note=f"handle {dy * 1000:.0f} mm from its end, w {w * 1000:.0f} mm"))
    return out


def _drag_pan(sk: Skills, pan: str, arm: str, T_target: np.ndarray) -> SkillResult:
    """Grip the dustpan's handle from above and slide it flat to ``T_target`` (object pose); keeps holding it."""
    w = sk.refresh()
    o = w.obj(pan)
    cands = [(0, g) for g in pan_handle_grasps(o)]
    pp = sk.plan_pick(pan, arm, grasps=[g for _, g in cands[:30]], keep=1, lift=0.0)
    if isinstance(pp, SkillResult):
        return sk._record(SkillResult(False, "drag", "ik_fail", 0, arm, note=f"no handle grasp: {pp.note[:120]}"))
    T_rel = pp.T_tcp_obj
    T_tcp_end = T_target @ inv_T(T_rel)
    n0 = sk.ex.steps_used
    # the pick without its lift: open, approach, descend, close (the lift op is a zero move)
    sk.ex.run({arm: pp.plan}, name=f"grip {pan}")
    q = sk.ex.q[arm].copy()
    mv = sk.motion.line(arm, q, T_tcp_end, SLOW, "drag", gripper=CLOSED, check=False, max_step_m=0.01)
    if not mv.ok:
        return sk._record(SkillResult(False, "drag", "ik_fail", sk.ex.steps_used - n0, arm, note=mv.note[:120]))
    sk.ex.run({arm: ArmPlan(arm, q, CLOSED).move(mv, "drag").hold(4, "settle")}, name=f"drag {pan}")
    o2 = sk.refresh().obj(pan)
    d = float(np.linalg.norm(o2.pos[:2] - T_target[:2, 3]))
    return sk._record(SkillResult(d < 0.03, "drag", "ok" if d < 0.03 else "place_miss", sk.ex.steps_used - n0, arm,
                                  note=f"{pan} {d * 1000:.0f} mm from its target, tilt {o2.tilt():.1f} deg"))


def _stroke(sk: Skills, arm: str, tool: str, obj: str, u: np.ndarray, dist: float):
    """One push of ``obj`` along ``u`` (world xy unit) by ``dist`` with the side of the held tool's head. A plan or a
    reason."""
    w = sk.refresh()
    t, o = w.obj(tool), w.obj(obj)
    T_tcp = sk.kin.tcp(arm, sk.ex.q[arm])
    T_rel = inv_T(T_tcp) @ t.T
    hd = tool_head(t)
    a0 = t.R[:, hd["axis"]].copy()
    a0[2] = 0.0
    if np.linalg.norm(a0) < 0.5:
        return "the tool is not held level"
    ang_a = math.atan2(a0[1], a0[0])
    ang_u = math.atan2(u[1], u[0])
    half_o = 0.5 * float(np.max(o.asset.extents[:2]))
    why = "no pose"
    for side in (1.0, -1.0):
        dpsi = (ang_u + side * math.pi / 2) - ang_a
        dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi
        Rz = rot_z(dpsi)
        R_t = Rz @ t.R
        P = (hd["pts"] @ R_t.T) if hd["pts"] is not None else np.zeros((1, 3))
        c_h = R_t @ hd["centre"]
        # the head's centre behind the object, its lowest point PUSH_Z over the table
        along = P @ np.r_[u, 0.0]
        half_u = float(along.max() - (c_h @ np.r_[u, 0.0]))
        B = np.r_[o.center_w[:2] - u * (half_u + half_o + 0.012), 0.0]
        z_low = float(P[:, 2].min() - c_h[2])
        B[2] = TABLE_Z + PUSH_Z - z_low
        T_t = np.eye(4)
        T_t[:3, :3] = R_t
        T_t[:3, 3] = B - c_h
        T_s = T_t @ inv_T(T_rel)
        T_e = T_s.copy()
        T_e[:2, 3] += u * dist
        T_ab = T_s.copy()
        T_ab[2, 3] = max(T_s[2, 3], T_tcp[2, 3]) + 0.0
        T_ab[2, 3] = max(T_ab[2, 3], T_s[2, 3] + float(o.top_z - TABLE_Z) + LIFT)
        T_up = T_e.copy()
        T_up[2, 3] = T_ab[2, 3]
        m1 = sk.motion.to_pose(arm, sk.ex.q[arm], T_ab, MEDIUM, "sweep_above", gripper=CLOSED, check=False)
        if not m1.ok:
            why = f"above: {m1.note[:60]}"
            continue
        m2 = sk.motion.line(arm, m1.q_end, T_s, MEDIUM, "sweep_down", gripper=CLOSED, check=False)
        if not m2.ok:
            why = f"down: {m2.note[:60]}"
            continue
        m3 = sk.motion.line(arm, m2.q_end, T_e, SLOW, "sweep", gripper=CLOSED, check=False, max_step_m=0.01)
        if not m3.ok:
            why = f"stroke: {m3.note[:60]}"
            continue
        m4 = sk.motion.line(arm, m3.q_end, T_up, MEDIUM, "sweep_up", gripper=CLOSED, check=False)
        p = ArmPlan(arm, sk.ex.q[arm].copy(), CLOSED).move(m1, "above").move(m2, "down").move(m3, "sweep")
        if m4.ok:
            p.move(m4, "up")
        return p
    return why


def push_leg(sk: Skills, arm: str, tool: str, obj: str, target_xy, tol: float = PUSH_TOL, max_strokes: int = 3,
             along=None) -> float:
    """Push ``obj`` towards ``target_xy`` with the held tool, re-read after every stroke (``along``: the push direction
    fixed, e.g. the pan's axis; else towards the target). Returns the distance left (m)."""
    target_xy = np.asarray(target_xy, float)[:2]
    left = float("inf")
    for _ in range(max_strokes):
        o = sk.refresh().obj(obj)
        d = target_xy - o.center_w[:2]
        left = float(np.linalg.norm(d))
        if left <= tol:
            return left
        u = np.asarray(along, float)[:2] if along is not None else d / left
        u = u / (np.linalg.norm(u) or 1.0)
        dist = min(PUSH_STROKE, max(0.0, float(d @ u)) + 0.004)
        if dist < 0.004:
            return left
        p = _stroke(sk, arm, tool, obj, u, dist)
        if isinstance(p, str):
            sk.log(f"[sweep] {obj}: no stroke ({p})")
            return left
        sk.ex.run({arm: p}, name=f"sweep {obj}")
        # mode B: nothing re-measures a pushed object; the expected outcome goes to the provider's confirm
        T_x = o.T.copy()
        T_x[:2, 3] += u * max(0.0, dist - 0.012)
        sk.confirm_moved(obj, T_x, "sweep")
        # the dustpan the other hand holds and the objects already in it: also moved by nothing we measure; the drag
        # and the pushes only touch the one object
    o = sk.refresh().obj(obj)
    return float(np.linalg.norm(target_xy - o.center_w[:2]))


def in_pan(pan_g: dict, xy) -> bool:
    return float(np.linalg.norm(np.asarray(xy, float)[:2] - pan_g["c"][:2])) < pan_g["r"]


def sweep_into(sk: Skills, objects: list[str], pan: str, tool: str) -> dict:
    """Sweep ``objects`` into the dustpan ``pan`` with the broom ``tool`` (module doc)."""
    w = sk.refresh()
    pool = [l for l in objects if l in w.objects]
    if not pool:
        return {"ok": False, "stage": "sweep", "note": "nothing to sweep"}
    # the dustpan and the broom as small boxes from their meshes: their bboxes are mostly air (a broom lying by the
    # dustpan's handle had every grasp "hit" the dustpan's box)
    from dojo.skills.motion import mesh_parts
    for lab in (pan, tool):
        if w.obj(lab).asset.mesh is not None:
            sk.coll.parts[lab] = mesh_parts(*w.obj(lab).asset.mesh, cell=0.015, step=0.004)
    try:
        P = np.array([w.obj(l).center_w[:2] for l in pool])
        po = w.obj(pan)
        side = -1.0 if float(np.mean(P[:, 0])) > float(po.pos[0]) else 1.0         # objects to the right: pan left
        pan_arm = "left" if side < 0 else "right"
        arm = _other(pan_arm)
        # 0. the broom within the sweeping arm's reach (else the other arm puts it there first)
        if isinstance(sk.plan_pick(tool, arm, grasps=handle_grasps(sk, w.obj(tool), arm), keep=1, lift=0.06),
                      SkillResult):
            r = relay_tool(sk, tool, arm, near_xy=(0.08 * (1 if arm == "right" else -1), -0.27))
            if not r.ok and r.status in ("ik_fail", "step_budget", "ended"):
                return {"ok": False, "stage": "broom relay", "note": r.note}
            w = sk.refresh()
            P = np.array([w.obj(l).center_w[:2] for l in pool])
            po = w.obj(pan)
        # 1. the dustpan beside the objects, its opening facing them, held by the arm on the far side
        e = np.array([side * 1.0, 0.0])                                              # into the pan: away from them
        near = P[np.argmin(P[:, 0] * -side)] if side < 0 else P[np.argmax(P[:, 0])]
        g0 = pan_geometry(po)
        lip_x = float(near[0]) + side * 0.05
        c_t = np.array([lip_x + side * g0["lip_in"], float(np.clip(np.mean(P[:, 1]), -0.2, 0.05))])
        # the asset's +y (into the pan) along e: yaw
        yaw = math.atan2(e[1], e[0]) - math.pi / 2
        R_t = rot_z(yaw) @ rot_z(-po.yaw()) @ po.R
        c_l = inv_T(po.T) @ np.r_[g0["c"], 1.0]
        T_t = np.eye(4)
        T_t[:3, :3] = R_t
        T_t[:3, 3] = np.r_[c_t, 0.0] - R_t @ c_l[:3]
        T_t[2, 3] = po.pos[2]
        r = _drag_pan(sk, pan, pan_arm, T_t)
        if not r.ok and r.status == "ik_fail":
            return {"ok": False, "stage": "dustpan", "note": r.note}
        sk.confirm_moved(pan, T_t, "drag")
        g = pan_geometry(sk.refresh().obj(pan))
        # 2. the broom in the other hand
        r = _pick_tool(sk, tool, arm)
        if not r.ok:
            return {"ok": False, "stage": "broom", "note": r.note}
        # 3. each object into the pan along its lane
        def key(l):
            return float(np.linalg.norm(sk.world.obj(l).center_w[:2] - g["lip"][:2]))
        done, failed = [], []
        lanes = [0.0, -LANE, LANE, -2 * LANE, 2 * LANE]
        for i, lab in enumerate(sorted(pool, key=key)):
            lane = lanes[i % len(lanes)]
            S = g["lip"][:2] - g["e"][:2] * STAGE_OUT + g["lat"][:2] * lane
            F = g["lip"][:2] + g["e"][:2] * INSIDE + g["lat"][:2] * lane
            o = sk.refresh().obj(lab)
            off = o.center_w[:2] - g["lip"][:2]
            lateral = float(off @ g["lat"][:2]) - lane
            front = float(off @ g["e"][:2])
            if abs(lateral) > 0.015 or front > -0.02:
                push_leg(sk, arm, tool, lab, S)
            left = push_leg(sk, arm, tool, lab, F, along=g["e"][:2])
            ok = in_pan(g, sk.refresh().obj(lab).center_w)
            sk._record(SkillResult(ok, "sweep", "ok" if ok else "place_miss", 0, arm,
                                   note=f"{lab} {'in' if ok else 'not in'} the pan ({left * 1000:.0f} mm from its lane end)"))
            (done if ok else failed).append(lab)
        # 4. the broom down beyond the objects' side, away from the pan; the pan let go
        T = sk.kin.tcp(arm, sk.ex.q[arm])
        dst = T.copy()
        dst[0, 3] = float(np.clip(g["c"][0] - side * 0.25, -0.4, 0.4))
        dst[1, 3] = -0.25
        dst[2, 3] = max(T[2, 3], TABLE_Z + 0.08)
        mv = sk.motion.to_pose(arm, sk.ex.q[arm], dst, FAST, "broom_away", gripper=CLOSED, check=False)
        p = ArmPlan(arm, sk.ex.q[arm].copy(), CLOSED)
        if mv.ok:
            p.move(mv, "away")
        p.grip(OPEN, 6, "drop broom").hold(6, "let_go")
        q = sk.ex.q[pan_arm].copy()
        T2 = sk.kin.tcp(pan_arm, q)
        T2[2, 3] += 0.06
        up = sk.motion.line(pan_arm, q, T2, MEDIUM, "pan_let_go", gripper=OPEN, check=False)
        p2 = ArmPlan(pan_arm, q, CLOSED).grip(OPEN, 6, "open")
        if up.ok:
            p2.move(up, "up")
        sk.ex.run({arm: p, pan_arm: p2}, name="let go")
        for a in ARMS:
            sk.held[a] = None
    except (BudgetExceeded, EpisodeEnded) as e:
        return {"ok": False, "stage": "sweep", "note": str(e)}
    if failed:
        return {"ok": False, "stage": "sweep", "note": f"in {done}, out {failed}"}
    return {"ok": True, "stage": "sweep", "done": done}


SKILLS = {"sweep_into": sweep_into}
