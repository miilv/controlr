# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/put_into_bin_rgb_strict_v3/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""put_into_bin_rgb_strict_v3 (RoboDojo, STRICT tier B): put_into_bin_rgb_strict_v2 plus clearance-aware motion and a side
grasp for tall standing objects. basket round r2 (Claude, 25 Oct 2026). Developed on seed 0 layout 9 only.

Loads the frozen v2 skill (skills/put_into_bin_rgb_strict_v2, unchanged) as a PRIVATE module instance and replaces, in
that instance only, the functions its stages call for motion:
  * every arm move (`_move2`, `safe_carry`) becomes `_move3`: the goal joints come from the server's free IK (`reach`
    from the measured joints, the same IK the move uses), the straight joint line the server executes is sampled, and
    forward kinematics from the public URDF (fk.py, checked to 0.2 mm against logged flange poses) places capsules on
    the arm links, points on the gripper and on the held object (carried in the flange frame from the close). Obstacles:
    every located object (standing = vertical cylinder to its top, lying = cylinders along the axis), objects already
    placed in containers (arg `placed`, plus those the belief dropped in this stage), basket walls (thin boxes to the
    rim), the table for links and held objects, and the OTHER arm's links at its current joints. A blocked line is
    replaced by legs through a safe height (rise at the current xy, traverse, descend), each leg checked the same way;
    the joint line is then forced (max_dev 1.0) so the executed path is the checked one.
    Targets the r1 put_bottles misses: L0 approach knocked a small bottle over, L2 rise from a low park pose swept a
    tall bottle off the table, L4 a carry swept a tall bottle along (all reproduced by this check on the r1 joint logs).
  * picks use `pick3`: standing objects get a pregrasp 5 cm above the top (v2: 1.2 cm), plus a SIDE GRASP for tall
    standing objects (h >= 0.15 m) when no top-down chain exists or `side=true`: horizontal tool from the picking arm's
    side, jaws horizontal, TCP at 62-70 % of the height (z >= 0.90: lower horizontal tools wrap the IK), pregrasp 10 cm back, slow line-in, close, lift 8 cm (r1 put_bottles
    L4: the 1.00 m-top bottle had no reachable neck grasp and was skipped). A side-held object is released with its
    current tool orientation (rotated about world z only) by `drop_in3`; top-down-held objects use v2's rim-aware drop.
  * placed objects (`placed`) also enter the wall screen of picks, and deep release is skipped in a container that
    already holds objects (r1 classify: placed objects knocked out of baskets by later stages in L2/L3/L4).
New stages: `pre` (move to the chosen pregrasp and stop, so the planner can re-locate from the wrist view), and `home`
checks each arm's joint line home and goes via the survey pose when it is blocked. Other stages as v2.
Every direct reply carries `placed_entry` (container, release xy, z range, radius) for the planner's `placed` list.
"""
import importlib.util
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_Z = 0.765
GRASP_OFF = 0.1501
_M = {}
CTX = {"target": None, "placed": [], "raws": {}, "containers": [], "table": None, "log": None, "transits": []}
SIDE_MIN_H = 0.15
SIDE_FRACS = (0.62, 0.70)               # grasp height as a fraction of the object height above the table
SIDE_ZMIN = 0.90
SIDE_FIRST_H = 0.20                    # standing objects taller than this: side grasp first, top-down as fallback
RELAY_SPOTS_EXTRA = [(-0.12, -0.30), (0.12, -0.30), (-0.15, -0.20), (0.15, -0.20), (0.0, -0.33)]
SIDE_LIFT = 0.11                       # side-held tall bottles: bottom ~11 cm above the table, clear of lying bottles
STAND_PRE = 0.05
LYING_PRE = 0.06                       # pregrasp above a lying body's axis (v2: 0.04): fingertips clear its top
SAFE_Z = (0.04, 0.08, 0.12)            # added to the highest obstacle top near the path
VIA_CORRIDOR = 0.15                    # obstacles within this of the straight xy path set the via height
Z_CAP = 1.12
DETOUR_DZ = (0.03, 0.07)              # detour heights above max(current, target) z (plus the current z)
DETOUR_X = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)
DETOUR_Y = (-0.35, -0.25, -0.15, -0.05, 0.05)
DETOUR_MAX_CHECK = 8                   # waypoints whose second leg is checked (IK call each)
HELD_R = 0.008
OBJ_MARGIN = 0.006
REFUSE_BLOCKED = True                  # practice L9: an unchecked fallback move knocked the coke off the table


def _load(key, rel, name):
    if key not in _M:
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _M[key] = m
    return _M[key]


def FK():
    return _load("fk", "fk.py", "fk_basket_r2")


def V():
    if "v2" not in _M:
        m = _load("v2", os.path.join("..", "put_into_bin_rgb_strict_v2", "skill_robodojo.py"), "pib_v2_ro_basket_r2")
        m._move2 = _move3                       # private instance only: every v2 move goes through the clearance check
        m.safe_carry = carry3
    return _M["v2"]


# ------------------------------------------------------------------------------------------------ obstacles
def _yaw_of_quat(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_obstacles(o, skip=()):
    """[('cyl', cx, cy, r, z0, z1, name) | ('box', lo, hi, name)] from the belief objects + placed + container walls."""
    obs = []
    for k, v in (o.get("objects") or {}).items():
        if k in skip or v.get("container") or not v.get("aabb"):
            continue
        lo, hi = np.asarray(v["aabb"][0], float), np.asarray(v["aabb"][1], float)
        raw = CTX["raws"].get(k, {})
        W = float(raw.get("wid", min(hi[0] - lo[0], hi[1] - lo[1])))
        if raw.get("kind") == "standing":
            c = (lo[:2] + hi[:2]) / 2
            obs.append(("cyl", float(c[0]), float(c[1]), W / 2 + OBJ_MARGIN, float(lo[2]), float(hi[2]), k))
            continue
        L = float(raw.get("len", max(hi[0] - lo[0], hi[1] - lo[1])))
        yaw = _yaw_of_quat(v.get("quat", [1, 0, 0, 0]))
        u = np.array([math.cos(yaw), math.sin(yaw)])
        c = np.asarray(v["pos"][:2], float)
        n = max(1, int(math.ceil(L / max(W, 0.02))))
        for s in np.linspace(-(L - W) / 2, (L - W) / 2, n) if L > W else [0.0]:
            p = c + s * u
            obs.append(("cyl", float(p[0]), float(p[1]), W / 2 + OBJ_MARGIN, float(lo[2]), float(hi[2]), k))
    for e in CTX["placed"]:
        if e.get("name") in skip:
            continue
        obs.append(("cyl", float(e["c"][0]), float(e["c"][1]), float(e.get("r", 0.04)) + OBJ_MARGIN,
                    float(e.get("z0", TABLE_Z)), float(e.get("z1", 0.85)), "placed:" + str(e.get("name"))))
    for c in CTX["containers"]:
        lo, hi = [float(v) for v in c["lo"]], [float(v) for v in c["hi"]]
        if hi[2] <= TABLE_Z + 0.005:
            continue
        t = 0.008
        for wlo, whi in (([lo[0], lo[1] - t, TABLE_Z], [hi[0], lo[1] + t, hi[2]]),
                         ([lo[0], hi[1] - t, TABLE_Z], [hi[0], hi[1] + t, hi[2]]),
                         ([lo[0] - t, lo[1], TABLE_Z], [lo[0] + t, hi[1], hi[2]]),
                         ([hi[0] - t, lo[1], TABLE_Z], [hi[0] + t, hi[1], hi[2]])):
            obs.append(("box", np.array(wlo), np.array(whi), "wall:" + c["name"]))
    return obs


def _hit(P, R, obstacles):
    for ob in obstacles:
        if ob[0] == "cyl":
            _, cx, cy, r, z0, z1, nm = ob
            m = (np.hypot(P[:, 0] - cx, P[:, 1] - cy) < r + R) & (P[:, 2] > z0 - R) & (P[:, 2] < z1 + R)
        else:
            _, lo, hi, nm = ob
            m = np.all((P > lo - R[:, None]) & (P < hi + R[:, None]), axis=1)
        if m.any():
            i = int(np.argmax(m))
            return nm, [round(float(v), 3) for v in P[i]]
    return None


def _on_table(P):
    tb = CTX["table"] or {"x": [-0.70, 0.66], "y": [-0.45, 0.40]}
    return (P[:, 0] > tb["x"][0]) & (P[:, 0] < tb["x"][1]) & (P[:, 1] > tb["y"][0]) & (P[:, 1] < tb["y"][1])


def held_points(cf):
    """27 points of the held box (flange frame) from its 8 corners."""
    if cf is None:
        return None
    lo, hi = cf.min(axis=0), cf.max(axis=0)
    # cf corners are a rotated box: use trilinear samples between the corner set's extreme pairs
    c = cf.mean(axis=0)
    pts = [c]
    for p in cf:
        pts.append(p)
        pts.append((p + c) / 2)
    return np.asarray(pts)


def path_clear(arm, q0, q1, obstacles, other_pts=None, held=None, table=True):
    """Sample the joint line q0 -> q1: arm links + gripper + held points vs obstacles, the other arm and the table."""
    F = FK()
    qs = F.joint_line(q0, q1, max_dq=0.05)
    fr0 = F.chain(arm, q0)
    held_floor = TABLE_Z + 0.003
    if held is not None:
        p60, R60 = F.flange(arm, q0)
        held_floor = min(held_floor, float((p60 + held @ R60.T)[:, 2].min()) - 0.002)   # a lift starts at the table
    nl = sum(max(2, int(np.linalg.norm(fr0[b][0] - fr0[a][0]) / 0.02) + 1) for a, b, _ in F.LINKS)   # link lengths are fixed
    for i, q in enumerate(qs):
        P, R = F.link_points(arm, q)
        h = _hit(P, R, obstacles)
        if h:
            return False, {"what": h[0], "at": h[1], "frac": round(i / max(1, len(qs) - 1), 2)}
        if table:
            L = P[:nl]
            m = (L[:, 2] - R[:nl] < TABLE_Z + 0.002) & _on_table(L)
            if m.any():
                return False, {"what": "table(link)", "at": [round(float(v), 3) for v in L[int(np.argmax(m))]],
                               "frac": round(i / max(1, len(qs) - 1), 2)}
        if other_pts is not None and len(other_pts[0]):
            OP, OR = other_pts
            d = np.linalg.norm(P[:, None, :] - OP[None, :, :], axis=2)
            if (d < R[:, None] + OR[None, :]).any():
                return False, {"what": "other_arm", "frac": round(i / max(1, len(qs) - 1), 2)}
        if held is not None:
            p6, R6 = F.flange(arm, q)
            HW = p6 + held @ R6.T
            HR = np.full(len(HW), HELD_R)
            h = _hit(HW, HR, obstacles)
            if h:
                return False, {"what": "held:" + h[0], "at": h[1], "frac": round(i / max(1, len(qs) - 1), 2)}
            m = (HW[:, 2] < held_floor) & _on_table(HW)
            if m.any():
                return False, {"what": "held:table", "frac": round(i / max(1, len(qs) - 1), 2)}
            if other_pts is not None and len(other_pts[0]):
                d = np.linalg.norm(HW[:, None, :] - other_pts[0][None, :, :], axis=2)
                if (d < HELD_R + other_pts[1][None, :]).any():
                    return False, {"what": "held:other_arm", "frac": round(i / max(1, len(qs) - 1), 2)}
    return True, None


# ------------------------------------------------------------------------------------------------ transit move
def _state(robot):
    o = robot.obs(cameras=[])
    return o


def _held_flange_pts(robot, arm):
    att = getattr(robot, "att", {}).get(arm) or {}
    if not att:
        return None, []
    pts = []
    for name in att:
        raw = CTX["raws"].get(name)
        cf = V().held_corners_flange(robot, arm, name, raw) if raw else None
        if cf is not None:
            pts.append(held_points(cf))
    return (np.vstack(pts) if pts else None), list(att)


def _reach_q(robot, arm, pos, quat, from_q):
    r = robot.reach(arm=arm, targets=[{"pos": [float(v) for v in pos], "quat": [float(v) for v in quat]}],
                    from_q=[float(v) for v in from_q], with_q=True)
    res = (r.get("results") or [{}])[0] if r.get("ok") else {}
    return res if res.get("ok") else None


def plan_legs(robot, arm, pos, quat, exclude=(), o=None):
    """Clear joint-line legs from the current joints to (pos, quat): [direct] or [rise, traverse, descend] through a
    safe height. Returns (legs [(pos, quat, q, steps_est)], info)."""
    o = o or _state(robot)
    e = o["eef"][arm]
    q0 = np.asarray(e["q"], float)
    other = "right" if arm == "left" else "left"
    held, held_names = _held_flange_pts(robot, arm)
    skip = set(exclude) | set(held_names) | {k for a in ("left", "right") for k in (getattr(robot, "att", {}).get(a) or {})}
    obstacles = build_obstacles(o, skip=skip)
    other_pts = FK().link_points(other, o["eef"][other]["q"])
    info = {"n_obst": len(obstacles)}
    # start-contact exemption: obstacles already touching the arm or the held object at the current joints are being
    # left, not entered (practice L9: a coke leaning on the held cylinder blocked the lift and every carry)
    touching = []
    for ob in obstacles:
        okc, _ = path_clear(arm, q0, q0, [ob], None, held, table=False)
        if not okc:
            touching.append(ob[-1])
    if touching:
        obstacles = [ob for ob in obstacles if ob[-1] not in touching]
        info["start_contact"] = sorted(set(touching))
    g = _reach_q(robot, arm, pos, quat, q0)
    if g is None:
        info["why"] = "ik"
        return None, info
    ok, why = path_clear(arm, q0, g["q"], obstacles, other_pts, held)
    if ok:
        info["mode"] = "direct"
        return [(pos, quat, g["q"], g["steps_est"])], info
    info["direct_block"] = why
    cur = np.asarray(e["pos"], float)
    cq = np.asarray(e["flange"]["quat"], float)
    # via height from the obstacles near the straight xy path only (practice L9: a far 0.99 m bottle forced z 1.12)
    a2, b2 = cur[:2], np.asarray(pos[:2], float)
    ab = b2 - a2
    L2 = float(ab @ ab)

    def _near(xy, r):
        t_ = 0.0 if L2 < 1e-9 else min(1.0, max(0.0, float((np.asarray(xy) - a2) @ ab) / L2))
        return float(np.linalg.norm(a2 + t_ * ab - np.asarray(xy))) < r + VIA_CORRIDOR
    tops = [ob[5] for ob in obstacles if ob[0] == "cyl" and _near(ob[1:3], ob[3])]
    tops += [ob[2][2] for ob in obstacles if ob[0] == "box" and _near((ob[1][:2] + ob[2][:2]) / 2, 0.15)]
    top = max(tops + [TABLE_Z])
    hang = 0.0
    if held is not None:
        p6, R6 = FK().flange(arm, q0)
        hang = max(0.0, float(np.asarray(e["pos"], float)[2] - (p6 + held @ R6.T)[:, 2].min()))   # held extent below the TCP
    tried = []
    tq = np.asarray(quat, float)
    T = np.asarray(pos, float)
    sp = np.asarray(V().SURVEY[arm], float)
    sq = np.asarray(V().tool_quat(0.0, 0.0, 0.0), float)
    for dz in SAFE_Z:
        zs = min(Z_CAP, max(top + dz + hang, cur[2], pos[2]))
        A = np.array([cur[0], cur[1], zs])
        B = np.array([pos[0], pos[1], zs])
        variants = [("v1", [(A, cq), (B, tq), (T, tq)]),           # rise, traverse in the target orientation, descend
                    ("v2", [(A, cq), (B, cq), (T, tq)]),           # rise, traverse as held, descend while rotating
                    ("v3", [(A, cq), (sp, sq), (B, tq), (T, tq)])]  # via the arm's survey pose
        for vn, seq in variants:
            legs, qprev, good, prevp = [], q0, True, cur
            for (pp, qq) in seq:
                if np.linalg.norm(pp - prevp) < 0.005 and abs(float(np.dot(qq, (legs[-1][1] if legs else cq)))) > 0.9999:
                    continue
                r = _reach_q(robot, arm, pp, qq, qprev)
                if r is None or r["joint_travel_rad"] > 3.5:
                    good = False
                    tried.append({"z": round(zs, 3), "v": vn, "why": "ik" if r is None else "travel %.1f" % r["joint_travel_rad"]})
                    break
                okl, whyl = path_clear(arm, qprev, r["q"], obstacles, other_pts, held)
                if not okl:
                    good = False
                    tried.append({"z": round(zs, 3), "v": vn, "why": (whyl or {}).get("what")})
                    break
                legs.append((pp, qq, r["q"], r["steps_est"]))
                qprev, prevp = r["q"], pp
            if good and legs:
                info.update(mode="via", via=vn, z=round(zs, 3), tried=tried)
                return legs, info
        if zs >= Z_CAP:
            break
    # detour (practice L9: carrying a held cylinder past a 0.99 m milk needed z 1.12, out of IK reach; a rise at the
    # current xy was out of reach too): one grid waypoint at the current height or a little above, in the current or
    # the target tool orientation, reached straight from the current joints, then the target; every leg checked
    zl = sorted({round(min(Z_CAP, max(cur[2], pos[2]) + dz), 3) for dz in DETOUR_DZ} | {round(float(cur[2]), 3)})
    for zs in zl:
        for qn, qw in (("cur", cq), ("tgt", tq)):
            W = [np.array([x, y, zs]) for x in DETOUR_X for y in DETOUR_Y
                 if np.linalg.norm(np.array([x, y]) - cur[:2]) > 0.06 and np.linalg.norm(np.array([x, y]) - T[:2]) > 0.06]
            W.sort(key=lambda w: float(np.linalg.norm(w - cur) + np.linalg.norm(T - w)))
            rr = robot.reach(arm=arm, targets=[{"pos": [float(v) for v in w], "quat": [float(v) for v in qw]} for w in W],
                             from_q=[float(v) for v in q0], with_q=True)
            res = (rr.get("results") or []) if rr.get("ok") else []
            n_ik, n_chk = 0, 0
            for w, r in zip(W, res):
                if not r.get("ok") or r.get("joint_travel_rad", 9) > 3.0:
                    continue
                n_ik += 1
                if not path_clear(arm, q0, r["q"], obstacles, other_pts, held)[0]:
                    continue
                n_chk += 1
                if n_chk > DETOUR_MAX_CHECK:
                    break
                rT = _reach_q(robot, arm, T, tq, r["q"])
                if rT is None or rT["joint_travel_rad"] > 3.5:
                    continue
                if not path_clear(arm, r["q"], rT["q"], obstacles, other_pts, held)[0]:
                    continue
                legs = [(w, qw, r["q"], r["steps_est"]), (T, tq, rT["q"], rT["steps_est"])]
                info.update(mode="via", via="v4" + qn, z=zs, wp=[round(float(v), 3) for v in w], tried=tried)
                return legs, info
            tried.append({"z": zs, "v": "v4" + qn, "why": f"ik {n_ik}/{len(W)}, leg1 clear {n_chk}"})
    info["tried"] = tried
    return None, info


def _move3(robot, arm, pos, quat, est, speed="normal", max_dev=None, tries=3, factor=1.6, reserve=None):
    """Drop-in replacement for v2's _move2 (same signature and reply): clearance-checked legs, joint line forced."""
    exclude = tuple(CTX.get("exclude") or ())
    legs, info = plan_legs(robot, arm, pos, quat, exclude=exclude)
    rec = {"arm": arm, "to": [round(float(v), 3) for v in pos], **{k: info.get(k) for k in ("mode", "via", "z", "wp", "direct_block", "why", "start_contact")}}
    CTX["transits"].append(rec)
    rec["tried"] = info.get("tried")
    if legs is None and REFUSE_BLOCKED and (info.get("direct_block") or {}).get("what") not in (None, "table(link)"):
        rec["refused"] = True
        return {"moved": False, "steps": 0, "plan_fail_reason": "no clear path", "hint": info.get("direct_block")}
    if legs is None:
        # no clear plan (IK or table-only block): the move as v2 did it (logged)
        rec["fallback"] = True
        kw = dict(arm=arm, pos=[float(v) for v in pos], quat=[float(v) for v in quat], speed=speed,
                  max_steps=int(est * factor + 10))
        if max_dev is not None:
            kw["max_dev"] = float(max_dev)
        if reserve is not None:
            kw["reserve"] = int(reserve)
        m = {}
        for _ in range(tries):
            m = robot.move_to(**kw)
            if m.get("moved") or m.get("plan_fail_reason") not in ("lease_too_short",):
                break
        return m
    m, tot = {}, 0
    for i, (pp, qq, q, se) in enumerate(legs):
        last = i == len(legs) - 1
        kw = dict(arm=arm, pos=[float(v) for v in pp], quat=[float(v) for v in qq],
                  speed=speed if last else "normal", max_steps=int(se * (factor if last else 1.6) + 10),
                  max_dev=1.0 if (info.get("mode") == "via" or max_dev is None or max_dev >= 0.03) else float(max_dev))
        if reserve is not None:
            kw["reserve"] = int(reserve)
        for _ in range(tries):
            m = robot.move_to(**kw)
            if m.get("moved") or m.get("plan_fail_reason") not in ("lease_too_short",):
                break
        tot += int(m.get("steps") or 0)
        if not m.get("moved"):
            rec["leg_fail"] = i
            break
    m = dict(m)
    m["steps"] = tot
    rec["steps"] = tot
    return m


def carry3(robot, arm, name, target, quat, est, o, log_t, reserve=None, factor=2.3, max_dev=0.03, z_cap=1.10):
    """Replacement for v2's safe_carry: the clearance-checked move (held object points included)."""
    n0 = len(CTX["transits"])
    m = _move3(robot, arm, target, quat, est, max_dev=max_dev, factor=factor, reserve=reserve)
    log_t["transit"] = CTX["transits"][n0:]
    return m


# ------------------------------------------------------------------------------------------------ picks
def side_cands(robot, arm, raw, walls, lift):
    """Side grasps of a tall standing object: horizontal tool from the arm's side, jaws horizontal."""
    Vm = V()
    x, y = float(raw["c"][0]), float(raw["c"][1])
    h = float(raw.get("h", float(raw["top_z"]) - TABLE_Z))
    top = float(raw["top_z"])
    W = float(raw.get("wid", 0.07))
    base = np.asarray(Vm.P().BASES[arm], float)
    a0 = np.array([x, y]) - base
    a0 = a0 / (np.linalg.norm(a0) + 1e-9)
    out = []
    # practice L9: a horizontal tool below TCP z ~0.88 makes the IK wrap (6+ rad); 0.90-0.94 gives 1-2 rad chains
    zs = sorted({round(min(top - 0.045, max(SIDE_ZMIN, TABLE_Z + f * h)), 3) for f in SIDE_FRACS})
    for d, gz in [(d, gz) for gz in zs for d in (0.0, -25.0, 25.0, -45.0, 45.0)]:
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        a = np.array([c * a0[0] - s * a0[1], s * a0[0] + c * a0[1]])
        psi = math.degrees(math.atan2(-a[0], a[1]))
        qq = Vm.tool_quat(psi, 90.0, 0.0)
        g = np.array([x, y, gz])
        pre = g - np.array([a[0], a[1], 0.0]) * 0.10
        out.append({"g": g, "pre": pre, "up": g + [0, 0, lift], "quat": qq, "yaw": psi, "d": d, "z": gz,
                    "width": min(Vm.OPEN_MAX, W + 0.03)})
    return out


def plan_pick3(robot, arm, raw, table, walls, lift=0.08, bad=(), edge=False, side=None):
    """v2's top-down candidates with a standing pregrasp 5 cm above the top; side grasps for tall standing objects."""
    Vm = V()
    Pm = Vm.P()
    best, diag = None, {"rejected": {"wall": 0, "ik": 0}, "n_cands": 0}
    standing = raw.get("kind") == "standing"
    h = float(raw.get("h", float(raw["top_z"]) - TABLE_Z))
    if side is None and standing and h >= SIDE_FIRST_H:
        side = "first"                          # practice L9: a neck grasp on the 1.0 m coke slipped, the side grasp held
    if side not in (True, "first"):
        top = float(raw["top_z"])
        old = Vm.plan_pick2
        best, diag = old(robot, arm, raw, table, walls, lift=lift, bad=bad, edge=edge,
                         pre_clear=(STAND_PRE + 0.045 if standing else LYING_PRE))
        if best is not None and standing:
            # v2 places the standing pregrasp at max(g + pre_clear, top + 0.012): with pre_clear = 0.095 it is 5 cm
            # above the top for a neck grasp 4.5 cm below the top; re-check that chain from the higher pregrasp
            pass
    if best is None and standing and h >= SIDE_MIN_H and side is not False:
        lift = max(lift, SIDE_LIFT)
        for c in side_cands(robot, arm, raw, walls, lift):
            if any(np.linalg.norm(c["g"][:2] - np.asarray(b[:2])) < 0.012 and abs(Vm.fold90(c["yaw"] - b[2])) < 20.0 for b in bad):
                continue
            a = Pm._reach(robot, arm, c["pre"], quat=c["quat"])
            if not a or a["joint_travel_rad"] > 4.5:
                diag["rejected"]["ik"] += 1
                continue
            b = Pm._reach(robot, arm, c["g"], from_q=a.get("q"), quat=c["quat"])
            if not b or b["joint_travel_rad"] > 1.0:
                diag["rejected"]["ik"] += 1
                continue
            u = Pm._reach(robot, arm, c["up"], from_q=b.get("q"), quat=c["quat"])
            if not u or u["joint_travel_rad"] > 1.2:
                diag["rejected"]["ik"] += 1
                continue
            cost = a["steps_est"] + b["steps_est"] + u["steps_est"] + abs(c["d"]) / 5.0
            if best is None or cost < best["cost"]:
                best = {"cost": cost, "g": c["g"], "yaw": c["yaw"], "pitch": 90.0, "roll": 0.0, "pre": c["pre"],
                        "up": c["up"], "quat": a["quat"], "est": (a["steps_est"], b["steps_est"], u["steps_est"]),
                        "width": c["width"], "note": f"side d={c['d']:+.0f} z={c['z']:.3f}", "side": True}
    if best is None and side == "first":
        return plan_pick3(robot, arm, raw, table, walls, lift=lift, bad=bad, edge=edge, side=False)
    return best, diag


def pick3(robot, arm, name, raw, table, walls, lift, max_picks, log, edge=False, reserve=None, side=None, stop_at_pre=False):
    """v2's pick2 flow (held_look after close and after lift) with plan_pick3 and clearance-checked moves; the target
    is excluded from the obstacles only for the final approach (descent / line-in) and the lift."""
    Vm = V()
    Pm = Vm.P()
    bad = []
    o = Pm._obs(robot)
    refusals, attempt = 0, -1
    while attempt + 1 < max_picks:
        attempt += 1
        p, diag = plan_pick3(robot, arm, raw, table, walls, lift=lift, bad=bad, edge=edge, side=side)
        t = {"op": "pick3", "arm": arm, "obj": name, "attempt": attempt, "step0": o.get("step"), **diag}
        log.append(t)
        if p is None:
            t["why"] = "nudge_first" if diag["rejected"]["wall"] else "no clean chain (top-down or side)"
            return False, o
        t.update(grasp=[round(float(v), 4) for v in p["g"]], yaw=round(p["yaw"], 1), pitch=p["pitch"], roll=p.get("roll"),
                 est=p["est"], width=round(p["width"], 3), note=p["note"], side=bool(p.get("side")),
                 pre_pos=[round(float(v), 4) for v in p["pre"]])
        gcmd = Vm.aperture_cmd(p["width"])[0]
        e = o["eef"][arm]
        if abs(float(e.get("grip_cmd", 1.0)) - gcmd) > 0.02:
            robot.gripper(arm=arm, action=gcmd, wait_steps=0)
        n0 = len(CTX["transits"])
        CTX["exclude"] = ()
        m = _move3(robot, arm, p["pre"], p["quat"], p["est"][0], reserve=reserve)
        t["pre"] = Vm._mv(m)
        t["transit_pre"] = CTX["transits"][n0:]
        if not m.get("moved") or (m.get("tracking_rad") or 0) > 0.3:
            bad.append([float(p["g"][0]), float(p["g"][1]), float(p["yaw"])])
            if m.get("plan_fail_reason") == "no clear path" and refusals < 4:
                refusals += 1
                attempt -= 1                      # a refused approach (0 steps) does not use up an attempt
                t["why"] = "approach refused: no clear path"
            o = Pm._obs(robot)
            continue
        if stop_at_pre:
            t["stopped_at_pre"] = True
            return None, Pm._obs(robot)
        CTX["exclude"] = (name,)
        m = _move3(robot, arm, p["g"], p["quat"], p["est"][1], speed="slow", max_dev=0.01, factor=2.3, reserve=reserve)
        t["descend"] = Vm._mv(m)
        saved = {k: v.get("fixed") for k, v in robot.objs.items()}
        for k, v in robot.objs.items():
            if k != name:
                v["fixed"] = True
        ax0, robot.attach_xy = robot.attach_xy, 0.30
        try:
            robot.gripper(arm=arm, action="close", wait_steps=3)
        finally:
            robot.attach_xy = ax0
            for k, v in robot.objs.items():
                if saved.get(k):
                    v["fixed"] = True
                else:
                    v.pop("fixed", None)
        lk = (robot.looks[-1] if getattr(robot, "looks", None) else {})
        t["look_close"] = {k: lk.get(k) for k in ("ask", "answer", "note")}
        if not robot.last_held.get(arm):
            bad.append([float(p["g"][0]), float(p["g"][1]), float(p["yaw"])])
            robot.gripper(arm=arm, action=1.0, wait_steps=0)
            if p.get("side"):
                robot.move_to(arm=arm, pos=[float(v) for v in p["pre"]], quat=[float(v) for v in p["quat"]], max_dev=0.01,
                              speed="slow")
            else:
                robot.move_delta(arm=arm, dpos=[0, 0, 0.05])
            t["why"] = "perceiver: not held after close"
            CTX["exclude"] = ()
            o = Pm._obs(robot)
            continue
        m = _move3(robot, arm, p["up"], p["quat"], p["est"][2], factor=2.3, reserve=reserve)
        if not m.get("moved"):
            m = robot.move_delta(arm=arm, dpos=[0, 0, lift])
        t["lift"] = Vm._mv(m)
        CTX["exclude"] = ()
        lk2 = robot.held_look(arm=arm, stage="put_into_bin_rgb_strict_v3:after_lift")
        t["look_lift"] = {k: lk2.get(k) for k in ("ask", "answer", "note")}
        o = Pm._obs(robot)
        held = bool(lk2.get("held") and float(o["eef"][arm].get("grip_cmd", 1.0)) < 0.5)
        t.update(held=held, step1=o.get("step"))
        if held:
            return True, o
        bad.append([float(p["g"][0]), float(p["g"][1]), float(p["yaw"])])
        robot.gripper(arm=arm, action="open", wait_steps=3)
        o = Pm._obs(robot)
        t["why"] = "not held after the lift: the object may have moved, re-locate before another attempt"
        break
    CTX["exclude"] = ()
    return False, o


# ------------------------------------------------------------------------------------------------ drops
def _side_held(robot, arm):
    o = robot.obs(cameras=[])
    R = V().qmat(o["eef"][arm]["flange"]["quat"])
    return abs(float(R[2, 0])) < 0.5


def _rotz(deg):
    h = math.radians(deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def drop_side(robot, arm, name, raw, cont, log, margin=0.02, clear=0.025, settle=3, reserve=None):
    """Release of a side-held object: the current tool orientation rotated about world z (0, +-20 .. +-90 deg); every
    held-box corner inside the container footprint shrunk by margin; lowest corner `clear` above the rim (and above
    the table while over it)."""
    Vm = V()
    Pm = Vm.P()
    o = Pm._obs(robot)
    e = o["eef"][arm]
    t = {"op": "drop_side", "arm": arm, "obj": name, "to": cont["name"], "step0": o.get("step")}
    log.append(t)
    cf = Vm.held_corners_flange(robot, arm, name, raw)
    if cf is None:
        t["why"] = "object not attached in the belief"
        return False, None
    lo, hi = np.asarray(cont["lo"], float), np.asarray(cont["hi"], float)
    ix0, ix1, iy0, iy1 = lo[0] + margin, hi[0] - margin, lo[1] + margin, hi[1] - margin
    rim = float(hi[2])
    q0 = np.asarray(e["flange"]["quat"], float)
    best = None
    for d in (0, 20, -20, 40, -40, 60, -60, 90, -90):
        qq = Vm._qmul(_rotz(d), q0)
        rel = Vm.corners_world(cf, [0.0, 0.0, 0.0], qq)
        pts = []
        for x in np.arange(ix0, ix1 + 1e-9, 0.03):
            for y in np.arange(iy0, iy1 + 1e-9, 0.03):
                c = rel[:, :2] + [x, y]
                if c[:, 0].min() >= ix0 and c[:, 0].max() <= ix1 and c[:, 1].min() >= iy0 and c[:, 1].max() <= iy1:
                    pts.append((float(x), float(y)))
        if not pts:
            continue
        zt = max(rim, TABLE_Z if hi[2] > TABLE_Z else rim) + clear - float(rel[:, 2].min())
        cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        base = np.asarray(Pm.BASES[arm], float)
        pts.sort(key=lambda p: math.hypot(p[0] - cx, p[1] - cy) + 0.5 * float(np.linalg.norm(np.asarray(p) - base)))
        tg = [{"pos": [p[0], p[1], zt], "quat": [float(v) for v in qq]} for p in pts[:10]]
        r = robot.reach(arm=arm, targets=tg, from_q=e.get("q"), with_q=True)
        for p, rr in zip(pts[:10], (r.get("results") or []) if r.get("ok") else []):
            if rr.get("ok") and rr["joint_travel_rad"] <= 3.5:
                cost = rr["steps_est"] + abs(d) / 10.0
                if best is None or cost < best[0]:
                    best = (cost, (p[0], p[1], zt), qq, rr["steps_est"], d)
                break
    if best is None:
        t["why"] = "no reachable side-held release pose"
        return False, None
    t.update(target=[round(v, 3) for v in best[1]], dyaw=best[4])
    n0 = len(CTX["transits"])
    m = _move3(robot, arm, best[1], best[2], best[3], max_dev=0.03, factor=2.3, reserve=reserve)
    t["carry"] = Vm._mv(m)
    t["transit"] = CTX["transits"][n0:]
    if not m.get("moved"):
        t["why"] = f"carry: {m.get('plan_fail_reason')}"
        return False, None
    o1 = Pm._obs(robot)
    cw = Vm.corners_world(cf, o1["eef"][arm]["pos"], o1["eef"][arm]["flange"]["quat"])
    t["release_inside"] = bool(cw[:, 0].min() >= lo[0] and cw[:, 0].max() <= hi[0] and cw[:, 1].min() >= lo[1]
                               and cw[:, 1].max() <= hi[1])
    robot.gripper(arm=arm, action="open", wait_steps=3)
    robot.hold(steps=settle)
    t["step1"] = Pm._obs(robot).get("step")
    return bool(t["release_inside"]), t


def drop_in3(robot, arm, name, raw, cont, log, avoid=None, **kw):
    Vm = V()
    if _side_held(robot, arm):
        return drop_side(robot, arm, name, raw, cont, log, reserve=kw.get("reserve"))
    holds = [e for e in CTX["placed"] if e.get("container") == cont["name"]]
    holds += [k for k, v in (robot.objs or {}).items() if v.get("in") == cont["name"]]
    deep0 = Vm.DEEP_MIN_H
    if holds:
        Vm.DEEP_MIN_H = 9.0                      # no deep release into a container that already holds objects
    try:
        res = Vm.drop_in(robot, arm, name, raw, cont, log, avoid=avoid, **kw)
        t = log[-1] if log else {}
        if not res[0] and t.get("why") == "no rim-safe reachable drop pose" and t.get("n_ik", 0) > 0:
            # practice L9 (classify): a tank picked at the far end of the left arm's reach left the wrist wound up, and
            # every IK-feasible drop pose failed v2's wound-wrist home check; a parallel gripper can release with the
            # tool turned 180 deg about its axis (same footprint, unwound wrist)
            orig = Vm.tool_yaw_of
            Vm.tool_yaw_of = lambda q: orig(q) + 180.0
            try:
                res = Vm.drop_in(robot, arm, name, raw, cont, log, avoid=avoid, **kw)
            finally:
                Vm.tool_yaw_of = orig
            if log:
                log[-1]["yaw_flip"] = True
        return res
    finally:
        Vm.DEEP_MIN_H = deep0


def relay_side(robot, giver, name, raw, others, o, log, receiver=None, settle_gap=0.003):
    """Table relay of a side-held (upright) object: v2's relay spots, the current tool orientation rotated about world
    z (0 .. +-120 deg), set-down height from the held box's lowest corner, receiver IK check at the spot (top-down neck or
    side), clearance-checked hover and lower, open, retreat 10 cm along the tool axis, park via the checked move."""
    Vm = V()
    Pm = Vm.P()
    e = o["eef"][giver]
    q_cur = np.asarray(e["flange"]["quat"], float)
    t = {"op": "relay_side", "arm": giver, "obj": name, "step0": o.get("step")}
    log.append(t)
    cf = Vm.held_corners_flange(robot, giver, name, raw)
    if cf is None:
        t["why"] = "object not attached in the belief"
        return False
    occupied = [np.asarray(o["objects"][k]["pos"][:2], float) for k in others if k in o["objects"]]
    h = float(raw.get("h", float(raw["top_z"]) - TABLE_Z))
    chosen, n_try = None, 0
    # practice L9: a released 23 cm coke toppled onto a bottle 12.5 cm away -> prefer spots with a wide clearance,
    # then fall back to v2's 12 cm (a wide-only rule found no spot at all in the next practice)
    spots = Vm.RELAY_SPOTS + RELAY_SPOTS_EXTRA
    cand = [(xy, cr) for cr in (0.12 + 0.5 * h, 0.12) for xy in spots]
    seen = set()
    for (x, y), clear_r in cand:
        if (x, y) in seen or any(np.linalg.norm(np.asarray((x, y)) - c) < clear_r for c in occupied):
            continue
        seen.add((x, y))
        if receiver:
            tg = [{"pos": [x, y, TABLE_Z + h - 0.045], "quat": [float(v) for v in Vm.tool_quat(yw, pp, 0.0)]}
                  for yw in (0.0, 45.0, 90.0, 135.0) for pp in (0.0, 15.0, 30.0)]
            rr = robot.reach(arm=receiver, targets=tg)
            if not (rr.get("ok") and any(r_.get("ok") for r_ in (rr.get("results") or []))):
                continue
        for d in (0, 30, -30, 60, -60, 90, -90, 120, -120):
            qq = Vm._qmul(_rotz(d), q_cur)
            rel = Vm.corners_world(cf, [0.0, 0.0, 0.0], qq)
            place_z = TABLE_Z + settle_gap - float(rel[:, 2].min())
            n_try += 1
            hov = Pm._reach(robot, giver, (x, y, place_z + 0.025), from_q=e.get("q"), quat=qq)
            if not hov or hov["joint_travel_rad"] > 3.5:
                continue
            dn = Pm._reach(robot, giver, (x, y, place_z), from_q=hov.get("q"), quat=qq)
            if not dn or dn["joint_travel_rad"] > 0.8:
                continue
            chosen = (x, y, place_z, qq, d, hov, dn)
            break
        if chosen:
            break
    t["n_try"] = n_try
    if not chosen:
        t["why"] = "no reachable side-held relay pose"
        return False
    t["clear_r"] = round(float(clear_r), 3)
    x, y, place_z, qq, d, hov, dn = chosen
    t.update(spot=[x, y, round(float(place_z), 4)], dyaw=d)
    n0 = len(CTX["transits"])
    m = _move3(robot, giver, (x, y, place_z + 0.025), qq, hov["steps_est"], max_dev=0.03, factor=2.3)
    t["hover"] = Vm._mv(m)
    t["transit"] = CTX["transits"][n0:]
    if not m.get("moved"):
        t["why"] = f"hover: {m.get('plan_fail_reason')}"
        return False
    m = _move3(robot, giver, (x, y, place_z), qq, dn["steps_est"], max_dev=0.01, factor=2.3)
    t["lower"] = Vm._mv(m)
    robot.gripper(arm=giver, action="open", wait_steps=6)     # practice L9: a short open wait dragged the coke over
    R = Vm.qmat(qq)
    back = np.array([x, y, place_z]) - 0.10 * R[:, 0]
    robot.move_to(arm=giver, pos=[float(v) for v in back], quat=[float(v) for v in qq], max_dev=0.01, speed="slow",
                  path="line")
    sx = 1.0 if giver == "right" else -1.0
    park = (x + sx * 0.22, y - 0.06, place_z + 0.05)
    m = _move3(robot, giver, park, Vm.tool_quat(0.0, 0.0, 0.0), 30)
    if not m.get("moved"):
        m = robot.return_home(arms=[giver], open=True)
    t["park_steps"] = m.get("steps")
    t["step1"] = Pm._obs(robot).get("step")
    return True


# ------------------------------------------------------------------------------------------------ home
def home_safe(robot, arms, mode, margin_m, margin_deg):
    """Joint line of each arm to its start joints checked; a blocked arm goes via its survey pose first."""
    Vm = V()
    o = _state(robot)
    via = {}
    for arm in arms:
        q0 = np.asarray(o["eef"][arm]["q"], float)
        obst = build_obstacles(o, skip=())
        other = "right" if arm == "left" else "left"
        ok, why = path_clear(arm, q0, np.zeros(6), obst, FK().link_points(other, o["eef"][other]["q"]), None)
        if not ok:
            sp = Vm.SURVEY[arm]
            m = _move3(robot, arm, sp, Vm.tool_quat(0.0, 0.0, 0.0), 30)
            via[arm] = {"blocked": why, "survey_steps": m.get("steps"), "moved": m.get("moved")}
            o = _state(robot)
    r = Vm.H().run(robot, mode=mode, margin_m=margin_m, margin_deg=margin_deg, arms=arms)
    r["via_survey"] = via
    return r


# ------------------------------------------------------------------------------------------------ run
def put_back(robot, arm, log):
    """A held object that could not be relayed: set it down where it was picked (the reverse of the vertical lift),
    open, retreat (side grasp: back to the pregrasp; top-down: up 6 cm), home. Practice L9: v2's 'open + home' dropped
    a side-held 23 cm coke from 11 cm up and it toppled."""
    pk = [t_ for t_ in log if t_.get("op") == "pick3" and t_.get("held")]
    e = robot.obs(cameras=[])["eef"][arm]
    q = [float(v) for v in e["flange"]["quat"]]
    t = {"op": "put_back", "arm": arm}
    log.append(t)
    if pk and pk[-1].get("grasp"):
        g = np.asarray(pk[-1]["grasp"], float) + np.array([0.0, 0.0, 0.004])
        m = robot.move_to(arm=arm, pos=[float(v) for v in g], quat=q, speed="slow", max_dev=0.01, path="line")
        t["lower"] = V()._mv(m)
    robot.gripper(arm=arm, action="open", wait_steps=3)
    e = robot.obs(cameras=[])["eef"][arm]
    if pk and pk[-1].get("side") and pk[-1].get("pre_pos"):
        back = [float(v) for v in pk[-1]["pre_pos"]]
    else:
        back = [float(e["pos"][0]), float(e["pos"][1]), float(e["pos"][2]) + 0.06]
    m = robot.move_to(arm=arm, pos=back, quat=q, speed="normal", max_dev=0.01, path="line")
    t["retreat"] = V()._mv(m)
    robot.return_home(arms=[arm], open=True)
    return True


def _placed_entry(obj, raw, c, t, o, arm):
    tg = (t or {}).get("target") or o["eef"][arm]["pos"]
    L, W = float(raw.get("len", 0.06)), float(raw.get("wid", 0.06))
    return {"name": obj, "container": c["name"], "c": [round(float(tg[0]), 3), round(float(tg[1]), 3)],
            "z0": float(c["lo"][2]), "z1": round(float(c["lo"][2]) + max(float(raw.get("h", 0.05)), W) + 0.01, 3),
            "r": round(max(L, W) / 2, 3)}


def run(robot, stage, objects=None, containers=None, obj=None, container=None, arm=None, groups=None, table=None,
        arms=None, final=False, lift=0.08, max_picks=2, attach_xy=0.05, margin=0.02, clear=0.025, d=0.06,
        mode="auto", margin_m=0.04, margin_deg=7.0, ignore_walls=None, edge=None, poses=None, avoid=None,
        placed=None, side=None):
    Vm = V()
    o = robot.obs(cameras=[])
    if o.get("objects") is not None:
        return {"ok": False, "verified": False, "error": "put_into_bin_rgb_strict_v3 is the strict tier-B path; the server is privileged"}
    Sm = Vm.S()
    tab = dict(Vm.DEFAULT_TABLE, **(table or {}))
    raws = list(objects or [])
    conts = list(containers or [])
    CTX.update(placed=list(placed or []), raws={b["name"]: b for b in raws}, containers=conts, table=tab, transits=[],
               exclude=(), target=None)
    if stage in ("plan", "survey", "check", "nudge"):
        r = Vm.run(robot, stage, objects=objects, containers=containers, obj=obj, container=container, arm=arm,
                   groups=groups, table=table, arms=arms, final=final, lift=lift, max_picks=max_picks,
                   attach_xy=attach_xy, margin=margin, clear=clear, d=d, mode=mode, margin_m=margin_m,
                   margin_deg=margin_deg, ignore_walls=ignore_walls, edge=edge, poses=poses, avoid=avoid)
        if stage == "plan":
            r = dict(r)
            r["v3_side"] = {}
            for s in r.get("steps") or []:
                if s.get("mode") == "skip":
                    raw = CTX["raws"][s["obj"]]
                    for a in ("left", "right"):
                        R = Sm.records(raws, conts)
                        br = Sm.make_drop_belief(robot, R, conts, attach_xy=attach_xy)
                        p, _ = plan_pick3(br, a, raw, tab, Vm.wall_boxes(conts), lift=lift, side=True)
                        if p is not None:
                            r["v3_side"].setdefault(s["obj"], []).append(a)
            r["tier"] = "B-strict"
        return r
    if stage == "home":
        R = Sm.records(raws, conts)
        br = Sm.make_drop_belief(robot, R, conts, attach_xy=attach_xy)
        return {**home_safe(br, arms or ["left", "right"], mode, margin_m, margin_deg), "tier": "B-strict",
                "transits": CTX["transits"]}
    byname = {b["name"]: b for b in raws}
    cmap = {c["name"]: c for c in conts}
    R = Sm.records(raws, conts)
    br = Sm.make_drop_belief(robot, R, conts, attach_xy=attach_xy)
    walls = Vm.wall_boxes(conts, skip=tuple(ignore_walls or ()))
    for e in CTX["placed"]:
        c = e["c"]
        r_ = float(e.get("r", 0.04))
        walls.append(("placed:" + str(e.get("name")), [c[0] - r_, c[1] - r_, float(e.get("z0", TABLE_Z))],
                      [c[0] + r_, c[1] + r_, float(e.get("z1", 0.85))]))
    log = []
    out = {}
    try:
        if stage == "pre":
            raw = byname[obj]
            _, o = pick3(br, arm, obj, raw, tab, walls, lift, 1, log, side=side, stop_at_pre=True)
            e = br.obs(cameras=[])
            return {"ok": bool(log and log[-1].get("stopped_at_pre")), "verified": True, "tier": "B-strict", "log": log,
                    "transits": CTX["transits"], "step": e.get("step"), "steps_left": e.get("steps_left")}
        if stage == "drop_held":
            # the named object is already in the hand (e.g. after a refused carry): attach it in the belief, then drop
            raw = byname[obj]
            o = br.obs(cameras=[])
            saved = {k: v.get("fixed") for k, v in br.objs.items()}
            for k, v in br.objs.items():
                if k != obj:
                    v["fixed"] = True
            ax0, br.attach_xy = br.attach_xy, 0.30
            br._attach(arm, o["eef"])
            br.attach_xy = ax0
            for k, v in br.objs.items():
                if saved.get(k):
                    v["fixed"] = True
                else:
                    v.pop("fixed", None)
            c = cmap[container or conts[0]["name"]]
            inside, t = drop_in3(br, arm, obj, raw, c, log, margin=margin, clear=clear, avoid=avoid, reserve=None)
            out = {"held": True, "in_container": inside}
            if inside:
                out["placed_entry"] = _placed_entry(obj, raw, c, t, o, arm)
            okk = bool(inside)
        elif stage in ("direct", "relay"):
            raw = byname[obj]
            is_edge = bool(edge) if edge is not None else (not Vm.on_table(raw["c"], tab, 0.0) or Vm._crosses_edge(raw, tab))
            p0, _ = plan_pick3(br, arm, raw, tab, walls, lift=lift, edge=is_edge, side=side)
            rsv = Vm.FINAL_RESERVE if final else None
            if p0 is not None:
                okb, g = Vm._budget(br, sum(p0["est"]) + (25 if stage == "direct" else 45), final, log)
                if not okb:
                    e = br.obs(cameras=[])
                    return {"ok": False, "verified": False, "tier": "B-strict", "why": "budget", "budget": g, "log": log,
                            "step": e.get("step"), "steps_left": e.get("steps_left")}
            held, o = pick3(br, arm, obj, raw, tab, walls, lift, max_picks, log, edge=is_edge, reserve=rsv, side=side)
            out["held"] = held
            if not held and log and log[-1].get("why") == "nudge_first":
                out["why"] = "nudge_first"
            if held and stage == "direct":
                c = cmap[container or conts[0]["name"]]
                inside, t = drop_in3(br, arm, obj, raw, c, log, margin=margin, clear=clear, avoid=avoid,
                                     steps_left=o.get("steps_left"), home_after=bool(final), reserve=rsv)
                out["in_container"] = inside
                if t is None and not inside:
                    out["why"] = log[-1].get("why")
                if inside:
                    out["placed_entry"] = _placed_entry(obj, raw, c, t, o, arm)
            elif held and stage == "relay":
                others = [k for k in byname if k != obj]
                recv = "left" if arm == "right" else "right"
                if _side_held(br, arm):
                    out["relayed"] = relay_side(br, arm, obj, raw, others, o, log, receiver=recv)
                else:
                    out["relayed"] = Vm.relay2(br, arm, obj, raw, others, o, log, receiver=recv)
                if not out["relayed"]:
                    out["put_back"] = put_back(br, arm, log)
                spot = [t_.get("spot") for t_ in log if t_.get("op") in ("relay2", "relay_side")]
                out["spot"] = spot[-1] if spot else None
            okk = bool(out.get("held")) and bool(out.get("in_container") if stage == "direct" else out.get("relayed"))
        else:
            return {"ok": False, "verified": False, "error": f"unknown stage {stage!r} (plan|pre|direct|relay|drop_held|nudge|survey|home|check)"}
    except RuntimeError as ex:
        return {"ok": False, "verified": False, "error": str(ex), "log": log, "transits": CTX["transits"]}
    e = br.obs(cameras=[])
    fc, _ = Vm.H().full_cost(e["eef"])
    out.update(log=log, step=e.get("step"), steps_left=e.get("steps_left"), ended=e.get("ended"), home_cost_now=fc,
               transits=CTX["transits"])
    return Sm.bel().scrub({"ok": okk, "verified": okk, "tier": "B-strict", "belief_events": br.events,
                           "looks": br.looks, **out})
