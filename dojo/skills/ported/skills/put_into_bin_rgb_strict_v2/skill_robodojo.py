# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/put_into_bin_rgb_strict_v2/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""put_into_bin_rgb_strict_v2 (RoboDojo, STRICT tier B): stage-wise put-into-container / sort-into-containers with the
five fixes of the basket round r1 (Claude, 25 Oct 2026). Developed on seed 0 layout 9 only.

What it reads: images are the planner's; this skill reads proprioception (joints, flange, commanded gripper), free IK
(`reach`, 0 env steps), 0-step move dry runs and the perceiver's held_look answers. No object poses, no finger joints.
It refuses a privileged server.

Frozen code it loads read-only (unchanged files):
  skills/put_into_bin_rgb_strict  records(), make_drop_belief() (BeliefRobotStrict + gravity on release), tilt_quat(),
                                  pib() = put_into_bin v3 with the strict held patch (its _reach/_move/_relay/_box/_obs);
  skills/return_home_budget       (new in this round) home costs and the budget-aware finish.

Changes against put_into_bin_rgb_strict (each targets a diagnosed miss of the next3 run):
  1. RIM-AWARE DROP (drop_in; classify L0/L3/L4 pens perched on rims or fell off the far side): the held object's 8 box
     corners are carried in the flange frame (belief attachment fixed at the close); a drop pose (TCP, tool yaw, tilt
     about world x) is accepted only if every corner's xy lies inside the container's rim footprint shrunk by `margin`
     and the lowest corner clears the rim by `clear`. Tool yaws tried: the current one and the one that turns the
     object's long axis along the container's long side. Chained free IK from the current joints, travel <= 3.5 rad.
  2. EDGE-AWARE PICK (put_bottles L2: a bottle across the table edge fell when the relay pick closed beside it): grasp
     points along a lying body's axis are restricted to the part over the table (>= edge_margin inside the table
     edge), and the pregrasp sits `pre_clear_edge` above the top so the descent is a slow vertical line.
  3. WALL-AWARE PICK + NUDGE (classify L2 pen against the basket fronts, 4 attempts; L1/L4 fingers on a basket wall):
     each candidate grasp is screened in 3D against every container's walls with sample points of the open fingers
     (aperture narrowed to the object width + 2.2 cm, never below 3.4 cm so moves stay at full speed) and of the
     gripper housing; orientations include rolls about the object's axis that lean the tool away from a wall. A
     candidate that would touch a wall below the rim is dropped. If none is left the stage returns
     "nudge_first"; stage `nudge` pushes the object away from the wall with closed fingers tilted 35 deg toward it.
  4. SEARCH MOVE (classify L3: a pen under the home-pose gripper in the head view was never located): stage `survey`
     moves arms to a top-down pose high over their own side (dry-run first); the planner re-looks with the head camera.
  5. BUDGET-AWARE FINISH (put_bottles L4: all 4 in, return_home refused at 24 steps left): every stage reports the
     steps it needs (lower bound) and the home cost after it; `final=True` refuses a stage at 0 steps when the stage
     plus the cheapest home cannot fit (nothing to gain then); drop poses are ranked with the home cost when the budget
     is tight; stage `home` = return_home_budget (exact home if it fits, else the cheapest within-tolerance moves).

Stages (skill name=put_into_bin_rgb_strict_v2 args={"stage": .., "objects": [...], "containers": [...], ...}):
  plan     0 steps: per object the arms with a clean pick chain (or "nudge_first"), per container the arms with a
           rim-aware drop, the group -> container map with the fewest relays, per object direct / relay / nudge.
  direct   arm, obj, container[, final]: pick (held_look after close and after lift) + rim-aware drop.
  relay    arm (giver), obj: pick + put_into_bin's table relay (the planner re-locates, then `direct` by the receiver).
  nudge    arm, obj, container [, d]: push the object d (0.06 m) away from that container's nearest wall.
  survey   arms: move to the survey pose(s), max 45 planned steps each.
  home     return_home_budget (args mode, margin_m, margin_deg).
  check    0 steps: home costs and steps left.
Records as put_into_bin_rgb: object {name, kind standing|lying, c [x,y], top_z, h, len, wid, yaw (long axis, deg),
optional grasp [x,y], grasp_yaw, cat}; container {name, lo [x0,y0,z_floor], hi [x1,y1,z_rim]}.
Optional `table` {"x": [x_left_edge, x_right_edge], "y": [y_near, y_far]} (fixed scene, measured on practice L9).
"""
import importlib.util
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_Z = 0.765
GRASP_OFF = 0.1501
OPEN_MAX = 0.088                  # jaw opening at gripper 1.0 (shim: 2 x clip(-0.01 + g * 0.054, 0, 0.044))
APERTURE_MIN = 0.034              # gripper 0.5: below that the shim treats the arm as holding (0.05 rad/step)
FINGER = {"th": 0.012, "hk": 0.014, "z_lo": -0.012, "z_hi": 0.066}     # one finger plate, TCP frame [believed]
WOUND_HOME = 40                   # drop poses whose straight joint line home needs > 40 steps are not used
DEEP_MIN_H = 0.05                 # held box taller than this at the release pose -> deep release (see drop_in)
ROLL_MAX_WID = 0.025              # practice L9: a rolled pinch slipped off a 4 cm boxy toy tank; rolls only for pens
WALL_PAD = 0.008                  # wall boxes grown by this in xy (practice L9: a finger caught a basket rim the model cleared)
HOUSING = {"hj": 0.055, "hk": 0.036, "z_lo": 0.058}                     # gripper housing above the fingers [believed]
PITCHES = (0.0, 20.0, 35.0, 45.0)
ORIENTS2 = [(0.0, 0.0, 0.0), (180.0, 0.0, 0.0), (0.0, 15.0, 0.0), (180.0, 15.0, 0.0), (0.0, 30.0, 0.0),
            (0.0, 0.0, 20.0), (0.0, 0.0, -20.0), (0.0, 0.0, 35.0), (0.0, 0.0, -35.0)]      # (yaw offset, pitch, roll)
SURVEY = {"left": [-0.33, -0.30, 1.00], "right": [0.33, -0.30, 1.00]}
DEFAULT_TABLE = {"x": [-0.60, 0.555], "y": [-0.45, 0.40]}
FINAL_RESERVE = 12                # smallest near-home move expected after a final drop (steps); see return_home_budget
_M = {}


def _load(key, rel):
    if key not in _M:
        spec = importlib.util.spec_from_file_location(f"{key}_ro_basket_r1", os.path.join(HERE, rel))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _M[key] = m
    return _M[key]


def S():
    return _load("pib_rgb_strict", os.path.join("..", "put_into_bin_rgb_strict", "skill_robodojo.py"))


def P():
    return S().pib()


def H():
    return _load("home_budget", os.path.join("..", "return_home_budget", "skill_robodojo.py"))


# ------------------------------------------------------------------------------------------------ geometry helpers
def qmat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tool_yaw_of(q):
    """Tool yaw (deg) of a near-top-down flange quat: jaws close along world x at yaw 0; the jaw axis is flange y,
    which is (-cos yaw, -sin yaw, 0) for Rz(yaw) * TOP_DOWN."""
    j = qmat(q)[:, 1]
    return math.degrees(math.atan2(-j[1], -j[0]))


def fold90(a):
    return (a + 90.0) % 180.0 - 90.0


def _rect_corners(c, j, hj, hk):
    j = np.asarray(j, float)[:2]
    j = j / (np.linalg.norm(j) + 1e-12)
    k = np.array([-j[1], j[0]])
    c = np.asarray(c, float)[:2]
    return np.array([c + sj * hj * j + sk * hk * k for sj in (-1, 1) for sk in (-1, 1)])


def _rect_box_overlap(corners, lo, hi):
    """Oriented rectangle (4 xy corners) vs axis-aligned box [lo, hi] (xy): separating-axis test."""
    if corners[:, 0].max() < lo[0] or corners[:, 0].min() > hi[0] or corners[:, 1].max() < lo[1] or corners[:, 1].min() > hi[1]:
        return False
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    for i in range(2):
        e = corners[(i + 1) % 4 if i == 0 else 2] - corners[0]
        n = np.array([-e[1], e[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        a, b = corners @ n, box @ n
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def wall_boxes(containers, skip=()):
    """Each container's walls as one solid box (rim footprint, top = rim); skip = names to ignore."""
    out = []
    for c in containers or []:
        if c["name"] in skip:
            continue
        lo, hi = [float(v) for v in c["lo"]], [float(v) for v in c["hi"]]
        if hi[2] <= TABLE_Z + 0.005:
            continue                                   # a bin below the table top cannot touch a table pick
        lo = [lo[0] - WALL_PAD, lo[1] - WALL_PAD, lo[2]]
        hi = [hi[0] + WALL_PAD, hi[1] + WALL_PAD, hi[2]]
        out.append((c["name"], lo, hi))
    return out


TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])


def _qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def _axq(axis, deg):
    q = np.zeros(4)
    q[0] = math.cos(math.radians(deg) / 2)
    q[axis + 1] = math.sin(math.radians(deg) / 2)
    return q


def tool_quat(yaw=0.0, pitch=0.0, roll=0.0):
    """The server's convention (harness/robodojo_server.tool_quat): Rz(yaw) Ry(roll) Rx(pitch) TOP_DOWN. pitch tilts
    about the jaw axis; roll tilts about the axis across the jaws (a lying object's long axis for a cross grasp)."""
    r = _axq(0, pitch)
    r = _qmul(_axq(1, roll), r)
    r = _qmul(_axq(2, yaw), r)
    return _qmul(r, TOP_DOWN)


def _box_pts(a0, a1, j0, j1, k0, k1, n=(4, 3, 3)):
    return np.array([[a, j, k] for a in np.linspace(a0, a1, n[0]) for j in np.linspace(j0, j1, n[1])
                     for k in np.linspace(k0, k1, n[2])])


def gripper_pts(width):
    """Sample points of the open finger plates and the gripper housing in the tool frame (a = toward the fingertips
    along flange x, j = jaw axis = flange y, k = flange z), TCP at the origin [believed geometry: URDF finger mounts
    0.0866 m from the flange, TCP 0.1501 m, fingers 0..0.044 m per side]."""
    h = width / 2
    f = FINGER
    fp = _box_pts(-f["z_hi"], -f["z_lo"], h, h + f["th"], -f["hk"], f["hk"])
    fm = _box_pts(-f["z_hi"], -f["z_lo"], -h - f["th"], -h, -f["hk"], f["hk"])
    hs = _box_pts(-0.15, -HOUSING["z_lo"], -HOUSING["hj"], HOUSING["hj"], -HOUSING["hk"], HOUSING["hk"], n=(4, 5, 3))
    return {"finger+": fp, "finger-": fm, "housing": hs}


def gripper_clear(gp, quat, width, walls):
    """3D screen of the open gripper at TCP gp with flange quat `quat` against each container's walls (solid rim
    footprint below rim + 4 mm). Returns (ok, what hit)."""
    if not walls:
        return True, None
    R = qmat(quat)
    g = np.asarray(gp, float)
    for nm, pts in gripper_pts(width).items():
        W_ = g + pts[:, 0:1] * R[:, 0] + pts[:, 1:2] * R[:, 1] + pts[:, 2:3] * R[:, 2]
        for cn, lo, hi in walls:
            inside = ((W_[:, 0] >= lo[0]) & (W_[:, 0] <= hi[0]) & (W_[:, 1] >= lo[1]) & (W_[:, 1] <= hi[1])
                      & (W_[:, 2] < hi[2] + 0.004))
            if inside.any():
                return False, f"{nm}:{cn}"
    return True, None


def aperture_cmd(width):
    w = min(OPEN_MAX, max(APERTURE_MIN, float(width)))
    return min(1.0, (w / 2 + 0.01) / 0.054), w


def on_table(p, table, margin):
    return (table["x"][0] + margin <= p[0] <= table["x"][1] - margin) and (table["y"][0] + margin <= p[1] <= table["y"][1] - margin)


# ------------------------------------------------------------------------------------------------ pick
def grasp_cands(raw, table, walls, neck_depth=0.045, edge_margin=0.035):
    """Candidate grasps {pos, yaw, width, pref, note} for one located object (no IK yet)."""
    x, y = float(raw["c"][0]), float(raw["c"][1])
    top = float(raw["top_z"])
    h = float(raw.get("h", top - TABLE_Z))
    L, W = float(raw.get("len", raw.get("wid", 0.06))), float(raw.get("wid", 0.06))
    out = []
    if raw.get("kind") == "standing":
        gz = max(top - neck_depth, TABLE_Z + 0.01)
        width = min(OPEN_MAX, W + 0.03)
        for yw in (0.0, 45.0, 90.0, 135.0):
            out.append({"pos": [x, y, gz], "yaw": yw, "width": width, "pref": 0.0, "note": "neck", "standing": True})
        return out
    yaw = float(raw.get("yaw", 0.0))
    u = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    gz = max(top - h / 2, TABLE_Z + 0.01)
    gyaw = fold90(yaw + 90.0)
    width = min(OPEN_MAX, W + 0.022)
    if raw.get("grasp") is not None:
        gx, gy = float(raw["grasp"][0]), float(raw["grasp"][1])
        g0 = np.array([gx, gy])
        if raw.get("grasp_yaw") is not None:
            gyaw = fold90(float(raw["grasp_yaw"]))
        s0 = float((g0 - np.array([x, y])) @ u)
    else:
        s0 = 0.0
    half = max(0.0, L / 2 - 0.02)
    ss = sorted(set([round(s0, 3)] + [round(v, 3) for v in np.arange(-half, half + 1e-9, 0.01)]), key=lambda s: abs(s - s0))
    for s in ss:
        p = np.array([x, y]) + s * u
        if raw.get("grasp") is not None and abs(s - s0) < 1e-6:
            p = g0
        if not on_table(p, table, edge_margin):
            continue
        out.append({"pos": [float(p[0]), float(p[1]), gz], "yaw": gyaw, "width": width,
                    "pref": 300.0 * abs(s - s0), "note": f"axis s={s:+.3f}"})
    if L <= 0.085 and W >= 0.025:
        # short, wide bodies (watch, car, camera): also the cross grasp along the long axis (practice L9: a watch face
        # slipped and every retry point was within 1.2 cm of the blacklisted one)
        for c in list(out):
            out.append(dict(c, yaw=fold90(c["yaw"] + 90.0), width=min(OPEN_MAX, L + 0.022), pref=c["pref"] + 8.0,
                            note=c["note"] + " long"))
    return out


def plan_pick2(robot, arm, raw, table, walls, lift=0.08, bad=(), edge=False, pre_clear=0.04, max_cands=6):
    """Chained free IK (pregrasp <= 4.5 rad, grasp <= 1.0 rad from it, lift <= 1.2 rad) over wall-clear candidates."""
    Pm = P()
    top = float(raw["top_z"])
    cands = grasp_cands(raw, table, walls)
    rej = {"wall": 0, "ik": 0}
    best, n_try = None, 0
    for g in cands:
        gp = np.asarray(g["pos"], float)
        if any(np.linalg.norm(gp[:2] - np.asarray(b[:2])) < 0.012 and abs(fold90(g["yaw"] - b[2])) < 20.0 for b in bad):
            continue
        aw = aperture_cmd(g["width"])[1]
        reached_ik = False
        for dyaw, pitch, roll in ORIENTS2:
            if roll and (g.get("standing") or float(raw.get("wid", 0.06)) > ROLL_MAX_WID):
                continue                          # rolled (diagonal) pinches only for thin round bodies (pens)
            yw = g["yaw"] + dyaw
            qq = tool_quat(yw, pitch, roll)
            ok, hit = gripper_clear(gp, qq, aw, walls)
            if not ok:
                rej["wall"] += 1
                continue
            pre = gp.copy()
            pre[2] = max(gp[2] + pre_clear, top + (0.06 if edge else 0.012))
            reached_ik = True
            a = Pm._reach(robot, arm, pre, quat=qq)
            if not a or a["joint_travel_rad"] > 4.5:
                rej["ik"] += 1
                continue
            b = Pm._reach(robot, arm, gp, from_q=a.get("q"), quat=qq)
            if not b or b["joint_travel_rad"] > 1.0:
                rej["ik"] += 1
                continue
            up = gp + [0, 0, lift]
            c = Pm._reach(robot, arm, up, from_q=b.get("q"), quat=qq)
            if not c or c["joint_travel_rad"] > 1.2:
                rej["ik"] += 1
                continue
            cost = a["steps_est"] + b["steps_est"] + c["steps_est"] + pitch / 3.0 + abs(roll) / 3.0 + g["pref"]
            if best is None or cost < best["cost"]:
                best = {"cost": cost, "g": gp, "yaw": yw, "pitch": pitch, "roll": roll, "pre": pre, "up": up, "quat": a["quat"],
                        "est": (a["steps_est"], b["steps_est"], c["steps_est"]), "width": aw, "note": g["note"],
                        "q_up": c.get("q")}
            break
        n_try += 1 if reached_ik else 0          # wall-rejected candidates cost no IK and do not count
        if best is not None and n_try >= 3:
            break
        if n_try >= max_cands:
            break
    return best, {"n_cands": len(cands), "rejected": rej}


def pick2(robot, arm, name, raw, table, walls, lift, max_picks, log, edge=False, reserve=None):
    """Top-down pick with the wall/edge screen and a narrowed aperture. Held = perceiver yes after the close (belief
    attach) and after the lift, and the gripper commanded closed. Returns (held, obs)."""
    Pm = P()
    bad = []
    o = Pm._obs(robot)
    for attempt in range(max_picks):
        p, diag = plan_pick2(robot, arm, raw, table, walls, lift=lift, bad=bad, edge=edge)
        t = {"op": "pick2", "arm": arm, "obj": name, "attempt": attempt, "step0": o.get("step"), **diag}
        log.append(t)
        if p is None:
            t["why"] = "nudge_first" if diag["rejected"]["wall"] else "no clean top-down chain"
            return False, o
        t.update(grasp=[round(float(v), 4) for v in p["g"]], yaw=round(p["yaw"], 1), pitch=p["pitch"], roll=p["roll"], est=p["est"],
                 width=round(p["width"], 3), note=p["note"])
        gcmd = aperture_cmd(p["width"])[0]
        e = o["eef"][arm]
        if abs(float(e.get("grip_cmd", 1.0)) - gcmd) > 0.02:
            robot.gripper(arm=arm, action=gcmd, wait_steps=0)
        m = _move2(robot, arm, p["pre"], p["quat"], p["est"][0], reserve=reserve)
        t["pre"] = _mv(m)
        if not m.get("moved") or (m.get("tracking_rad") or 0) > 0.3:
            bad.append([float(p["g"][0]), float(p["g"][1]), float(p["yaw"])])
            if m.get("moved"):
                robot.return_home(arms=[arm], open=True)
            o = Pm._obs(robot)
            continue
        m = _move2(robot, arm, p["g"], p["quat"], p["est"][1], speed="slow", max_dev=0.01, factor=2.3, reserve=reserve)
        t["descend"] = _mv(m)
        # belief attach: only the named object is a candidate (others marked fixed for this close), any distance
        saved = {k: v.get("fixed") for k, v in robot.objs.items()}
        for k, v in robot.objs.items():
            if k != name:
                v["fixed"] = True
        ax0, robot.attach_xy = robot.attach_xy, 0.30
        try:
            robot.gripper(arm=arm, action="close", wait_steps=3)      # belief: held_look after_close, attach on yes
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
            robot.move_delta(arm=arm, dpos=[0, 0, 0.05])
            t["why"] = "perceiver: not held after close"
            o = Pm._obs(robot)
            continue
        m = _move2(robot, arm, p["up"], p["quat"], p["est"][2], factor=2.3, reserve=reserve)
        if not m.get("moved"):
            m = robot.move_delta(arm=arm, dpos=[0, 0, lift])
        t["lift"] = _mv(m)
        lk2 = robot.held_look(arm=arm, stage="put_into_bin_rgb_strict_v2:after_lift")
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
        break                                     # practice L9: a blind retry at the stale location closed on air
    return False, o


def _mv(m):
    return {k: m.get(k) for k in ("steps", "residual_m", "tracking_rad", "plan_fail_reason")}


def _move2(robot, arm, pos, quat, est, speed="normal", max_dev=None, tries=3, factor=1.6, reserve=None):
    """put_into_bin's leased move, plus an explicit per-move reserve (None = the server's start reserve)."""
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


# ------------------------------------------------------------------------------------------------ rim-aware drop
def held_corners_flange(robot, arm, name, raw):
    """The held object's 8 box corners in the flange frame (from the belief attachment made at the close)."""
    att = robot.att[arm].get(name)
    if att is None:
        return None
    r, Ro = att
    if raw.get("kind") == "standing":
        L = W = float(raw.get("wid", 0.06))
    else:
        L, W = float(raw.get("len", raw.get("wid", 0.06))), float(raw.get("wid", 0.06))
    H_ = float(raw.get("h", float(raw["top_z"]) - TABLE_Z))
    return np.array([np.asarray(r, float) + np.asarray(Ro, float) @ np.array([sx * L / 2, sy * W / 2, sz * H_ / 2])
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def corners_world(cf, tcp, quat):
    R = qmat(quat)
    fl = np.asarray(tcp, float) - GRASP_OFF * R[:, 0]
    return fl + cf @ R.T


def drop_in(robot, arm, name, raw, cont, log, margin=0.02, clear=0.025, settle=3, steps_left=None, home_after=False,
            reserve=None, avoid=None):
    """Rim-aware release: whole held box inside the rim footprint shrunk by margin, lowest corner >= rim + clear."""
    Pm = P()
    Sm = S()
    o = Pm._obs(robot)
    e = o["eef"][arm]
    t = {"op": "drop_in", "arm": arm, "obj": name, "to": cont["name"], "step0": o.get("step")}
    log.append(t)
    cf = held_corners_flange(robot, arm, name, raw)
    if cf is None:
        t["why"] = "object not attached in the belief"
        return False, None
    lo, hi = np.asarray(cont["lo"], float), np.asarray(cont["hi"], float)
    ix0, ix1, iy0, iy1 = lo[0] + margin, hi[0] - margin, lo[1] + margin, hi[1] - margin
    rim = float(max(hi[2], TABLE_Z))
    q0 = e["flange"]["quat"]
    y0 = tool_yaw_of(q0)
    # object long axis yaw now (world), and the tool yaw that lays it along the container's long side
    R0 = qmat(q0)
    Ro = robot.att[arm][name][1]
    ax = (R0 @ np.asarray(Ro, float))[:, 0]
    th0 = math.degrees(math.atan2(ax[1], ax[0]))
    long_x = (hi[0] - lo[0]) >= (hi[1] - lo[1])
    want = 0.0 if long_x else 90.0
    d = fold90(want - th0)
    yaws = [y0] if abs(d) < 5.0 else [y0 + d, y0]
    cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
    xs = np.arange(ix0, ix1 + 1e-9, 0.02) if ix1 > ix0 else np.array([cx])
    ys = np.arange(iy0, iy1 + 1e-9, 0.02) if iy1 > iy0 else np.array([cy])
    base = np.asarray(Pm.BASES[arm], float)
    best, n_geom, n_ik = None, 0, 0
    for yw in yaws:
        for pitch in PITCHES:
            qq = np.asarray(Sm.tilt_quat(yw, pitch), float)
            rel = corners_world(cf, [0.0, 0.0, 0.0], qq)          # corners relative to the TCP
            zt = rim + clear - float(rel[:, 2].min())
            pts = []
            for x in xs:
                for y in ys:
                    c = rel[:, :2] + [x, y]
                    if c[:, 0].min() >= ix0 and c[:, 0].max() <= ix1 and c[:, 1].min() >= iy0 and c[:, 1].max() <= iy1:
                        pts.append((float(x), float(y)))
            n_geom += len(pts)
            if not pts:
                continue
            def _avoid_pen(p):
                if not avoid:
                    return 0.0
                return 3.0 * max(0.0, 0.09 - min(math.hypot(p[0] - float(a[0]), p[1] - float(a[1])) for a in avoid))
            pts.sort(key=lambda p: math.hypot(p[0] - cx, p[1] - cy) + 0.3 * float(np.linalg.norm(np.asarray(p) - base))
                     + _avoid_pen(p))
            tg = [{"pos": [p[0], p[1], zt], "quat": [float(v) for v in qq]} for p in pts[:10]]
            r = robot.reach(arm=arm, targets=tg, from_q=e.get("q"), with_q=True)
            res = (r.get("results") or []) if r.get("ok") else []
            for p, rr in zip(pts[:10], res):
                if not rr.get("ok") or rr["joint_travel_rad"] > 3.5:
                    continue
                n_ik += 1
                home_n = H().trapezoid_steps(float(np.abs(np.asarray(rr["q"], float)).max()))
                if home_n > WOUND_HOME:           # practice L9: a wound-up wrist (home 45 steps) wedged the arm on its way home
                    continue
                cost = rr["steps_est"] + pitch / 5.0 + 60.0 * math.hypot(p[0] - cx, p[1] - cy) + 2.0 * max(0, home_n - 30)
                if avoid:                         # spread drops: earlier releases into this container (practice L9 stacked a toy on a toy)
                    dmin = min(math.hypot(p[0] - float(a[0]), p[1] - float(a[1])) for a in avoid)
                    cost += 250.0 * max(0.0, 0.09 - dmin)
                if home_after and steps_left is not None:
                    cost += home_n
                if best is None or cost < best["cost"]:
                    best = {"cost": cost, "pos": (p[0], p[1], zt), "quat": qq, "pitch": pitch, "yaw": yw,
                            "est": rr["steps_est"], "home_after": home_n}
                break
    t.update(n_geom=n_geom, n_ik=n_ik, yaws=[round(v, 1) for v in yaws], obj_axis_yaw=round(th0, 1))
    if best is None:
        t["why"] = "no rim-safe reachable drop pose"
        return False, None
    t.update(target=[round(v, 3) for v in best["pos"]], pitch=best["pitch"], tool_yaw=round(best["yaw"], 1),
             est=best["est"], home_after=best["home_after"])
    m = safe_carry(robot, arm, name, best["pos"], best["quat"], best["est"], o, t, reserve=reserve)
    t["carry"] = _mv(m)
    if m.get("moved"):
        # deep release for tall objects into shallow containers (practice L9: an 8 cm figure released 9 cm above the
        # basket floor toppled over the front rim): lower until the object's bottom is 2 cm above the floor, if every
        # gripper point below the rim stays 12 mm inside the rim footprint and above the floor
        rel = corners_world(cf, [0.0, 0.0, 0.0], best["quat"])
        tall = float(rel[:, 2].max() - rel[:, 2].min())
        floor = float(lo[2])
        if tall > DEEP_MIN_H and (rim - floor) < 0.15:
            z_low = floor + 0.02 - float(rel[:, 2].min())
            if z_low < best["pos"][2] - 0.01:
                R = qmat(best["quat"])
                g = np.array([best["pos"][0], best["pos"][1], z_low])
                wid = min(OPEN_MAX, float(raw.get("wid", 0.05)) + 0.022)
                okd = True
                for pts in gripper_pts(wid).values():
                    W_ = g + pts[:, 0:1] * R[:, 0] + pts[:, 1:2] * R[:, 1] + pts[:, 2:3] * R[:, 2]
                    low = W_[:, 2] < rim + 0.004
                    ins = ((W_[:, 0] >= lo[0] + 0.012) & (W_[:, 0] <= hi[0] - 0.012) & (W_[:, 1] >= lo[1] + 0.012)
                           & (W_[:, 1] <= hi[1] - 0.012) & (W_[:, 2] >= floor + 0.005))
                    if np.any(low & ~ins):
                        okd = False
                        break
                t["deep"] = {"tall": round(tall, 3), "z_low": round(float(z_low), 3), "clear": okd}
                if okd:
                    md = robot.move_to(arm=arm, pos=[float(v) for v in g], quat=[float(v) for v in best["quat"]],
                                       speed="slow", max_dev=0.01, path="line",
                                       **({"reserve": int(reserve)} if reserve is not None else {}))
                    t["deep"]["move"] = _mv(md)
    if not m.get("moved"):
        t["why"] = f"carry: {m.get('plan_fail_reason')}"
        return False, None
    o1 = Pm._obs(robot)
    fl_tcp, fl_q = o1["eef"][arm]["pos"], o1["eef"][arm]["flange"]["quat"]
    cw = corners_world(cf, fl_tcp, fl_q)
    t["release_box"] = {"x": [round(float(cw[:, 0].min()), 3), round(float(cw[:, 0].max()), 3)],
                        "y": [round(float(cw[:, 1].min()), 3), round(float(cw[:, 1].max()), 3)],
                        "z_min": round(float(cw[:, 2].min()), 3)}
    t["release_inside"] = bool(cw[:, 0].min() >= lo[0] and cw[:, 0].max() <= hi[0] and cw[:, 1].min() >= lo[1]
                               and cw[:, 1].max() <= hi[1])
    robot.gripper(arm=arm, action="open", wait_steps=3)
    robot.hold(steps=settle)
    o2 = Pm._obs(robot)
    t["step1"] = o2.get("step")
    return bool(t["release_inside"]), t


def _path_conflicts(a_xy, b_xy, obstacles, bottom_z, clear_xy=0.07):
    """Obstacles {name: (xy, top_z)} closer than clear_xy to the segment a->b (xy) whose top is above bottom_z - 0.02."""
    a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
    ab = b - a
    L2 = float(ab @ ab)
    out = []
    for nm, (xy, top) in obstacles.items():
        p = np.asarray(xy, float)
        t = 0.0 if L2 < 1e-9 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
        if np.linalg.norm(a + t * ab - p) < clear_xy and top > bottom_z - 0.02:
            out.append(nm)
    return out


def _obstacles(o, skip):
    out = {}
    for k, v in (o.get("objects") or {}).items():
        if k in skip or v.get("container") or not v.get("aabb"):
            continue
        lo, hi = np.asarray(v["aabb"][0], float), np.asarray(v["aabb"][1], float)
        out[k] = (((lo[:2] + hi[:2]) / 2).tolist(), float(hi[2]))
    return out


def safe_carry(robot, arm, name, target, quat, est, o, log_t, reserve=None, factor=2.3, max_dev=0.03, z_cap=1.10):
    """Carry the held object to `target`; if the straight xy path passes within 7 cm of another located object whose
    top is above the held object's lowest point (practice L9: a hanging 23 cm bottle swept a lying bottle 19 cm), first
    rise to a via height that clears them (capped at z_cap, IK-checked), cross at that height, then descend."""
    e = o["eef"][arm]
    tcp = np.asarray(e["pos"], float)
    obj = o["objects"].get(name)
    low = float(obj["aabb"][0][2]) if obj and obj.get("aabb") else tcp[2] - 0.05
    hang = float(tcp[2] - low)
    obs = _obstacles(o, skip={name})
    bottom = min(low, float(target[2]) - hang)
    hits = _path_conflicts(tcp[:2], target[:2], obs, bottom)
    log_t["path_hits"] = hits
    if hits:
        zv = min(z_cap, max(obs[h][1] for h in hits) + 0.03 + hang)
        if zv > max(tcp[2], target[2]) + 0.005:
            up = (float(tcp[0]), float(tcp[1]), zv)
            ov = (float(target[0]), float(target[1]), zv)
            r = robot.reach(arm=arm, targets=[{"pos": list(up), "quat": [float(v) for v in quat]},
                                              {"pos": list(ov), "quat": [float(v) for v in quat]}], from_q=e.get("q"))
            res = r.get("results") or []
            if r.get("ok") and len(res) == 2 and all(x.get("ok") for x in res):
                m1 = _move2(robot, arm, up, quat, res[0]["steps_est"], max_dev=max_dev, factor=factor, reserve=reserve)
                m2 = _move2(robot, arm, ov, quat, res[1]["steps_est"] + 4, max_dev=max_dev, factor=factor, reserve=reserve) if m1.get("moved") else {}
                log_t["via"] = {"z": round(zv, 3), "up": _mv(m1), "over": _mv(m2)}
            else:
                log_t["via"] = {"z": round(zv, 3), "why": "via unreachable"}
    return _move2(robot, arm, target, quat, est, max_dev=max_dev, factor=factor, reserve=reserve)


# ------------------------------------------------------------------------------------------------ relay with wrist options
RELAY_SPOTS = [(0.0, -0.24), (-0.04, -0.24), (0.04, -0.24), (0.0, -0.20), (-0.06, -0.20), (0.06, -0.20), (0.0, -0.28),
               (-0.07, -0.28), (0.07, -0.28), (-0.10, -0.22), (0.10, -0.22), (0.0, -0.14)]


def relay2(robot, giver, name, raw, others, o, log, receiver=None, settle_gap=0.008):
    """put_into_bin's table relay (same spots, same set-down: lowest point 8 mm above the table, open, rise 6 cm, park
    20 cm to the giver's side) with two additions found on practice L9: (a) besides the held wrist pose it tries the
    tool yaw +-90 and pitches 0/15/30 (put_into_bin keeps the held pose only; a tall standing bottle held at the neck
    with a 15 deg pitch had no reachable spot), with the set-down height from the held box's lowest corner for that
    pose; (b) the receiver arm must reach a grasp at the spot (free IK) before the spot is used."""
    Pm = P()
    e = o["eef"][giver]
    q_cur = np.asarray(e["flange"]["quat"], float)
    t = {"op": "relay2", "arm": giver, "obj": name, "step0": o.get("step")}
    log.append(t)
    cf = held_corners_flange(robot, giver, name, raw)
    if cf is None:
        t["why"] = "object not attached in the belief"
        return False
    ty = tool_yaw_of(q_cur)
    quats = [("held", q_cur)]
    for dy in (0.0, 90.0, -90.0):
        for p in (0.0, 15.0, 30.0):
            quats.append((f"y{ty + dy:.0f}p{p:.0f}", tool_quat(ty + dy, p, 0.0)))
    occupied = [np.asarray(o["objects"][k]["pos"][:2], float) for k in others if k in o["objects"]]
    standing = raw.get("kind") == "standing"
    top_off = float(raw.get("h", 0.06))
    chosen = None
    n_try = 0
    tcp0 = np.asarray(e["pos"], float)
    low0 = float(o["objects"][name]["aabb"][0][2]) if o["objects"].get(name, {}).get("aabb") else tcp0[2] - 0.05
    obs = _obstacles(o, skip={name})
    spots = sorted(RELAY_SPOTS, key=lambda sp: len(_path_conflicts(tcp0[:2], sp, obs, low0)))
    for (x, y) in spots:
        if any(np.linalg.norm(np.asarray((x, y)) - c) < 0.11 for c in occupied):
            continue
        if receiver:
            gz = TABLE_Z + (top_off - 0.045 if standing else top_off / 2)
            tg = [{"pos": [x, y, gz], "quat": [float(v) for v in tool_quat(yw, pp, 0.0)]}
                  for yw in (0.0, 45.0, 90.0, 135.0) for pp in (0.0, 15.0, 30.0)]
            rr = robot.reach(arm=receiver, targets=tg)
            if not (rr.get("ok") and any(r_.get("ok") for r_ in (rr.get("results") or []))):
                continue
        for qn, qq in quats:
            rel = corners_world(cf, [0.0, 0.0, 0.0], qq)
            place_z = TABLE_Z + settle_gap - float(rel[:, 2].min())
            n_try += 1
            hov = Pm._reach(robot, giver, (x, y, place_z + 0.025), from_q=e.get("q"), quat=qq)
            if not hov or hov["joint_travel_rad"] > 3.5:
                continue
            dn = Pm._reach(robot, giver, (x, y, place_z), from_q=hov.get("q"), quat=qq)
            if not dn or dn["joint_travel_rad"] > 0.8:
                continue
            chosen = (x, y, place_z, qq, qn, hov, dn)
            break
        if chosen:
            break
    t["n_try"] = n_try
    if not chosen:
        t["why"] = "no free reachable relay spot (held pose, yaw +-90, pitch 0/15/30)"
        return False
    x, y, place_z, qq, qn, hov, dn = chosen
    t.update(spot=[x, y, round(float(place_z), 4)], wrist=qn)
    m = safe_carry(robot, giver, name, (x, y, place_z + 0.025), qq, hov["steps_est"], o, t)
    t["hover"] = _mv(m)
    if not m.get("moved"):
        t["why"] = f"hover: {m.get('plan_fail_reason')}"
        return False
    m = _move2(robot, giver, (x, y, place_z), qq, dn["steps_est"], max_dev=0.01, factor=2.3)
    t["lower"] = _mv(m)
    robot.gripper(arm=giver, action="open", wait_steps=3)
    robot.move_delta(arm=giver, dpos=[0, 0, 0.06])
    sx = 1.0 if giver == "right" else -1.0
    park = (x + sx * 0.20, y - 0.04, place_z + 0.08)
    pk = Pm._reach(robot, giver, park, quat=qq)
    m = _move2(robot, giver, park, qq, pk["steps_est"]) if pk and pk["joint_travel_rad"] <= 2.0 else {}
    if not m.get("moved"):
        m = robot.return_home(arms=[giver], open=True)
    t["park_steps"] = m.get("steps")
    o2 = Pm._obs(robot)
    t["step1"] = o2.get("step")
    return True


# ------------------------------------------------------------------------------------------------ nudge, survey
def nudge(robot, arm, raw, cont, d=0.06, tilt=35.0, log=None):
    """Push a lying object away from the nearest wall of `cont` with the closed fingers tilted toward the wall.
    Wall faces: the container's 4 rim-footprint sides; the one nearest the object's centre is used."""
    Pm = P()
    Sm = S()
    log = log if log is not None else []
    x, y = float(raw["c"][0]), float(raw["c"][1])
    top = float(raw["top_z"])
    h = float(raw.get("h", top - TABLE_Z))
    W = float(raw.get("wid", 0.02))
    lo, hi = cont["lo"], cont["hi"]
    faces = {"front": abs(y - lo[1]), "back": abs(y - hi[1]), "left": abs(x - lo[0]), "right": abs(x - hi[0])}
    face = min(faces, key=faces.get)
    n = {"front": (0, -1), "back": (0, 1), "left": (-1, 0), "right": (1, 0)}[face]     # push direction (away)
    n = np.asarray(n, float)
    wall = {"front": lo[1], "back": hi[1], "left": lo[0], "right": hi[0]}[face]
    axis_z = max(top - h / 2, TABLE_Z + 0.006)
    # start: the fingertip line between the object and the wall (or on the object's wall-side top when flush)
    if face in ("front", "back"):
        s = np.array([x, wall + n[1] * 0.010])
        yaw = 0.0                                    # closed jaws along x: the narrow side of the fingers faces the wall
        tq = Sm.tilt_quat(yaw, tilt if face == "front" else -tilt)
    else:
        s = np.array([wall + n[0] * 0.010, y])
        yaw = 90.0
        tq = None
    z = axis_z + 0.008
    t = {"op": "nudge", "arm": arm, "obj": raw["name"], "face": face, "wall": round(float(wall), 3), "d": d,
         "start": [round(float(s[0]), 3), round(float(s[1]), 3), round(z, 3)], "tilt": tilt, "obj_wid": W}
    log.append(t)
    if tq is None:
        r = robot.reach(arm=arm, targets=[{"pos": [float(s[0]), float(s[1]), z], "yaw_deg": yaw, "pitch_deg": -tilt if face == "left" else tilt}])
        tq = ((r.get("results") or [{}])[0]).get("quat") if r.get("ok") else None
        if tq is None:
            t["why"] = "no side-wall tilt"
            return False, t
    tq = [float(v) for v in tq]
    robot._r.gripper(arm=arm, action="close", wait_steps=0)        # direct call: no held_look for a push
    above = [float(s[0]), float(s[1]), z + 0.06]
    start = [float(s[0]), float(s[1]), z]
    end = [float(s[0] + n[0] * d), float(s[1] + n[1] * d), z + 0.002]
    chk = robot.reach(arm=arm, targets=[{"pos": above, "quat": tq}, {"pos": start, "quat": tq}, {"pos": end, "quat": tq}])
    res = chk.get("results") or []
    if not chk.get("ok") or len(res) < 3 or not all(r.get("ok") for r in res):
        t["why"] = "nudge path unreachable"
        t["reach"] = [r.get("why") for r in res]
        robot._r.gripper(arm=arm, action=1.0, wait_steps=0)
        return False, t
    m = Pm._move(robot, arm, above, tq, res[0]["steps_est"])
    t["above"] = _mv(m)
    if not m.get("moved"):
        t["why"] = f"above: {m.get('plan_fail_reason')}"
        return False, t
    m = robot.move_to(arm=arm, pos=start, quat=tq, speed="slow", max_dev=0.01, path="line")
    t["down"] = _mv(m)
    m = robot.move_to(arm=arm, pos=end, quat=tq, speed="slow", max_dev=0.01, path="line")
    t["push"] = _mv(m)
    m = robot.move_delta(arm=arm, dpos=[0, 0, 0.07])
    t["up"] = _mv(m)
    robot._r.gripper(arm=arm, action=1.0, wait_steps=0)
    return True, t


def survey(robot, arms, poses=None, max_steps=45):
    out = {}
    for arm in arms:
        p = (poses or {}).get(arm) or SURVEY[arm]
        d = robot.move_to(arm=arm, pos=p, yaw_deg=0.0, dry_run=True)
        out[arm] = {"planned": d.get("planned_steps"), "why": d.get("plan_fail_reason")}
        if d.get("planned_steps") is None or d["planned_steps"] > max_steps:
            out[arm]["moved"] = False
            continue
        m = robot.move_to(arm=arm, pos=p, yaw_deg=0.0, max_steps=max_steps)
        out[arm].update(_mv(m), moved=m.get("moved"))
    return out


# ------------------------------------------------------------------------------------------------ plan
def drop_arms2(robot, cont, raws):
    """Arms with at least one rim-safe drop pose for a mid-sized object (the largest located footprint, top-down
    or tilted, centre grasp)."""
    lo, hi = np.asarray(cont["lo"], float), np.asarray(cont["hi"], float)
    rim = float(max(hi[2], TABLE_Z))
    c = (lo[:2] + hi[:2]) / 2
    ok = []
    for arm in ("left", "right"):
        tg = [{"pos": [float(c[0]), float(c[1]), z], "yaw_deg": 0.0} for z in (1.00, 0.95, 0.90)]
        tg += [{"pos": [float(c[0]), float(c[1]), rim + 0.05], "quat": S().tilt_quat(0.0, p)} for p in PITCHES]
        tg += [{"pos": [float(c[0]), float(c[1]), rim + 0.05], "quat": S().tilt_quat(90.0, p)} for p in PITCHES]
        r = robot.reach(arm=arm, targets=tg)
        if r.get("ok") and any(x.get("ok") for x in (r.get("results") or [])):
            ok.append(arm)
    return ok


def plan2(robot, raws, containers, groups, table, lift=0.08):
    import itertools
    walls = wall_boxes(containers)
    names = [b["name"] for b in raws]
    byname = {b["name"]: b for b in raws}
    if groups:                                  # objects already placed may be left out of `objects`
        groups = {g: [k for k in v if k in byname] for g, v in groups.items()}
        groups = {g: v for g, v in groups.items() if v}
    pick = {}
    for k in names:
        pick[k] = {}
        for arm in ("left", "right"):
            p, diag = plan_pick2(robot, arm, byname[k], table, walls, lift=lift)
            pick[k][arm] = None if p is None else int(sum(p["est"]))
            if p is None and diag["rejected"]["wall"]:
                pick[k][arm] = "nudge_first"
    pick_arms = {k: [a for a in ("left", "right") if isinstance(pick[k][a], int)] for k in names}
    cmap = {c["name"]: c for c in containers}
    darms = {c["name"]: drop_arms2(robot, c, raws) for c in containers}

    def cost_item(k, c):
        both = [a for a in pick_arms[k] if a in darms[c]]
        dd = float(np.linalg.norm(np.asarray(byname[k]["c"][:2], float) - (np.asarray(cmap[c]["lo"][:2]) + np.asarray(cmap[c]["hi"][:2])) / 2))
        if both:
            return 0, dd
        if pick_arms[k] and darms[c]:
            return 1, dd
        return 10, dd

    cn = [c["name"] for c in containers]
    if groups:
        gn = sorted(groups)
        best = None
        for perm in itertools.permutations(cn, len(gn)):
            cst = [cost_item(k, c) for g, c in zip(gn, perm) for k in groups[g]]
            key = (sum(a for a, _ in cst), sum(b for _, b in cst))
            if best is None or key < best[0]:
                best = (key, dict(zip(gn, perm)))
        assign = {k: best[1][g] for g in gn for k in groups[g]}
        gmap = best[1]
    else:
        assign = {k: cn[0] for k in names}
        gmap = None
    steps = []
    for k in names:
        c = assign[k]
        both = [a for a in pick_arms[k] if a in darms[c]]
        nud = [a for a in ("left", "right") if pick[k][a] == "nudge_first"]
        if both:
            cx = float(byname[k]["c"][0])
            arm = both[0] if len(both) == 1 else ("left" if cx < 0 else "right")
            steps.append({"obj": k, "container": c, "mode": "direct", "arm": arm})
        elif pick_arms[k] and darms[c]:
            steps.append({"obj": k, "container": c, "mode": "relay", "giver": pick_arms[k][0], "receiver": darms[c][0]})
        elif nud:
            steps.append({"obj": k, "container": c, "mode": "nudge_first", "arm": nud[0]})
        else:
            steps.append({"obj": k, "container": c, "mode": "skip", "why": "unreachable"})
    return {"assign": assign, "group_to_container": gmap, "steps": steps, "pick": pick, "drop_arms": darms}


# ------------------------------------------------------------------------------------------------ run
def _budget(robot, est_lb, final, log):
    o = robot.obs(cameras=[])
    left = int(o.get("steps_left", 0))
    fc, _ = H().full_cost(o["eef"])
    g = {"steps_left": left, "stage_lb": int(est_lb), "home_now": fc, "final": bool(final)}
    log.append({"op": "budget", **g})
    if final and est_lb + FINAL_RESERVE > left:
        return False, g
    return True, g


def run(robot, stage, objects=None, containers=None, obj=None, container=None, arm=None, groups=None, table=None,
        arms=None, final=False, lift=0.08, max_picks=2, attach_xy=0.05, margin=0.02, clear=0.025, d=0.06,
        mode="auto", margin_m=0.04, margin_deg=7.0, ignore_walls=None, edge=None, poses=None, avoid=None):
    o = robot.obs(cameras=[])
    if o.get("objects") is not None:
        return {"ok": False, "verified": False, "error": "put_into_bin_rgb_strict_v2 is the strict tier-B path; the server is privileged"}
    if stage == "home":
        return {**H().run(robot, mode=mode, margin_m=margin_m, margin_deg=margin_deg, arms=arms), "tier": "B-strict"}
    if stage == "check":
        rep = H().run(robot, dry=True, margin_m=margin_m, margin_deg=margin_deg)
        return {"ok": True, "verified": True, "tier": "B-strict", **rep}
    if stage == "survey":
        r = survey(robot, arms or ["left", "right"], poses=poses)
        e = robot.obs(cameras=[])
        return {"ok": all(v.get("moved") for v in r.values()), "verified": True, "tier": "B-strict", "survey": r,
                "step": e.get("step"), "steps_left": e.get("steps_left")}
    Sm = S()
    tab = dict(DEFAULT_TABLE, **(table or {}))
    raws = list(objects or [])
    byname = {b["name"]: b for b in raws}
    conts = list(containers or [])
    cmap = {c["name"]: c for c in conts}
    if stage == "plan":
        r = plan2(robot, raws, conts, groups, tab, lift=lift)
        return {"ok": True, "verified": True, "tier": "B-strict", **r}
    R = Sm.records(raws, conts)
    br = Sm.make_drop_belief(robot, R, conts, attach_xy=attach_xy)
    walls = wall_boxes(conts, skip=tuple(ignore_walls or ()))
    log = []
    out = {}
    try:
        if stage == "nudge":
            c = cmap[container]
            ok, t = nudge(br, arm, byname[obj], c, d=d, log=log)
            out = {"nudged": ok}
            okk = ok
        elif stage in ("direct", "relay"):
            raw = byname[obj]
            is_edge = bool(edge) if edge is not None else (not on_table(raw["c"], tab, 0.0) or _crosses_edge(raw, tab))
            p0, _ = plan_pick2(br, arm, raw, tab, walls, lift=lift, edge=is_edge)
            if p0 is not None:
                okb, g = _budget(br, sum(p0["est"]) + (25 if stage == "direct" else 45), final, log)
                if not okb:
                    e = br.obs(cameras=[])
                    return {"ok": False, "verified": False, "tier": "B-strict", "why": "budget", "budget": g, "log": log,
                            "step": e.get("step"), "steps_left": e.get("steps_left")}
            rsv = FINAL_RESERVE if final else None
            held, o = pick2(br, arm, obj, raw, tab, walls, lift, max_picks, log, edge=is_edge, reserve=rsv)
            out["held"] = held
            if not held and log and log[-1].get("why") == "nudge_first":
                out["why"] = "nudge_first"
            if held and stage == "direct":
                c = cmap[container or conts[0]["name"]]
                inside, t = drop_in(br, arm, obj, raw, c, log, margin=margin, clear=clear, avoid=avoid,
                                    steps_left=o.get("steps_left"), home_after=bool(final), reserve=rsv)
                out["in_container"] = inside
                if t is None and not inside:
                    out["why"] = log[-1].get("why")
            elif held and stage == "relay":
                others = [k for k in byname if k != obj]
                recv = "left" if arm == "right" else "right"
                out["relayed"] = relay2(br, arm, obj, raw, others, o, log, receiver=recv)
                if not out["relayed"]:
                    br.gripper(arm=arm, action="open", wait_steps=3)
                    br.return_home(arms=[arm], open=True)
                spot = [t.get("spot") for t in log if t.get("op") == "relay"]
                out["spot"] = spot[-1] if spot else None
            okk = bool(out.get("held")) and bool(out.get("in_container") if stage == "direct" else out.get("relayed"))
        else:
            return {"ok": False, "verified": False, "error": f"unknown stage {stage!r} (plan|direct|relay|nudge|survey|home|check)"}
    except RuntimeError as ex:
        return {"ok": False, "verified": False, "error": str(ex), "log": log}
    e = br.obs(cameras=[])
    fc, _ = H().full_cost(e["eef"])
    out.update(log=log, step=e.get("step"), steps_left=e.get("steps_left"), ended=e.get("ended"), home_cost_now=fc)
    return Sm.bel().scrub({"ok": okk, "verified": okk, "tier": "B-strict", "belief_events": br.events,
                           "looks": br.looks, **out})


def _crosses_edge(raw, tab):
    if raw.get("kind") == "standing":
        return False
    L = float(raw.get("len", 0.0))
    yaw = math.radians(float(raw.get("yaw", 0.0)))
    c = np.asarray(raw["c"][:2], float)
    u = np.array([math.cos(yaw), math.sin(yaw)])
    return not (on_table(c + u * L / 2, tab, 0.0) and on_table(c - u * L / 2, tab, 0.0))
