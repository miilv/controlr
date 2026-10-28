# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/return_home_budget/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""return_home_budget (RoboDojo, any tier incl. strict): finish an episode with both arms inside RoboDojo's home tolerance
at the least env-step cost.  basket round r1 agent (Claude), 25 Oct 2026.

Why: the server's return_home always drives both arms back to the exact start joints. RoboDojo's home rule only needs
each arm's flange within 0.15 m per axis and 20 deg of its start pose (rules/TASKS-robodojo.md 1.4, FP:1631-1644), and
put_bottles L4 (next3 run) lost a 4-bottles-in episode because return_home needed more than the 24 steps left.

run(robot, mode="auto", home_q=None, home_flange=None, margin_m=0.04, margin_deg=7.0, dry=False)
  1. cost of the exact return_home, computed with the shim's own timing model (gripper ramp 0.25/step, then one joint
     line to the start joints at 0.08 rad/step with a 2-step trapezoid ramp; both arms in parallel);
  2. per arm not already inside the tolerance (minus the margins): a "near-home" target = the start flange pose moved
     toward the current pose, clipped to 0.15 - margin_m per axis, with the orientation slerped to within
     20 - margin_deg of the start; its cost from a 0-step move_to dry run (reserve 0). Arms move one after the other;
  3. mode "auto": the exact return_home if it fits in steps_left, else the near-home moves if they fit (cheaper),
     else whichever is cheaper (best effort). mode "full" / "near" force one. dry=True only reports the costs.
Grippers: an arm whose gripper is commanded closed is opened first (the ladders want >= 80 % open).
Reads only proprioception (eef q, flange pose, commanded gripper, home deltas) and dry-run replies.
"""
import math

import numpy as np

HOME_TOL_M = 0.15
HOME_TOL_DEG = 20.0
LIM_EMPTY = 0.08
GRIP_STEP = 0.25
GRASP_OFF = 0.1501
# start pose of both arms (seed 0 layout 9 practice reset obs; also the shim's home_q = start joints = 0)
DEFAULT_HOME_FLANGE = {"left": ([-0.2995, -0.3523, 0.9215], [0.707, 0.0, 0.0, 0.7072]),
                       "right": ([0.3005, -0.3523, 0.9215], [0.707, 0.0, 0.0, 0.7072])}


STAGING = {"left": [-0.33, -0.30, 1.00], "right": [0.33, -0.30, 1.00]}   # top-down pose high over each arm's own side


def trapezoid_steps(L, vmax=LIM_EMPTY, ramp=2):
    """Same as rd_shim.trapezoid's n: minimal steps so each step moves <= vmax with `ramp` linear ramp steps."""
    if L <= 1e-9:
        return 0
    n = max(1, int(math.ceil(L / vmax - 1e-9)))
    while True:
        t = np.arange(1, n + 1, dtype=float)
        v = vmax * np.minimum(1.0, np.minimum(t, n + 1 - t) / (ramp + 1.0))
        if float(v.sum()) >= L - 1e-9:
            return n
        n += 1


def grip_steps(cmd):
    return int(math.ceil(max(0.0, 1.0 - float(cmd)) / GRIP_STEP - 1e-9))


def full_cost(eef, home_q=None):
    """Env steps of the server's return_home from the current joints (both arms, parallel)."""
    L = 0.0
    g = 0
    for arm in ("left", "right"):
        q = np.asarray(eef[arm]["q"], float)
        hq = np.zeros_like(q) if home_q is None else np.asarray(home_q[arm], float)
        L = max(L, float(np.abs(q - hq).max()))
        g = max(g, grip_steps(eef[arm].get("grip_cmd", eef[arm].get("gripper", 1.0))))
    return g + trapezoid_steps(L), round(L, 4)


def _qmat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _qangle(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(a @ b)))))


def _slerp(a, b, t):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def near_target(eef_arm, home_flange, margin_m, margin_deg):
    """Flange pose inside the tolerance box nearest the current pose; returns (tcp_pos, quat)."""
    hp, hq = np.asarray(home_flange[0], float), np.asarray(home_flange[1], float)
    fp = np.asarray(eef_arm["flange"]["pos"], float)
    fq = np.asarray(eef_arm["flange"]["quat"], float)
    lim = HOME_TOL_M - margin_m
    tp = hp + np.clip(fp - hp, -lim, lim)
    ang = _qangle(hq, fq)
    amax = HOME_TOL_DEG - margin_deg
    tq = fq if ang <= amax else _slerp(hq, fq, amax / ang)
    tcp = tp + GRASP_OFF * _qmat(tq)[:, 0]
    return [float(v) for v in tcp], [float(v) for v in tq]


def run(robot, mode="auto", home_q=None, home_flange=None, margin_m=0.04, margin_deg=7.0, dry=False, arms=None):
    hf = home_flange or DEFAULT_HOME_FLANGE
    which = tuple(arms or ("left", "right"))
    o = robot.obs(cameras=[])
    eef = o["eef"]
    left = int(o.get("steps_left", 0))
    fc, L = full_cost(eef, home_q)
    near = {}
    near_cost = 0
    for arm in which:
        e = eef[arm]
        inside = e["home_dpos_max_m"] <= HOME_TOL_M - margin_m and e["home_ddeg"] <= HOME_TOL_DEG - margin_deg
        g = grip_steps(e.get("grip_cmd", e.get("gripper", 1.0)))
        if inside:
            near[arm] = {"inside": True, "grip_steps": g}
            near_cost += g
            continue
        tcp, tq = near_target(e, hf[arm], margin_m, margin_deg)
        d = robot.move_to(arm=arm, pos=tcp, quat=tq, dry_run=True, reserve=0)
        n = d.get("planned_steps")
        near[arm] = {"inside": False, "tcp": [round(v, 4) for v in tcp], "quat": [round(v, 5) for v in tq],
                     "planned": n, "why": d.get("plan_fail_reason"), "grip_steps": g}
        near_cost = None if (n is None or near_cost is None) else near_cost + n + g
    rep = {"steps_left": left, "full_cost": fc, "full_travel_rad": L, "near_cost": near_cost, "near": near}
    if dry:
        return {"ok": True, "verified": False, "dry": True, **rep}
    if mode == "full":
        choice = "full"
    elif mode == "near":
        choice = "near" if near_cost is not None else "full"
    elif fc <= left:
        choice = "full"
    elif near_cost is not None and near_cost <= left:
        choice = "near"
    else:
        choice = "near" if (near_cost is not None and near_cost < fc) else "full"
    rep["choice"] = choice
    if choice == "full":
        # unwind first (practice L9, classify): from a wound-up wrist (straight joint line home > 3.0 rad on some joint)
        # the joint-line return_home swept the left arm into the head-camera mount; move such an arm to a top-down
        # staging pose over its own side first when that plans in <= 60 steps
        unwound = {}
        for arm in which:
            qa = np.asarray(eef[arm]["q"], float)
            hq = np.zeros_like(qa) if home_q is None else np.asarray(home_q[arm], float)
            if float(np.abs(qa - hq).max()) > 3.0:
                sp = STAGING[arm]
                d = robot.move_to(arm=arm, pos=sp, yaw_deg=0.0, dry_run=True, reserve=0)
                n = d.get("planned_steps")
                if n is not None and n <= 60 and n + fc <= left:
                    m = robot.move_to(arm=arm, pos=sp, yaw_deg=0.0, reserve=0, max_steps=70)
                    unwound[arm] = {"planned": n, "steps": m.get("steps"), "moved": m.get("moved")}
                else:
                    unwound[arm] = {"planned": n, "moved": False}
        if unwound:
            rep["unwind"] = unwound
        h = robot.return_home(arms=list(which), open=True)
        rep["home_reply"] = {k: h.get(k) for k in ("steps", "move_steps", "grip_steps", "all_home", "plan_fail_reason",
                                                  "within_tolerance")}
    else:
        moves = {}
        for arm in which:
            n = near[arm]
            if n["grip_steps"]:
                robot.gripper(arm=arm, action="open", wait_steps=0)
            if n["inside"]:
                continue
            m = robot.move_to(arm=arm, pos=n["tcp"], quat=n["quat"], reserve=0, max_dev=0.05)
            moves[arm] = {k: m.get(k) for k in ("moved", "steps", "residual_m", "tracking_rad", "plan_fail_reason")}
            if o.get("ended") or m.get("ended"):
                break
        rep["near_moves"] = moves
    o2 = robot.obs(cameras=[])
    homed = {a: bool(o2["eef"][a]["home_dpos_max_m"] <= HOME_TOL_M and o2["eef"][a]["home_ddeg"] <= HOME_TOL_DEG)
             for a in which}
    rep.update(homed=homed, step=o2.get("step"), steps_left_after=o2.get("steps_left"), ended=o2.get("ended"))
    ok = all(homed.values())
    return {"ok": ok, "verified": ok, **rep}
