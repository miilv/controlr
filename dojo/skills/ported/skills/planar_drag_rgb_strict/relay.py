# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/planar_drag_rgb_strict/relay.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""relay.py: the yaw relay of planar_drag_rgb_strict (night T5, 25 Oct 2026). Strict tier B, contract reads only.

Why: planar_drag v1 drags the T with its whole turn done in contact. In the J1b block one push episode needed a 156 deg
tool rotation while pinched; three legs left the T unmoved (results/RESULTS-J1b-2026-10-25.md).

Rule (installed by skill_robodojo.py into its PRIVATE planar_drag instance only, like the strict `_leg` patch):
  * v1 plans first (`_plan`, unchanged). If the turn |wrap(pad yaw - T yaw)| is at most `relay_max_deg`, or every leg
    of the v1 plan already turns at most `relay_max_deg`, the v1 plan is returned unchanged (same object, no extra
    robot call). `relay_max_deg` None or <= 0 disables the relay.
  * Otherwise the drag becomes a relay of n = ceil(|turn| / relay_max_deg) legs, each turning the T by turn / n.
    Every leg is v1's own `_leg`: hover at hover_z over the T at the leg's grasp yaw, slow descent, close, held_look,
    drag, open, lift clear_dz. So between legs the tool is lifted above the part and any wrist re-turn (the other grasp
    variant, +180) happens in the air; in contact the tool turns at most turn / n. The last leg ends on the target.
  * Waypoints, in order of preference: legs that translate along one principal axis of the T (stem or bar, at the
    leg's mid heading), at most two translating legs, shortest path first and at most MAX_DETOUR_M off the start-goal
    segment; then a straight line. A half turn (180 deg) tries both directions. Every leg must pass v1's free-IK chain
    check (`_leg_check`, sane). One arm for every leg first; an arm switch (the giver homes, as in v1's table relay,
    plan mode "relay") only when no single arm serves every leg.
  * Budget: per leg est = the leg check's cost (reach steps_est for hover and descent + drag joint travel / 0.05 rad
    per step) + LEG_OVERHEAD_STEPS, + HANDOVER_HOME_STEPS before an arm switch. The relay runs only if every leg starts
    with at least min_steps_leg steps left and relay_reserve steps remain after the last; otherwise the v1 plan runs
    unchanged and the skill reply says relay: "budget_fallback". No feasible relay: v1 plan, relay: "infeasible".
  * Within one drag call the relay's legs run back to back (pd.run re-plans from the belief after each leg; the
    continuation keeps the leg count, so each later leg turns the remainder / legs left, at most relay_max_deg +
    YAW_SLACK_DEG). A failed leg (held_look "no", refused move) stops the relay: the caller re-looks.
Reads: free IK (`reach`) and `obs` (steps_left) through the belief robot; poses are the skill's located args.
"""
import math

import numpy as np

LEG_OVERHEAD_STEPS = 22     # close (3), open (2), the slow 5 cm retreat, chunk dry runs, fine moves
# Re-pinned 27 Oct 2026 (coord/DECISIONS.md D-C12): 35 [believed] -> 22. In Isaac every relay leg came in 9-14 steps
# under its estimate on 8 legs, 12.6 on average [measured: docs/notes/COMPUTE4-STRICT-2026-10-27.md section 4].
HANDOVER_HOME_STEPS = 30    # the giver's return_home at an arm switch [believed]
YAW_SLACK_DEG = 1.0         # a continuation leg may exceed the planned per-leg turn by the drag's end tolerance (0.8)
XY_EPS = 0.002              # m: a translation below this counts as none
MAX_DETOUR_M = 0.08         # axis waypoints at most this far from the start-goal segment
MAX_AXIS_CANDIDATES = 4     # axis decompositions tried by free IK before the straight line
PATH_SLACK_M = 0.005        # axis path length at most sqrt(2) * |translation| + this


class RelayState:
    """Per skill call. `log` holds one entry per plan call that engaged the relay."""

    def __init__(self, max_deg=60.0, reserve=40, min_steps_leg=110):
        self.max_deg = None if max_deg is None or float(max_deg) <= 0 else float(max_deg)
        self.reserve = int(reserve)
        self.min_steps_leg = int(min_steps_leg)
        self.log = []
        self.active = False       # a planned relay is running in this call
        self.planned_n = 0
        self.legs_done = 0
        self.last_leg = None      # (ok, why, arm) of the last relay leg
        self.last_mode = None     # mode of the last relay plan returned ("relay" = pd.run homed its first leg's arm)
        self.memo = None          # (key, plan): the drag pre-plan, reused by pd.run's first _plan call

    def engaged_by_args(self, t_yaw, pad_yaw):
        return self.max_deg is not None and abs(_wrap(float(pad_yaw) - float(t_yaw))) > self.max_deg + 1e-6

    def status(self):
        st = [e["status"] for e in self.log]
        if not st:
            return None
        return "budget_fallback" if "budget_fallback" in st else st[0]


def _wrap(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def _unit(deg):
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)])


def _seg_dist(w, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(w - a))
    t = min(1.0, max(0.0, float((w - a) @ ab) / L2))
    return float(np.linalg.norm(w - (a + t * ab)))


# ------------------------------------------------------------------ waypoint decompositions
def _finish(S, G, trans, axes, family):
    W = [S.copy()]
    for t in trans:
        W.append(W[-1] + t)
    W[-1] = G.copy()                          # the last leg ends exactly on the target (absorbs <= XY_EPS)
    moving = [k for k, t in enumerate(trans) if float(np.linalg.norm(t)) > XY_EPS]
    return {"family": family, "axes": axes, "W": W,
            "path_m": float(sum(np.linalg.norm(W[k + 1] - W[k]) for k in range(len(trans)))),
            "detour_m": max(_seg_dist(w, S, G) for w in W), "last_move": moving[-1] if moving else -1}


def axis_candidates(S, yaw, G, turn, n):
    """Decompositions whose translating legs each move along one T principal axis (stem = the T's yaw direction,
    bar = yaw + 90) at that leg's mid heading; at most two legs translate, the others turn in place.
    Sorted: shortest path, then smallest detour, then translation as late as possible."""
    S, G = np.asarray(S, float), np.asarray(G, float)
    d = G - S
    L = float(np.linalg.norm(d))
    r = turn / n
    heads = [yaw + (k + 0.5) * r for k in range(n)]
    name = ("stem", "bar")
    zero = [np.zeros(2) for _ in range(n)]
    out = []
    if L <= XY_EPS:
        out.append(_finish(S, G, zero, ["turn"] * n, "axis"))
    else:
        for k in range(n):
            for a in (0, 1):
                u = _unit(heads[k] + 90.0 * a)
                s = float(d @ u)
                if float(np.linalg.norm(d - s * u)) <= XY_EPS:
                    tr = [z.copy() for z in zero]
                    tr[k] = s * u
                    out.append(_finish(S, G, tr, [name[a] if j == k else "turn" for j in range(n)], "axis"))
        for i in range(n):
            for j in range(i + 1, n):
                for ai in (0, 1):
                    for aj in (0, 1):
                        ui, uj = _unit(heads[i] + 90.0 * ai), _unit(heads[j] + 90.0 * aj)
                        M = np.column_stack([ui, uj])
                        if abs(float(np.linalg.det(M))) < 0.2:
                            continue
                        si, sj = np.linalg.solve(M, d)
                        if min(abs(si), abs(sj)) <= XY_EPS:
                            continue                  # a one-leg solution, listed above when it exists
                        tr = [z.copy() for z in zero]
                        tr[i], tr[j] = si * ui, sj * uj
                        ax = ["turn"] * n
                        ax[i], ax[j] = name[ai], name[aj]
                        out.append(_finish(S, G, tr, ax, "axis"))
    keep, seen = [], set()
    for c in out:
        if c["path_m"] > math.sqrt(2.0) * L + PATH_SLACK_M or c["detour_m"] > MAX_DETOUR_M:
            continue
        sig = tuple(round(float(v), 4) for w in c["W"] for v in w)
        if sig in seen:
            continue
        seen.add(sig)
        keep.append(c)
    keep.sort(key=lambda c: (round(c["path_m"], 3), round(c["detour_m"], 3), -c["last_move"], c["axes"]))
    return keep


def line_candidate(S, yaw, G, turn, n):
    S, G = np.asarray(S, float), np.asarray(G, float)
    return _finish(S, G, [(G - S) / n for _ in range(n)], ["line"] * n, "line")


def _yaws(yaw, goal_yaw, turn, n):
    return [_wrap(yaw + k * turn / n) for k in range(n)] + [float(goal_yaw)]


# ------------------------------------------------------------------ search with v1's free-IK leg check
def _arm_sequences(pd, arm, W, n, prefer=None, first=None):
    """Arm per leg: one arm for every leg first, then one switch (start-side arm gives). `first` pins the first leg's
    arm (a continuation: the arm that is over the T, or the receiver after pd.run homed the giver)."""
    if arm:
        return [[arm] * n]
    seqs = [s for s in _all_sequences(pd, W, n, prefer) if first is None or s[0] == first]
    return seqs


def _all_sequences(pd, W, n, prefer):
    far = {a: max(float(np.linalg.norm(w - pd.BASES[a])) for w in W) for a in ("left", "right")}
    order = sorted(("left", "right"), key=lambda a: (round(far[a] / 0.10), 0 if a == prefer else 1, a))
    seqs = [[a] * n for a in order]
    if n > 1:
        start_near = sorted(("left", "right"), key=lambda a: (float(np.linalg.norm(W[0] - pd.BASES[a])), a))
        for A in start_near:
            B = pd.OTHER[A]
            for m in sorted(range(1, n), key=lambda m: (abs(m - n / 2.0), m)):
                seqs.append([A] * m + [B] * (n - m))
    return seqs


def search(pd, robot, S, yaw, G, goal_yaw, gz, hover_z, offsets, arm, n, max_leg_deg, prefer=None, first=None):
    """First (candidate, arm sequence) whose every leg passes v1's `_leg_check` (sane, turn <= max_leg_deg).
    Returns (candidate, legs, tried) or (None, None, tried)."""
    S, G = np.asarray(S, float), np.asarray(G, float)
    turn = _wrap(goal_yaw - yaw)
    turns = [turn] if abs(abs(turn) - 180.0) > 1e-6 else [turn, -turn]
    cache, tried = {}, []
    for tn in turns:
        cands = axis_candidates(S, yaw, G, tn, n)[:MAX_AXIS_CANDIDATES] + [line_candidate(S, yaw, G, tn, n)]
        for c in cands:
            W, Y = c["W"], _yaws(yaw, goal_yaw, tn, n)
            for seq in _arm_sequences(pd, arm, W, n, prefer, first):
                legs = []
                for k, a in enumerate(seq):
                    key = (a, tuple(round(float(v), 6) for v in (W[k][0], W[k][1], Y[k], W[k + 1][0], W[k + 1][1], Y[k + 1])))
                    if key not in cache:
                        cache[key] = pd._leg_check(robot, a, W[k], Y[k], W[k + 1], Y[k + 1], gz, hover_z, offsets)
                    rec = cache[key]
                    if rec is None or not rec["sane"] or abs(rec["dyaw"]) > max_leg_deg + 1e-6:
                        tried.append([c["family"], "".join(x[0] for x in seq), k])
                        break
                    legs.append(dict(rec, goal_xy=[float(W[k + 1][0]), float(W[k + 1][1])], goal_yaw=float(Y[k + 1])))
                else:
                    return dict(c, turn=tn), legs, tried
    return None, None, tried


def estimate(legs):
    """Steps per leg: v1's leg-check cost + LEG_OVERHEAD_STEPS, + HANDOVER_HOME_STEPS when the next leg switches arm."""
    est = []
    for k, lg in enumerate(legs):
        e = int(math.ceil(float(lg["cost"]))) + LEG_OVERHEAD_STEPS
        if k + 1 < len(legs) and legs[k + 1]["arm"] != lg["arm"]:
            e += HANDOVER_HOME_STEPS
        est.append(e)
    return est


def fits(est, steps_left, min_steps_leg, reserve):
    if steps_left is None:
        return True
    left = int(steps_left)
    for e in est:
        if left < min_steps_leg:
            return False
        left -= e
    return left >= reserve


def _relay_plan(cand, legs, est, info):
    mode = "relay" if len(legs) > 1 and legs[0]["arm"] != legs[1]["arm"] else "single"
    return {"mode": mode, "legs": [dict(lg, est_steps=e) for lg, e in zip(legs, est)], "yaw_relay": info}


def _info(st, status, n, turn, cand=None, legs=None, est=None, steps_left=None, v1=None, tried=None):
    i = {"status": status, "max_deg": st.max_deg, "turn_deg": round(float(turn), 2), "n_legs": int(n),
         "leg_turn_deg": round(float(turn) / n, 2)}
    if cand is not None:
        i.update(family=cand["family"], axes=cand["axes"], path_m=round(cand["path_m"], 4),
                 detour_m=round(cand["detour_m"], 4), arms=[lg["arm"] for lg in legs],
                 waypoints=[[round(float(w[0]), 4), round(float(w[1]), 4)] for w in cand["W"]])
    if est is not None:
        i.update(est_steps=est, est_total=int(sum(est)), steps_left=steps_left, reserve=st.reserve,
                 min_steps_leg=st.min_steps_leg)
    if v1 is not None:
        i["v1"] = {"mode": v1.get("mode"), "leg_turns_deg": [lg.get("dyaw") for lg in v1.get("legs") or []]}
    if tried:
        i["tried"] = tried[:12]
    return i


# ------------------------------------------------------------------ the wrappers installed into the private instance
def plan(pd, st, robot, objs, inst, goal_xy, goal_yaw, gz, hover_z, offsets, arm=None):
    p, yaw, _ = pd._pose(objs, inst)
    S, G = np.asarray(p[:2], float), np.asarray(goal_xy, float)
    key = (st.legs_done, tuple(round(float(v), 9) for v in (S[0], S[1], yaw, G[0], G[1], goal_yaw, gz, hover_z)),
           tuple(float(o) for o in offsets), arm)
    if st.memo is not None and st.memo[0] == key:
        return st.memo[1]
    if st.active:
        left = st.planned_n - st.legs_done
        if st.last_leg is not None and not st.last_leg[0]:
            st.active = False
            return {"mode": "none", "why": f"yaw relay stopped: leg {st.legs_done} of {st.planned_n} failed "
                                           f"({st.last_leg[1]}); re-look", "yaw_relay": {"status": "stopped"}}
        if left > 0:
            turn = _wrap(goal_yaw - yaw)
            last_arm = (st.last_leg or (None, None, None))[2]
            # no arm switch without the giver homed: after a "single" plan the same arm continues, after a "relay"
            # plan (pd.run homed its first leg's arm) the other arm does
            first = None if last_arm is None else (last_arm if st.last_mode != "relay" else pd.OTHER[last_arm])
            cand, legs, tried = search(pd, robot, S, yaw, G, goal_yaw, gz, hover_z, offsets, arm, left,
                                       st.max_deg + YAW_SLACK_DEG, prefer=last_arm, first=first)
            if cand is None:
                st.active = False
                return {"mode": "none", "why": "yaw relay: no feasible continuation from the belief pose; re-look",
                        "yaw_relay": _info(st, "stopped", left, turn, tried=tried)}
            info = _info(st, "continue", left, cand["turn"], cand, legs, estimate(legs))
            st.log.append({k: info[k] for k in ("status", "n_legs", "leg_turn_deg", "family", "axes", "arms")})
            out = _relay_plan(cand, legs, estimate(legs), info)
            st.last_mode = out["mode"]
            return out
        st.active = False
    plan0 = pd._plan_v1(robot, objs, inst, goal_xy, goal_yaw, gz, hover_z, offsets, arm)
    turn = _wrap(goal_yaw - yaw)
    if st.max_deg is None or abs(turn) <= st.max_deg + 1e-6:
        return plan0
    if plan0.get("mode") != "none" and all(abs(float(lg["dyaw"])) <= st.max_deg + 1e-6 for lg in plan0.get("legs") or []):
        return plan0
    n = int(math.ceil(abs(turn) / st.max_deg - 1e-9))
    o = robot.obs(cameras=[])
    steps_left = o.get("steps_left")
    prefer = plan0["legs"][0]["arm"] if plan0.get("mode") == "single" else None
    cand, legs, tried = search(pd, robot, S, yaw, G, goal_yaw, gz, hover_z, offsets, arm, n, st.max_deg, prefer)
    if cand is None:
        info = _info(st, "infeasible", n, turn, steps_left=steps_left, v1=plan0, tried=tried)
        st.log.append({"status": "infeasible", "n_legs": n, "turn_deg": info["turn_deg"]})
        return dict(plan0, yaw_relay=info)
    est = estimate(legs)
    if not fits(est, steps_left, st.min_steps_leg, st.reserve):
        info = _info(st, "budget_fallback", n, cand["turn"], cand, legs, est, steps_left, v1=plan0)
        st.log.append({k: info[k] for k in ("status", "n_legs", "turn_deg", "est_steps", "est_total", "steps_left",
                                             "reserve", "min_steps_leg")})
        return dict(plan0, yaw_relay=info)
    info = _info(st, "planned", n, cand["turn"], cand, legs, est, steps_left, v1=plan0)
    st.log.append({k: info[k] for k in ("status", "n_legs", "turn_deg", "leg_turn_deg", "family", "axes", "arms",
                                         "est_steps", "est_total", "steps_left")})
    st.active, st.planned_n, st.legs_done, st.last_leg = True, n, 0, None
    out = _relay_plan(cand, legs, est, info)
    st.last_mode = out["mode"]
    st.memo = (key, out)
    return out


def leg(pd, st, robot, inst, lg, **kw):
    ok, rec = pd._leg_v1(robot, inst, lg, **kw)
    if st.active:
        st.legs_done += 1
        st.last_leg = (bool(ok), rec.get("why"), lg.get("arm"))
        rec["yaw_relay_leg"] = [st.legs_done, st.planned_n]
    return ok, rec


def install(pd, st):
    """Point the private planar_drag instance's _plan / _leg at this call's relay state (v1 kept as _plan_v1 / _leg_v1)."""
    if not hasattr(pd, "_plan_v1"):
        pd._plan_v1, pd._leg_v1 = pd._plan, pd._leg
    pd._plan = lambda robot, objs, inst, goal_xy, goal_yaw, gz, hover_z, offsets, arm=None: plan(
        pd, st, robot, objs, inst, goal_xy, goal_yaw, gz, hover_z, offsets, arm)
    pd._leg = lambda robot, inst, lg, **kw: leg(pd, st, robot, inst, lg, **kw)
    return st
