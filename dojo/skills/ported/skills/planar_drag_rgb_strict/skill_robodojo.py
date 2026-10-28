# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/planar_drag_rgb_strict/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""planar_drag_rgb_strict (RoboDojo, STRICT tier B: images + joints + flange + commanded gripper; calibration computed
from the public config; NO finger joints, NO object poses). Copy of planar_drag_rgb with the held signal replaced.

Runs planar_drag v1 (skills/planar_drag/skill_robodojo.py, loaded read-only into a PRIVATE module instance) behind
BeliefRobotStrict (belief_strict.py). STRICT changes, nothing else:
  * BeliefRobotStrict.gripper(close) asks the perceiver held_look ("object between the fingers? yes/no", wrist image,
    0 env steps) and attaches the T only on "yes";
  * planar_drag's `_leg` is replaced, in that private instance only, by the same function source with its grasp test
    `grasp_detected or min(finger_q) > 0.003` swapped for that perceiver answer (and its rec["grip"] log for the
    commanded gripper). On "no" the leg does what v1 does: open, lift clear_dz, return "grasp not detected".
The rest, as planar_drag_rgb:
the T and the pad poses come from the planner's pixel picks; while the T is pinched its pose follows the flange.
Each call runs ONE leg (grasp, drag, release, lift 5 cm) by default, so the planner re-looks and re-locates the T after
every drag segment and calls again for a correction or for the second leg of a relay.

stage="plan":  0 env steps. planar_drag's own planner (`_plan`, free IK) on the located poses: mode single / relay /
               none, arm, tool yaw. Use it to choose the arm whose wrist camera refines the T before the drag.
stage="drag":  t_pose=[x, y, yaw], pad_pose=[x, y, yaw] (the same object-frame point and heading convention on both,
               e.g. the T axis midpoint and the stem->bar heading), t_top (located top z of the T),
               legs=1, home=False, arm=None, plus any planar_drag argument (chunk_m, speed, ...).
Returns planar_drag's reply with `details` (its belief-side errors are NOT measurements: they are ~0 by construction),
plus `belief_events` (attach/detach with the xy offset) and `tier`: "B".

Yaw relay (night T5, relay.py; relay_max_deg=60, relay_reserve=40): when the drag would turn the tool by more than
relay_max_deg while pinching the T, it becomes a relay of legs that each turn at most relay_max_deg, with the tool lifted
above the part and re-approached between legs; all its legs run in this one drag call (they count as one of `legs`).
If the relay's step estimate does not fit steps_left, the v1 plan runs and the reply carries relay: "budget_fallback".
Turns up to relay_max_deg keep the v1 plan byte for byte. relay_max_deg <= 0 or None disables it.
"""
import importlib.util
import inspect
import os

TABLE_Z = 0.765
_PD = None
_BL = None
_RL = None


def _load(name, fname):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_LEG_OLD = (
    '    rec["grip"] = {k: g.get(k) for k in ("finger_q", "gap_m", "fingers_stopped", "grasp_detected")}\n'
    '    if not (g.get("grasp_detected") or min(g.get("finger_q") or [0]) > 0.003):\n')
_LEG_NEW = (
    '    rec["grip"] = {k: g.get(k) for k in ("grip_cmd", "cmd_width_m")}\n'
    '    rec["look"] = (getattr(robot, "looks", None) or [None])[-1]\n'
    '    if not (getattr(robot, "last_held", None) or {}).get(arm):\n')


def _pd():
    global _PD
    if _PD is None:
        m = _load("planar_drag_v1_ro_strict", os.path.join("..", "planar_drag", "skill_robodojo.py"))
        src = inspect.getsource(m._leg)
        if src.count(_LEG_OLD) != 1:
            raise RuntimeError("planar_drag._leg changed: strict patch does not apply")
        exec(compile(src.replace(_LEG_OLD, _LEG_NEW), m.__file__ + ":_leg[strict]", "exec"), m.__dict__)
        _PD = m
    return _PD


def _bl():
    global _BL
    if _BL is None:
        _BL = _load("rgb_belief_strict", "belief_strict.py")
    return _BL


def _rl():
    global _RL
    if _RL is None:
        _RL = _load("planar_drag_relay_strict", "relay.py")
    return _RL


def _relay_state(pd, relay_max_deg, relay_reserve, kw):
    msl = kw.get("min_steps_leg", inspect.signature(pd.run).parameters["min_steps_leg"].default)
    return _rl().install(pd, _rl().RelayState(relay_max_deg, relay_reserve, msl))


def _objects(t_pose, pad_pose, t_top, pad_z):
    bl = _bl()
    x, y, yw = [float(v) for v in t_pose]
    px, py, pyw = [float(v) for v in pad_pose]
    top = float(t_top)
    return {
        "t": {"label": "t", "category": "T", "pos": [x, y, TABLE_Z + 0.5 * (top - TABLE_Z)], "quat": bl.yaw_quat(yw),
              "aabb": [[x - 0.05, y - 0.05, TABLE_Z], [x + 0.05, y + 0.05, top]]},
        "target_t": {"label": "target_t", "category": "T_pad", "fixed": True, "pos": [px, py, float(pad_z)],
                     "quat": bl.yaw_quat(pyw), "aabb": [[px - 0.05, py - 0.05, TABLE_Z], [px + 0.05, py + 0.05, float(pad_z)]]},
    }


def run(robot, t_pose, pad_pose, t_top, stage="drag", pad_z=0.767, legs=1, home=False, arm=None, hover_z=0.83,
        grasp_yaw_offsets=(90,), relay_max_deg=60.0, relay_reserve=40, **kw):
    pd = _pd()
    br = _bl().BeliefRobotStrict(robot, _objects(t_pose, pad_pose, t_top, pad_z), tag="planar_drag_rgb_strict")
    if stage not in ("plan", "drag"):
        return {"ok": False, "verified": False, "error": f"stage must be plan or drag, got {stage!r}"}
    st = _relay_state(pd, relay_max_deg, relay_reserve, kw)
    if stage == "plan":
        o = br.obs()
        gz = max(float(t_top) - 0.004, TABLE_Z + 0.009)
        plan = pd._plan(br, o["objects"], "t", [float(pad_pose[0]), float(pad_pose[1])], float(pad_pose[2]), gz, hover_z,
                        [float(v) for v in grasp_yaw_offsets], arm)
        plan.pop("candidates", None)
        rep = {"ok": True, "verified": plan.get("mode") != "none", "plan": plan, "grasp_z": round(gz, 4), "tier": "B-strict"}
        if st.log:
            rep.update(relay=st.status(), relay_log=st.log)
        return rep
    max_legs = int(legs)
    if max_legs >= 1 and st.engaged_by_args(t_pose[2], pad_pose[2]):
        # pre-plan with exactly pd.run's inputs so a relay's legs all run in this call; pd.run reuses this plan
        objs = br.obs()["objects"]
        tp, ty, _ = pd._pose(objs, "target_t")
        gz = max(float(objs["t"]["aabb"][1][2]) - 0.004, TABLE_Z + 0.009)
        pre = pd._plan(br, objs, "t", tp[:2].astype(float), float(ty), gz, hover_z, [float(x) for x in grasp_yaw_offsets], arm)
        if (pre.get("yaw_relay") or {}).get("status") == "planned":
            max_legs = int(legs) - 1 + st.planned_n
    rep = pd.run(br, obj="t", target="target_t", arm=arm, max_legs=max_legs, home=bool(home), hover_z=hover_z,
                 grasp_yaw_offsets=grasp_yaw_offsets, **kw)
    rep = _bl().scrub(dict(rep or {}))
    rep.update(tier="B-strict", belief_events=br.events, looks=br.looks,
               belief_final={k: [round(float(v), 4) for v in br.objs["t"]["pos"]] for k in ("t",)},
               note="details.final / verified are belief-side (open loop); re-look to measure the real pose")
    if st.log:
        rep.update(relay=st.status(), relay_log=st.log)
    return rep
