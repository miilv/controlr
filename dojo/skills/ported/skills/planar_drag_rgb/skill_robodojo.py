# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/planar_drag_rgb/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""planar_drag_rgb (RoboDojo, tier B: images + proprioception + camera calibration; NO object poses).

Runs planar_drag v1 (skills/planar_drag/skill_robodojo.py, loaded read-only, unchanged) behind BeliefRobot (belief.py):
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
"""
import importlib.util
import os

TABLE_Z = 0.765
_PD = None
_BL = None


def _load(name, fname):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pd():
    global _PD
    if _PD is None:
        _PD = _load("planar_drag_v1_ro", os.path.join("..", "planar_drag", "skill_robodojo.py"))
    return _PD


def _bl():
    global _BL
    if _BL is None:
        _BL = _load("rgb_belief", "belief.py")
    return _BL


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
        grasp_yaw_offsets=(90,), **kw):
    pd = _pd()
    br = _bl().BeliefRobot(robot, _objects(t_pose, pad_pose, t_top, pad_z))
    if stage == "plan":
        o = br.obs()
        gz = max(float(t_top) - 0.004, TABLE_Z + 0.009)
        plan = pd._plan(br, o["objects"], "t", [float(pad_pose[0]), float(pad_pose[1])], float(pad_pose[2]), gz, hover_z,
                        [float(v) for v in grasp_yaw_offsets], arm)
        plan.pop("candidates", None)
        return {"ok": True, "verified": plan.get("mode") != "none", "plan": plan, "grasp_z": round(gz, 4), "tier": "B"}
    if stage != "drag":
        return {"ok": False, "verified": False, "error": f"stage must be plan or drag, got {stage!r}"}
    rep = pd.run(br, obj="t", target="target_t", arm=arm, max_legs=int(legs), home=bool(home), hover_z=hover_z,
                 grasp_yaw_offsets=grasp_yaw_offsets, **kw)
    rep = dict(rep or {})
    rep.update(tier="B", belief_events=br.events,
               belief_final={k: [round(float(v), 4) for v in br.objs["t"]["pos"]] for k in ("t",)},
               note="details.final / verified are belief-side (open loop); re-look to measure the real pose")
    return rep
