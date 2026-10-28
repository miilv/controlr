"""Stream P's port of Mikhail's skills (dojo/skills/ported): the HarnessRobot adapter, the ported routes, and the legality
rule (outside G3 mode gt the skills read only our belief: the env's ground truth raises). CPU only, kinematic env, no
catalogue, no network."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from dojo.runtime.belief import KinematicBelief
from dojo.skills import planner
from dojo.skills.env import KinematicEnv
from dojo.skills.kinematics import ARMS, X5Kin
from dojo.skills.ported import TIER_A, ported
from dojo.skills.ported.harness_robot import (TOP_DOWN, HarnessRobot, TruthDenied, TruthGuard, qmat, tool_quat,
                                              trapezoid)
from dojo.skills.skills import GTProvider, Skills, _objects_from_rows_nodb
from dojo.skills.world import WorldState, arms_from_proprio

Z = 0.765


def rows_pickup():
    return [dict(label="target", category="toy_car", category_idx=3, pos=[0.12, -0.12, Z + 0.02], quat=[1, 0, 0, 0],
                 half=[0.04, 0.015, 0.02]),
            dict(label="clutter_a", category="dice", category_idx=0, pos=[-0.15, -0.05, Z + 0.01], quat=[1, 0, 0, 0],
                 half=[0.01, 0.01, 0.01])]


def rows_push():
    return [dict(label="t", category="t", category_idx=0, pos=[-0.05, -0.15, Z + 0.0075], quat=[1, 0, 0, 0],
                 half=[0.05, 0.05, 0.0075]),
            dict(label="target_t", category="t_cushion", category_idx=0, type="Geometry", pos=[0.05, -0.10, Z + 0.001],
                 quat=[math.cos(0.2), 0, 0, math.sin(0.2)], half=[0.05, 0.05, 0.001])]


def rows_swap():
    return [dict(label="t0", category="t", category_idx=0, pos=[-0.06, -0.17, Z + 0.0075], quat=[1, 0, 0, 0],
                 half=[0.05, 0.05, 0.0075]),
            dict(label="t1", category="t", category_idx=1, pos=[0.06, -0.16, Z + 0.0075],
                 quat=[math.cos(0.4), 0, 0, math.sin(0.4)], half=[0.05, 0.05, 0.0075])]


def rows_cover():
    out = []
    for i, (lab, idx, x) in enumerate((("red", 4, -0.12), ("green", 6, 0.0), ("blue", 5, 0.12))):
        out.append(dict(label=lab, category="cube", category_idx=idx, pos=[x, -0.12, Z + 0.02], quat=[1, 0, 0, 0],
                        half=[0.02, 0.02, 0.02]))
        out.append(dict(label=f"cup{i}", category="cup", category_idx=1, pos=[x, 0.02, Z + 0.04], quat=[1, 0, 0, 0],
                        half=[0.035, 0.035, 0.04]))
    return out


def rows_pour():
    out = [dict(label="cup", category="cup", category_idx=6, pos=[0.15, -0.12, Z + 0.045], quat=[1, 0, 0, 0],
                half=[0.035, 0.035, 0.045]),
           dict(label="vase", category="vase", category_idx=2, pos=[-0.10, -0.05, Z + 0.08], quat=[1, 0, 0, 0],
                half=[0.05, 0.05, 0.08])]
    for k in range(3):
        out.append(dict(label=f"sphere_{k}", category="sphere", category_idx=0, pos=[0.15, -0.12, Z + 0.02 + 0.015 * k],
                        quat=[1, 0, 0, 0], half=[0.007, 0.007, 0.007]))
    return out


def rows_bottle():
    return [dict(label="bottle", category="wuliangye", category_idx=0, pos=[0.12, -0.12, Z + 0.11], quat=[1, 0, 0, 0],
                 half=[0.04, 0.04, 0.11]),
            dict(label="cup", category="goblet", category_idx=6, pos=[-0.05, -0.10, Z + 0.05], quat=[1, 0, 0, 0],
                 half=[0.04, 0.04, 0.05])]


SCENES = {
    "pick_up": (rows_pickup, "Pick up the toy car by 10 cm."),
    "planar_drag": (rows_push, "Push the T-shaped block to align it precisely with the gray T-shaped pad."),
    "swap_poses": (rows_swap, "Pick up the two T-shaped blocks, swap their positions, and place them back with the "
                              "correct orientations."),
    "cover_uncover_v2": (rows_cover, "Cover the blocks from left to right, remember their colors, then uncover them "
                                     "in the order: red, green, and blue."),
    "pour": (rows_pour, "Pour all the balls from the cup into the vase."),
    "pour_bottle": (rows_bottle, "Pour the liquid from the bottle into the cup."),
}


def gt_skills(rows, instr, step_lim=400):
    env = KinematicEnv(rows, step_lim=step_lim, instruction=instr)
    return env, Skills(env, GTProvider(env, None, instruction=instr), reserve=30)


class TruthRaisingEnv(KinematicEnv):
    """The world as camera mode sees it: actions and proprioception work, every ground-truth read raises (the env keeps
    simulating its objects internally)."""

    def gt_objects(self):
        raise AssertionError("ground truth read: gt_objects")

    def finger_joints(self):
        raise AssertionError("ground truth read: finger_joints")


class BeliefEnv:
    """Our runtime's env shape: each action also advances the kinematic belief (dojo.runtime.belief)."""

    def __init__(self, env, belief):
        self._env, self.belief = env, belief

    @property
    def step_lim(self):
        return self._env.step_lim

    @property
    def steps_used(self):
        return self._env.steps_used

    def proprio(self):
        return self._env.proprio()

    def step(self, action):
        self.belief.step(action)
        return self._env.step(action)


class BeliefProvider:
    """A camera-mode provider: the WorldState is the belief (started from a perception record), never the env's truth."""

    def __init__(self, env, record_rows, instruction):
        self.env, self.instruction = env, instruction
        self.belief = KinematicBelief(_objects_from_rows_nodb(record_rows), X5Kin(), env.proprio())
        self.home_q, self.notes = None, {}

    def refresh(self, holding=None):
        st = self.env.proprio()
        arms = arms_from_proprio(st, holding)
        if self.home_q is None:
            self.home_q = {a: arms[a].q.copy() for a in ARMS}
        return WorldState(task="", instruction=self.instruction, objects=copy.deepcopy(self.belief.objects), arms=arms,
                          home_q=self.home_q, step=int(self.env.steps_used), step_lim=int(self.env.step_lim),
                          notes=self.notes)


def belief_skills(record_rows, true_rows, instr, step_lim=400):
    env0 = TruthRaisingEnv(true_rows, step_lim=step_lim, instruction=instr)
    prov = BeliefProvider(env0, record_rows, instr)
    env = BeliefEnv(env0, prov.belief)
    sk = Skills(env, prov, reserve=30)
    return env0, sk


# ------------------------------------------------------------------ math copied from the shim
def test_tool_quat_top_down_closes_along_world_x():
    R = qmat(tool_quat(0.0))
    assert R[:, 0] == pytest.approx([0, 0, -1], abs=1e-9)        # tool axis down
    assert abs(R[0, 1]) == pytest.approx(1.0, abs=1e-9)          # jaws (flange y) along world x
    assert qmat(TOP_DOWN) == pytest.approx(R)
    R90 = qmat(tool_quat(90.0))
    assert abs(R90[1, 1]) == pytest.approx(1.0, abs=1e-9)


def test_trapezoid_respects_the_speed_cap_and_ends_on_the_goal():
    n, s = trapezoid(1.0, 0.08, 2)
    assert s[-1] == pytest.approx(1.0) and n == len(s)
    assert np.diff(np.r_[0.0, s]).max() <= 0.08 + 1e-9
    assert trapezoid(0.0, 0.08, 2)[0] == 0


# ------------------------------------------------------------------ the adapter's verbs (G3 mode gt)
def test_obs_names_objects_by_identity_never_by_label():
    env, sk = gt_skills(rows_pickup(), "Pick up the toy car by 10 cm.")
    robot = HarnessRobot(sk)
    assert robot.truth is True
    o = robot.obs(cameras=[], grasps=["*"])
    assert o["ok"] and set(o["objects"]) == {"toy_car_3_0", "dice_0_0"}
    for rec in o["objects"].values():
        assert rec["label"] is None
        lo, hi = rec["aabb"]
        assert hi[2] > lo[2] and rec["grasps"]
    e = o["eef"]["right"]
    assert e["at_home"] and e["grip_cmd"] == 1.0 and len(e["finger_q"]) == 2
    assert o["steps_left"] == 400 and o["ended"] is False and o["instruction"].startswith("Pick up")


def test_motion_verbs_reach_move_grip_home():
    env, sk = gt_skills(rows_pickup(), "Pick up the toy car by 10 cm.")
    robot = HarnessRobot(sk)
    r = robot.reach(arm="right", targets=[{"pos": [0.12, -0.12, 0.85], "yaw_deg": 0.0}], with_q=True)
    assert r["ok"] and r["results"][0]["ok"] and len(r["results"][0]["q"]) == 6 and r["results"][0]["steps_est"] > 2
    d = robot.move_to(arm="right", pos=[0.12, -0.12, 0.85], yaw_deg=0.0, dry_run=True)
    assert d["moved"] is False and d["planned_steps"] > 0 and d["steps"] == 0 and env.steps_used == 0
    m = robot.move_to(arm="right", pos=[0.12, -0.12, 0.85], yaw_deg=0.0)
    assert m["moved"] and m["reached"] and m["residual_m"] < 0.01 and m["path_mode"] == "joint_line"
    assert m["steps"] == env.steps_used > 0
    far = robot.move_to(arm="left", pos=[0.6, 0.3, 0.8], yaw_deg=0.0)
    assert far["moved"] is False and far["plan_fail_reason"] in ("workspace_limit", "ik_unreachable")
    g = robot.gripper(arm="right", action="close")
    assert g["ok"] and g["grip_cmd"] == 0.0 and g["steps"] >= 4 + 3
    h = robot.return_home()
    assert h["ok"] and h["all_home"] and sk.ex.g["right"] == 1.0


def test_budget_reserve_refuses_without_stepping():
    env, sk = gt_skills(rows_pickup(), "Pick up the toy car by 10 cm.", step_lim=30)
    robot = HarnessRobot(sk, reserve=20)
    m = robot.move_to(arm="right", pos=[0.12, -0.12, 0.80], yaw_deg=0.0)
    assert m["moved"] is False and m["plan_fail_reason"] == "budget" and env.steps_used == 0


def test_label_handles_resolve_to_instance_names():
    env, sk = gt_skills(rows_push(), SCENES["planar_drag"][1])
    robot = HarnessRobot(sk)
    assert robot.resolve_args({"obj": "@t", "target": "@target_t", "x": 1}) == {"obj": "t_0_0", "target": "t_cushion_0_0",
                                                                               "x": 1}


# ------------------------------------------------------------------ routes
@pytest.mark.parametrize("skill", sorted(SCENES))
def test_each_instruction_routes_to_its_ported_skill_only_when_switched_on(skill):
    rows_f, instr = SCENES[skill]
    env, sk = gt_skills(rows_f(), instr)
    w = sk.refresh()
    assert planner.plan(instr, w).skill == "none"                 # the default: unchanged
    p = planner.plan(instr, w, ported=True)
    assert p.skill == "ported" and p.args["skill"] == skill
    assert skill in TIER_A


def test_our_own_routes_win_over_ported_ones():
    rows = [dict(label=f"block{i}", category="block", category_idx=i, pos=[-0.1 + 0.1 * i, -0.1, Z + 0.02],
                 quat=[1, 0, 0, 0], half=[0.02, 0.02, 0.02]) for i in range(3)]
    instr = "Stack the three blocks with different textures on top of each other."
    env, sk = gt_skills(rows, instr)
    w = sk.refresh()
    assert planner.plan(instr, w, ported=True).skill == planner.plan(instr, w).skill == "stack_objects"


def test_planar_drag_route_passes_handles_not_label_text():
    env, sk = gt_skills(rows_push(), SCENES["planar_drag"][1])
    p = planner.plan(SCENES["planar_drag"][1], sk.refresh(), ported=True)
    assert p.args["args"] == {"obj": "@t", "target": "@target_t"}


def test_g3_skill_attr_switches_the_routes():
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["ported_routes"] == ("off", "on")


def test_runtime_flag_defaults_off():
    from dojo.runtime.policy import RuntimeConfig
    assert RuntimeConfig().ported_routes is False


# ------------------------------------------------------------------ a whole skill, G3 mode gt (kinematic)
def test_pick_up_end_to_end_on_the_kinematic_env():
    env, sk = gt_skills(rows_pickup(), "Pick up the toy car by 10 cm.", step_lim=200)
    sk.ported_routes = "on"
    lifted = []
    sk.ex.on_step = lambda ex: lifted.append(env.objects[0]["pos"][2])
    res = planner.solve(sk)
    assert res["skill"] == "ported" and res["verified"] is True
    assert max(lifted) > Z + 0.02 + 0.10
    run = sk.ported_runs[-1]
    assert run["truth"] is True and run["skill"] == "pick_up"


# ------------------------------------------------------------------ legality: camera mode never reads truth
def test_truth_guard_raises_on_ground_truth_and_passes_proprio():
    env = KinematicEnv(rows_pickup())
    g = TruthGuard(env)
    assert "left_arm_joint_state" in g.proprio()
    for name in ("gt_objects", "finger_joints", "measured_gripper", "objects", "attached", "reward"):
        with pytest.raises(TruthDenied):
            getattr(g, name)


def test_truth_needs_a_plain_gt_provider():
    env0, sk = belief_skills(rows_pickup(), rows_pickup(), "Pick up the toy car by 10 cm.")
    assert HarnessRobot(sk).truth is False
    with pytest.raises(TruthDenied):
        HarnessRobot(sk, truth=True)


def _hidden_change(rows):
    """Another truth behind the same belief: every object 2 cm off, other labels, other config indices."""
    out = []
    for i, r in enumerate(copy.deepcopy(rows)):
        r["pos"] = [r["pos"][0] + 0.02, r["pos"][1] - 0.02, r["pos"][2]]
        r["label"] = f"hidden{i}"
        r["category_idx"] = int(r.get("category_idx", 0)) + 7
        out.append(r)
    return out


@pytest.mark.parametrize("skill", sorted(SCENES))
def test_camera_mode_reads_only_the_belief(skill):
    rows_f, instr = SCENES[skill]
    record = rows_f()                          # perception's record = the belief's start
    logs = []
    for truth in (rows_f(), _hidden_change(rows_f())):
        env0, sk = belief_skills(record, truth, instr, step_lim=400)
        p = planner.plan(instr, sk.refresh(), ported=True)
        assert p.skill == "ported" and p.args["skill"] == skill
        res = ported(sk, **p.args)             # no TruthDenied, no AssertionError from the truth-raising env
        assert sk.ported_runs[-1]["truth"] is False
        assert "error" in res and res["skill"] == f"ported:{skill}"
        logs.append([{k: np.asarray(v).round(6).tolist() for k, v in a.items()} for a in sk.ex.log.actions])
    assert logs[0] == logs[1]                  # the hidden truth changed nothing the robot did
    assert len(logs[0]) > 0 or skill == "cover_uncover_v2"


def test_pick_up_grounds_the_phrase_by_the_public_description(monkeypatch):
    import dojo.skills.ported as P
    desc = {"Rigid/hoe/00000": ("yellow ice axe",), "Rigid/dice/00000": ("white dice",)}
    monkeypatch.setattr(P, "asset_descriptions", lambda key: desc.get(key, ()))
    rows = [dict(label="target", category="hoe", category_idx=0, pos=[0.12, -0.12, Z + 0.02], quat=[1, 0, 0, 0],
                 half=[0.08, 0.015, 0.02]),
            dict(label="d", category="dice", category_idx=0, pos=[-0.12, -0.12, Z + 0.01], quat=[1, 0, 0, 0],
                 half=[0.01, 0.01, 0.01], clutter=True)]
    env, sk = gt_skills(rows, "Pick up the yellow ice axe by 10 cm.")
    p = planner.plan("Pick up the yellow ice axe by 10 cm.", sk.refresh(), ported=True)
    assert p.args == {"skill": "pick_up", "args": {"obj": "@target"}}
    p = planner.plan("Pick up the white dice by 10 cm.", sk.refresh(), ported=True)
    assert p.args["args"] == {"obj": "@d"}                         # a clutter object can be the one named
    p = planner.plan("Pick up the red mug by 10 cm.", sk.refresh(), ported=True)
    assert p.args == {"skill": "pick_up", "args": {}}              # no description match: the skill grounds by names
