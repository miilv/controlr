"""DOJO_GRIP_OBS (dojo.simlab.gripobs): what a dev harness hands the policy as the gripper reading. ``robodojo`` passes
RoboDojo's observation through unchanged (the default). ``command`` gives the strict contract's value: the last
command, the step-0 observation before any command, rounded as the strict server reports it. CPU only."""

import numpy as np
import pytest

from dojo.simlab.gripobs import GripObs, grip_obs_mode


def _st(gl, gr):
    return {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": np.array([gl]),
            "right_arm_joint_state": np.zeros(6), "right_ee_joint_state": np.array([gr])}


def _act(gl, gr):
    return {"left_arm_joint_state": np.zeros(6, np.float32), "left_ee_joint_state": np.array([gl], np.float32),
            "right_arm_joint_state": np.zeros(6, np.float32), "right_ee_joint_state": np.array([gr], np.float32)}


def test_robodojo_mode_is_a_pass_through(monkeypatch):
    monkeypatch.delenv("DOJO_GRIP_OBS", raising=False)
    g = GripObs()
    assert g.mode == "robodojo"
    st = _st(1.0, 1.0)
    assert g.apply(st) is st
    g.stepped(_act(0.0, 1.0))
    st2 = _st(0.32, 1.0)                     # RoboDojo's reading while the jaws are blocked by a block
    assert g.apply(st2) is st2


def test_command_mode_reads_the_last_command_like_the_strict_server(monkeypatch):
    monkeypatch.setenv("DOJO_GRIP_OBS", "command")
    g = GripObs()
    assert g.mode == "command"
    assert float(g.apply(_st(1.0, 1.0))["left_ee_joint_state"][0]) == 1.0        # step 0: the observation itself
    g.stepped(_act(0.667, 1.0))
    g.stepped(_act(0.0, 1.0))
    out = g.apply(_st(0.32, 0.98))                                                # blocked jaws, a settling right arm
    assert float(out["left_ee_joint_state"][0]) == 0.0 and float(out["right_ee_joint_state"][0]) == 1.0
    np.testing.assert_array_equal(out["left_arm_joint_state"], np.zeros(6))      # nothing else changes
    g.stepped(_act(0.123456, 1.0))
    assert float(g.apply(_st(0.2, 1.0))["left_ee_joint_state"][0]) == 0.1235      # rounded as grip_cmd (1e-4)
    g.reset()
    assert float(g.apply(_st(0.9, 1.0))["left_ee_joint_state"][0]) == 0.9         # a new episode starts from its obs


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        grip_obs_mode("finger")
    assert grip_obs_mode(" Command ") == "command"


def test_runner_env_and_rendered_episode_apply_it(monkeypatch):
    monkeypatch.setenv("DOJO_GRIP_OBS", "command")
    from dojo.simlab.proxy_eval import RenderedEpisode
    from dojo.skills.env import RunnerEnv

    class FakeEnv:
        def __init__(self):
            self.end_flag, self.take_action_cnt, self.step_lim = [False], [0], 10

        def take_action_batch(self, acts, env_idx_list):
            self.take_action_cnt[0] += 1

    class FakeRunner:
        def __init__(self):
            self.env, self.g = FakeEnv(), 1.0

        def obs_state(self, i):
            return _st(self.g, 1.0)

        def obs(self, i):
            return {"images": {}, "state": _st(self.g, 1.0), "instruction": "x"}

        def step(self, a):
            self.env.take_action_cnt[0] += 1
            return [False]

        def reward(self, i):
            return {"steps_used": self.env.take_action_cnt[0], "step_lim": 10}

    r = FakeRunner()
    env = RunnerEnv(r)
    assert float(env.proprio()["left_ee_joint_state"][0]) == 1.0
    env.step(_act(0.0, 1.0))
    r.g = 0.33                                                                    # RoboDojo: the blocked jaws
    assert float(env.proprio()["left_ee_joint_state"][0]) == 0.0
    assert float(RunnerEnv(r, grip_obs="robodojo").proprio()["left_ee_joint_state"][0]) == 0.33

    ep = object.__new__(RenderedEpisode)
    ep.run, ep.n_obs, ep.grip = FakeRunner(), 0, __import__("dojo.simlab.gripobs", fromlist=["GripObs"]).GripObs()
    assert float(ep.observe()["state"]["left_ee_joint_state"][0]) == 1.0
    ep.step(_act(0.0, 1.0))
    ep.run.g = 0.3
    assert float(ep.observe()["state"]["left_ee_joint_state"][0]) == 0.0
