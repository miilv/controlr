"""MockRobot: contract, rendering, grasp rule, scripted episodes reach the goal."""

from __future__ import annotations

import numpy as np
import pytest

from controlr.config import Config, SafetyConfig
from controlr.robot import make_robot
from controlr.robot.base import Robot
from controlr.robot.mock import MockRobot
from controlr.robot.safety import SafetyEnvelope
from controlr.types import Action, ActionMode, Observation


def _goto(robot: MockRobot, env: SafetyEnvelope, target, gripper=None, max_lines=20):
    """Scripted 'controller': ee_delta steps (<= max_step) toward a target,
    through the safety envelope like the loop does."""
    for _ in range(max_lines):
        st = robot.state()
        d = np.asarray(target, float) - st.tcp_pos
        if np.linalg.norm(d) < 5e-4:
            break
        out, ev = env.filter([Action(ActionMode.EE_DELTA, tuple(d))], st)
        assert out, ev
        robot.execute(out)
    if gripper is not None:
        out, _ = env.filter([Action(None, None, gripper=gripper)], robot.state())
        return robot.execute(out)
    return None


def test_contract_and_factory():
    cfg = Config()
    cfg.robot.backend = "mock"
    r = make_robot(cfg)
    assert isinstance(r, MockRobot) and isinstance(r, Robot)
    obs = r.reset({"name": "reach", "instruction": "", "params": {}}, seed=0)
    assert isinstance(obs, Observation)
    img = obs.images["scene"]
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    cam = obs.cameras["scene"]
    assert cam.K.shape == (3, 3) and cam.T_cam_base.shape == (4, 4)
    assert r.task_instruction and "red ball" in r.task_instruction
    with pytest.raises(ValueError):
        make_robot(Config(robot=type(cfg.robot)(backend="nope")))


def test_rendering_is_deterministic_and_reflects_state():
    r = MockRobot()
    a = r.reset({"name": "pick_place", "params": {}}, seed=3).images["scene"]
    b = r.observe().images["scene"]
    assert np.array_equal(a, b)
    assert a.std() > 10                                  # not a blank frame
    env = SafetyEnvelope(r.spec, SafetyConfig())
    out, _ = env.filter([Action(ActionMode.EE_DELTA, (0.05, 0.0, -0.05))], r.state())
    r.execute(out)
    c = r.observe().images["scene"]
    assert np.abs(c.astype(int) - a.astype(int)).sum() > 1000


def test_camera_projects_tcp_into_the_image():
    r = MockRobot({"width": 320, "height": 240})
    obs = r.reset({"name": "reach"}, seed=1)
    cam = obs.cameras["scene"]
    p = cam.T_cam_base @ np.r_[obs.state.tcp_pos, 1.0]
    u, v = (cam.K @ p[:3])[:2] / p[2]
    assert p[2] > 0 and 0 <= u < 320 and 0 <= v < 240
    assert obs.images["scene"].shape == (240, 320, 3)


def test_reach_scripted_episode_reaches_goal():
    r = MockRobot()
    r.reset({"name": "reach", "params": {}}, seed=5)
    env = SafetyEnvelope(r.spec, SafetyConfig())
    assert not r.check_goal().success
    _goto(r, env, r._scene.target)
    g = r.check_goal()
    assert g.success and g.progress == 1.0, g


def test_pick_place_scripted_episode_reaches_goal():
    r = MockRobot()
    r.reset({"name": "pick_place", "params": {}}, seed=11)
    env = SafetyEnvelope(r.spec, SafetyConfig())
    sc = r._scene
    cube, zone = sc.cube.copy(), sc.zone.copy()
    above = 0.15
    t0 = r.state().t
    _goto(r, env, [cube[0], cube[1], above])
    _goto(r, env, cube, gripper=0.0)
    assert r.state().holding and r.state().gripper_closed
    assert np.isclose(r.state().gripper_mm, sc.cube_size * 1000)
    p_mid = r.check_goal().progress
    _goto(r, env, [cube[0], cube[1], above])
    assert sc.cube[2] > cube[2] + 0.05                   # cube lifted with the gripper
    _goto(r, env, [zone[0], zone[1], above])
    _goto(r, env, [zone[0], zone[1], cube[2]])
    assert not r.check_goal().success                    # still held
    rep = _goto(r, env, [zone[0], zone[1], cube[2]], gripper=0.085)
    assert any(e.kind == "release" for e in rep.events)
    g = r.check_goal()
    assert g.success, g
    assert g.progress == 1.0 and p_mid >= 0.5
    assert r.state().t > t0                              # time advanced (travel + settle)


def test_grasp_misses_when_far_from_cube():
    r = MockRobot()
    r.reset({"name": "pick_place", "params": {"cube_xy": [-0.32, -0.20], "zone_xy": [-0.30, 0.05]}}, seed=0)
    env = SafetyEnvelope(r.spec, SafetyConfig())
    cube = r._scene.cube.copy()
    rep = _goto(r, env, cube + [0.03, 0.0, 0.0], gripper=0.0)     # 30 mm off
    assert not r.state().holding
    assert any("nothing" in e.message for e in rep.events)


def test_unfiltered_actions_are_resolved_by_fallback_envelope():
    r = MockRobot()
    r.reset({"name": "reach"}, seed=0)
    p0 = r.state().tcp_pos.copy()
    rep = r.execute([Action(ActionMode.EE_DELTA, (0.0, 0.0, -0.5))])   # no q_target
    assert rep.executed and rep.executed[0].q_target is not None
    assert r.state().tcp_pos[2] >= p0[2] - 0.1 - 1e-3                    # step-limited
    assert any(e.kind == "step_limit" for e in rep.events)


def test_task_params_and_unknown_task():
    r = MockRobot()
    r.reset({"name": "reach", "params": {"target": [-0.3, -0.1, 0.1], "tolerance_m": 0.02}})
    assert np.allclose(r._scene.target, [-0.3, -0.1, 0.1])
    r.reset({"name": "reach", "instruction": "custom words"})
    assert r.task_instruction == "custom words"
    with pytest.raises(ValueError):
        r.reset({"name": "juggle"})
