"""Isaac Sim integration tests (GPU host only).

Run on compute3:  CONTROLR_ISAAC=1 uv run --extra dev pytest -q tests/test_isaac_sim.py
The module launches its own server (scripts/isaac_server.sh) on
CONTROLR_ISAAC_TEST_PORT (default 7821, never the deployment port) and shuts
it down afterwards. ~15 s startup + ~3 min of physics. No LLM calls.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.isaac,
              pytest.mark.skipif(os.environ.get("CONTROLR_ISAAC") != "1", reason="needs Isaac Sim (CONTROLR_ISAAC=1)")]

from controlr.robot.isaac.tasks import START_Q, scripted_pick_place_plan  # noqa: E402
from controlr.robot.kinematics import UR3Kinematics, matrix_to_rotvec  # noqa: E402
from controlr.types import Action, ActionMode  # noqa: E402

KIN = UR3Kinematics()


@pytest.fixture(scope="module")
def robot():
    from controlr.robot.isaac.client import IsaacRobot
    r = IsaacRobot({"port": int(os.environ.get("CONTROLR_ISAAC_TEST_PORT", "7821")), "launch": True,
                    "startup_timeout_s": 600})
    yield r
    r.close()


def joint_action(q, gripper=None) -> Action:
    q = tuple(float(v) for v in q)
    return Action(ActionMode.JOINT_ABS, q, gripper=gripper, q_target=q)


def test_startup_info(robot):
    info = robot.server_info
    assert info["timings"]["startup_total_s"] < 300
    assert robot.camera.width == 640 and robot.camera.height == 480
    # PHANTOM factory D435 RGB intrinsics
    assert robot.camera.K[0, 0] == pytest.approx(609.28, abs=0.01)
    # the gripper opens to roughly the Robotiq stroke; pad midpoint ~ the 180 mm TCP
    assert 0.08 < info["gripper"]["max_m"] < 0.10
    assert np.asarray(info["gripper"]["pad_midpoint_in_tool_m"])[2] == pytest.approx(0.18, abs=0.003)


@pytest.mark.parametrize("task", ["waffle_pick_place", "reach", "push"])
def test_reset_observe_roundtrip(robot, task):
    obs = robot.reset({"name": task}, seed=11)
    img = obs.images["scene"]
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8 and img.std() > 10
    np.testing.assert_allclose(obs.state.q, START_Q, atol=2e-3)
    assert obs.state.gripper_mm > 80 and not obs.state.holding
    assert robot.episode["settle_drift_m"] < 0.005
    again = robot.observe()                        # physics paused: nothing moves
    np.testing.assert_allclose(again.state.q, obs.state.q, atol=1e-6)
    assert again.state.t == pytest.approx(obs.state.t)
    goal = robot.check_goal()
    assert not goal.success and goal.message


def test_small_ee_delta_tracking_within_2mm(robot):
    robot.reset({"name": "reach", "params": {"nominal": True}}, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(6):
        d = rng.uniform(-0.02, 0.02, 3)
        rep = robot.execute([Action(ActionMode.EE_DELTA, tuple(d))])
        achieved = rep.state_after.tcp_pos - rep.state_before.tcp_pos
        assert np.linalg.norm(achieved - d) < 0.002, (d, achieved)
        assert not rep.stopped and not any(e.kind == "collision" for e in rep.events)
        # orientation kept (rotation=none)
        dR = KIN.fk(rep.state_after.q)[1] - KIN.fk(rep.state_before.q)[1]
        assert np.linalg.norm(dR) < 0.01


def test_closing_on_nothing_is_not_holding(robot):
    """Regression (integration 2026-10-02): pad-on-pad contact must not count
    as holding — the pad forces are filtered to the object."""
    robot.reset({"name": "reach"}, seed=11)
    rep = robot.execute([Action(None, None, gripper=0.0)])
    assert rep.state_after.gripper_mm < 10
    assert rep.state_after.holding is False
    robot.execute([Action(None, None, gripper=0.085)])


def test_kinematics_match_sim_tcp_for_random_q(robot):
    """controlr FK(q_measured) == sim tool0 pose + 180 mm (same TCP)."""
    robot.reset({"name": "reach", "params": {"nominal": True}}, seed=0)
    rng = np.random.default_rng(1)
    checked = 0
    while checked < 5:
        q = np.asarray(START_Q) + rng.uniform(-0.3, 0.3, 6)
        if KIN.fk(q)[0][2] < 0.12:                 # keep clear of the table / packet
            continue
        rep = robot.execute([joint_action(q)])
        st = rep.state_after
        p_fk, rv_fk = KIN.fk(st.q)
        assert np.linalg.norm(st.tcp_pos - p_fk) < 0.001, (q, st.tcp_pos, p_fk)
        R_err = KIN.fk_matrix(st.q)[:3, :3].T @ _rotvec_matrix(st.tcp_rotvec)
        assert np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) < 0.5
        checked += 1


def _rotvec_matrix(rv):
    from controlr.robot.kinematics import rotvec_to_matrix
    return rotvec_to_matrix(rv)


def run_scripted_pick_place(robot, seed: int) -> dict:
    obs = robot.reset({"name": "waffle_pick_place"}, seed=seed)
    ep = robot.episode
    w, z = ep["object_quat_wxyz"][0], ep["object_quat_wxyz"][3]
    plan = scripted_pick_place_plan(ep["object_pos"], 2 * np.arctan2(z, w),
                                    robot.server_info["scene_info"]["bin"]["center"])
    q = np.asarray(obs.state.q)
    for ph in plan:
        p0 = KIN.fk(q)[0]
        n = max(1, int(np.ceil(np.linalg.norm(ph["pos"] - p0) / 0.02))) if ph["linear"] else 1
        acts = []
        for k in range(1, n + 1):
            tp = p0 + (ph["pos"] - p0) * k / n if ph["linear"] else ph["pos"]
            q = KIN.ik(tp, matrix_to_rotvec(ph["R"]), q)
            assert q is not None, ph["name"]
            acts.append(joint_action(q, ph["gripper"] if k == n else None))
        rep = robot.execute(acts)
        assert not rep.stopped, (ph["name"], [e.message for e in rep.events])
        if ph["name"] == "lift":
            assert rep.state_after.holding, "packet not held after the lift"
    goal = robot.check_goal()
    return {"goal": goal, "state": robot.state()}


def test_scripted_waffle_pick_place_succeeds(robot):
    res = run_scripted_pick_place(robot, seed=0)
    goal = res["goal"]
    assert goal.success, goal
    assert goal.metrics["inside_bin"] and goal.metrics["unloaded"]
