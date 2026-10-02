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


def test_force_stop_into_the_mat_is_tight_and_holds_then_backs_off(robot):
    """Review control-safety #5/#6: contacts are sampled every step during motion (stops
    overshot 80 N up to 1297 N at 10 ms sampling), and a stop freezes the target at the
    measured pose (the arm used to keep pushing during settling)."""
    obs = robot.reset({"name": "reach", "params": {"nominal": True}}, seed=0)
    q0 = np.asarray(obs.state.q)
    p0, rv0 = KIN.fk(q0)
    target = p0.copy()
    target[2] = robot.spec.table_z - 0.03                   # TCP 30 mm into the mat (no envelope)
    q_in = KIN.ik(target, rv0, q0)
    assert q_in is not None
    rep = robot.execute([joint_action(q_in)])
    assert rep.stopped and rep.events[0].level.value == "stop", [e.message for e in rep.events]
    peak = float(rep.events[0].message.split("contact force ")[1].split(" N")[0])
    assert peak < 250.0, rep.events[0].message
    st = robot.state()
    np.testing.assert_allclose(st.q, rep.state_after.q, atol=1e-6)     # paused, holding
    up = robot.execute([Action(ActionMode.EE_DELTA, (0.0, 0.0, 0.04))])
    assert not up.stopped and up.state_after.tcp_pos[2] > rep.state_after.tcp_pos[2] + 0.03


def test_reach_seed0_marker_is_reachable_in_the_sim(robot):
    """Review control-safety #1: drive the TCP to the (now feasible) seed-0 marker along the
    envelope's straight-line waypoints; no collision on the way."""
    from controlr.config import SafetyConfig
    from controlr.robot.safety import SafetyEnvelope

    obs = robot.reset({"name": "reach"}, seed=0)
    marker = np.asarray(robot.episode["marker"], float)
    env = SafetyEnvelope(robot.spec, SafetyConfig())
    env.reset(robot.reference_state())
    st = obs.state
    for _ in range(12):
        d = marker - st.tcp_pos
        if np.linalg.norm(d) < 0.005:
            break
        acts, _ = env.filter([Action(ActionMode.EE_DELTA, tuple(d))], st)
        assert acts, "envelope refused the move"
        rep = robot.execute(acts)
        assert not rep.stopped, [e.message for e in rep.events]
        st = rep.state_after
    assert robot.check_goal().success, (st.tcp_pos, marker)


# ---------------------------------------------------------------------------
# rotation=yaw (docs/ROTATION_REPORT.md)
# ---------------------------------------------------------------------------

SIM_START_Q = (0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406)      # configs/sim_waffle.yaml


def _yaw_envelope(robot):
    from controlr.config import SafetyConfig
    from controlr.robot.safety import SafetyEnvelope
    env = SafetyEnvelope(robot.spec, SafetyConfig(), rotation="yaw")
    env.reset(robot.reference_state())
    return env


def _rpy_measured(st):
    from controlr.robot.kinematics import matrix_to_rpy
    return matrix_to_rpy(_rotvec_matrix(st.tcp_rotvec))


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def test_yaw_delta_moves_track_on_the_tilted_tool(robot):
    """ee_delta dyaw through the rotation=yaw envelope: achieved yaw within 1 deg of the
    executed command, TCP within 2 mm of the commanded translation (0 for pure turns),
    roll/pitch held at the reference tilt within 1 deg."""
    obs = robot.reset({"name": "waffle_pick_place", "params": {"start_q": list(SIM_START_Q), "nominal": True}},
                      seed=0)
    env = _yaw_envelope(robot)
    ref = _rpy_measured(robot.reference_state())
    st = obs.state
    moves = [((0, 0, 0), 20), ((0, 0, 0), -20), ((0, 0, -0.06), 25), ((0.02, -0.02, 0), -30),
             ((0, 0, 0), -15), ((-0.02, 0.01, 0.03), 10), ((0, 0, 0), 30), ((0, 0, 0.03), -40)]
    worst = {"yaw": 0.0, "pos": 0.0, "tilt": 0.0}
    for d, dyaw in moves:
        acts, ev = env.filter([Action(ActionMode.EE_DELTA, (*d, 0.0, 0.0, np.radians(dyaw)))], st)
        assert acts, [e.message for e in ev]
        cmd = acts[0].values                                   # after any step-limit clamp
        rep = robot.execute(acts)
        assert not rep.stopped, [e.message for e in rep.events]
        r0, r1 = _rpy_measured(rep.state_before), _rpy_measured(rep.state_after)
        yaw_err = abs(_wrap_deg(np.degrees(r1[2] - r0[2]) - np.degrees(cmd[5])))
        pos_err = float(np.linalg.norm(rep.state_after.tcp_pos - rep.state_before.tcp_pos - np.asarray(cmd[:3])))
        tilt_err = float(np.max(np.abs([_wrap_deg(v) for v in np.degrees(np.asarray(r1[:2]) - ref[:2])])))
        worst = {k: max(worst[k], v) for k, v in (("yaw", yaw_err), ("pos", pos_err * 1000), ("tilt", tilt_err))}
        assert yaw_err < 1.0, (d, dyaw, yaw_err)
        assert pos_err < 0.002, (d, dyaw, pos_err)
        assert tilt_err < 1.0, (d, dyaw, tilt_err)
        st = rep.state_after
    print("yaw tracking worst:", {k: round(v, 3) for k, v in worst.items()})


@pytest.mark.parametrize("offset", [-45, 45])
def test_packet_stands_still_at_large_yaw(robot, offset):
    robot.reset({"name": "waffle_pick_place", "params": {"yaw_offset_deg": offset}}, seed=5)
    rec = robot.scene_record()
    assert rec["packet"]["tilt_deg"] < 1.0 and rec["packet"]["settle_drift_mm"] < 3.0, rec["packet"]
    assert abs(rec["packet"]["yaw_offset_deg"] - offset) < 1.0, rec["packet"]
    goal = robot.check_goal()
    assert goal.metrics["object_tilt_deg"] < 1.0 and goal.metrics["object_speed_m_s"] < 0.01


@pytest.mark.parametrize("offset,start_yaw", [(-40, 20), (-15, 0), (15, -20), (30, 0)])
def test_scripted_yaw_expert_succeeds(robot, offset, start_yaw):
    """A scripted expert in the model's action space (ee_delta + dyaw + GRIP, through the
    rotation=yaw envelope) turns the jaws across the rotated packet, grasps, turns back
    while low, and places it in the box."""
    from controlr.robot.isaac.tasks import run_scripted_yaw_pick_place
    from controlr.robot.kinematics import pose_to_matrix

    params = {"start_q": list(SIM_START_Q), "yaw_offset_deg": offset, "start_yaw_deg": [start_yaw, start_yaw],
              "nominal": True}                  # nominal packet xy (+30 deg needs it; see test_isaac_tasks)
    obs = robot.reset({"name": "waffle_pick_place", "params": params}, seed=offset + 100)
    rec = robot.scene_record()
    assert abs(rec["start"]["yaw_offset_deg"] - start_yaw) < 0.01
    env = _yaw_envelope(robot)
    cur = {"st": obs.state}
    holding: list[bool] = []
    img_dir = os.environ.get("CONTROLR_ISAAC_IMG_DIR")

    def act(vals, grip, phase=""):
        acts, ev = env.filter([Action(ActionMode.EE_DELTA, vals, grip)], cur["st"])
        bad = [e.message for e in ev if e.kind in ("reach", "ik_fail", "table", "workspace")]
        assert acts and not bad, bad
        rep = robot.execute(acts)
        assert not rep.stopped, [e.message for e in rep.events]
        cur["st"] = rep.state_after
        holding.append(bool(rep.state_after.holding))
        return pose_to_matrix(rep.state_after.tcp_pos, rep.state_after.tcp_rotvec)

    ep = robot.episode
    w, z = ep["object_quat_wxyz"][0], ep["object_quat_wxyz"][3]
    log = run_scripted_yaw_pick_place(ep["object_pos"], 2 * np.arctan2(z, w),
                                      robot.server_info["scene_info"]["bin"]["center"], act,
                                      pose_to_matrix(obs.state.tcp_pos, obs.state.tcp_rotvec))
    phases = [r["phase"] for r in log]
    i_lift = len(phases) - 1 - phases[::-1].index("lift0")
    assert holding[i_lift], "packet not held after the first lift"
    goal = robot.check_goal()
    if img_dir:
        from PIL import Image
        Image.fromarray(robot.observe().images["scene"]).save(f"{img_dir}/yaw_expert_{offset:+d}_done.png")
    assert goal.success, (offset, start_yaw, goal)
    print(f"yaw expert offset {offset:+d} start {start_yaw:+d}: {len(log)} actions, success")
