"""SO-101 backend without hardware: URDF kinematics, the 5-joint orientation rules, and
``So101Robot`` against the real server code (``FakeArm``, test image) over a real
``multiprocessing.connection`` — moves, gripper, a blocked joint, a whole FakeLLM episode."""

from __future__ import annotations

import math
import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest

from controlr.config import load_config
from controlr.llm.fake import FakeLLM
from controlr.loop import run_episode
from controlr.robot.safety import make_envelope
from controlr.robot.so101 import kinematics as K
from controlr.robot.so101 import server as S
from controlr.robot.so101.client import So101Robot
from controlr.types import Action, ActionMode, EventLevel

FAST = {"reset_duration_s": 0.2, "connect_timeout_s": 2.0}


# --------------------------------------------------------------------------- kinematics

def test_fk_zero_pose_matches_the_urdf():
    T = K.fk(np.zeros(5))
    # upper arm up, forearm and tool straight forward (so101_new_calib.urdf at q = 0)
    assert np.allclose(T[:3, 3], [0.391, 0.0, 0.226], atol=1e-3)
    assert np.allclose(T[:3, 2], [1.0, 0.0, 0.0], atol=1e-3)


def test_ik_round_trip_from_a_nearby_seed():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = np.radians(rng.uniform([-60, -60, -60, -60, -90], [60, 60, 60, 60, 90]))
        T = K.fk(q)
        q2, pe, ae = K.ik(T[:3, 3], T[:3, :3], q + np.radians(rng.uniform(-8, 8, 5)))
        assert pe < 5e-4 and ae < 5e-3
        assert np.allclose(K.fk(q2)[:3, 3], T[:3, 3], atol=5e-4)


def test_tool_rotation_and_reachable_rotation():
    assert np.allclose(K.tool_rotation(math.pi / 2, 0.0), K.R_DOWN)
    R = K.tool_rotation(math.radians(60), 0.0, math.radians(90))
    # a vertical tool keeps any heading; a tilted one is turned to point away from the base
    assert np.allclose(K.reachable_rotation([0.2, 0.15, 0.05], K.R_DOWN), K.R_DOWN)
    R2 = K.reachable_rotation([0.2, 0.15, 0.05], R)
    az = math.atan2(R2[1, 2], R2[0, 2])
    assert az == pytest.approx(math.atan2(0.15, 0.2 - K.PAN_AXIS_XY[0]), abs=1e-9)
    assert R2[2, 2] == pytest.approx(R[2, 2])            # same pitch


def test_line_path_reaches_or_says_why():
    R = K.tool_rotation(math.radians(60), 0.0, math.radians(90))
    q0, pe, _ = K.ik([0.24, 0.0, 0.10], R, np.radians([0, -30, 60, 60, -90]), iters=300)
    assert pe < 1e-3
    p = np.array([0.26, 0.05, 0.06])
    path, frac, why = K.line_path(q0, p, K.reachable_rotation(p, R))
    assert frac == 1.0 and why == "" and len(path) >= 10
    assert np.allclose(K.fk(path[-1])[:3, 3], p, atol=2e-3)
    path, frac, why = K.line_path(q0, [0.60, 0.0, 0.10], R)        # far beyond the reach
    assert frac < 1.0 and "out of reach" in why


def test_default_camera_sees_the_jaws_from_above_and_upright_at_home():
    R = K.tool_rotation(math.radians(60), 0.0, math.radians(90))
    q, _, _ = K.ik([0.24, 0.0, 0.10], R, np.radians([0, -30, 60, 60, -90]), iters=300)
    Tc = np.linalg.inv(K.camera_T_cam_base(q, K.DEFAULT_T_TCP_CAM))
    assert Tc[2, 3] > K.fk(q)[2, 3] + 0.05            # camera above the TCP
    assert Tc[:3, 0] @ [0, -1, 0] > 0.9               # image right = the arm's right (-y)
    assert Tc[:3, 2] @ [0, 0, -1] > 0.9               # looking down


# --------------------------------------------------------------------------- server + client

@pytest.fixture
def rig():
    arm = S.FakeArm(speed_rad_s=30.0)
    server = S.Server(arm, S.Camera(None, 320, 180, 30), control_hz=200.0, max_joint_speed=20.0,
                      settle_s=0.02, settle_max_s=0.2)
    listener = Listener(("127.0.0.1", 0), authkey=b"test")
    t = threading.Thread(target=S.serve, args=(server, "", 0, b"test"),
                         kwargs={"listener": listener, "max_clients": 1, "handle_sigint": False}, daemon=True)
    t.start()
    yield arm, listener.address
    t.join(5)


def _robot(address, monkeypatch, **params):
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    return So101Robot({"host": address[0], "port": address[1], **FAST, **params})


def test_reset_moves_home_and_observes(rig, monkeypatch):
    arm, address = rig
    robot = _robot(address, monkeypatch)
    obs = robot.reset({})
    assert np.allclose(obs.state.tcp_pos, [0.24, 0.0, 0.10], atol=2e-3)
    assert obs.images["wrist"].shape == (180, 320, 3)
    assert obs.state.gripper_mm == pytest.approx(28.0, abs=1.0)        # GRIP open = 40 %
    assert obs.notes and obs.notes[0].startswith("VIEW: from the jaw tips in the wrist image, +x points up")
    assert robot.spec.kinematics == "backend" and len(robot.spec.joints) == 5
    robot.close()


def test_move_and_grip_through_the_envelope(rig, monkeypatch):
    from controlr.config import SafetyConfig

    arm, address = rig
    robot = _robot(address, monkeypatch)
    obs = robot.reset({})
    env = make_envelope(robot.spec, SafetyConfig(max_step_m=0.05))
    env.reset(robot.reference_state())
    acts, events = env.filter([Action(ActionMode.EE_DELTA, (0.03, 0.03, -0.02), gripper=0.0, raw="MOVE")],
                              obs.state)
    assert not events
    rep = robot.execute(acts)
    assert not rep.stopped and not [e for e in rep.events if e.level != EventLevel.INFO]
    assert np.allclose(rep.state_after.tcp_pos, obs.state.tcp_pos + [0.03, 0.03, -0.02], atol=3e-3)
    assert rep.state_after.gripper_closed and rep.state_after.gripper_mm < 2.0
    assert len(rep.backend["runs"]) == 2                # the move, then the jaws
    robot.close()


def test_sag_is_corrected_and_the_offset_carries_over(rig, monkeypatch):
    arm, address = rig
    arm.sag = np.radians([0.0, 3.0, -2.5, 1.5, 0.0])
    robot = _robot(address, monkeypatch)
    obs = robot.reset({})
    assert np.allclose(obs.state.tcp_pos, [0.24, 0.0, 0.10], atol=2e-3)
    tgt = tuple(obs.state.tcp_pos + [0.02, 0.0, -0.02]) + tuple(obs.state.tcp_rotvec)
    rep = robot.execute([Action(ActionMode.EE_DELTA, (0.02, 0.0, -0.02), raw="MOVE", tcp_target=tgt)])
    assert not rep.events
    assert np.allclose(rep.state_after.tcp_pos, tgt[:3], atol=2e-3)
    assert np.max(np.abs(rep.backend["runs"][-1]["offset_rad"])) > np.radians(1.0)
    robot.close()


def test_gripper_never_opens_past_the_cap(rig, monkeypatch):
    arm, address = rig
    robot = _robot(address, monkeypatch)
    robot.reset({})
    rep = robot.execute([Action(None, None, gripper=0.07, raw="GRIP 70 mm")])
    assert rep.state_after.gripper_mm == pytest.approx(28.0, abs=1.0)
    assert robot.spec.gripper_max_mm == pytest.approx(28.0)
    robot.close()


def test_unreachable_move_is_skipped_and_a_blocked_joint_stops(rig, monkeypatch):
    arm, address = rig
    robot = _robot(address, monkeypatch)
    obs = robot.reset({})
    far = tuple(obs.state.tcp_pos + [0.4, 0, 0]) + tuple(obs.state.tcp_rotvec)
    rep = robot.execute([Action(ActionMode.EE_DELTA, (0.4, 0, 0), raw="MOVE far", tcp_target=far)])
    assert [e.kind for e in rep.events] == ["clamp"] and "% of the way" in rep.events[0].message
    obs = robot.reset({})
    arm.block = {1: float(arm.q[1]) + 0.02}            # shoulder_lift cannot go further
    tgt = tuple(obs.state.tcp_pos + [0.05, 0, -0.05]) + tuple(obs.state.tcp_rotvec)
    rep = robot.execute([Action(ActionMode.EE_DELTA, (0.05, 0, -0.05), raw="MOVE", tcp_target=tgt)])
    assert rep.stopped and rep.events[-1].level == EventLevel.STOP and rep.events[-1].kind == "contact"
    robot.close()


def _bare_server(arm):
    return S.Server(arm, S.Camera(None, 320, 180, 30), control_hz=200.0, max_joint_speed=20.0,
                    settle_s=0.02, settle_max_s=0.2)


def test_a_garbled_bus_read_is_retried_and_the_move_completes():
    arm = S.FakeArm(speed_rad_s=30.0)
    server = _bare_server(arm)
    target = arm.q + np.radians([10.0, 5.0, -5.0, 0.0, 0.0])
    arm.fail_after, arm.fail_reads = 6, 2            # two reads in a row fail mid-move: retried
    res = server.run_path([target], None, 0.3)
    assert arm.fail_reads == 0
    assert not res["stopped"] and res["stop_reason"] == ""
    assert np.allclose(res["state"]["q"], target, atol=np.radians(0.5))


def test_a_persistent_bus_error_stops_the_move_without_raising():
    arm = S.FakeArm(speed_rad_s=30.0)
    server = _bare_server(arm)
    q0 = arm.q.copy()
    target = q0 + np.radians([30.0, 0.0, 0.0, 0.0, 0.0])
    arm.fail_after, arm.fail_reads = 6, 10**6        # the bus dies mid-move
    res = server.run_path([target], None, 1.0)
    assert res["stopped"] and res["stop_reason"].startswith("bus error: ConnectionError")
    assert res["ticks"] < 10 and res["wall_s"] < 0.5
    assert np.allclose(arm.goal_q, server.last_cmd[0])           # frozen at the last goal sent
    assert abs(arm.goal_q[0] - q0[0]) < np.radians(15.0)         # well short of the target
    assert set(res) >= {"state", "stopped", "stop_reason", "stop_joint", "final_err_rad", "offset_rad"}


def test_whole_episode_with_fake_llm(rig, monkeypatch, tmp_path):
    arm, address = rig
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    cfg = load_config("configs/so101_touch.yaml", [f"log.root={tmp_path}", f"robot.params.host={address[0]}",
                                                   f"robot.params.port={address[1]}",
                                                   "robot.params.reset_duration_s=0.2"])
    replies = ["MOVE ee_delta 20 0 0\nSTATUS OK", "GRIP close\nSTATUS OK", "MOVE ee_delta 0 0 -30\nSTATUS OK",
               "HOLD\nSTATUS DONE touching"]
    res = run_episode(cfg, llm=FakeLLM(replies, ttft_s=0.0))
    assert res.outcome == "done_unverified" and res.turns == 4
    from pathlib import Path
    sysp = (Path(res.run_dir) / "system_prompt.md").read_text()
    assert "VIEW line" in sysp and "same in every" not in sysp


# --------------------------------------------------------------------------- config

def test_so101_config_rules(monkeypatch):
    with pytest.raises(ValueError, match="system_v3"):
        load_config("configs/so101.yaml", ["prompt.system=system_v0"])
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    with pytest.raises(ValueError, match="rotation=yaw"):
        So101Robot({"tool_pitch_deg": 60}, rotation="yaw", conn=object())
