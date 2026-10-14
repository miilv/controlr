"""SO-101 joint zero calibration: dwell detection in a teleop recording, the offset fit on
synthetic touches with known offsets, and ``So101Robot``'s ``joint_offsets_deg`` (model angle =
measured + offset) against the real server code (``FakeArm``) over a real connection."""

from __future__ import annotations

import importlib.util
import json
import math
import threading
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import pytest

from controlr.robot.so101 import kinematics as K
from controlr.robot.so101 import server as S
from controlr.robot.so101 import zero_fit as Z
from controlr.robot.so101.client import So101Robot
from controlr.types import Action, ActionMode

TRUE_OFF = np.array([0.0, 4.0, -6.0, 7.0, 0.0])          # deg, [pan, lift, elbow, wrist, roll]
CAL = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 800, "range_max": 3200}
       for i, j in enumerate(K.MOTORS)}


def touch_poses(offsets_deg, n_reach=4, n_side=2, pitches=(35.0, 60.0, 85.0), seed=0, z=0.0):
    """MEASURED joint angles of closed-jaw touches on the table (z = 0) for an arm whose true zero
    offsets are ``offsets_deg``: IK in the true (model) frame, then measured = model - offset."""
    rng = np.random.default_rng(seed)
    out = []
    for x in np.linspace(0.16, 0.32, n_reach):
        for y in np.linspace(-0.08, 0.08, n_side):
            for pitch in pitches:
                p = np.array([x, y, z])
                az = math.atan2(y - K.PAN_AXIS_XY[1], x - K.PAN_AXIS_XY[0])
                R = K.tool_rotation(math.radians(pitch), az, math.radians(90.0))
                q, pe, ae = K.ik(p, R, np.radians([math.degrees(az), 0, 60, 60, -90]), iters=300)
                if pe < 2e-4 and ae < 0.02:
                    out.append(np.degrees(q) - offsets_deg + rng.normal(0, 0.05, 5))
    return np.array(out)


# --------------------------------------------------------------------------- fit

def test_conversion_matches_lerobot_degrees():
    assert Z.norm_to_deg(0.0, 800, 3200) == pytest.approx(0.0)
    assert Z.norm_to_deg(100.0, 800, 3200) == pytest.approx(1200 * 360 / 4095)
    assert Z.norm_to_deg(-100.0, 773, 3094) == pytest.approx(-(3094 - 773) / 2 * 360 / 4095)


def test_fit_recovers_known_offsets_from_synthetic_touches():
    Q = touch_poses(TRUE_OFF)
    assert len(Q) >= 12
    before = np.array([Z.tcp_mm(q)[2] for q in Q])
    assert np.max(np.abs(before)) > 5.0                     # the offsets matter: touches look off the table
    fit = Z.fit_offsets(Q)
    assert np.allclose(fit.offsets_deg, TRUE_OFF, atol=0.5), fit.offsets_deg
    assert np.max(np.abs(fit.residuals_mm)) < 1.0
    assert not fit.warnings and fit.cond < 1e3


def test_fit_with_base_height_needs_two_surface_heights():
    dz = 6.0          # the table is 6 mm below base_link: FK(q + offset).z + dz = surface height
    Q0 = touch_poses(TRUE_OFF, z=-dz / 1000)
    Q1 = touch_poses(TRUE_OFF, z=(60.0 - dz) / 1000, seed=1)            # on a 60 mm block
    fit = Z.fit_offsets(np.vstack([Q0, Q1]), tip_z_mm=np.r_[np.zeros(len(Q0)), np.full(len(Q1), 60.0)],
                        fit_dz=True)
    assert fit.dz_mm == pytest.approx(dz, abs=1.5)
    assert np.allclose(fit.offsets_deg, TRUE_OFF, atol=0.8)
    assert np.max(np.abs(fit.residuals_mm)) < 1.0
    fit = Z.fit_offsets(Q0, fit_dz=True)                                # one height: warned
    assert any("one height" in w for w in fit.warnings)


def test_too_few_or_similar_touches_warn():
    Q = touch_poses(TRUE_OFF, n_reach=1, n_side=1, pitches=(60.0,))
    fit = Z.fit_offsets(Q)
    assert any("touches for" in w for w in fit.warnings)
    assert any("prior" in w or "ill-conditioned" in w for w in fit.warnings)
    assert np.all(np.abs(fit.offsets_deg) < 15)             # the prior keeps it sane
    with pytest.raises(ValueError, match="vertical"):
        Z.fit_offsets(Q, joints=("shoulder_pan",))


# --------------------------------------------------------------------------- recordings

def _deg_to_norm(deg, c):
    raw = deg * 4095 / 360 + (c["range_min"] + c["range_max"]) / 2
    return (raw - c["range_min"]) / (c["range_max"] - c["range_min"]) * 200 - 100


def write_recording(d: Path, Q_deg, rest_first=True, dwell_s=1.0, move_s=1.0, fps=60):
    """A teleop recording (scripts/so101_teleop_record.py format) that visits each pose and
    dwells there, jaws closed; optionally starts in the folded rest pose."""
    d.mkdir(parents=True)
    (d / "calib_so101_follower.json").write_text(json.dumps(CAL))
    poses = ([np.array([1.0, -101.6, 98.1, 83.3, -89.5])] if rest_first else []) + list(Q_deg)
    lines = [json.dumps({"t": 0.0, "event": "start"})]
    t, prev = 0.0, poses[0]
    rng = np.random.default_rng(1)
    for q in poses:
        for k in range(int(move_s * fps)):                  # move from the previous pose
            s = (k + 1) / (move_s * fps)
            qq = prev + s * (q - prev)
            lines.append(_tick(t, qq, 1.4))
            t += 1 / fps
        for _ in range(int(dwell_s * fps)):
            lines.append(_tick(t, q + rng.normal(0, 0.03, 5), 1.4))
            t += 1 / fps
        prev = q
    lines.append(json.dumps({"t": t, "event": "stop"}))
    (d / "joints.jsonl").write_text("\n".join(lines) + "\n")


def _tick(t, q_deg, g):
    f = {j: round(float(_deg_to_norm(v, CAL[j])), 3) for j, v in zip(K.ARM_JOINTS, q_deg)}
    f["gripper"] = g
    return json.dumps({"t": round(t, 4), "leader": f, "follower": f})


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "so101_fit_offsets.py"
    spec = importlib.util.spec_from_file_location("so101_fit_offsets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dwells_found_and_fit_end_to_end(tmp_path, capsys):
    Q = touch_poses(TRUE_OFF, n_reach=3, n_side=2, pitches=(40.0, 80.0))
    rec = tmp_path / "touch"
    write_recording(rec, Q)
    t, q, g = Z.load_recording(rec)
    dwells, rej = Z.find_dwells(t, q, g)
    assert rej["rest_pose"] == 1 and len(dwells) == len(Q)
    assert np.allclose([d.q_deg for d in dwells], Q, atol=0.1)
    out = tmp_path / "off.json"
    assert _script().main([str(rec), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert np.allclose(res["joint_offsets_deg"], TRUE_OFF, atol=0.5)
    assert "rms" in capsys.readouterr().out


def test_no_touches_is_reported_not_raised(tmp_path, capsys):
    rec = tmp_path / "rest_only"
    write_recording(rec, [])                                # only the rest pose
    assert _script().main([str(rec), "--out", str(tmp_path / "x.json")]) == 2
    assert "no touch dwells" in capsys.readouterr().out
    assert not (tmp_path / "x.json").exists()


# --------------------------------------------------------------------------- client

@pytest.fixture
def rig():
    arm = S.FakeArm(speed_rad_s=30.0)
    server = S.Server(arm, S.Camera(None, 320, 180, 30), control_hz=200.0, max_joint_speed=20.0,
                      settle_s=0.02, settle_max_s=0.2)
    listener = Listener(("127.0.0.1", 0), authkey=b"test")
    t = threading.Thread(target=S.serve, args=(server, "", 0, b"test"),
                         kwargs={"listener": listener, "max_clients": 1, "handle_sigint": False}, daemon=True)
    t.start()
    yield arm, server, listener.address
    t.join(5)


def test_client_applies_joint_offsets_both_ways(rig, monkeypatch):
    arm, server, address = rig
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    off = np.radians(TRUE_OFF)
    robot = So101Robot({"host": address[0], "port": address[1], "reset_duration_s": 0.2, "connect_timeout_s": 2.0,
                        "joint_offsets_deg": TRUE_OFF.tolist()})
    assert np.allclose(robot.limits, server.lim + off[:, None])          # limits in the model frame
    obs = robot.reset({})
    # the server got model - offset; the client reports measured + offset, FK in the model frame
    assert np.allclose(arm.goal_q, robot.home_q - off, atol=1e-6)
    assert np.allclose(obs.state.q, arm.q + off, atol=1e-6)
    assert np.allclose(obs.state.tcp_pos, [0.24, 0.0, 0.10], atol=2e-3)
    assert np.allclose(obs.state.tcp_pos, K.fk(np.asarray(arm.q) + off)[:3, 3], atol=1e-6)
    cam = np.linalg.inv(obs.cameras["wrist"].T_cam_base)
    assert np.allclose(cam, K.fk(arm.q + off) @ robot.T_tcp_cam, atol=1e-6)
    tgt = tuple(obs.state.tcp_pos + [0.02, 0.01, -0.03]) + tuple(obs.state.tcp_rotvec)
    rep = robot.execute([Action(ActionMode.EE_DELTA, (0.02, 0.01, -0.03), raw="MOVE", tcp_target=tgt)])
    assert not rep.events and not rep.stopped                           # no false arrival WARN
    assert np.allclose(rep.state_after.tcp_pos, tgt[:3], atol=2e-3)
    assert np.allclose(rep.state_after.q, arm.q + off, atol=1e-6)
    assert robot.scene_record()["joint_offsets_deg"] == pytest.approx(TRUE_OFF.tolist())
    robot.close()


def test_client_rejects_a_wrong_offset_length(monkeypatch):
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    with pytest.raises(ValueError, match="5 values"):
        So101Robot({"joint_offsets_deg": [1.0, 2.0]}, conn=object())
