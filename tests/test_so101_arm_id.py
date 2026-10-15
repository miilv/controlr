"""arm_id: which SO-101 arm is on which port, from encoder readings against the calibration
files. Real numbers from the rig (2026-10-15): port A = follower, port B = leader, and each arm's
motors held the OTHER arm's calibration (a swapped-port lerobot connect had written it)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from controlr.robot.so101 import server as S

# loaded by path, as the rig does (no controlr imports allowed in it)
_spec = importlib.util.spec_from_file_location(
    "so101_arm_id_t", Path(__file__).parents[1] / "controlr/robot/so101/arm_id.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def _calib(rows):
    keys = ("homing_offset", "range_min", "range_max")
    return {m: {"id": i, "drive_mode": 0, **dict(zip(keys, r))} for i, (m, r) in enumerate(zip(A.MOTORS, rows), 1)}


FOLLOWER = _calib([(-820, 651, 3321), (-1907, 773, 3094), (1842, 820, 3084),
                   (1393, 791, 3263), (-1411, 77, 3916), (1188, 2044, 3452)])
LEADER = _calib([(1653, 752, 3397), (-1650, 797, 3296), (-1399, 885, 3101),
                 (1735, 822, 3130), (-1283, 148, 3945), (364, 2032, 3299)])
CALIBS = {"follower_v1": FOLLOWER, "leader_v1": LEADER}

PORT_A = ({"shoulder_pan": 3722, "shoulder_lift": 551, "elbow_flex": 2184, "wrist_flex": 2676,
           "wrist_roll": 848, "gripper": 2893},
          {"shoulder_pan": 1653, "shoulder_lift": -1650, "elbow_flex": -1399, "wrist_flex": 1735,
           "wrist_roll": -1283, "gripper": 364})
PORT_B = ({"shoulder_pan": 370, "shoulder_lift": 1054, "elbow_flex": 3946, "wrist_flex": 3202,
           "wrist_roll": 1208, "gripper": 1220},
          {"shoulder_pan": -820, "shoulder_lift": -1907, "elbow_flex": 1842, "wrist_flex": 1393,
           "wrist_roll": -1411, "gripper": 1188})


def test_the_right_file_fits_and_the_wrong_one_does_not():
    assert A.score(*PORT_A, FOLLOWER).out == 0
    assert A.score(*PORT_B, LEADER).out == 0
    wrong_a, wrong_b = A.score(*PORT_A, LEADER), A.score(*PORT_B, FOLLOWER)
    assert wrong_a.out == 2 and {m for m, j in wrong_a.joints.items() if j.out} == {"shoulder_pan", "shoulder_lift"}
    assert wrong_b.out == 3 and {m for m, j in wrong_b.joints.items() if j.out} == {
        "shoulder_pan", "elbow_flex", "gripper"}
    # pred = (raw + motor offset - file offset) mod 4096
    assert A.score(*PORT_A, FOLLOWER).joints["shoulder_pan"].pred == (3722 + 1653 + 820) % 4096


def test_a_joint_resting_on_its_stop_is_tolerated():
    # leader B: shoulder_lift reads 797 = range_min exactly (rest pose on the stop); 30 ticks past it is still in
    raw, off = PORT_B
    s = A.score({**raw, "shoulder_lift": raw["shoulder_lift"] - 30}, off, LEADER)
    assert s.out == 0 and s.joints["shoulder_lift"].pred == 767
    assert A.score({**raw, "shoulder_lift": raw["shoulder_lift"] - 30}, off, LEADER, tol=0).out == 1


def test_score_accepts_lerobot_calibration_objects():
    class Cal:
        def __init__(self, d):
            self.__dict__.update(d)

    assert A.score(*PORT_A, {m: Cal(c) for m, c in FOLLOWER.items()}).out == 0


def test_assign_real_numbers_and_swapped_ports():
    assert A.assign({"/dev/ttyUSB0": PORT_A, "/dev/ttyUSB1": PORT_B}, CALIBS) == {
        "follower_v1": "/dev/ttyUSB0", "leader_v1": "/dev/ttyUSB1"}
    assert A.assign({"/dev/ttyUSB0": PORT_B, "/dev/ttyUSB1": PORT_A}, CALIBS) == {
        "follower_v1": "/dev/ttyUSB1", "leader_v1": "/dev/ttyUSB0"}
    # the server looks for the follower only; the leader's port is left out
    assert A.assign({"/dev/ttyUSB0": PORT_B, "/dev/ttyUSB1": PORT_A}, {"follower_v1": FOLLOWER}) == {
        "follower_v1": "/dev/ttyUSB1"}


def test_motor_offsets_say_what_the_motors_hold_not_which_arm():
    # the trap: the follower's motors held the leader's calibration (lerobot's is_calibrated would pass)
    assert A.holder(PORT_A[1], CALIBS) == "leader_v1"
    assert A.holder(PORT_B[1], CALIBS) == "follower_v1"
    assert A.holder({m: 0 for m in A.MOTORS}, CALIBS) is None
    text = A.format_scores({"/dev/ttyUSB0": PORT_A}, CALIBS)
    assert "as follower_v1: 0/6" in text and "as leader_v1: 2/6" in text and "hold: leader_v1's" in text


def test_identical_arms_are_ambiguous():
    with pytest.raises(A.ArmIdError, match="ambiguous"):
        A.assign({"/dev/ttyUSB0": PORT_A, "/dev/ttyUSB1": PORT_A}, {"follower_v1": FOLLOWER})
    # two calibrations that read the same: both assignments fit
    with pytest.raises(A.ArmIdError, match="ambiguous"):
        A.assign({"/dev/ttyUSB0": PORT_A, "/dev/ttyUSB1": PORT_A}, {"a": FOLLOWER, "b": FOLLOWER})


def test_an_arm_far_outside_its_range_is_refused():
    raw, off = PORT_A
    far = ({**raw, "elbow_flex": (raw["elbow_flex"] + 1500) % 4096}, off)    # elbow 130 deg past
    assert A.score(*far, FOLLOWER).out == 1
    with pytest.raises(A.ArmIdError, match="still has 1 joint"):
        A.assign({"/dev/ttyUSB0": far, "/dev/ttyUSB1": PORT_B}, CALIBS)


# a third arm, calibrated on its own (its offsets put both files' ranges in the wrong place)
UNKNOWN = ({m: 2047 for m in A.MOTORS},
           {"shoulder_pan": -2047, "shoulder_lift": 1900, "elbow_flex": 0, "wrist_flex": -1800,
            "wrist_roll": 0, "gripper": -1500})


def test_a_third_unknown_arm():
    assert A.score(*UNKNOWN, FOLLOWER).out > 0 and A.score(*UNKNOWN, LEADER).out > 0
    ports = {"/dev/ttyUSB0": PORT_B, "/dev/ttyUSB1": UNKNOWN, "/dev/ttyUSB2": PORT_A}
    assert A.assign(ports, CALIBS) == {"follower_v1": "/dev/ttyUSB2", "leader_v1": "/dev/ttyUSB0"}
    # the unknown arm instead of the leader: refused, not guessed
    with pytest.raises(A.ArmIdError, match="no assignment fits"):
        A.assign({"/dev/ttyUSB0": UNKNOWN, "/dev/ttyUSB1": PORT_A}, CALIBS)


def test_too_few_ports():
    with pytest.raises(A.ArmIdError, match="1 readable port"):
        A.assign({"/dev/ttyUSB0": PORT_A}, CALIBS)


def test_find_scans_ports_and_skips_dead_ones(monkeypatch):
    data = {"/dev/ttyUSB0": PORT_B, "/dev/ttyUSB1": PORT_A}

    def read_port(p):
        if p not in data:
            raise ConnectionError("no status packet")
        return data[p]

    monkeypatch.setattr(A, "read_port", read_port)
    monkeypatch.setattr(A, "candidate_ports", lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"])
    logs = []
    mapping, readings = A.find(CALIBS, log=logs.append)
    assert mapping == {"follower_v1": "/dev/ttyUSB1", "leader_v1": "/dev/ttyUSB0"}
    assert set(readings) == set(data)
    assert any("ttyUSB2: no answer" in s for s in logs) and "follower_v1 = /dev/ttyUSB1" in logs[-1]


def test_pure_part_imports_without_lerobot_and_server_loads_it():
    src = (Path(__file__).parents[1] / "controlr/robot/so101/arm_id.py").read_text()
    assert "import controlr" not in src and "from controlr" not in src
    assert S.arm_id.assign is not None and S.arm_id.MOTORS == S.MOTORS


def test_server_cli_defaults_to_auto():
    import inspect
    assert 'default="auto"' in inspect.getsource(S.main)
    sh = (Path(__file__).parents[1] / "scripts/so101_server.sh").read_text()
    assert 'PORT="${SO101_PORT:-auto}"' in sh
