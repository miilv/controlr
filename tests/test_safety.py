"""SafetyEnvelope: step limits, workspace, table clearance, joint limits,
IK failure, clamp vs reject, chaining, near-limit warnings."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from controlr.config import SafetyConfig
from controlr.robot.kinematics import UR3Kinematics, matrix_to_rpy, rotation_angle, rotvec_to_matrix
from controlr.robot.safety import SafetyEnvelope
from controlr.robot.spec import HOME_Q_TOPDOWN, ur3_cb3_spec
from controlr.types import Action, ActionMode, EventLevel, RobotState

KIN = UR3Kinematics()
# envelope logic tested against a raised 53 mm table, TCP-only clearance (finger geometry is
# tested separately below with the real pad sizes)
SPEC = ur3_cb3_spec(table_z=0.053, finger_pad=None)


def _state(q=HOME_Q_TOPDOWN) -> RobotState:
    q = np.asarray(q, float)
    pos, rv = KIN.fk(q)
    return RobotState(t=0.0, q=q, tcp_pos=pos, tcp_rotvec=rv, gripper_mm=85.0, gripper_closed=False)


def _env(**kw) -> SafetyEnvelope:
    return SafetyEnvelope(SPEC, dataclasses.replace(SafetyConfig(), **kw))


def _tcp(a: Action) -> np.ndarray:
    return KIN.fk(a.q_target)[0]


def _kinds(events):
    return [e.kind for e in events]


def test_small_ee_delta_passes_unchanged():
    st = _state()
    a = Action(ActionMode.EE_DELTA, (0.02, -0.01, -0.03), raw="MOVE ee_delta 20 -10 -30")
    out, ev = _env().filter([a], st)
    assert len(out) == 1 and not ev
    assert out[0].values == a.values and out[0].raw == a.raw and out[0].mode is a.mode
    assert np.linalg.norm(_tcp(out[0]) - (st.tcp_pos + a.values)) < 1e-3
    # orientation kept (rotation=none locks the gripper orientation)
    R = KIN.fk_matrix(out[0].q_target)[:3, :3]
    assert rotation_angle(R @ rotvec_to_matrix(st.tcp_rotvec).T) < np.deg2rad(0.5)


def test_step_limit_scales_translation():
    st = _state()
    a = Action(ActionMode.EE_DELTA, (0.15, 0.0, 0.0))
    out, ev = _env(max_step_m=0.05).filter([a], st)
    assert out and np.allclose(out[0].values, (0.05, 0.0, 0.0))
    e = [e for e in ev if e.kind == "step_limit"][0]
    assert e.level == EventLevel.WARN and "150 mm" in e.message and "50 mm" in e.message
    assert np.linalg.norm(_tcp(out[0]) - st.tcp_pos - [0.05, 0, 0]) < 1e-3


def test_reject_mode_drops_action():
    st = _state()
    a = Action(ActionMode.EE_DELTA, (0.15, 0.0, 0.0))
    out, ev = _env(max_step_m=0.05, clamp=False).filter([a], st)
    assert out == [] and "rejected" in ev[0].message


def test_table_clearance_clamps_z():
    st = _state()            # TCP z = 150 mm, table 53 mm, clearance 5 mm -> floor 58 mm
    a = Action(ActionMode.EE_DELTA, (0.0, 0.0, -0.10))
    out, ev = _env().filter([a], st)
    assert out
    assert abs(_tcp(out[0])[2] - 0.058) < 1e-3
    assert np.allclose(out[0].values, (0.0, 0.0, 0.058 - st.tcp_pos[2]), atol=1e-9)
    e = [e for e in ev if e.kind == "table"][0]
    assert "table" in e.message and "58 mm" in e.message and "50 mm" in e.message


def test_workspace_box_with_margin():
    st = _state()
    lo_x = SPEC.workspace_lo[0] + 0.02
    a = Action(ActionMode.EE_ABS, (-0.70, st.tcp_pos[1], st.tcp_pos[2]))
    out, ev = _env(workspace_margin_m=0.02, max_step_m=1.0).filter([a], st)
    kinds = _kinds(ev)
    assert "workspace" in kinds
    if out and "reach" not in kinds:       # clamped target may still be out of reach -> shortened
        assert abs(out[0].values[0] - lo_x) < 1e-9
    else:
        assert "ik_fail" in kinds or "reach" in kinds


def test_unreachable_target_is_shortened_to_the_reachable_part():
    """Review control-safety #4: an IK failure used to drop the whole move although most
    of it was feasible; now the TCP goes as far along the line as IK allows."""
    st = _state()
    # inside the box but tool-down at 400 mm height is not reachable
    a = Action(ActionMode.EE_ABS, (-0.35, -0.15, 0.40))
    out, ev = _env(max_step_m=1.0).filter([a], st)
    e = [e for e in ev if e.kind == "reach"][0]
    assert "not reachable" in e.message and "400 mm" in e.message and "% of the way" in e.message
    assert out and e.level == EventLevel.WARN
    p = _tcp(out[0])
    line = np.array([-0.35, -0.15, 0.40]) - st.tcp_pos
    off = (p - st.tcp_pos) - line * np.dot(p - st.tcp_pos, line) / np.dot(line, line)
    assert np.linalg.norm(off) < 2e-3 and p[2] > st.tcp_pos[2] + 0.05       # went up along the line
    assert np.allclose(out[0].values, p, atol=1e-3)


def test_completely_unreachable_target_is_skipped():
    st = _state()
    out, ev = _env(max_step_m=1.0).filter([Action(ActionMode.EE_ABS, (-0.35, -0.15, 0.40))], st)
    assert out                                    # (partial, see above) — now from the edge itself:
    q_edge = np.asarray(out[0].q_target)
    pos, rv = KIN.fk(q_edge)
    st2 = RobotState(t=0.0, q=q_edge, tcp_pos=pos, tcp_rotvec=rv, gripper_mm=85.0, gripper_closed=False)
    out2, ev2 = _env(max_step_m=1.0).filter([Action(ActionMode.EE_ABS, (-0.35, -0.15, 0.40))], st2)
    assert out2 == [] and "ik_fail" in _kinds(ev2)


def test_chunk_chains_from_previous_target():
    st = _state()
    acts = [Action(ActionMode.EE_DELTA, (0.03, 0.0, 0.0)),
            Action(ActionMode.EE_DELTA, (0.0, 0.03, 0.0)),
            Action(ActionMode.EE_DELTA, (0.0, 0.0, -0.03))]
    out, ev = _env().filter(acts, st)
    assert len(out) == 3 and not ev
    assert np.linalg.norm(_tcp(out[-1]) - (st.tcp_pos + [0.03, 0.03, -0.03])) < 2e-3


def test_ik_fail_in_chunk_keeps_chain_from_last_good_target():
    st = _state()
    acts = [Action(ActionMode.EE_DELTA, (0.02, 0.0, 0.0)),
            Action(ActionMode.EE_ABS, (-0.35, -0.15, 0.40)),       # unreachable
            Action(ActionMode.EE_DELTA, (0.0, 0.02, 0.0))]
    out, ev = _env(max_step_m=1.0).filter(acts, st)
    assert len(out) == 3 and "reach" in _kinds(ev)           # the middle one is shortened
    p_mid = _tcp(out[1])
    assert np.linalg.norm(_tcp(out[-1]) - (p_mid + [0.0, 0.02, 0.0])) < 2e-3


def test_yaw_rotation_and_rotation_step_limit():
    st = _state()
    R0 = rotvec_to_matrix(st.tcp_rotvec)
    yaw0 = matrix_to_rpy(R0)[2]
    out, ev = _env().filter([Action(ActionMode.EE_DELTA, (0.0, 0.0, 0.0, np.deg2rad(20)))], st)
    assert out and not ev
    R = KIN.fk_matrix(out[0].q_target)[:3, :3]
    assert abs(rotation_angle(R @ R0.T) - np.deg2rad(20)) < np.deg2rad(0.5)
    assert np.allclose(R[:, 2], R0[:, 2], atol=0.01)            # still pointing down
    # 60 deg -> clamped to 30 deg
    out, ev = _env().filter([Action(ActionMode.EE_DELTA, (0.0, 0.0, 0.0, np.deg2rad(60)))], st)
    assert out and "step_limit" in _kinds(ev)
    assert abs(out[0].values[3] - SafetyConfig().max_step_rad) < 1e-9
    # absolute yaw
    out, ev = _env().filter([Action(ActionMode.EE_ABS, tuple(st.tcp_pos) + (yaw0 + 0.2,))], st)
    assert out
    assert abs(matrix_to_rpy(KIN.fk_matrix(out[0].q_target)[:3, :3])[2] - (yaw0 + 0.2)) < np.deg2rad(0.5)


def test_full_rpy_delta():
    st = _state()
    d = (0.0, 0.0, 0.0, np.deg2rad(10), np.deg2rad(-5), np.deg2rad(15))
    out, ev = _env().filter([Action(ActionMode.EE_DELTA, d)], st)
    assert out and not ev
    from controlr.robot.kinematics import rpy_to_matrix
    R_exp = rpy_to_matrix(d[3:]) @ rotvec_to_matrix(st.tcp_rotvec)
    R = KIN.fk_matrix(out[0].q_target)[:3, :3]
    assert rotation_angle(R @ R_exp.T) < np.deg2rad(0.5)


def test_joint_delta_step_limit_scales_whole_move():
    st = _state()
    a = Action(ActionMode.JOINT_DELTA, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    out, ev = _env().filter([a], st)
    assert out and "step_limit" in _kinds(ev)
    assert np.isclose(out[0].values[5], SafetyConfig().max_step_rad)
    assert np.allclose(np.array(out[0].q_target) - st.q, out[0].values)
    assert "wrist_3" in ev[0].message


def test_joint_soft_limit_clamp_and_near_limit_warning():
    q = np.array(HOME_Q_TOPDOWN)
    q[5] = -2 * np.pi + 0.2                 # 0.2 rad from the hard limit, inside soft+near band
    st = _state(q)
    a = Action(ActionMode.JOINT_DELTA, (0.0, 0.0, 0.0, 0.0, 0.0, -0.3))
    out, ev = _env().filter([a], st)
    soft = -2 * np.pi + SafetyConfig().joint_margin_rad
    assert out and np.isclose(out[0].q_target[5], soft)
    kinds = _kinds(ev)
    assert "joint_limit" in kinds and "joint_limit_near" in kinds
    near = [e for e in ev if e.kind == "joint_limit_near"][0]
    assert "wrist_3" in near.message and "deg" in near.message and near.level == EventLevel.WARN


def test_joint_move_into_table_is_shortened():
    st = _state()
    # shoulder lift down a lot: TCP would dive toward the table
    a = Action(ActionMode.JOINT_DELTA, (0.0, 0.5, 0.0, 0.0, 0.0, 0.0))
    p_full = KIN.fk(st.q + np.array(a.values))[0]
    assert p_full[2] < 0.058 or not (np.all(p_full >= SPEC.workspace_lo) and np.all(p_full <= SPEC.workspace_hi))
    out, ev = _env().filter([a], st)
    assert "workspace" in _kinds(ev)
    if out:
        p = _tcp(out[0])
        assert p[2] >= 0.058 - 1e-6
        assert 0 < out[0].values[1] < 0.5


def test_joint_abs_and_wrong_arity():
    st = _state()
    tgt = tuple(np.array(HOME_Q_TOPDOWN) + 0.05)
    out, ev = _env().filter([Action(ActionMode.JOINT_ABS, tgt)], st)
    assert out and out[0].values == tgt and np.allclose(out[0].q_target, tgt)
    out, ev = _env().filter([Action(ActionMode.JOINT_ABS, (0.1, 0.2))], st)
    assert out == [] and ev[0].kind == "invalid"


def test_gripper_only_and_gripper_clamp():
    st = _state()
    out, ev = _env().filter([Action(None, None, gripper=0.2, raw="GRIP 200")], st)
    assert out and np.isclose(out[0].gripper, 0.085)
    assert np.allclose(out[0].q_target, st.q)
    assert ev[0].kind == "clamp" and "85 mm" in ev[0].message
    out, ev = _env().filter([Action(None, None, gripper=0.0)], st)
    assert out[0].gripper == 0.0 and not ev


def test_tcp_offset_from_spec_is_used():
    spec = ur3_cb3_spec(tcp_offset=(0, 0, 0.15, 0, 0, 0))
    env = SafetyEnvelope(spec, SafetyConfig())
    assert np.allclose(env.kin.tcp_offset, (0, 0, 0.15, 0, 0, 0))


def test_messages_are_llm_readable_units():
    st = _state()
    _, ev = _env(max_step_m=0.05).filter([Action(ActionMode.EE_DELTA, (0.0, 0.0, -0.2))], st)
    for e in ev:
        assert "mm" in e.message or "deg" in e.message
        assert "0.0" not in e.message.split("->")[0] or "mm" in e.message


@pytest.mark.parametrize("clamp", [True, False])
def test_no_events_for_safe_chunk(clamp):
    st = _state()
    acts = [Action(ActionMode.EE_DELTA, (0.01, 0.01, 0.0), gripper=0.04)]
    out, ev = _env(clamp=clamp).filter(acts, st)
    assert len(out) == 1 and ev == []


# ---------------------------------------------------------------------------
# review fixes (docs/reviews/2026-10-02-fixlog.md): fingertip clearance, straight paths, reference
# orientation, joint-mode TCP step / path checks
# ---------------------------------------------------------------------------

from controlr.robot.isaac.tasks import START_Q  # noqa: E402
from controlr.robot.safety import finger_drop  # noqa: E402

REAL = ur3_cb3_spec()            # real pad geometry, real table


def test_finger_drop_geometry():
    R_down = np.diag([1.0, -1.0, -1.0])                 # tool z = -base z
    assert finger_drop(REAL, R_down, 0.085) == pytest.approx(REAL.finger_pad[0])
    assert finger_drop(ur3_cb3_spec(finger_pad=None), R_down, 0.085) == 0.0
    R_tilt = KIN.fk_matrix(START_Q)[:3, :3]             # Isaac start: tilted tool + tilted jaw line
    assert finger_drop(REAL, R_tilt, 0.091) > finger_drop(REAL, R_tilt, 0.0) > 0.02


def test_table_clearance_protects_the_lowest_fingertip():
    """Review control-safety #2 / contracts #4: with the tilted tool fully open the
    lower pad sits far below the TCP; the TCP-only floor let it into the mat."""
    q = np.asarray(START_Q)
    pos, rv = KIN.fk(q)
    st = RobotState(t=0.0, q=q, tcp_pos=pos, tcp_rotvec=rv, gripper_mm=91.0, gripper_closed=False)
    env = SafetyEnvelope(REAL, SafetyConfig(max_step_m=1.0))
    env.reset(st)
    out, ev = env.filter([Action(ActionMode.EE_DELTA, (0.0, 0.0, -(pos[2] - REAL.table_z) - 0.05))], st)
    e = [e for e in ev if e.kind == "table"][0]
    assert "lowest fingertip" in e.message
    if out:
        T = KIN.fk_matrix(out[0].q_target)
        lowest = T[2, 3] - finger_drop(REAL, T[:3, :3], 0.091)
        assert lowest >= REAL.table_z + SafetyConfig().table_clearance_m - 1.5e-3


def test_ee_move_follows_a_straight_line_with_waypoints():
    """Review control-safety #3: joint-linear interpolation bent 100 mm moves by up to
    115 mm; the envelope now hands the backend IK waypoints every path_step_m."""
    st = _state()
    a = Action(ActionMode.EE_DELTA, (0.06, 0.04, -0.03))
    out, ev = _env().filter([a], st)
    assert out and out[0].q_path is not None
    path = [np.asarray(x) for x in out[0].q_path] + [np.asarray(out[0].q_target)]
    d = np.asarray(a.values)
    assert len(path) >= int(np.linalg.norm(d) / 0.005)
    for qk in path:
        p = KIN.fk(qk)[0] - st.tcp_pos
        assert np.linalg.norm(p - d * np.dot(p, d) / np.dot(d, d)) < 1.5e-3
    for qa, qb in zip([st.q] + path[:-1], path):      # no joint jumps between waypoints
        assert np.max(np.abs(qb - qa)) < 0.2


def test_reference_orientation_undoes_a_contact_tilt():
    """Review control-safety #8: rotation=none re-anchored to the measured pose, so a
    tilt from a collision was locked in for the rest of the episode."""
    st0 = _state()
    env = _env()
    env.reset(st0)
    q_t = np.asarray(st0.q) + np.array([0, 0, 0, 0.12, 0, 0])      # wrist knocked ~7 deg
    pos, rv = KIN.fk(q_t)
    st = RobotState(t=0.0, q=q_t, tcp_pos=pos, tcp_rotvec=rv, gripper_mm=85.0, gripper_closed=False)
    out, ev = env.filter([Action(ActionMode.EE_DELTA, (0.01, 0.0, 0.0))], st)
    assert out and "tilt" in _kinds(ev)
    R = KIN.fk_matrix(out[0].q_target)[:3, :3]
    assert rotation_angle(R @ KIN.fk_matrix(st0.q)[:3, :3].T) < np.deg2rad(0.6)
    assert np.allclose(out[0].values, (0.01, 0.0, 0.0))          # translation as asked


def test_joint_move_respects_tcp_step_limit():
    """Review control-safety #9: 30 deg of shoulder_pan moved the TCP ~230 mm."""
    st = _state()
    out, ev = _env(max_step_m=0.05).filter([Action(ActionMode.JOINT_DELTA, (0.4, 0, 0, 0, 0, 0))], st)
    assert out and "step_limit" in _kinds(ev)
    assert np.linalg.norm(_tcp(out[0]) - st.tcp_pos) <= 0.05 + 1e-4


def test_joint_path_is_checked_between_the_end_points():
    env = _env()
    st = _state()
    q = np.asarray(st.q)
    # all samples of a safe small move are inside
    assert env._path_inside(q, q + np.array([0.05, 0, 0, 0, 0, 0]), 0.085)
    # a path that dips under the floor and comes back is rejected although both ends are fine
    calls = []
    orig = env._inside
    env._inside = lambda qx, w: (calls.append(1), orig(qx, w) and not np.allclose(qx, q + 0.5 * 0.1))[1]
    assert not env._path_inside(q, q + 0.1, 0.085)


def test_elbow_limit_matches_the_urdf():
    names = [j.name for j in REAL.joints]
    e = REAL.joints[names.index("elbow")]
    assert e.lower == pytest.approx(-np.pi) and e.upper == pytest.approx(np.pi)


# ---------------------------------------------------------------------------
# rotation=yaw on the tilted Isaac start pose (docs/experiments/2026-10-02-rotation-yaw.md)
# ---------------------------------------------------------------------------

Q_TILT = (0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406)     # configs/sim_waffle.yaml start_q
SPEC_RIG = ur3_cb3_spec(table_z=-0.0095)


def _yaw_env(**kw) -> SafetyEnvelope:
    env = SafetyEnvelope(SPEC_RIG, dataclasses.replace(SafetyConfig(), **kw), rotation="yaw")
    env.reset(_state(Q_TILT))
    return env


def _rpy(q):
    return matrix_to_rpy(KIN.fk_matrix(q)[:3, :3])


def _yaw_move(env, q, d=(0, 0, 0), dyaw=0.0):
    return env.filter([Action(ActionMode.EE_DELTA, (*d, 0.0, 0.0, dyaw))], _state(q))


def test_yaw_delta_turns_the_heading_and_holds_the_tilt():
    env = _yaw_env()
    out, ev = _yaw_move(env, Q_TILT, (0.01, -0.01, -0.02), np.radians(20))
    assert out and not [e for e in ev if e.kind in ("reach", "ik_fail", "step_limit")], [e.message for e in ev]
    r0, r1 = _rpy(Q_TILT), _rpy(out[0].q_target)
    assert np.degrees(r1[2] - r0[2]) == pytest.approx(20, abs=0.5)
    np.testing.assert_allclose(r1[:2], r0[:2], atol=np.radians(0.5))
    assert np.linalg.norm(_tcp(out[0]) - KIN.fk(Q_TILT)[0] - [0.01, -0.01, -0.02]) < 1e-3
    # the straight-line waypoints keep the TCP on the line while turning
    for qw in out[0].q_path or ():
        p = KIN.fk(qw)[0] - KIN.fk(Q_TILT)[0]
        assert np.linalg.norm(np.cross(p, [0.01, -0.01, -0.02])) / np.linalg.norm([0.01, -0.01, -0.02]) < 1.5e-3


def test_yaw_is_kept_by_later_translations_and_a_contact_tilt_is_undone():
    env = _yaw_env()
    out, _ = _yaw_move(env, Q_TILT, dyaw=np.radians(25))
    q1 = np.asarray(out[0].q_target)
    out, ev = _yaw_move(env, q1, (0.0, 0.0, -0.02), 0.0)          # translation only: yaw stays
    assert np.degrees(_rpy(out[0].q_target)[2] - _rpy(Q_TILT)[2]) == pytest.approx(25, abs=0.5)
    # a measured pose tilted by a contact (4 deg about base x): the next move restores the
    # reference roll/pitch, keeps the turned heading, and says so
    T = KIN.fk_matrix(q1)
    c, s = np.cos(np.radians(4)), np.sin(np.radians(4))
    R_tilted = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ T[:3, :3]
    from controlr.robot.kinematics import matrix_to_rotvec
    q_t = KIN.ik(T[:3, 3], matrix_to_rotvec(R_tilted), q1)
    out, ev = _yaw_move(env, q_t, (0.0, 0.0, 0.01), 0.0)
    assert any(e.kind == "tilt" and "yaw is kept" in e.message for e in ev)
    r = _rpy(out[0].q_target)
    np.testing.assert_allclose(r[:2], _rpy(Q_TILT)[:2], atol=np.radians(0.6))
    assert np.degrees(r[2] - _rpy(Q_TILT)[2]) == pytest.approx(25, abs=1.0)


def test_yaw_step_limit_clamps_the_turn_only():
    env = _yaw_env()
    out, ev = _yaw_move(env, Q_TILT, dyaw=np.radians(45))
    e = [e for e in ev if e.kind == "step_limit"]
    assert e and "45.0 deg" in e[0].message and "30.0 deg" in e[0].message
    assert np.degrees(out[0].values[5]) == pytest.approx(30, abs=0.1)
    assert np.degrees(_rpy(out[0].q_target)[2] - _rpy(Q_TILT)[2]) == pytest.approx(30, abs=0.5)


def test_yaw_turn_at_the_reach_edge_names_the_reason_and_the_turned_angle():
    """At the high start pose a -30 deg turn straightens the elbow (edge of reach)."""
    env = _yaw_env()
    out, ev = _yaw_move(env, Q_TILT, dyaw=np.radians(-30))
    msg = " ".join(e.message for e in ev if e.kind in ("reach", "ik_fail"))
    assert "edge of its reach" in msg and "turned" in msg and "with yaw" in msg, msg


def test_yaw_mode_without_rotation_flag_is_unchanged():
    """rotation unset (None): 6 ee_delta values stay a free extrinsic rotation of the measured pose."""
    env = SafetyEnvelope(SPEC_RIG, SafetyConfig())
    env.reset(_state(Q_TILT))
    out, _ = _yaw_move(env, Q_TILT, dyaw=np.radians(10))
    assert np.degrees(_rpy(out[0].q_target)[2] - _rpy(Q_TILT)[2]) == pytest.approx(10, abs=0.5)


# live run 20261002T184337Z (sonnet-5-5 seed 0): a reach-clamped lift parked the arm at elbow
# 0.0 deg; from there 17 turns of moves were refused as "large joint swing" (any inward move needs
# a big first-step elbow change at the singularity) and the episode ran out of turns.
Q_LIFT_184337 = (0.51103, -0.94978, 0.63688, 0.97799, 1.9256, -2.46481)
Q_STUCK_184337 = (0.50874, -0.72156, 0.00037, 1.38219, 1.92205, -2.46824)


def test_reach_clamped_lift_stops_before_the_stretched_elbow():
    env = _yaw_env()
    out, ev = _yaw_move(env, Q_LIFT_184337, (0.0, 0.0, 0.06))
    assert out and any(e.kind == "reach" and "edge of its reach" in e.message for e in ev)
    assert abs(np.degrees(out[0].q_target[2])) >= 8.0


@pytest.mark.parametrize("d", [(0.03, 0.0, 0.0), (0.0, 0.0, -0.015), (0.0, 0.1, 0.0), (0.04, 0.04, -0.04)])
def test_arm_can_back_out_of_a_stretched_elbow(d):
    env = _yaw_env()
    out, ev = _yaw_move(env, Q_STUCK_184337, d)
    assert out, [e.message for e in ev]
    assert np.degrees(out[0].q_target[2]) > 8.0                      # bent back on the start branch
    assert np.linalg.norm(_tcp(out[0]) - KIN.fk(Q_STUCK_184337)[0] - d) < 1e-3
