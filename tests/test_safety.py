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
SPEC = ur3_cb3_spec(table_z=0.053)   # envelope logic tested against a raised 53 mm table


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
    if out:                                # clamped target may still be out of reach -> ik_fail
        assert abs(out[0].values[0] - lo_x) < 1e-9
    else:
        assert "ik_fail" in kinds


def test_unreachable_target_is_ik_fail_and_dropped():
    st = _state()
    # inside the box but tool-down at 400 mm height is not reachable
    a = Action(ActionMode.EE_ABS, (-0.35, -0.15, 0.40))
    out, ev = _env(max_step_m=1.0).filter([a], st)
    assert out == []
    e = [e for e in ev if e.kind == "ik_fail"][0]
    assert "not reachable" in e.message and "400 mm" in e.message


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
    assert len(out) == 2 and "ik_fail" in _kinds(ev)
    assert np.linalg.norm(_tcp(out[-1]) - (st.tcp_pos + [0.02, 0.02, 0.0])) < 2e-3


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
