"""Feedback text: deterministic formatting, event grouping, goal visibility, STATE line."""

from __future__ import annotations

import math

import numpy as np
import pytest

from controlr.config import Config
from controlr.protocol.feedback import (
    format_action,
    format_feedback,
    format_state,
    matrix_to_rpy,
    rotvec_to_matrix,
    rpy_to_matrix,
)
from controlr.protocol.grammar import parse_reply
from controlr.types import (
    Action,
    ActionMode,
    EventLevel,
    ExecReport,
    GoalReport,
    JointSpec,
    Observation,
    RobotSpec,
    RobotState,
    SafetyEvent,
)

SPEC = RobotSpec(
    name="UR3 test", joints=tuple(JointSpec(f"j{i}", -6.0, 6.0, 1.0) for i in range(6)),
    base_frame_doc="z up", tcp_doc="tcp", gripper_max_mm=85.0,
    workspace_lo=(0.1, -0.35, 0.0), workspace_hi=(0.45, 0.35, 0.4), home_q=(0.0,) * 6)


def rotvec_from_rpy(r, p, y) -> np.ndarray:
    R = rpy_to_matrix(r, p, y)
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-6:   # axis from the symmetric part
        B = (R + np.eye(3)) / 2
        ax = np.sqrt(np.clip(np.diag(B), 0, None))
        i = int(np.argmax(ax))
        ax = B[:, i] / ax[i]
        return ax / np.linalg.norm(ax) * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return ax * th


def state(pos=(0.312, -0.045, 0.088), yaw_deg=12.0, grip=42.0, closed=False, holding=False, q=None):
    return RobotState(t=0.0, q=np.zeros(6) if q is None else np.asarray(q, float),
                      tcp_pos=np.asarray(pos, float),
                      tcp_rotvec=rotvec_from_rpy(math.pi, 0.0, math.radians(yaw_deg)),
                      gripper_mm=grip, gripper_closed=closed, holding=holding)


def obs(st: RobotState) -> Observation:
    return Observation(t=0.0, images={}, cameras={}, state=st)


def test_rpy_roundtrip_and_rotvec():
    for rpy in [(0.1, -0.2, 0.3), (math.pi, 0.0, 0.5), (-2.0, 0.4, -3.0)]:
        R = rpy_to_matrix(*rpy)
        R2 = rpy_to_matrix(*matrix_to_rpy(R))
        assert np.allclose(R, R2, atol=1e-9)
        assert np.allclose(rotvec_to_matrix(rotvec_from_rpy(*rpy)), R, atol=1e-6)


def test_state_line_matches_contract_example():
    cfg = Config()
    cfg.action.rotation = "yaw"
    assert format_state(state(), cfg) == \
        "STATE: tcp x=312 y=-45 z=88 mm yaw=12 deg | grip 42 mm open | holding: no"


def test_state_line_variants():
    cfg = Config()   # rotation none: no orientation
    assert format_state(state(holding=None), cfg) == \
        "STATE: tcp x=312 y=-45 z=88 mm | grip 42 mm open | holding: unknown"
    cfg.action.rotation = "full"
    s = format_state(state(), cfg)
    assert "roll=180 pitch=0 yaw=12 deg" in s
    cfg = Config()
    cfg.action.mode = "joint_abs"
    s = format_state(state(q=[0.1, -1.5, 1.5, 0, 0, 0]), cfg)
    assert "q=[5.7 -85.9 85.9 0.0 0.0 0.0] deg" in s
    cfg = Config()
    cfg.action.pos_unit = "m"
    assert "x=0.312 y=-0.045 z=0.088 m" in format_state(state(), cfg)
    # no negative zero
    assert "-0" not in format_state(state(pos=(0.3, -0.0001, 0.1)), Config())


def _report(before, after, actions, events=(), dur=0.41, executed=None, stopped=False):
    return ExecReport(requested=list(actions), executed=list(actions if executed is None else executed),
                      events=list(events), state_before=before, state_after=after,
                      duration_s=dur, stopped=stopped)


def test_full_feedback_shape_and_order():
    cfg = Config()
    cfg.action.rotation = "yaw"
    cfg.episode.goal_feedback = "always"
    parsed = parse_reply("MOVE ee_delta 20 0 -10 0\nSTATUS OK", cfg.action, SPEC)
    b = state(pos=(0.2924, -0.0452, 0.0978))
    a = state(pos=(0.312, -0.045, 0.088))
    events = [
        SafetyEvent(EventLevel.INFO, "contact", "contact finger-object"),
        SafetyEvent(EventLevel.WARN, "joint_limit_near", "wrist_2 at 171 deg, soft limit 175 deg"),
        SafetyEvent(EventLevel.WARN, "clamp", "z target 3 mm -> 20 mm (table clearance)"),
        SafetyEvent(EventLevel.WARN, "clamp", "z target 3 mm -> 20 mm (table clearance)"),
    ]
    fb = format_feedback(7, parsed, _report(b, a, parsed.actions, events), GoalReport(False, 0.4, ""),
                         obs(a), cfg, SPEC)
    assert fb.splitlines() == [
        "TURN 7",
        "EXEC: MOVE ee_delta 20 0 -10 0 -> achieved dx=19.6 dy=0.2 dz=-9.8 mm dyaw=0.0 deg (0.41 s)",
        "CLAMP: z target 3 mm -> 20 mm (table clearance) (x2)",
        "WARN: wrist_2 at 171 deg, soft limit 175 deg",
        "EVENT: contact finger-object",
        "GOAL: not reached (progress 40%)",
        "STATE: tcp x=312 y=-45 z=88 mm yaw=12 deg | grip 42 mm open | holding: no",
    ]
    # deterministic
    assert fb == format_feedback(7, parsed, _report(b, a, parsed.actions, events),
                                 GoalReport(False, 0.4, ""), obs(a), cfg, SPEC)


def test_stop_event_and_stopped_flag():
    cfg = Config()
    st = state()
    parsed = parse_reply("MOVE ee_delta 0 0 -50\nSTATUS OK", cfg.action, SPEC)
    fb = format_feedback(3, parsed, _report(st, st, parsed.actions,
                                            [SafetyEvent(EventLevel.STOP, "collision", "collision with box wall")],
                                            stopped=True), None, obs(st), cfg)
    assert "STOP: collision with box wall" in fb and "execution stopped early" not in fb
    fb = format_feedback(3, parsed, _report(st, st, parsed.actions, stopped=True), None, obs(st), cfg)
    assert "STOP: execution stopped early" in fb
    fb = format_feedback(3, parsed, _report(st, st, parsed.actions, executed=[]), None, obs(st), cfg)
    assert "-> nothing executed" in fb


@pytest.mark.parametrize("mode,status,shown", [
    ("never", "DONE", False), ("on_done", "OK", False), ("on_done", "DONE", True),
    ("always", "OK", True),
])
def test_goal_visibility(mode, status, shown):
    cfg = Config()
    cfg.episode.goal_feedback = mode
    parsed = parse_reply(f"HOLD\nSTATUS {status}", cfg.action, SPEC)
    st = state()
    fb = format_feedback(2, parsed, _report(st, st, []), GoalReport(True, 1.0, "packet in box"), obs(st), cfg)
    assert ("GOAL: reached - packet in box" in fb) is shown


def test_parse_errors_and_grammar_reminder():
    cfg = Config()
    parsed = parse_reply("MOVE ee_delta 1 2\nI think", cfg.action, SPEC)
    st = state()
    fb = format_feedback(4, parsed, None, None, obs(st), cfg, SPEC)
    lines = fb.splitlines()
    assert lines[1] == "EXEC: HOLD"
    errs = [ln for ln in lines if ln.startswith("PARSE ERROR:")]
    assert len(errs) == 2
    assert any(ln.startswith("GRAMMAR: ") and "MOVE ee_delta dx dy dz" in ln for ln in lines)
    # without spec the reminder is still produced
    assert "GRAMMAR:" in format_feedback(4, parsed, None, None, obs(st), cfg)


def test_turn0_and_state_text_off():
    cfg = Config()
    st = state()
    assert format_feedback(0, None, None, None, obs(st), cfg) == "TURN 0\n" + format_state(st, cfg)
    cfg.observation.state_text = False
    assert format_feedback(0, None, None, None, obs(st), cfg) == "TURN 0"


def test_gripper_only_and_joint_achieved():
    cfg = Config()
    b, a = state(grip=85.0), state(grip=31.6, closed=True, holding=True)
    parsed = parse_reply("GRIP close\nSTATUS OK", cfg.action, SPEC)
    fb = format_feedback(5, parsed, _report(b, a, parsed.actions, dur=0.6), None, obs(a), cfg)
    assert "EXEC: GRIP close -> achieved grip 85->32 mm (0.60 s)" in fb
    cfg.action.mode = "joint_delta"
    parsed = parse_reply("MOVE joint_delta 10 0 0 0 0 0\nSTATUS OK", cfg.action, SPEC)
    b, a = state(q=np.zeros(6)), state(q=[math.radians(9.93), 0, 0, 0, 0, 0])
    fb = format_feedback(1, parsed, _report(b, a, parsed.actions), None, obs(a), cfg)
    assert "EXEC: MOVE joint_delta 10.0 0.0 0.0 0.0 0.0 0.0 -> achieved dq=[9.9 0.0 0.0 0.0 0.0 0.0] deg" in fb


def test_format_action_canonical():
    cfg = Config()
    a = Action(ActionMode.EE_DELTA, (0.02, 0.0, -0.0104), 0.085, raw="move ee_delta 20,0,-10.4 grip open")
    assert format_action(a, cfg.action) == "MOVE ee_delta 20 0 -10 GRIP open"
    assert format_action(Action(None, None, 0.0), cfg.action) == "GRIP close"
    assert format_action(Action(None, None, None), cfg.action) == "HOLD"
    cfg.action.gripper = "width"
    assert format_action(Action(None, None, 0.04), cfg.action) == "GRIP 40"
