"""Reply grammar: spec rendering, tolerant parsing, completeness predicate."""

from __future__ import annotations

import math
import re

import pytest

from controlr.config import ActionConfig
from controlr.protocol.grammar import grammar_reminder, grammar_spec, is_complete, parse_reply
from controlr.types import ActionMode, JointSpec, RobotSpec, Status

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def make_spec() -> RobotSpec:
    return RobotSpec(
        name="UR3 test", joints=tuple(JointSpec(n, -2 * math.pi, 2 * math.pi, 1.0) for n in JOINTS),
        base_frame_doc="z up", tcp_doc="between fingertips", gripper_max_mm=85.0,
        workspace_lo=(0.1, -0.35, 0.0), workspace_hi=(0.45, 0.35, 0.4),
        home_q=(0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0))


SPEC = make_spec()


def acfg(**kw) -> ActionConfig:
    return ActionConfig(**kw)


def test_basic_ee_delta_mm_to_si():
    r = parse_reply("MOVE ee_delta 20 0 -10\nSTATUS OK going down", acfg(), SPEC)
    assert r.errors == [] and r.complete and r.status == Status.OK
    assert r.note == "going down"
    (a,) = r.actions
    assert a.mode == ActionMode.EE_DELTA
    assert a.values == pytest.approx((0.02, 0.0, -0.01))
    assert a.gripper is None and a.raw.startswith("MOVE")


def test_markdown_lowercase_prose_tolerated():
    text = ("Sure! I'll move a bit.\n```text\n- move EE_DELTA 20, 0, -10\n```\n"
            "**STATUS:** ok approaching\nThanks.")
    r = parse_reply(text, acfg(), SPEC)
    assert r.errors == []
    assert r.status == Status.OK and r.note == "approaching"
    assert r.actions[0].values == pytest.approx((0.02, 0.0, -0.01))


def test_wrong_arity_and_non_numeric_are_errors_not_actions():
    r = parse_reply("MOVE ee_delta 20 0\nSTATUS OK", acfg(), SPEC)
    assert r.actions == [] and len(r.errors) == 1 and "needs 3 numbers" in r.errors[0]
    r = parse_reply("MOVE ee_delta 20 zero 0\nSTATUS OK", acfg(), SPEC)
    assert r.actions == [] and "not a number" in r.errors[0]


def test_mode_mismatch_and_missing_mode():
    r = parse_reply("MOVE joint_delta 1 2 3 4 5 6\nSTATUS OK", acfg(), SPEC)
    assert r.actions == [] and "use MOVE ee_delta" in r.errors[0]
    r = parse_reply("MOVE 5 5 5\nSTATUS OK", acfg(), SPEC)   # omitted mode = configured mode
    assert r.errors == [] and r.actions[0].values == pytest.approx((0.005, 0.005, 0.005))


def test_chunk_overflow_keeps_first_max_chunk():
    text = "MOVE ee_delta 1 0 0\nMOVE ee_delta 2 0 0\nMOVE ee_delta 3 0 0\nSTATUS OK"
    r = parse_reply(text, acfg(max_chunk=2), SPEC)
    assert [a.values[0] for a in r.actions] == pytest.approx([0.001, 0.002])
    assert any("max is 2" in e for e in r.errors)
    r1 = parse_reply(text, acfg(max_chunk=1), SPEC)
    assert len(r1.actions) == 1 and "dropped 2" in r1.errors[0]


@pytest.mark.parametrize("unit,val,expect", [("mm", "20", 0.02), ("cm", "2", 0.02), ("m", "0.02", 0.02)])
def test_position_units(unit, val, expect):
    r = parse_reply(f"MOVE ee_delta {val} 0 0\nSTATUS OK", acfg(pos_unit=unit), SPEC)
    assert r.actions[0].values[0] == pytest.approx(expect)


def test_unit_suffix_and_key_value_tokens():
    r = parse_reply("MOVE ee_delta dx=20mm dy=0 dz=-5\nSTATUS OK", acfg(), SPEC)
    assert r.errors == [] and r.actions[0].values == pytest.approx((0.02, 0.0, -0.005))
    r = parse_reply("MOVE ee_delta 2cm 0 0\nSTATUS OK", acfg(), SPEC)
    assert r.actions == [] and "use mm" in r.errors[0]


def test_rotation_layouts():
    d = parse_reply("MOVE ee_delta 0 0 0 30\nSTATUS OK", acfg(rotation="yaw"), SPEC)
    assert d.actions[0].values == pytest.approx((0, 0, 0, 0, 0, math.radians(30)))
    a = parse_reply("MOVE ee_abs 300 0 100 -45\nSTATUS OK", acfg(mode="ee_abs", rotation="yaw"), SPEC)
    assert a.actions[0].values == pytest.approx((0.3, 0, 0.1, math.pi, 0, math.radians(-45)))
    f = parse_reply("MOVE ee_abs 300 0 100 3.1416 0 0.5\nSTATUS OK",
                    acfg(mode="ee_abs", rotation="full", ang_unit="rad"), SPEC)
    assert f.actions[0].values == pytest.approx((0.3, 0, 0.1, 3.1416, 0, 0.5))
    bad = parse_reply("MOVE ee_delta 0 0 0\nSTATUS OK", acfg(rotation="yaw"), SPEC)
    assert bad.actions == [] and "needs 4 numbers" in bad.errors[0]


def test_joint_modes_one_value_per_joint_deg_to_rad():
    r = parse_reply("MOVE joint_delta 10 0 0 0 0 -10\nSTATUS OK", acfg(mode="joint_delta"), SPEC)
    v = r.actions[0].values
    assert len(v) == 6 and v[0] == pytest.approx(math.radians(10)) and v[5] == pytest.approx(-math.radians(10))
    r = parse_reply("MOVE joint_abs 0 -90 90 -90 -90\nSTATUS OK", acfg(mode="joint_abs"), SPEC)
    assert r.actions == [] and "needs 6 numbers" in r.errors[0]


def test_gripper_binary_and_width():
    r = parse_reply("GRIP open\nSTATUS OK", acfg(), SPEC)
    assert r.actions[0].mode is None and r.actions[0].gripper == pytest.approx(0.085)
    r = parse_reply("MOVE ee_delta 0 0 -5 GRIP close\nSTATUS OK", acfg(), SPEC)
    assert r.actions[0].gripper == 0.0 and r.actions[0].values == pytest.approx((0, 0, -0.005))
    r = parse_reply("GRIP 40\nSTATUS OK", acfg(), SPEC)
    assert r.actions == [] and "binary" in r.errors[0]
    r = parse_reply("GRIP 40\nSTATUS OK", acfg(gripper="width"), SPEC)
    assert r.errors == [] and r.actions[0].gripper == pytest.approx(0.04)
    r = parse_reply("GRIP -1\nSTATUS OK", acfg(gripper="width"), SPEC)
    assert r.actions == [] and ">= 0" in r.errors[0]


def test_hold_is_no_action():
    r = parse_reply("HOLD\nSTATUS STUCK need a better view", acfg(), SPEC)
    assert r.actions == [] and r.errors == [] and r.status == Status.STUCK


def test_missing_status_and_lines_after_status():
    r = parse_reply("MOVE ee_delta 1 2 3", acfg(), SPEC)
    assert not r.complete and r.status is None and len(r.actions) == 1
    assert any("missing STATUS" in e for e in r.errors)
    r = parse_reply("STATUS DONE\nMOVE ee_delta 1 2 3", acfg(), SPEC)
    assert r.actions == [] and r.status == Status.DONE and any("after STATUS" in e for e in r.errors)
    r = parse_reply("STATUS MAYBE\nSTATUS FAIL", acfg(), SPEC)
    assert r.status == Status.FAIL and any("STATUS needs" in e for e in r.errors)


def test_parse_never_raises_on_garbage():
    for t in ["", "\n\n", "MOVE", "GRIP", "STATUS", "MOVE ee_delta nan inf 1e999\nSTATUS OK", "```"]:
        parse_reply(t, acfg(), SPEC)


@pytest.mark.parametrize("text,done", [
    ("STATUS O", False), ("STATUS DON", False), ("STATUS", False), ("MOVE ee_delta 1 2 3\n", False),
    ("STATUS OK", True), ("STATUS OK\n", True), ("MOVE ee_delta 1 0 0\nSTATUS DONE all set", True),
    ("status done", True), ("**STATUS:** OK", True), ("- STATUS LIMIT\n", True),
    ("the STATUS OK is next", False), ("STATUS OKAY", False), ("STATUS STUCK.", True),
])
def test_is_complete(text, done):
    assert is_complete(text) is done


def _example_replies(spec_text: str) -> list[str]:
    """Indented example blocks after 'Examples:' in the grammar spec."""
    body = spec_text.split("Examples:", 1)[1]
    blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
    return ["\n".join(ln.strip() for ln in b.splitlines()) for b in blocks]


CONFIGS = [
    dict(), dict(rotation="yaw"), dict(rotation="full"), dict(mode="ee_abs"),
    dict(mode="ee_abs", rotation="yaw"), dict(mode="ee_abs", rotation="full", ang_unit="rad"),
    dict(mode="joint_delta"), dict(mode="joint_abs", ang_unit="rad"), dict(pos_unit="cm"),
    dict(pos_unit="m", max_chunk=3), dict(gripper="width", max_chunk=2),
]


@pytest.mark.parametrize("kw", CONFIGS)
def test_spec_examples_parse_cleanly(kw):
    cfg = acfg(**kw)
    text = grammar_spec(cfg, SPEC)
    assert cfg.mode in text and cfg.pos_unit in text
    examples = _example_replies(text)
    assert len(examples) >= 3
    for ex in examples:
        r = parse_reply(ex, cfg, SPEC)
        assert r.errors == [] and r.complete, (ex, r.errors)


def test_spec_mentions_configuration():
    t = grammar_spec(acfg(mode="joint_delta", max_chunk=3, gripper="width"), SPEC)
    assert "dshoulder_pan" in t and "At most 3" in t and "<width mm>" in t
    assert grammar_spec(acfg(), SPEC) == grammar_spec(acfg(), SPEC)
    rem = grammar_reminder(acfg(), SPEC)
    assert "\n" not in rem and "MOVE ee_delta dx dy dz" in rem
    assert "<one value per joint>" in grammar_reminder(acfg(mode="joint_abs"), None)
