"""Operating manual + planner prompt assembly."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from controlr.config import Config
from controlr.prompts.builder import (
    build_planner_messages,
    build_planner_prompt,
    build_system_prompt,
    cache_warning,
    estimate_tokens,
    fill,
    load_template,
)
from controlr.protocol.grammar import grammar_spec
from controlr.types import CameraInfo, JointSpec, RobotSpec

NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
SPEC = RobotSpec(
    name="UR3 CB3 + Robotiq 2F-85",
    joints=tuple(JointSpec(n, -2 * math.pi, 2 * math.pi, 1.0) for n in NAMES),
    base_frame_doc="Origin at the base mounting plate centre, z up; {braces} stay literal.",
    tcp_doc="between the fingertips, 180 mm from the flange.", gripper_max_mm=85.0,
    workspace_lo=(0.10, -0.35, 0.0), workspace_hi=(0.45, 0.35, 0.40),
    home_q=(0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0))


def top_down_cam() -> CameraInfo:
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([0.275, 0.0, 1.0])
    return CameraInfo("scene", 640, 480, np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]), T)


def test_default_manual_contents_and_size():
    cfg = Config()
    s = build_system_prompt(cfg, SPEC, "Put the waffle packet in the box.")
    for needle in ["Operating manual: UR3 CB3 + Robotiq 2F-85", "shoulder_pan", "| 2 | shoulder_lift |",
                   "{braces} stay literal", "180 mm from the flange", grammar_spec(cfg.action, SPEC),
                   "x 100..450, y -350..350, z 0..400 mm", "Put the waffle packet in the box.",
                   "STATE: tcp x=300 y=-50 z=150 mm", "Appendix B", "(none)"]:
        assert needle in s, needle
    assert not re.search(r"\{[a-z_]+\}", s.replace("{braces}", ""))
    assert "PARSE ERROR" not in s.split("Appendix B")[1]        # generated example parses
    assert estimate_tokens(s) >= 1200
    assert s == build_system_prompt(cfg, SPEC, "Put the waffle packet in the box.")   # byte-stable
    assert "(given in the first user turn)" in build_system_prompt(cfg, SPEC)


@pytest.mark.parametrize("kw", [
    dict(mode="joint_delta"), dict(mode="joint_abs", gripper="width", max_chunk=3),
    dict(mode="ee_abs", rotation="yaw"), dict(mode="ee_abs", rotation="full", ang_unit="rad", pos_unit="cm"),
    dict(rotation="full"), dict(pos_unit="m"),
])
def test_all_action_spaces_render(kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg.action, k, v)
    s = build_system_prompt(cfg, SPEC, "task")
    assert f"MOVE {cfg.action.mode}" in s
    assert "PARSE ERROR" not in s.split("Appendix B")[1]
    assert not re.search(r"\{[a-z_]+\}", s.replace("{braces}", ""))


def test_config_switches_change_the_manual():
    cfg = Config()
    cfg.prompt.extra_rules = ["Never push the box.", "The packet is fragile."]
    cfg.episode.goal_feedback = "never"
    cfg.observation.renderers = ["grid", "ee_marker", "diff"]
    cfg.safety.clamp = False
    s = build_system_prompt(cfg, SPEC, "t")
    assert "- Never push the box.\n- The packet is fragile." in s
    assert "GOAL:" not in s
    assert "Grid overlay" in s and "TCP marker" in s and "Diff image" in s
    assert "rejected (not executed)" in s
    cfg.episode.goal_feedback = "always"
    assert "GOAL: <whether the task is complete" in build_system_prompt(cfg, SPEC, "t")


def test_camera_axes_from_calibration():
    s = build_system_prompt(Config(), SPEC, "t", cameras={"scene": top_down_cam()})
    assert "- `scene`: +x points right, +y points up, +z points toward the camera in the image." in s


def test_fill_semantics():
    assert fill("a {x} b {y}", {"x": "{y}", "y": "2"}) == "a {y} b 2"   # single pass
    assert fill("json {\"k\": 1}", {}) == "json {\"k\": 1}"              # non-identifier braces untouched
    with pytest.raises(KeyError):
        fill("{missing}", {})
    with pytest.raises(FileNotFoundError):
        load_template("no_such_prompt")


def test_planner_prompt_and_messages():
    cfg = Config()
    sysp = build_planner_prompt(cfg, SPEC, cameras={"scene": top_down_cam()})
    assert sysp.startswith(build_system_prompt(cfg, SPEC, cameras={"scene": top_down_cam()}))
    assert "PLANNING CALL" in sysp and "at most 250 words" in sysp and "SCENE" in sysp
    assert not re.search(r"\{[a-z_]+\}", sysp.replace("{braces}", ""))

    class Img:
        jpeg = b"\xff\xd8\xff"
    msgs = build_planner_messages(cfg, SPEC, "Put the packet in the box.", [Img(), "data:image/jpeg;base64,AAA"],
                                  obs_text="TURN 0\nSTATE: tcp x=1 y=2 z=3 mm")
    assert [m["role"] for m in msgs] == ["system", "user"]
    parts = msgs[1]["content"]
    assert parts[0] == {"type": "text", "text": "TASK: Put the packet in the box."}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,/9j/"
    assert parts[2]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert parts[3]["text"].startswith("TURN 0")


def test_cache_warning():
    s = build_system_prompt(Config(), SPEC, "t")
    assert cache_warning("claude/claude-sonnet-5", s) is None
    assert cache_warning("claude/claude-opus-5-5", s) is None
    assert "4096" in cache_warning("no-think/claude/claude-haiku-4-5-20251001", s)
    assert cache_warning("gpt-5.6-luna-low", s) is None
