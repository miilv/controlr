"""Operating manual + planner prompt assembly."""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np
import pytest

from controlr.config import Config
from controlr.prompts.builder import (
    build_planner_prompt,
    build_system_prompt,
    cache_warning,
    estimate_tokens,
    fill,
    load_template,
)
from controlr.protocol.grammar import grammar_spec, parse_reply
from controlr.robot.spec import ur3_cb3_spec
from controlr.types import CameraInfo

# the real rig's spec (negative-x workspace); review contracts #2: a positive-x fixture
# hid examples that lay outside the real workspace
SPEC = dataclasses.replace(ur3_cb3_spec(),
                           base_frame_doc="Origin at the base mounting plate centre, z up; {braces} stay literal.")


def top_down_cam() -> CameraInfo:
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([-0.35, -0.10, 1.0])
    return CameraInfo("scene", 640, 480, np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]), T)


def test_default_manual_contents_and_size():
    cfg = Config()
    s = build_system_prompt(cfg, SPEC, "Put the waffle packet in the box.")
    for needle in ["Operating manual: UR3 CB3 + Robotiq 2F-85", "shoulder_pan", "| 2 | shoulder_lift |",
                   "{braces} stay literal", "180 mm from the tool flange", grammar_spec(cfg.action, SPEC),
                   "x -550..-150, y -420..220, z ", "Put the waffle packet in the box.",
                   "STATE: tcp x=-350 y=-100 z=140 mm", "Appendix B", "(none)",
                   "the lowest fingertip at least"]:
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
    assert ("- `scene`: +x points right (35 px per 100 mm), +y points up (35 px per 100 mm), "
            "+z points toward the camera in the 448-px image.") in s


def test_camera_axes_separate_depth_from_height_on_a_tilted_camera():
    """Review contracts #3: on the real 25-deg tilted D435, +y and +z both point "up";
    the manual must give the pixel lengths and say how to tell them apart."""
    from controlr.robot.mock import d435_camera
    s = build_system_prompt(Config(), SPEC, "t", cameras={"scene": d435_camera()})
    line = [ln for ln in s.splitlines() if ln.startswith("- `scene`:")][0]
    px = [int(v) for v in re.findall(r"\((\d+) px per 100 mm\)", line)]
    assert len(px) == 3 and px[1] > 1.5 * px[2]               # depth (y) moves much more than height (z)
    assert "a pure +y move and a pure +z move both shift the gripper up" in s
    cfg = Config()
    cfg.action.pos_unit = "cm"
    assert "px per 10.0 cm" in build_system_prompt(cfg, SPEC, "t", cameras={"scene": d435_camera()})


def test_fill_semantics():
    assert fill("a {x} b {y}", {"x": "{y}", "y": "2"}) == "a {y} b 2"   # single pass
    assert fill("json {\"k\": 1}", {}) == "json {\"k\": 1}"              # non-identifier braces untouched
    with pytest.raises(KeyError):
        fill("{missing}", {})
    with pytest.raises(FileNotFoundError):
        load_template("no_such_prompt")


def test_planner_prompt():
    cfg = Config()
    sysp = build_planner_prompt(cfg, SPEC, cameras={"scene": top_down_cam()})
    assert sysp.startswith(build_system_prompt(cfg, SPEC, cameras={"scene": top_down_cam()}))
    assert "PLANNING CALL" in sysp and "at most 250 words" in sysp and "SCENE" in sysp
    assert not re.search(r"\{[a-z_]+\}", sysp.replace("{braces}", ""))


def test_cache_warning():
    s = build_system_prompt(Config(), SPEC, "t")
    assert cache_warning("claude/claude-sonnet-5", s) is None
    assert cache_warning("claude/claude-opus-5-5", s) is None
    assert "4096" in cache_warning("no-think/claude/claude-haiku-4-5-20251001", s)
    assert cache_warning("gpt-5.6-luna-low", s) is None


def test_tool_orientation_doc_from_state0():
    """rotation=none: the manual states where the (fixed) tool points and the jaw line."""
    from controlr.prompts.builder import build_system_prompt
    from controlr.robot.kinematics import UR3Kinematics, matrix_to_rotvec
    from controlr.robot.spec import ur3_cb3_spec
    from controlr.types import RobotState

    cfg = Config()
    spec = ur3_cb3_spec()
    # tool tilted 35 deg below horizontal toward -x/-y (the Isaac waffle start)
    a = np.array([-0.70, -0.41, -0.58]); a /= np.linalg.norm(a)
    x = np.array([-0.27, 0.91, -0.33]); x -= a * (x @ a); x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(a, x), a])
    st = RobotState(t=0.0, q=np.zeros(6), tcp_pos=np.array([-0.3, -0.2, 0.2]), tcp_rotvec=matrix_to_rotvec(R),
                    gripper_mm=85.0, gripper_closed=False, holding=False)
    text = build_system_prompt(cfg, spec, state0=st)
    assert "Tool orientation" in text and "36 deg below horizontal, toward -x and -y" in text
    assert "(+0.70, +0.41, +0.58)" in text            # the side the gripper body is on
    assert "fingertips straddle the TCP" in text and "lowest fingertip is" in text
    assert "Tool orientation" not in build_system_prompt(cfg, spec)          # no state -> no line
    cfg.action.rotation = "yaw"
    yaw_text = build_system_prompt(cfg, spec, state0=st)
    tool = [ln for ln in yaw_text.splitlines() if ln.startswith("Tool orientation")][0]
    assert "you command the heading (yaw) only" in tool and "36 deg below horizontal" in tool
    assert "heading of the object's long side) ± 90 deg" in tool and "edge of the arm's reach" in tool
    # the paragraph is invariant under a turn about the vertical (randomised start yaw)
    c, s_ = np.cos(0.3), np.sin(0.3)
    st2 = RobotState(t=0.0, q=np.zeros(6), tcp_pos=st.tcp_pos,
                     tcp_rotvec=matrix_to_rotvec(np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ R),
                     gripper_mm=85.0, gripper_closed=False, holding=False)
    tool2 = [ln for ln in build_system_prompt(cfg, spec, state0=st2).splitlines()
             if ln.startswith("Tool orientation")][0]
    assert tool2 == tool


# ---------------------------------------------------------------------------
# review fixes
# ---------------------------------------------------------------------------

def _all_manual_positions(cfg, spec, text):
    """Every TCP position a manual shows: STATE lines, MOVE ee_abs lines (parsed)."""
    pts = []
    for m in re.finditer(r"tcp x=(-?[\d.]+) y=(-?[\d.]+) z=(-?[\d.]+)", text):
        pts.append(tuple(float(v) for v in m.groups()))
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line.startswith("MOVE ee_abs"):
            r = parse_reply(line + "\nSTATUS OK", cfg.action, spec)
            for a in r.actions:
                pts.append(tuple(v * 1000 for v in a.values[:3]))
    return pts


@pytest.mark.parametrize("mode", ["ee_delta", "ee_abs"])
def test_every_example_position_is_inside_the_workspace(mode):
    cfg = Config()
    cfg.action.mode = mode
    text = build_system_prompt(cfg, SPEC, "t")
    pts = _all_manual_positions(cfg, SPEC, text)
    assert len(pts) >= 4
    lo = np.asarray(SPEC.workspace_lo) * 1000
    hi = np.asarray(SPEC.workspace_hi) * 1000
    for p in pts:
        assert np.all(np.asarray(p) >= lo - 1) and np.all(np.asarray(p) <= hi + 1), p
    assert "table clearance)" not in text.split("Appendix B")[1]      # no blind descent into the clamp


def test_state_text_off_manual_never_mentions_a_state_line():
    """Review contracts #7: the manual said "no STATE line" and then showed one."""
    cfg = Config()
    cfg.observation.state_text = False
    text = build_system_prompt(cfg, SPEC, "t")
    assert "STATE" not in text, [ln for ln in text.splitlines() if "STATE" in ln]
    assert "does not report the measured robot state" in text
    assert "STATE: tcp" in build_system_prompt(Config(), SPEC, "t")


def test_fewshot_file_and_run_dir_land_in_the_manual(tmp_path):
    """Review contracts #1: prompt.fewshot was never read."""
    demo = tmp_path / "demo.md"
    demo.write_text("user:\n    TURN 0\nassistant:\n    MOVE ee_delta 10 0 0\n    STATUS OK demo-marker")
    cfg = Config()
    cfg.prompt.fewshot = str(demo)
    text = build_system_prompt(cfg, SPEC, "t")
    assert "Appendix C" in text and "demo-marker" in text
    run = tmp_path / "run"
    run.mkdir()
    (run / "messages.jsonl").write_text(
        '{"role": "system", "content": "manual"}\n'
        '{"role": "user", "content": ["TURN 0", {"image_sha": "abc"}, "IMAGES: scene"]}\n'
        '{"role": "assistant", "content": "MOVE ee_delta 0 0 -5\\nSTATUS OK run-marker"}\n')
    cfg.prompt.fewshot = str(run)
    text = build_system_prompt(cfg, SPEC, "t")
    assert "run-marker" in text and "<image>" in text and "manual" not in text.split("Appendix C")[1]
    cfg.prompt.fewshot = str(tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        build_system_prompt(cfg, SPEC, "t")


def test_yaw_manual_reads_headings_from_the_calibrated_camera():
    """rotation=yaw on the Isaac D435: the manual states the image-angle rule, a pure-turn
    example, the STATE yaw of the real tilted tool and an Appendix B turn that parses."""
    from controlr.prompts.builder import build_system_prompt
    from controlr.robot.kinematics import UR3Kinematics, matrix_to_rpy
    from controlr.robot.spec import ur3_cb3_spec
    from controlr.types import CameraInfo, RobotState

    cam_t_wc = np.array([
        [0.9999720414746357, -0.005669109656577374, 0.00487621418213748, -0.35399058583569004],
        [-0.007215535335001398, -0.9026929426481264, 0.4302248102365732, -0.44643379477503714],
        [0.0019627325028448834, -0.4302479662810364, -0.9027086103456388, 0.9535],
        [0.0, 0.0, 0.0, 1.0]])
    K = np.array([[609.28, 0, 337.81], [0, 608.13, 249.65], [0, 0, 1]])
    cams = {"scene": CameraInfo("scene", 640, 480, K, np.linalg.inv(cam_t_wc))}
    kin = UR3Kinematics()
    q = np.array([0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406])
    p, rv = kin.fk(q)
    st = RobotState(0.0, q, p, rv, 91.0, False, False)
    cfg = Config()
    cfg.action.rotation = "yaw"
    spec = ur3_cb3_spec(table_z=-0.0095)
    text = build_system_prompt(cfg, spec, cameras=cams, state0=st)
    assert "a positive dyaw turns the jaw line counter-clockwise in the image" in text
    assert "`MOVE ee_delta 0 0 0 20`" in text
    yaw = round(float(np.degrees(matrix_to_rpy(kin.fk_matrix(q)[:3, :3])[2])))
    assert f"yaw={yaw} deg | grip" in text                       # example STATE with the rig's heading
    assert f"yaw={yaw + 15} deg" in text and "MOVE ee_delta 30 20 0 15 GRIP open" in text
