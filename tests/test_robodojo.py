"""RoboDojo backend without Isaac: the multi-arm grammar, the Cartesian envelope, the
flange <-> TCP frames, the shim (``shim/deploy.py``) against a fake ``TASK_ENV`` and a fake
cuRobo planner, and a whole episode through ``controlr robodojo-serve`` <-> shim over a real
``multiprocessing.connection`` with FakeLLM."""

from __future__ import annotations

import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest

from controlr.config import ActionConfig, SafetyConfig, load_config
from controlr.llm.fake import FakeLLM
from controlr.prompts.builder import build_system_prompt
from controlr.protocol.feedback import format_state
from controlr.protocol.grammar import grammar_spec, parse_reply
from controlr.robot.kinematics import matrix_to_rotvec, rotvec_to_matrix
from controlr.robot.robodojo import protocol as P
from controlr.robot.robodojo.client import (
    flange_to_tcp, matrix_to_quat, quat_to_matrix, tcp_to_flange,
)
from controlr.robot.robodojo.shim import deploy as shim
from controlr.robot.safety import CartesianEnvelope, make_envelope
from controlr.robot.spec import ARX_X5_TABLE_Z, arx_x5_dual_spec, ur3_cb3_spec
from controlr.types import Action, ActionMode, RobotState

SPEC = arx_x5_dual_spec()
FULL = ActionConfig(rotation="full")
TOP_DOWN = (0.5, -0.5, 0.5, 0.5)          # flange quaternion of a straight-down grasp (wxyz)


# --------------------------------------------------------------------------- grammar

def test_two_arm_line_is_one_simultaneous_step():
    r = parse_reply("MOVE L ee_delta 10 0 0 0 0 0 R ee_delta 0 0 -20 0 0 5 GRIP close\nSTATUS OK", FULL, SPEC)
    assert not r.errors and r.status.value == "OK"
    assert [(a.arm, a.step) for a in r.actions] == [("L", 0), ("R", 0)]
    assert r.actions[0].values[:3] == pytest.approx((0.01, 0.0, 0.0))
    assert r.actions[1].gripper == 0.0 and r.actions[1].values[5] == pytest.approx(np.radians(5))


def test_two_arm_grip_lines_and_errors():
    r = parse_reply("GRIP R close L open\nSTATUS OK", FULL, SPEC)
    assert [(a.arm, a.gripper) for a in r.actions] == [("R", 0.0), ("L", 0.08)]
    r = parse_reply("MOVE L GRIP close R ee_delta 0 0 5 0 0 0\nSTATUS OK", FULL, SPEC)
    assert [(a.arm, a.mode) for a in r.actions] == [("L", None), ("R", ActionMode.EE_DELTA)]
    r = parse_reply("MOVE ee_delta 10 0 0 0 0 0\nSTATUS OK", FULL, SPEC)
    assert not r.actions and "start each part with an arm" in r.errors[0]
    r = parse_reply("MOVE L ee_delta 1 0 0 0 0 0 L ee_delta 1 0 0 0 0 0\nSTATUS OK", FULL, SPEC)
    assert not r.actions and "at most once" in r.errors[0]


def test_two_arm_grammar_examples_parse():
    text = grammar_spec(FULL, SPEC)
    blocks = text.split("Examples")[1].split("\n\n")
    for b in blocks:
        lines = "\n".join(ln.strip() for ln in b.splitlines() if ln.startswith("    "))
        if lines:
            assert not parse_reply(lines, FULL, SPEC).errors, lines


# --------------------------------------------------------------------------- envelope

def _arm_state(x, y, z, grip_mm=80.0) -> RobotState:
    R = np.diag([1.0, -1.0, -1.0])        # tool z down
    return RobotState(t=0.0, q=np.zeros(6), tcp_pos=np.array([x, y, z]), tcp_rotvec=matrix_to_rotvec(R),
                      gripper_mm=grip_mm, gripper_closed=False)


def _dual(l_xyz=(-0.25, -0.1, 0.95), r_xyz=(0.25, -0.1, 0.95)) -> RobotState:
    arms = {"L": _arm_state(*l_xyz), "R": _arm_state(*r_xyz)}
    s = arms["L"]
    return RobotState(t=0.0, q=s.q, tcp_pos=s.tcp_pos, tcp_rotvec=s.tcp_rotvec, gripper_mm=80.0,
                      gripper_closed=False, arms=arms)


def test_make_envelope_picks_by_kinematics():
    assert isinstance(make_envelope(SPEC, SafetyConfig(), "full"), CartesianEnvelope)
    assert not isinstance(make_envelope(ur3_cb3_spec(), SafetyConfig()), CartesianEnvelope)


def test_cartesian_envelope_clamps_per_arm_and_sets_targets():
    env = make_envelope(SPEC, SafetyConfig(), "full")
    st = _dual()
    env.reset(st)
    acts = parse_reply("MOVE L ee_delta 0 0 -500 0 0 0 R ee_delta 300 0 0 0 0 0\nSTATUS OK", FULL, SPEC).actions
    out, ev = env.filter(acts, st)
    assert [a.arm for a in out] == ["L", "R"] and all(a.q_target is None for a in out)
    l, r = out
    # L: step limit (100 mm) first, then the table clearance would not bind at 850 mm
    assert l.tcp_target[2] == pytest.approx(0.85)
    assert r.tcp_target[0] == pytest.approx(0.35)
    assert {e.kind for e in ev} == {"step_limit"}
    # a second line chains from each arm's own target
    out2, ev2 = env.filter(parse_reply("MOVE L ee_delta 0 0 -100 0 0 0\nSTATUS OK", FULL, SPEC).actions,
                           _dual(l_xyz=(-0.25, -0.1, 0.80)))
    assert out2[0].tcp_target[2] == pytest.approx(ARX_X5_TABLE_Z + 0.005)
    assert any(e.kind == "table" for e in ev2)


def test_cartesian_envelope_rejects_joint_modes_and_unknown_arms():
    env = make_envelope(SPEC, SafetyConfig(), "full")
    st = _dual()
    env.reset(st)
    out, ev = env.filter([Action(ActionMode.JOINT_DELTA, (0.0,) * 6, arm="L"),
                          Action(ActionMode.EE_DELTA, (0.0,) * 6, arm="X")], st)
    assert out == [] and {e.kind for e in ev} == {"invalid"}


def test_state_lines_per_arm():
    cfg = load_config("configs/robodojo.yaml")
    lines = format_state(_dual(), cfg).splitlines()
    assert lines[0].startswith("STATE L: tcp x=-250 y=-100 z=950 mm roll=180 pitch=0 yaw=0 deg")
    assert lines[1].startswith("STATE R: tcp x=250")


def test_manual_renders_for_robodojo_and_v0_still_refuses_two_arms():
    cfg = load_config("configs/robodojo.yaml")
    text = build_system_prompt(cfg, SPEC)
    assert "MOVE L ee_delta dx dy dz droll dpitch dyaw" in text and "{" not in text
    cfg.prompt.system = "system_v0"
    with pytest.raises(KeyError):
        build_system_prompt(cfg, SPEC)


# --------------------------------------------------------------------------- frames

def test_top_down_flange_is_tool_z_down_and_grasp_point_below():
    pos, rv = flange_to_tcp([0.3, 0.0, 1.0, *TOP_DOWN])
    R = rotvec_to_matrix(rv)
    assert R[:, 2] == pytest.approx([0, 0, -1], abs=1e-9)
    assert pos == pytest.approx([0.3, 0.0, 1.0 - 0.1501])


def test_flange_tcp_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        f = np.concatenate([rng.uniform(-0.5, 0.5, 3), q * np.sign(q[0])])
        back = tcp_to_flange(*flange_to_tcp(f))
        assert back[:3] == pytest.approx(f[:3], abs=1e-9)
        assert quat_to_matrix(back[3:]) == pytest.approx(quat_to_matrix(f[3:]), abs=1e-9)
        assert matrix_to_quat(quat_to_matrix(f[3:])) == pytest.approx(f[3:], abs=1e-9)


# --------------------------------------------------------------------------- fake RoboDojo

class FakeTaskEnv:
    """A RoboDojo EvalEnv stand-in. A fake "joint" vector IS the flange pose (xyz + rotvec),
    so the fake planner is a straight line in that space. Success: the right gripper is
    closed and its flange has risen 50 mm above where it started (general_pickup-like)."""

    def __init__(self, step_lim=200):
        self.task_name = "general_pickup"
        self.env_seeds = [3]
        self.step_lim = step_lim
        self.take_action_cnt = [0]
        self.end_flag = [False]
        self.success = [True]
        self.num_envs = 1
        self.instruction = ["Pick up the blue car by 10 cm."]
        rv = matrix_to_rotvec(quat_to_matrix(TOP_DOWN))
        self.q = {"left": np.array([-0.3, -0.1, 1.05, *rv]), "right": np.array([0.3, -0.1, 1.05, *rv])}
        self.g = {"left": 1.0, "right": 1.0}
        self.z0 = self.q["right"][2]
        self.actions: list[dict] = []
        self.obs_calls = 0

    def _flange(self, arm):
        q = self.q[arm]
        return np.concatenate([q[:3], matrix_to_quat(rotvec_to_matrix(q[3:6]))])

    def get_obs(self):
        self.obs_calls += 1
        img = np.zeros((48, 64, 3), np.uint8)
        st = {}
        for arm in ("left", "right"):
            st[f"{arm}_arm_joint_state"] = self.q[arm].copy()
            st[f"{arm}_ee_pose"] = self._flange(arm)
            st[f"{arm}_ee_joint_state"] = [self.g[arm]]
        return {"vision": {c: {"color": img} for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "state": st, "instruction": self.instruction[0]}

    def take_action(self, action):
        assert set(action) == {"left_arm_joint_state", "left_ee_joint_state",
                               "right_arm_joint_state", "right_ee_joint_state"}
        if self.take_action_cnt[0] >= self.step_lim:
            return
        self.actions.append(action)
        self.take_action_cnt[0] += 1
        for arm in ("left", "right"):
            self.q[arm] = np.asarray(action[f"{arm}_arm_joint_state"], float)
            self.g[arm] = float(action[f"{arm}_ee_joint_state"][0])

    def is_episode_end(self):
        if not self.end_flag[0]:
            if self.g["right"] < 0.5 and self.q["right"][2] > self.z0 + 0.05:
                self.end_flag[0], self.success[0] = True, True
            elif self.take_action_cnt[0] >= self.step_lim or not self.success[0]:
                self.end_flag[0], self.success[0] = True, False
        return self.end_flag[0]


def fake_planner(arm, flange, q0):
    flange = np.asarray(flange, float)
    if flange[2] < 0.7:
        return {"status": "Fail"}
    qt = np.concatenate([flange[:3], matrix_to_rotvec(quat_to_matrix(flange[3:]))])
    return {"status": "Success", "position": np.linspace(q0, qt, 12)}


def test_shim_moves_arm_then_gripper_and_holds_the_other_arm():
    env = FakeTaskEnv()
    rig = shim.Rig(env, planner=fake_planner)
    rig.observe()
    target = env._flange("right").copy()
    target[1] += 0.2
    res = rig.execute([{"right": {"flange": target.tolist(), "grip": 0.0}}])
    n = res["env_steps"]
    assert res["steps"][0]["arms"]["right"]["status"] == "Success"
    # 0.2 m in fake joint space at 0.05 per step = 4 arm steps, then 4 gripper steps (1 -> 0)
    assert n == 8 and env.take_action_cnt[0] == 8
    assert env.q["right"][1] == pytest.approx(target[1])
    assert [a["right_ee_joint_state"][0] for a in env.actions] == [1.0] * 4 + [0.0] * 4
    assert all(np.allclose(a["left_arm_joint_state"], env.actions[0]["left_arm_joint_state"]) for a in env.actions)
    assert env.obs_calls >= 1 + n      # one get_obs per env step: RoboDojo's mp4


def test_shim_reports_unreachable_and_step_limit():
    env = FakeTaskEnv(step_lim=3)
    rig = shim.Rig(env, planner=fake_planner)
    rig.observe()
    low = env._flange("left").copy()
    low[2] = 0.5
    res = rig.execute([{"left": {"flange": low.tolist(), "grip": None}}])
    assert res["steps"][0]["arms"]["left"]["status"] == "Fail" and res["env_steps"] == 0
    far = env._flange("left").copy()
    far[0] += 0.5
    res = rig.execute([{"left": {"flange": far.tolist(), "grip": None}}])
    assert res["env_steps"] == 3 and res["ended"] and not res["success"]


def test_shim_finish_marks_unfinished_episode_failed():
    env = FakeTaskEnv()
    rig = shim.Rig(env, planner=fake_planner)
    rig.finish()
    assert env.end_flag == [True] and env.success == [False]


# --------------------------------------------------------------------------- end to end

PICK = [
    "MOVE R ee_delta 0 0 0 0 0 0 GRIP open\nSTATUS OK",
    "MOVE R ee_delta 0 0 -40 0 0 0 L ee_delta 0 -30 0 0 0 0\nSTATUS OK descending",
    "GRIP R close\nSTATUS OK",
    "MOVE R ee_delta 0 0 100 0 0 0\nSTATUS OK lifting",
    "HOLD\nSTATUS DONE",
]


def _run_pair(tmp_path, replies, env, max_turns=10):
    cfg = load_config("configs/robodojo.yaml", [f"log.root={tmp_path}", f"episode.max_turns={max_turns}"])
    listener = Listener(("127.0.0.1", 0), authkey=b"test")
    host, port = listener.address
    env_vars = {P.HOST_ENV: host, P.PORT_ENV: str(port), P.AUTHKEY_ENV: "test"}
    errors = []

    def sim():
        import os
        old = {k: os.environ.get(k) for k in env_vars}
        try:
            os.environ.update(env_vars)
            shim.eval_one_episode(env, None, planner=fake_planner)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    t = threading.Thread(target=sim, daemon=True)
    t.start()
    from controlr.robot.robodojo.serve import serve
    results = serve(cfg, llm_factory=lambda: FakeLLM(list(replies), ttft_s=0.0), max_episodes=1,
                    listener=listener)
    t.join(10)
    listener.close()
    assert not errors, errors
    return results[0]


def test_episode_through_serve_and_shim_ends_on_robodojo_success(tmp_path):
    env = FakeTaskEnv()
    res = _run_pair(tmp_path, PICK, env)
    assert env.success == [True] and env.end_flag == [True]
    assert res.outcome == "env_end" and res.success      # RoboDojo latched success while lifting
    assert res.turns == 4
    import json
    from pathlib import Path
    run = Path(res.run_dir)
    assert run.name.endswith("robodojo_general_pickup_L3")
    turns = [json.loads(x) for x in (run / "turns.jsonl").read_text().splitlines()]
    assert "STEPS:" in turns[0]["feedback"] and "STATE R:" in turns[0]["feedback"]
    assert turns[1]["actions"][1]["arm"] == "L" and turns[1]["executed"][1]["tcp_target"]
    setup = json.loads((run / "setup.json").read_text())
    assert setup["instruction"] == "Pick up the blue car by 10 cm." and setup["scene"]["layout_id"] == 3


def test_episode_that_gives_up_is_failed_in_robodojo(tmp_path):
    env = FakeTaskEnv()
    res = _run_pair(tmp_path, ["MOVE L ee_delta 0 0 0 0 0 0 Z\nSTATUS FAIL nope"], env)
    assert res.outcome == "fail" and not res.success
    assert env.end_flag == [True] and env.success == [False]
