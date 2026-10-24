"""dojo.runtime: firewall, instruction-only skill choice, thread bridge, belief, whole episodes on the
kinematic env through the proxy_eval policy interface. CPU only (no Isaac, no perception service)."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from dojo.runtime.belief import KinematicBelief
from dojo.runtime.bridge import Bridge
from dojo.runtime.estimate import est_objects, estimate_to_layout, flagged, objects_from_gt_rows
from dojo.runtime.firewall import DEV_GT_KEY, FirewallError, check_obs, sanitize_info
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.runtime.supervisor import SKILLS, parse_instruction
from dojo.skills.env import KinematicEnv
from dojo.skills.kinematics import X5Kin
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]
RUNTIME = Path(__file__).resolve().parents[2] / "dojo" / "runtime"


def _env(positions=POS, half=0.015, step_lim=550, instruction=STACK):
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [half] * 3} for i, p in enumerate(positions)]
    return KinematicEnv(rows, step_lim=step_lim, instruction=instruction)


class FakeAssets:
    def __init__(self, half=0.015):
        self.half = half

    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -self.half), bbox_max=np.full(3, self.half))


class FakeEpisode:
    """proxy_eval's ProxyEpisode stand-in over the kinematic env."""

    def __init__(self, env, task="stack_blocks"):
        self.env, self.task = env, task

    def info(self):
        return {"stable": True, "task": self.task, "layout_id": 1, "instruction": self.env.instruction,
                "step_lim": self.env.step_lim, "reset_s": 0.0, "render_stage_s": 0.0,
                "layout_file": "/dev/null/our_dev_record_with_gt.json"}

    def observe(self):
        img = np.zeros((48, 64, 3), np.uint8)
        return {"images": {c: img for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "state": self.env.proprio(), "instruction": self.env.instruction, "steps_used": self.env.steps_used,
                "step_lim": self.env.step_lim, "renderer": "fake"}


def run_episode(policy, ep, mode, max_calls=2000):
    """proxy_eval.run_eval's loop (+ the integ driver's dev GT hook in mode gt)."""
    policy.reset(ep.info())
    for _ in range(max_calls):
        obs = ep.observe()
        if mode == "gt":
            obs[DEV_GT_KEY] = ep.env.gt_objects()
        d = policy.next(obs)
        if d.get("done"):
            return policy.episode_report()
        for a in d["actions"]:
            if ep.env.step(a):
                return policy.episode_report()
    raise AssertionError("episode did not finish")


def _stacked(env):
    p = sorted((np.asarray(o["pos"]) for o in env.objects), key=lambda v: v[2])
    return all(np.linalg.norm(a[:2] - b[:2]) < 0.006 and 0.025 < b[2] - a[2] < 0.035 for a, b in zip(p[:-1], p[1:]))


# ------------------------------------------------------------------ firewall
def _obs(**extra):
    st = {"left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6), "left_ee_pose": np.zeros(7),
          "right_ee_pose": np.zeros(7), "left_ee_joint_state": [1.0], "right_ee_joint_state": [1.0]}
    o = {"images": {"cam_head": np.zeros((4, 4, 3), np.uint8)}, "state": st, "instruction": "x", "steps_used": 0,
         "step_lim": 10}
    o.update(extra)
    return o


def test_firewall_refuses_anything_but_the_contract():
    clean, gt = check_obs(_obs(renderer="isaac"), "perception")
    assert gt is None and set(clean) == {"images", "state", "instruction", "steps_used", "step_lim"}
    for bad in ({DEV_GT_KEY: []}, {"gt": {}}, {"objects": []}, {"reward": 1}, {"cameras": {}}):
        with pytest.raises(FirewallError):
            check_obs(_obs(**bad), "perception")
    o = _obs()
    o["state"]["object_pose"] = np.zeros(7)
    with pytest.raises(FirewallError):
        check_obs(o, "perception")
    o = _obs()
    o["images"]["cam_depth"] = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(FirewallError):
        check_obs(o, "perception")
    with pytest.raises(FirewallError):              # mode gt needs the driver's GT, and only that extra key
        check_obs(_obs(), "gt")
    _, gt = check_obs(_obs(**{DEV_GT_KEY: [1]}), "gt")
    assert gt == [1]


def test_info_drops_layout_file():
    keep, dropped = sanitize_info({"task": "t", "instruction": "i", "step_lim": 5, "layout_file": "/x.json",
                                   "layout_record": {}})
    assert "layout_file" not in keep and dropped == ["layout_file", "layout_record"]


def test_mode_perception_never_touches_gt(monkeypatch):
    """Whole episode in mode perception: the observation carries only the contract, the GT hook raises if
    anything calls it, and the policy is fed by a fake perception service that saw only the frames."""
    env = _env()
    ep = FakeEpisode(env)

    class FakePerception:
        calls = []

        def estimate(self, frames, vocabulary, fold):
            self.calls.append(("estimate", vocabulary, len(frames)))
            for f in frames:
                assert set(f) <= {"images", "state", "instruction", "steps_used", "step_lim"}
            rng = np.random.default_rng(0)
            recs = [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                     "default_pos": (np.asarray(p) + np.r_[rng.normal(0, 0.001, 2), 0.0]).tolist(),
                     "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]
            return {"Rigid": {"block": recs}}

        def check(self, frame, objects, fold):
            self.calls.append(("check",))
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

    pc = FakePerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(), perception_client=pc)
    import dojo.runtime.belief as B
    monkeypatch.setattr(B.RuntimeProvider, "_gt_objects", lambda self: (_ for _ in ()).throw(AssertionError("GT read")))
    rep = run_episode(pol, ep, "perception")
    assert rep["error"] is None, rep["error"]
    assert rep["info_dropped"] == ["layout_file"]
    assert _stacked(env), [o["pos"] for o in env.objects]
    assert pc.calls[0] == ("estimate", "stack_blocks", 1)
    assert rep["provider"]["resync"] >= 1                  # checkpoints after releases


def test_runtime_sources_never_read_gt_or_env_internals():
    """Static check: decision / perception-path modules reference no GT accessor or env internals; the only
    GT entry is the provider's ``gt_source`` used in mode gt."""
    forbidden = re.compile(r"objects_gt|robot_gt|cameras_gt|\.gt\(|reward_manager|scene_manager|layout_manager|"
                           r"Eval_Layout|layout_file|\.env\.|take_action_cnt")
    for name in ("policy.py", "supervisor.py", "belief.py", "verifier.py", "bridge.py", "estimate.py",
                 "perception_service.py", "firewall.py"):
        src = (RUNTIME / name).read_text()
        body = src.split('"""', 2)[-1] if src.startswith('"""') else src        # module docstring may name them
        hits = [ln for ln in body.splitlines() if forbidden.search(ln) and "sanitize" not in ln]
        assert not hits, (name, hits)


# ------------------------------------------------------------------ instruction-only decisions
INSTR = {
    "stack_three_textured_blocks": STACK,
    "build_tower_blocks_boards": "Build a tower using the wooden blocks and wooden boards.",
    "stack_three_bowls": "Stack the three bowls together.",
    "bottles_into_dustbin": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
    "swap_blocks_via_empty_mat": "Swap the two blocks using the empty mat, pressing the button after each move.",
    "press_buttons_by_number_cards": "Press the two red buttons the required number of times according to the number "
                                     "cards, then press the blue button to confirm.",
}


def test_instruction_selects_the_skill():
    assert {s.name for s in SKILLS} == set(INSTR)
    for name, text in INSTR.items():
        assert parse_instruction(text).name == name
    for other in ("stack the blocks from bottom to top in the order of red, green, and blue, then reset the robot arm.",
                  "Push the T-shaped block to align it precisely with the gray T-shaped pad.", ""):
        assert parse_instruction(other) is None


def test_decisions_never_read_the_task_name():
    """The task name in info is a decoy: the skill follows the instruction."""
    env = _env()
    pol = RuntimePolicy(RuntimeConfig(mode="gt"), assets=FakeAssets())
    rep = run_episode(pol, FakeEpisode(env, task="push_T"), "gt")
    assert rep["skill"] == "stack_three_textured_blocks" and _stacked(env)
    env2 = _env(instruction="Push the T-shaped block to align it precisely with the gray T-shaped pad.")
    pol2 = RuntimePolicy(RuntimeConfig(mode="gt"), assets=FakeAssets())
    rep2 = run_episode(pol2, FakeEpisode(env2, task="stack_blocks"), "gt")
    assert rep2["skill"] is None and env2.steps_used == 0
    for name in ("policy.py", "supervisor.py", "belief.py", "verifier.py"):
        for ln in (RUNTIME / name).read_text().splitlines():
            if re.search(r"""info(\.get\(|\[)["']task["']""", ln):
                assert "log_task" in ln, (name, ln)


# ------------------------------------------------------------------ episodes, bridge, belief
def test_mode_gt_episode_stacks_and_goes_home():
    env = _env()
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off"), assets=FakeAssets())
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert rep["error"] is None, rep["error"]
    assert _stacked(env)
    assert all(np.abs(env.q[a]).max() < 0.02 for a in ("left", "right")) and env.g["left"] == env.g["right"] == 1.0
    assert rep["verify"].startswith("off") and rep["decisions"][0]["action"] == "run"
    assert env.steps_used < 320 and rep["timings"]["observations"] > 3


def test_bridge_unblocks_on_cancel_after_episode_end():
    b = Bridge()

    def prog():
        for _ in range(5):
            b.step({"x": np.zeros(1)})
        b.observe()
        b.step({"x": np.zeros(1)})
        b.observe()                     # the harness never answers this one (episode ended)
        return "unreachable"

    out = b.start(prog, {"steps_used": 0, "step_lim": 10, "state": {}})
    assert len(out["actions"]) == 5
    out = b.resume({"steps_used": 5, "step_lim": 10, "state": {}})
    assert len(out["actions"]) == 1
    b.cancel(timeout_s=5)
    assert not b.thread.is_alive() and b.result is None and b.error is None


def test_kinematic_belief_follows_jaws_and_drops_on_support():
    env = _env([(-0.30, -0.15, 0.78), (-0.20, -0.15, 0.78), (0.3, -0.2, 0.78)])
    objs = objects_from_gt_rows(env.gt_objects(), None)
    kb = KinematicBelief(objs, X5Kin(), env.proprio())
    from dojo.skills.programs import stack_blocks
    from dojo.skills.skills import GTProvider, Skills
    sk = Skills(env, GTProvider(env, None, "stack_blocks"))
    n0 = 0
    ex_log = sk.ex.log
    stack_blocks.run(sk)
    for a in ex_log.actions[n0:]:
        kb.step(a)
    truth = {o["label"]: np.asarray(o["pos"]) for o in env.objects}
    for lab, o in kb.objects.items():
        assert np.linalg.norm(o.pos - truth[lab]) < 0.004, (lab, o.pos, truth[lab])
    assert not kb.attached


def test_estimate_to_layout_fills_fields_from_public_template():
    tmpl = {"Room": {"default": "r"}, "Table": {"default": "t"}, "Ground": {}, "Background": {},
            "Rigid": {"block": [{"category": "block", "category_idx": 0, "label": "block_0", "group": 0,
                                 "place_tag": "up", "default_pos": [9, 9, 9], "default_ori": [1, 0, 0, 0]}]},
            "Geometry": {"camera_stand": [{"category": "camera_stand", "category_idx": 0, "label": "camera_stand",
                                           "default_pos": [0, 0.4, 0.765], "default_ori": [1, 0, 0, 0]}]}}
    objs = [{"type": "Rigid", "category": "block", "category_idx": 7, "label": "block_0", "pos": [0.1, -0.2, 0.78],
             "quat": [1, 0, 0, 0]}]
    lay = estimate_to_layout("stack_blocks", objs, template=tmpl, fixed={"camera_stand"},
                             meta_fn=lambda t, c, i: {"physics": {"mass": 0.1}})
    b = lay["Rigid"]["block"][0]
    assert b["category_idx"] == 7 and b["default_pos"] == [0.1, -0.2, 0.78] and b["place_tag"] == "up"
    assert b["physics"] == {"mass": 0.1, "type": "rigid"} and lay["Geometry"]["camera_stand"][0]["label"] == "camera_stand"
    assert {"Room", "Table", "Ground", "Background"} <= set(lay)


def test_flagged_records():
    rec = {"Rigid": {"block": [{"label": "a", "_perception": {"confidence": 0.9}},
                               {"label": "b", "_perception": {"confidence": 0.5}},
                               {"label": "c", "_perception": {"confidence": 0.95, "ambiguous": True}}]},
           "_not_found": [{"label": "d"}]}
    assert [f["label"] for f in flagged(rec, 0.7)] == ["b", "c", "d"]


def test_est_objects_skip_clutter():
    rows = [{"label": "a", "category": "block", "category_idx": 1, "pos": [0, 0, 0.78], "quat": [1, 0, 0, 0]},
            {"label": None, "inst_name": "cl0", "category": "cup", "category_idx": 2, "pos": [0, 0, 0.78],
             "quat": [1, 0, 0, 0], "clutter": True}]
    objs = objects_from_gt_rows(rows, None)
    assert [o["label"] for o in est_objects(objs)] == ["a"]
