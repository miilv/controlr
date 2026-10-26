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
from dojo.runtime.supervisor import perception_vocabulary
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


def run_episode(policy, ep, mode, max_calls=2000, success_fn=None):
    """proxy_eval.run_eval's loop (+ the integ driver's dev GT hook in mode gt). ``success_fn(env)``: end the
    episode at that step, as RoboDojo does on success (the policy is not called again)."""
    policy.reset(ep.info())
    for _ in range(max_calls):
        obs = ep.observe()
        if mode == "gt":
            obs[DEV_GT_KEY] = ep.env.gt_objects()
        d = policy.next(obs)
        if d.get("done"):
            return policy.episode_report()
        for a in d["actions"]:
            if ep.env.step(a) or (success_fn is not None and success_fn(ep.env)):
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

        def estimate(self, frames, vocabulary, fold, prior_record=None):
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
    "stack_blocks": STACK,
    "build_tower": "Build a tower using the wooden blocks and wooden boards.",
    "stack_bowls": "Stack the three bowls together.",
    "put_bottles_into_dustbin": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
    "swap_blocks": "Swap the two blocks using the empty mat, pressing the button after each move.",
    "press_by_number": "Press the two red buttons the required number of times according to the number "
                       "cards, then press the blue button to confirm.",
    "push_T": "Push the T-shaped block to align it precisely with the gray T-shaped pad.",
    "cover_blocks": "Cover the blocks from left to right, remember their colors, then uncover them in the order: red, "
                    "green, and blue.",
    "general_pickup": "Pick up the toy car by 10 cm.",
    "align_blocks": "Use the set square to push the three blocks into a straight, aligned row, then reset the robot arm.",
    "stack_blocks_by_language": "stack the blocks from bottom to top in the order of cyan, yellow, and green, then reset "
                                "the robot arm.",
    "plug_in_charger": "Plug the charger into the power strip.",
    "make_kong": "Wait for the opponent to discard a tile, then declare a kong with the matching tiles.",
}


def test_instruction_selects_the_perception_vocabulary():
    for name, text in INSTR.items():
        assert perception_vocabulary(text).name == name
    assert perception_vocabulary("stack the blocks from bottom to top in the order of red, green, and blue, "
                                 "then reset the robot arm.").name == "stack_blocks_by_language"
    assert perception_vocabulary("Cover the blocks from left to right, remember their colors, then uncover them in the "
                                 "order: red, green, and blue.").name == "cover_blocks"
    assert perception_vocabulary("") is None


def test_decisions_never_read_the_task_name():
    """The task name in info is a decoy: the skill follows the instruction."""
    env = _env()
    pol = RuntimePolicy(RuntimeConfig(mode="gt"), assets=FakeAssets())
    rep = run_episode(pol, FakeEpisode(env, task="push_T"), "gt")
    assert rep["skill"] == "stack_objects" and rep["vocabulary"] == "stack_blocks" and _stacked(env)
    env2 = _env(instruction="Push the T-shaped block to align it precisely with the gray T-shaped pad.")
    pol2 = RuntimePolicy(RuntimeConfig(mode="gt"), assets=FakeAssets())
    rep2 = run_episode(pol2, FakeEpisode(env2, task="stack_blocks"), "gt")
    assert rep2["skill"] is None and rep2["decisions"][0]["action"] == "finish"
    assert all(np.abs(env2.q[a]).max() < 0.02 for a in ("left", "right"))
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
    from dojo.skills import planner
    from dojo.skills.skills import GTProvider, Skills
    sk = Skills(env, GTProvider(env, None, "", instruction=STACK))
    n0 = 0
    ex_log = sk.ex.log
    assert planner.solve(sk)["ok"]
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


# ------------------------------------------------------------------ bundle verifier against a fake twin service
class FakeTwinClient:
    """twin_service.TwinClient stand-in: rollouts on copies of a kinematic env built from the estimate."""

    def __init__(self, task="stack_blocks", half=0.015):
        self.task, self.half = task, half
        self.snaps, self.base, self.loads, self.verifies = {}, None, [], []

    def info(self):
        return {"task": self.task}

    def _env_from(self, layout, robot):
        rows = [{"label": r["label"], "category": c, "category_idx": r["category_idx"], "pos": r["default_pos"],
                 "quat": r["default_ori"], "half": [self.half] * 3} for c, lst in (layout.get("Rigid") or {}).items()
                for r in lst]
        env = KinematicEnv(rows, step_lim=10_000)
        for a in ("left", "right"):
            env.q[a] = np.asarray(robot[f"{a}_q"], float)
            env.g[a] = float(robot[f"{a}_grip"])
        return env

    def load_from_estimate(self, layout, robot=None, settle_steps=50):
        self.loads.append(layout)
        self.base = self._env_from(layout, robot)
        return {"fast": True, "stable": True}

    def verify(self, world, candidates, perturb=None, k=1, record_traj=False, seed=0, keep=None):
        if "layout" in world:                       # the real service loads the estimate inside verify
            self.load_from_estimate(world["layout"], world.get("robot"))
        base = self.snaps[world["snapshot_id"]] if "snapshot_id" in world else self.base
        self.verifies.append({"from": "snapshot" if "snapshot_id" in world else "estimate",
                              "n": [len(c["actions"]) for c in candidates], "labels": (perturb or {}).get("labels"),
                              "prefix": [c["meta"]["prefix"] for c in candidates]})
        out = []
        for ci, c in enumerate(candidates):
            for si in range(k):
                e = copy.deepcopy(base)
                for a in c["actions"]:
                    e.step(a)
                o = {"candidate": ci, "sample": si, "success": False, "score": 0.0, "ended": False,
                     "reward": {"success": False, "success_flag": True, "ended": False, "steps_used": 0, "step_lim": 1000},
                     "final_objects": {r["label"]: list(r["pos"]) + list(r["quat"]) for r in e.gt_objects()},
                     "final_robot": {f"{a}_q": e.q[a].tolist() for a in ("left", "right")}}
                if keep == "nominal" and si == 0:
                    sid = max(self.snaps, default=0) + 1
                    self.snaps[sid] = e
                    o["snapshot_id"] = sid
                out.append(o)
        return out

    def drop_snapshot(self, sid):
        self.snaps.pop(int(sid), None)


@pytest.mark.parametrize("mode", ["gt", "perception"])
def test_bundle_verify_chains_through_the_grasp(mode, monkeypatch):
    import dojo.runtime.estimate as E
    monkeypatch.setattr(E, "estimate_to_layout", lambda scene, objs: {"Rigid": {"block": [
        {"label": o["label"], "category_idx": o["category_idx"], "default_pos": o["pos"], "default_ori": o["quat"]}
        for o in objs]}})
    env = _env()
    tc = FakeTwinClient()

    class FakePerception:
        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for o in env.objects]}}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

    pol = RuntimePolicy(RuntimeConfig(mode=mode, verify="bundle", n_cands=2, verify_k=2), assets=FakeAssets(),
                        twin_client=tc, perception_client=FakePerception())
    rep = run_episode(pol, FakeEpisode(env), mode)
    assert rep["error"] is None, rep["error"]
    assert _stacked(env)
    v = rep["verify"]
    assert v["segments"] >= 4 and v["calls"] == v["segments"], v
    assert v["from_snapshot"] >= 2 and v["from_estimate"] >= 2, v          # places chain from the kept pick
    for vv in tc.verifies:
        if vv["from"] == "snapshot":
            assert vv["labels"] is not None and len(vv["labels"]) == 2       # the held block is not perturbed
            assert max(vv["prefix"]) <= 20, vv         # only what ran after the kept pick (a park), not the pick itself
    assert len(tc.snaps) <= 3                                                  # unused snapshots are dropped
    assert rep["provider"]["adopted"] >= 4


def test_pose_errors_match_interchangeable_labels():
    from dojo.runtime.estimate import pose_errors
    gt = objects_from_gt_rows([{"label": f"block_{i}", "category": "block", "category_idx": i, "pos": p, "quat": [1, 0, 0, 0]}
                               for i, p in enumerate([(0.1, -0.1, 0.78), (0.3, -0.2, 0.78)])], None)
    est = objects_from_gt_rows([{"label": "block_0", "category": "block", "category_idx": 1, "pos": (0.301, -0.2, 0.78),
                                 "quat": [1, 0, 0, 0]},
                                {"label": "block_1", "category": "block", "category_idx": 0, "pos": (0.1, -0.102, 0.78),
                                 "quat": [1, 0, 0, 0]}], None)
    e = pose_errors(est, gt)
    assert e["block_0"]["est_label"] == "block_1" and e["block_0"]["xy_mm"] == 2.0 and e["block_0"]["id_ok"]
    assert e["block_1"]["xy_mm"] == 1.0 and e["block_1"]["shape_ok"]


def test_report_survives_success_ending_the_episode(monkeypatch):
    """RoboDojo ends the episode the step the stack is complete and both arms are home: the worker is then
    parked inside finish(); stats, history and result must still reach the report."""
    import dojo.runtime.estimate as E
    monkeypatch.setattr(E, "estimate_to_layout", lambda scene, objs: {"Rigid": {"block": [
        {"label": o["label"], "category_idx": o["category_idx"], "default_pos": o["pos"], "default_ori": o["quat"]}
        for o in objs]}})
    env = _env()
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="bundle", n_cands=2, verify_k=1), assets=FakeAssets(),
                        twin_client=FakeTwinClient())

    def success(e):
        return _stacked(e) and all(np.abs(e.q[a]).max() < 0.05 for a in ("left", "right"))
    rep = run_episode(pol, FakeEpisode(env), "gt", success_fn=success)
    assert _stacked(env) and rep["error"] is None
    assert isinstance(rep["verify"], dict) and rep["verify"]["segments"] >= 4 and rep["verify"]["calls"] >= 4
    assert rep["provider"]["refresh"] > 0 and len(rep["history"]) >= 4
    assert rep["result"] and rep["result"]["ok"]


def test_perception_service_serves_its_ops(monkeypatch):
    """The service process: both classes exist, every client op is served, plain types only."""
    import secrets
    import threading
    from multiprocessing.connection import Listener

    import dojo.runtime.perception_service as ps
    for op in ("estimate", "look_plan", "look_poses", "check", "refine"):
        assert callable(getattr(ps._Perception, op)) and callable(getattr(ps.PerceptionClient, op))
    calls = []
    for op in ("estimate", "look_plan", "look_poses", "check", "refine"):
        monkeypatch.setattr(ps._Perception, op, lambda self, *a, _op=op, **k: calls.append(_op) or
                            ({} if _op in ("estimate", "check", "refine") else []))
    key = secrets.token_bytes(16)
    lst = Listener(("127.0.0.1", 0), authkey=key)
    port = lst.address[1]
    lst.close()
    th = threading.Thread(target=ps.serve, args=(port, key), daemon=True)
    th.start()
    pc = ps.PerceptionClient(port, key, timeout_s=10)
    obs = {"images": {"cam_head": np.zeros((4, 4, 3), np.uint8)}, "state": {"x": np.zeros(2)}, "instruction": "i"}
    assert pc.estimate([obs], "stack_blocks", 0) == {}
    assert pc.look_plan(obs, {"Rigid": {}}) == []
    assert pc.look_poses([0, 0, 0.8], "left") == []
    assert pc.check(obs, [], 1) == {}
    assert pc.refine(obs, [], 1) == {}
    assert calls == ["estimate", "look_plan", "look_poses", "check", "refine"]
    pc.close(stop_service=True)
    th.join(5)
    assert not th.is_alive()


def test_unconfirmed_divergence_does_not_replan():
    """A resync check that fails while the re-estimate puts the object where the belief has it: log, go on."""
    env = _env()

    class DoubtfulPerception:
        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for o in env.objects]}}

        def check(self, frame, objects, fold):            # always claims block_0 is not where predicted
            return {o["label"]: {"head": {"prec": 0.0 if o["label"] == "block_0" else 1.0, "n_vis": 500, "n_px": 600}}
                    for o in objects}

        def look_plan(self, *a, **k):
            return []

    pol = RuntimePolicy(RuntimeConfig(mode="perception"), assets=FakeAssets(), perception_client=DoubtfulPerception())
    rep = run_episode(pol, FakeEpisode(env), "perception")
    kinds = [e["kind"] for e in rep["events"]]
    assert "divergence_unconfirmed" in kinds and "divergence_replan" not in kinds, kinds
    assert _stacked(env) and rep["result"]["ok"]


def test_report_gap_table_pairs_layouts():
    from dojo.runtime.report import gap_table
    rows = [{"task": "t", "verify": "off", "mode": "gt", "layout_id": i, "score": 1.0, "success": True} for i in (1, 2)]
    rows += [{"task": "t", "verify": "off", "mode": "perception", "layout_id": 1, "score": 0.15, "success": False,
              "category": "perception_gross"},
             {"task": "t", "verify": "off", "mode": "perception", "layout_id": 3, "score": 1.0, "success": True}]
    t = gap_table(rows)
    assert "| t | off | 1 | 100.0 | 15.0 | +85.0 | 1 | 0 | 1: perception_gross |" in t


def test_look_macro_falls_back_and_fuses():
    """A flagged object triggers the look: the planned camera pose is unreachable, a variant works, the arm
    goes there, the frame after the move is fused with the first one, then both arms return home."""
    from dojo.skills.geometry import T_to_pose7
    from dojo.skills.kinematics import down_R
    from dojo.skills.geometry import T_from
    env = _env()
    kin = X5Kin()
    reachable = T_to_pose7(kin.flange_from_tcp(T_from((-0.25, -0.15, 0.90), down_R(0.0)))).tolist()

    class LookingPerception:
        calls = []

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            self.calls.append(len(frames))
            conf = 0.5 if len(frames) == 1 else 0.9
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": conf}} for o in env.objects]}}

        def look_plan(self, frame, record, max_looks=2):
            return [{"arm": "left", "purpose": "low_confidence", "label": "block_1", "target": [-0.26, -0.19, 0.78],
                     "flange_pose": [5.0, 5.0, 5.0, 1, 0, 0, 0]}]                       # unreachable

        def look_poses(self, target, arm, variants=None):
            return [{"arm": arm, "dist": 0.14, "elev": 55, "flange_pose": [9.0, 9.0, 9.0, 1, 0, 0, 0]},
                    {"arm": arm, "dist": 0.18, "elev": 70, "flange_pose": reachable}]

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

    pc = LookingPerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception"), assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    lk = rep["looks"][0]
    assert lk["status"] == "ok" and lk["arm"] == "left" and lk["pose"] == "d0.18/e70" and lk["tried"] == 3
    assert pc.calls[:2] == [1, 2] and rep["records"].get("fused") and rep["look_steps"] > 0
    assert _stacked(env)


def test_supervisor_hook_is_pluggable():
    """A custom supervisor sees instruction + WorldState + events and decides; here it refuses to act."""
    from dojo.runtime.supervisor import Decision
    seen = []

    class Hook:
        def decide(self, instruction, world, events):
            seen.append((instruction, sorted(world.objects), list(events)))
            return Decision("finish", reason="hook says no")

    env = _env()
    pol = RuntimePolicy(RuntimeConfig(mode="gt"), assets=FakeAssets(), supervisor=Hook())
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert seen and seen[0][0] == STACK and seen[0][1] == ["block_0", "block_1", "block_2"]
    assert rep["decisions"][0]["reason"] == "hook says no" and not _stacked(env)


def test_refine_hook_updates_the_belief_only_when_accepted():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class RefinePerception:
        def __init__(self, ok):
            self.ok, self.calls = ok, []

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            self.calls.append([o["label"] for o in objects])
            p = list(objects[0]["pos"])
            p[0] += 0.005
            return {objects[0]["label"]: {"ok": self.ok, "pos": p, "quat": objects[0]["quat"], "shift_mm": 5.0}}

    for ok in (True, False):
        pc = RefinePerception(ok)
        pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=pc)
        pol.reset(FakeEpisode(env).info())
        b = pol.bridge
        b.obs, b.fresh = FakeEpisode(env).observe(), True
        prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
        prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
        x0 = float(prov.belief.objects["block_1"].pos[0])
        out = pol._refine_hook(prov, None, "block_1", "left", "pick")
        x1 = float(prov.belief.objects["block_1"].pos[0])
        assert pc.calls[0][0] == "block_1" and len(pc.calls[0]) == 3          # the target first, others as occluders
        if ok:
            assert out and abs(x1 - x0 - 0.005) < 1e-9
        else:
            assert out is None and x1 == x0
        assert pol.report["events"][-1]["kind"] == "refine"


def test_refine_runs_at_every_hover_in_an_episode():
    """Mode perception + refine: skills calls the hook at each pick / place hover; accepted re-fits move the
    belief (here: the fake perception corrects a 6 mm initial error), the stack still completes."""
    env = _env()

    class RefinePerception:
        def __init__(self):
            self.n = 0

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": [o["pos"][0] + 0.006, o["pos"][1], o["pos"][2]],
                                         "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}}
                                        for o in env.objects]}}

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            assert arm in ("left", "right")
            self.n += 1
            o = objects[0]
            true = next(r for r in env.objects if r["label"] == o["label"])
            return {o["label"]: {"ok": True, "pos": list(true["pos"]), "quat": list(true["quat"]), "shift_mm": 6.0}}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

        def look_plan(self, *a, **k):
            return []

    pc = RefinePerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    kinds = [(e.get("for"), e.get("ok")) for e in rep["events"] if e["kind"] == "refine"]
    assert ("pick", True) in kinds and len(kinds) >= 3, kinds
    assert _stacked(env)


def test_merge_looked_keeps_unflagged_frame0():
    from dojo.runtime.estimate import merge_looked
    r0 = {"Rigid": {"block": [{"label": "block_0", "category": "block", "category_idx": 1, "default_pos": [0.1, -0.1, 0.78],
                               "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}},
                              {"label": "block_1", "category": "block", "category_idx": 2, "default_pos": [0.3, -0.2, 0.78],
                               "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.4}}]},
          "_not_found": [{"label": "block_2"}]}
    r1 = {"Rigid": {"block": [{"label": "block_1", "category": "block", "category_idx": 1, "default_pos": [0.11, -0.1, 0.78],
                               "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}},
                              {"label": "block_0", "category": "block", "category_idx": 3, "default_pos": [0.305, -0.2, 0.78],
                               "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.95}},
                              {"label": "block_2", "category": "block", "category_idx": 4, "default_pos": [-0.3, -0.2, 0.78],
                               "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.8}}]}}
    m = merge_looked(r0, r1, ["block_1", "block_2"])
    by = {r["label"]: r for r in m["Rigid"]["block"]}
    assert by["block_0"]["default_pos"] == [0.1, -0.1, 0.78]                       # unflagged: frame 0 kept
    assert by["block_1"]["default_pos"] == [0.305, -0.2, 0.78] and by["block_1"]["category_idx"] == 3   # matched by position
    assert by["block_2"]["default_pos"] == [-0.3, -0.2, 0.78] and not m["_not_found"]


def test_refine_not_ok_off_prior_asks_for_a_rehover():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class OffCentre:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            return {o["label"]: {"ok": False, "pos": [o["pos"][0] + 0.02, o["pos"][1], o["pos"][2]], "quat": o["quat"],
                                 "shift_mm": 20.0, "iou": 0.2}}

    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=OffCentre())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    x0 = float(prov.belief.objects["block_0"].pos[0])
    out = pol._refine_hook(prov, None, "block_0", "right", "pick")
    assert "rehover" in out and abs(float(out["rehover"][0][0]) - x0 - 0.02) < 1e-9
    assert float(prov.belief.objects["block_0"].pos[0]) == x0          # the belief waits for the second look


def test_report_arm_table_counts_rescued_and_lost():
    from dojo.runtime.report import arm_table
    a = [{"task": "t", "layout_id": i, "success": s, "score": 1.0 if s else 0.15, "category": None if s else "x"}
         for i, s in ((1, True), (2, False), (3, True))]
    b = [{"task": "t", "layout_id": i, "success": s, "score": 1.0 if s else 0.15, "category": None if s else "plan"}
         for i, s in ((1, True), (2, True), (3, False))]
    t = arm_table({"base": a, "refine": b})
    assert "| t | 3 | 2 | 2 |" in t and "| 1 [2] | 1 3: plan |" in t


def test_twin_error_falls_back_to_single_plan_and_still_finishes(monkeypatch):
    import dojo.runtime.estimate as E
    monkeypatch.setattr(E, "estimate_to_layout", lambda scene, objs: {"Rigid": {}})
    env = _env()

    class BrokenTwin(FakeTwinClient):
        def verify(self, *a, **k):
            raise RuntimeError("twin service: TypeError: 'NoneType' object is not subscriptable")

    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="bundle"), assets=FakeAssets(), twin_client=BrokenTwin())
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert rep["error"] is None and _stacked(env)
    assert rep["verify"]["errors"] == 1 and rep["verify"]["calls"] == 0


def test_place_refine_passes_the_held_object_and_rehovers_higher():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    seen = {}

    class HeldAware:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            seen["held"], seen["labels"], seen["idv"] = held, [o["label"] for o in objects], identity_vocabulary
            o = objects[0]
            return {o["label"]: {"ok": False, "pos": None, "quat": None, "iou": 0.1, "visible_frac": 0.5}}

    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=HeldAware())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.belief.attached["left"] = ("block_2", np.eye(4))

    class Sk:
        refine_ctx = {"kind": "place", "label": "block_0", "arm": "left", "held": ("Rigid/block/00005", [0, 0, 0.9, 1, 0, 0, 0])}
    out = pol._refine_hook(prov, Sk(), "block_0", "left", "place")
    assert seen["held"][0] == "Rigid/block/00005" and seen["labels"] == ["block_0", "block_1"] and seen["idv"] is None
    assert out["rehover_dz"] == 0.03 and np.allclose(out["rehover"][0], prov.belief.objects["block_0"].pos)


def test_add_found_assigns_missing_labels_to_new_objects_only():
    from dojo.runtime.estimate import add_found
    r0 = {"Rigid": {"block": [{"label": "block_0", "category": "block", "category_idx": 1, "default_pos": [0.1, -0.1, 0.78],
                               "default_ori": [1, 0, 0, 0]}]},
          "_not_found": [{"label": "block_1", "candidates": ["Rigid/block/00001"]}]}
    r1 = {"Rigid": {"block": [{"label": "block_1", "category": "block", "category_idx": 1, "default_pos": [0.102, -0.1, 0.78],
                               "default_ori": [1, 0, 0, 0]},
                              {"label": "block_0", "category": "block", "category_idx": 2, "default_pos": [-0.3, -0.2, 0.78],
                               "default_ori": [1, 0, 0, 0]}]}}
    m, found = add_found(r0, r1)
    by = {r["label"]: r for r in m["Rigid"]["block"]}
    assert found == ["block_1"] and by["block_1"]["default_pos"] == [-0.3, -0.2, 0.78] and not m["_not_found"]
    assert by["block_0"]["default_pos"] == [0.1, -0.1, 0.78]


def test_search_finds_a_missing_block_before_planning():
    """Frame 0 misses block_2 (and the look does not find it); the search moves a wrist over a region, the
    re-estimate with that view finds it, the episode plans with all three blocks and stacks them."""
    env = _env()

    class Hidden:
        def __init__(self):
            self.searched = 0

        def _rec(self, labels):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for o in env.objects if o["label"] in labels]},
                    "_not_found": [{"label": l, "candidates": ["Rigid/block/00005"]}
                                   for l in ("block_0", "block_1", "block_2") if l not in labels]}

        def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None):
            if fuse_mode == "full":
                self.searched += 1
                return self._rec({"block_0", "block_1", "block_2"})
            return self._rec({"block_0", "block_1"})

        def look_plan(self, *a, **k):
            return []

        def search_regions(self, frame, record, vocabulary):
            return [{"target": [0.3, -0.2, 0.78], "arm": "right", "why": "test"}]

        def look_poses(self, target, arm, variants=None):
            from dojo.skills.geometry import T_from, T_to_pose7
            from dojo.skills.kinematics import down_R
            x = -0.25 if arm == "left" else 0.25
            return [{"arm": arm, "dist": 0.18, "elev": 70,
                     "flange_pose": T_to_pose7(X5Kin().flange_from_tcp(T_from((x, -0.15, 0.90), down_R(0.0)))).tolist()}]

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

    pc = Hidden()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", search=True), assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    assert pc.searched == 1 and rep["search"][0]["found"] == ["block_2"]
    assert not any(e["kind"] == "perception_missing" for e in rep["events"]) and _stacked(env)


def test_post_place_refine_writes_the_measured_pose():
    """After a release, the placed block is re-measured from the releasing arm's wrist and the belief moves."""
    env = _env()

    class Measuring:
        calls = []

        def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for o in env.objects]}}

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            true = next(r for r in env.objects if r["label"] == o["label"])
            self.calls.append((o["label"], arm))
            return {o["label"]: {"ok": True, "pos": [true["pos"][0] + 0.004, true["pos"][1], true["pos"][2]],
                                 "quat": list(true["quat"]), "shift_mm": 4.0}}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

        def look_plan(self, *a, **k):
            return []

    pc = Measuring()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", post_place=True, rehover_when_far=False), assets=FakeAssets(),
                        perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    pp = [e for e in rep["events"] if e["kind"] == "post_place"]
    assert len(pp) >= 2 and all(e["ok"] and e.get("to") for e in pp), pp
    assert rep["provider"]["post_place_moved"] >= 2


def test_search_refine_fits_the_missing_block_at_the_region():
    env = _env()
    true2 = next(o for o in env.objects if o["label"] == "block_2")

    class WristFinds:
        refined = []

        def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for o in env.objects if o["label"] != "block_2"]},
                    "_not_found": [{"label": "block_2", "candidates": ["Rigid/block/00005", "Rigid/block/00006"]}]}

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            self.refined.append((o["label"], o["key"], max_shift_m))
            if o["label"] == "block_2":
                return {"block_2": {"ok": True, "pos": list(true2["pos"]), "quat": [1, 0, 0, 0], "shift_mm": 40.0, "iou": 0.9}}
            return {o["label"]: {"ok": True, "pos": list(o["pos"]), "quat": o["quat"], "shift_mm": 0.0}}

        def look_plan(self, *a, **k):
            return []

        def search_regions(self, frame, record, vocabulary):
            return [{"target": [true2["pos"][0] - 0.03, true2["pos"][1], 0.78], "arm": "right", "why": "hidden_by_gripper"}]

        def look_poses(self, target, arm, variants=None):
            from dojo.skills.geometry import T_from, T_to_pose7
            from dojo.skills.kinematics import down_R
            return [{"arm": arm, "dist": 0.18, "elev": 70,
                     "flange_pose": T_to_pose7(X5Kin().flange_from_tcp(T_from((0.25, -0.15, 0.90), down_R(0.0)))).tolist()}]

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

    pc = WristFinds()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", search=True), assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    assert rep["search"][0]["found"] == ["block_2"] and rep["search"][0]["via"] == "refine"
    assert pc.refined[0][:2] == ("block_2", "Rigid/block/00005") and pc.refined[0][2] == 0.10
    assert _stacked(env)


def test_large_borderline_correction_needs_a_confirming_second_refine():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class Seq:
        def __init__(self, second_shift):
            self.n, self.second = 0, second_shift

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            self.n += 1
            if self.n == 1:
                return {o["label"]: {"ok": True, "pos": [o["pos"][0] + 0.04, o["pos"][1], o["pos"][2]], "quat": o["quat"],
                                     "shift_mm": 40.0, "iou": 0.65}}
            return {o["label"]: {"ok": True, "pos": [o["pos"][0] + self.second / 1000, o["pos"][1], o["pos"][2]],
                                 "quat": o["quat"], "shift_mm": self.second, "iou": 0.9}}

    for second, accepted in ((2.0, True), (25.0, False)):
        pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07), assets=FakeAssets(),
                            perception_client=Seq(second))
        pol.reset(FakeEpisode(env).info())
        b = pol.bridge
        b.obs, b.fresh = FakeEpisode(env).observe(), True
        prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
        prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
        x0 = float(prov.belief.objects["block_1"].pos[0])
        h1 = pol._refine_hook(prov, None, "block_1", "left", "pick")
        assert "rehover" in h1 and pol.report["events"][-1]["rehover"] == "confirm"
        pol._refine_hook(prov, None, "block_1", "left", "pick")
        x1 = float(prov.belief.objects["block_1"].pos[0])
        if accepted:
            assert abs(x1 - (x0 + 0.042)) < 1e-9
        else:
            assert x1 == x0


def test_program_seam_runs_on_the_episode_context():
    """RuntimePolicy(program=fn): fn(ctx) replaces the supervisor loop; setup and the finishing tail stay."""
    env = _env()
    seen = {}

    def program(ctx):
        seen.update(instruction=ctx.instruction, scene=ctx.scene, labels=sorted(ctx.sk.world.objects))
        res = ctx.library.SKILLS["stack_objects"](ctx.sk, objects=["block_0", "block_1", "block_2"])
        if res.get("ok"):
            ctx.sk.finish()                       # like the supervisor loop: a successful program finishes itself
        return {"ok": bool(res.get("ok")), "stage": "agent"}

    pol = RuntimePolicy(RuntimeConfig(mode="gt"), assets=FakeAssets(), program=program)
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert rep["error"] is None, rep["error"]
    assert seen == {"instruction": STACK, "scene": "stack_blocks", "labels": ["block_0", "block_1", "block_2"]}
    assert _stacked(env) and rep["result"]["stage"] == "agent"
    assert all(np.abs(env.q[a]).max() < 0.05 for a in ("left", "right"))


def test_divergence_update_matches_by_position_not_label():
    """A fresh estimate may give the interchangeable labels to other physical blocks: the provider re-keys it by
    position, so a diverged block is not 'moved' onto another block's spot."""
    from dojo.runtime.belief import RuntimeProvider
    env = _env([(-0.30, -0.15, 0.78), (0.10, -0.10, 0.78), (0.30, -0.20, 0.78)])

    class Swapped:
        def check(self, obs, objects):                 # the provider's checker interface (policy._Checker)
            return {o.label: {"head": {"prec": 0.0 if o.label == "block_1" else 1.0, "n_vis": 500, "n_px": 600}}
                    for o in objects}

        def estimate(self, obs):
            # labels permuted; block_1 really moved 20 mm in x
            return {"Rigid": {"block": [
                {"category": "block", "category_idx": 5, "label": "block_2", "default_pos": [0.12, -0.10, 0.78],
                 "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}},
                {"category": "block", "category_idx": 5, "label": "block_1", "default_pos": [0.30, -0.20, 0.78],
                 "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}},
                {"category": "block", "category_idx": 5, "label": "block_0", "default_pos": [-0.30, -0.15, 0.78],
                 "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}}]}}

    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets(), checker=Swapped(), replan_on_divergence=False)
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.resync(b.obs)
    assert abs(float(prov.belief.objects["block_1"].pos[0]) - 0.12) < 1e-9          # moved by its own 20 mm
    assert abs(float(prov.belief.objects["block_2"].pos[0]) - 0.30) < 1e-9          # untouched


def test_failed_pick_refine_triggers_a_local_search():
    """block_1's estimate is 35 mm off; at its pick hover refine finds nothing (iou 0.05, no suggestion): the
    policy treats it as lost, searches around its belief pose with a wrist look, refits it there and re-plans."""
    env = _env()
    true1 = next(o for o in env.objects if o["label"] == "block_1")

    class LostOnce:
        def __init__(self):
            self.pick_calls = 0

        def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None):
            recs = []
            for o in env.objects:
                p = list(o["pos"])
                if o["label"] == "block_1":
                    p[0] += 0.035
                recs.append({"category": "block", "category_idx": 5, "label": o["label"], "default_pos": p,
                             "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}})
            return {"Rigid": {"block": recs}}

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            if o["label"] == "block_1" and max_shift_m < 0.1:          # a hover refine (not the search's 10 cm fit)
                self.pick_calls += 1
                if abs(o["pos"][0] - true1["pos"][0]) > 0.02:
                    return {"block_1": {"ok": False, "pos": None, "quat": None, "iou": 0.05}}
            if max_shift_m >= 0.1:                                       # the search fit at the region
                return {o["label"]: {"ok": True, "pos": list(true1["pos"]), "quat": [1, 0, 0, 0], "shift_mm": 35.0,
                                     "iou": 0.9}}
            return {o["label"]: {"ok": True, "pos": list(o["pos"]), "quat": o["quat"], "shift_mm": 0.0, "iou": 0.9}}

        def search_regions(self, frame, record, vocabulary):
            return []

        def look_poses(self, target, arm, variants=None):
            from dojo.skills.geometry import T_from, T_to_pose7
            from dojo.skills.kinematics import down_R
            x = -0.25 if arm == "left" else 0.25
            return [{"arm": arm, "dist": 0.18, "elev": 70,
                     "flange_pose": T_to_pose7(X5Kin().flange_from_tcp(T_from((x, -0.15, 0.90), down_R(0.0)))).tolist()}]

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

        def look_plan(self, *a, **k):
            return []

    pc = LostOnce()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True, search_lost=True, rehover_when_far=False),
                        assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    ls = [e for e in rep["events"] if e["kind"] == "lost_search"]
    assert ls and ls[0]["label"] == "block_1" and ls[0]["found"], rep["events"][-6:]
    assert _stacked(env)


def test_merge_looked_does_not_add_a_held_object_under_a_missing_label():
    from dojo.runtime.estimate import merge_looked
    r0 = {"Rigid": {"block": [{"label": "block_0", "category": "block", "category_idx": 1, "default_pos": [0.1, -0.1, 0.78],
                               "default_ori": [1, 0, 0, 0]}]}, "_not_found": [{"label": "block_1"}]}
    r1 = {"Rigid": {"block": [{"label": "block_1", "category": "block", "category_idx": 1, "default_pos": [0.101, -0.1, 0.78],
                               "default_ori": [1, 0, 0, 0]}]}}            # the same physical block, other label
    m = merge_looked(r0, r1, [])
    assert [r["label"] for r in m["Rigid"]["block"]] == ["block_0"] and m["_not_found"]


def test_held_refine_measures_the_in_hand_object_and_reanchors_the_grasp():
    """kind "held": the object in the jaws is measured from that wrist; a flipped pose sticks across refreshes."""
    from dojo.runtime.belief import RuntimeProvider
    from dojo.skills.geometry import T_from
    env = _env()
    flipped = [0.0, 0.0, 0.0, 1.0]                       # 180 deg about z (wxyz)

    class InHand:
        seen = {}

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            self.seen.update(label=objects[0]["label"], prior=objects[0]["pos"], arm=arm, held=held)
            o = objects[0]
            return {o["label"]: {"ok": True, "pos": list(o["pos"]), "quat": flipped, "shift_mm": 0.5, "turned_deg": 180.0}}

    pc = InHand()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=pc)
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    o = prov.belief.objects["block_1"]
    prov.belief.attached["left"] = ("block_1", np.linalg.inv(prov.belief.kin.tcp("left", prov.belief.q["left"])) @ o.T)

    class Sk:
        refine_ctx = {"kind": "held", "label": "block_1", "arm": "left",
                      "held": ("Rigid/block/00005", list(o.pos) + [1.0, 0.0, 0.0, 0.0])}
    out = pol._refine_hook(prov, Sk(), "block_1", "left", "held")
    assert pc.seen["label"] == "block_1" and pc.seen["arm"] == "left" and pc.seen["held"] is None
    assert np.allclose(out["block_1"][1], flipped)
    prov.belief.sync_robot(b.obs["state"])                     # a refresh keeps the measured in-hand pose
    assert abs(abs(float(np.dot(prov.belief.objects["block_1"].quat, flipped))) - 1.0) < 1e-6


def test_deployed_code_reads_the_deploy_stamp(tmp_path, monkeypatch):
    import json as _json

    import dojo.runtime.eval as EV
    stamp = {"head": "abc1234", "skills": "def5678"}
    fake = tmp_path / "dojo" / "runtime" / "eval.py"
    fake.parent.mkdir(parents=True)
    (tmp_path / ".deployed.json").write_text(_json.dumps(stamp))
    monkeypatch.setattr(EV, "__file__", str(fake))
    assert EV.deployed_code() == stamp


def _place_setup(cfg, pc, env):
    from dojo.runtime.belief import RuntimeProvider
    pol = RuntimePolicy(cfg, assets=FakeAssets(), perception_client=pc)
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.events = pol.report["events"]
    return pol, prov


class _PlaceSeq:
    """Place refines answering a scripted list of (dx mm, visible fraction); records each call's prior."""

    def __init__(self, answers):
        self.answers, self.priors = list(answers), []

    def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
        o = objects[0]
        self.priors.append(list(o["pos"]))
        dx, vis, base = self.answers.pop(0)
        x = (base if base is not None else o["pos"][0]) + dx / 1000
        return {o["label"]: {"ok": True, "pos": [x, o["pos"][1], o["pos"][2]], "quat": o["quat"],
                             "shift_mm": abs(x - o["pos"][0]) * 1000, "iou": 0.85, "visible_frac": vis}}


class _PlaceSk:
    refine_ctx = {"kind": "place", "label": "block_0", "arm": "left", "held": None}


def test_place_confirm_applies_a_big_place_correction_only_when_a_second_view_agrees():
    env = _env()
    cfg = RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07, place_confirm=True)
    for second_dx, agrees in ((3.0, True), (20.0, False)):
        pc = _PlaceSeq([(25.0, 1.0, None)])
        pol, prov = _place_setup(cfg, pc, env)
        x0 = float(prov.belief.objects["block_0"].pos[0])
        x_true = x0 + 0.025
        pc.answers.append((second_dx, 1.0, x_true))                       # the second fit, relative to the first
        h = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
        assert "rehover" in h and h["rehover_dz"] == cfg.place_confirm_dz
        assert np.isclose(h["rehover"][0][0], x_true)                      # hover over the found pose
        assert float(prov.belief.objects["block_0"].pos[0]) == x0           # not applied before the second view
        assert pol.report["events"][-1]["confirm_needed"] == ["shift"]
        out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
        assert np.isclose(pc.priors[1][0], x0)                              # the second fit starts from the prior
        x1 = float(prov.belief.objects["block_0"].pos[0])
        if agrees:
            assert out and np.isclose(x1, x_true + 0.003) and prov.stats["place_confirmed"] == 1
        else:
            assert out is None and x1 == x0 and prov.stats["place_unconfirmed"] == 1


def test_place_confirm_small_full_view_shift_applies_partial_view_or_our_measurement_needs_a_look():
    env = _env()
    cfg = RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07, place_confirm=True,
                        rehover_when_far=False)
    # small shift, full view: applied at once (the closed2 behaviour)
    pol, prov = _place_setup(cfg, _PlaceSeq([(6.0, 1.0, None)]), env)
    x0 = float(prov.belief.objects["block_0"].pos[0])
    out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
    assert "block_0" in out and np.isclose(float(prov.belief.objects["block_0"].pos[0]), x0 + 0.006)
    # the same shift seen on a partial view: a confirming look
    pol, prov = _place_setup(cfg, _PlaceSeq([(6.0, 0.6, None)]), env)
    out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
    assert "rehover" in out and pol.report["events"][-1]["confirm_needed"] == ["partial_view"]
    # a support our own post-place refine measured: a confirming look even in full view
    pol, prov = _place_setup(cfg, _PlaceSeq([(6.0, 1.0, None)]), env)
    pol.report["events"].append({"kind": "post_place", "label": "block_0", "ok": True})
    out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
    assert "rehover" in out and pol.report["events"][-1]["confirm_needed"] == ["measured_support"]
    # ... unless it was picked since
    pol, prov = _place_setup(cfg, _PlaceSeq([(6.0, 1.0, None)]), env)
    pol.report["events"] += [{"kind": "post_place", "label": "block_0", "ok": True},
                             {"kind": "refine", "for": "pick", "label": "block_0", "ok": True}]
    out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
    assert "block_0" in out
    # below the noise floor on a partial view: dropped, no look
    pol, prov = _place_setup(cfg, _PlaceSeq([(2.0, 0.6, None)]), env)
    out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
    assert out is None and float(prov.belief.objects["block_0"].pos[0]) == x0
    # off (default): the same partial-view shift is applied as before
    pol, prov = _place_setup(RuntimeConfig(mode="perception", refine=True, rehover_when_far=False),
                             _PlaceSeq([(6.0, 0.6, None)]), env)
    out = pol._refine_hook(prov, _PlaceSk(), "block_0", "left", "place")
    assert "block_0" in out


def test_verify_select_risky_verifies_places_and_skips_plain_picks(monkeypatch):
    import dojo.runtime.estimate as E
    monkeypatch.setattr(E, "estimate_to_layout", lambda scene, objs: {"Rigid": {"block": [
        {"label": o["label"], "category_idx": o["category_idx"], "default_pos": o["pos"], "default_ori": o["quat"]}
        for o in objs]}})
    env = _env()
    tc = FakeTwinClient()
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="bundle", n_cands=2, verify_k=2, verify_select="risky"),
                        assets=FakeAssets(), twin_client=tc)
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert rep["error"] is None, rep["error"]
    assert _stacked(env)
    v = rep["verify"]
    sel = [e for e in rep["events"] if e["kind"] == "verify_select"]
    picks = [e for e in sel if e["segment"] == "pick"]
    places = [e for e in sel if e["segment"] == "place"]
    assert picks and places and not any(e["verify"] for e in picks) and all(e["verify"] for e in places)
    assert v["skipped"] == len(picks) and v["calls"] == len(places), v
    assert v["from_snapshot"] == 0                    # no kept pick to chain from: places load the checkpoint
    assert all(vv["from"] == "estimate" for vv in tc.verifies)


def test_verify_select_risky_marks_picks_after_a_failed_refine_or_a_repeat():
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="bundle", verify_select="risky"))
    pol.reset({"instruction": STACK, "step_lim": 550})
    assert pol._risky_segment("place", "block_0") == "place"
    assert pol._risky_segment("pick", "block_1") is None
    assert pol._risky_segment("pick", "block_1") == "repick"
    pol.report["events"].append({"kind": "refine", "for": "pick", "label": "block_2", "ok": False})
    assert pol._risky_segment("pick", "block_2") == "refine_failed"
    pol.report["events"].append({"kind": "refine", "for": "pick", "label": "block_0", "ok": True,
                                 "why": " (large correction not confirmed, prior kept)"})
    assert pol._risky_segment("pick", "block_0") == "refine_failed"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(verify_select="some"))


def test_watch_lays_the_knocked_tile_face_up_after_two_equal_reads():
    from dojo.skills.library import face_up, is_standing
    env = _env()

    class Watch:
        def __init__(self, reads, supported=True):
            self.reads, self.supported, self.n = list(reads), supported, 0

        def watch(self, frame0, frame, vocabulary):
            self.n += 1
            return {"supported": self.supported, "knocked": self.reads.pop(0) if self.reads else []}

    up = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]            # place "up": 90 deg about x (the face points up)
    tile = AssetInfo(key="Rigid/mahjong/00005", bbox_min=np.array([-0.012, -0.008, 0.0]),
                     bbox_max=np.array([0.012, 0.008, 0.034]), place={"up": np.asarray(up)})
    pc = Watch([[], ["block_0"], ["block_0"], ["block_0"]])
    cfg = RuntimeConfig(mode="perception", watch=True, watch_every=0)
    pol, prov = _place_setup(cfg, pc, env)
    o = prov.belief.objects["block_0"]
    o.asset = tile
    z_bottom = float(o.aabb()[0][2])
    assert is_standing(o) and not face_up(o)
    w = pol._make_watcher(prov, pol.bridge.obs, "make_kong")
    obs = pol.bridge.obs
    w(obs)
    w(obs)
    assert not face_up(o)                                             # one read is not enough
    w(obs)
    assert face_up(o) and abs(float(o.aabb()[0][2]) - z_bottom) < 1e-9   # lies where it stood
    ev = [e for e in pol.report["events"] if e["kind"] == "watch"]
    assert ev and ev[0]["laid"] == ["block_0"]
    w(obs)
    assert pc.n == 3                                                  # done: no more calls
    # a vocabulary without a watch read: one call, then never again
    pc2 = Watch([], supported=False)
    pol2, prov2 = _place_setup(cfg, pc2, env)
    w2 = pol2._make_watcher(prov2, pol2.bridge.obs, "stack_blocks")
    w2(obs)
    w2(obs)
    assert pc2.n == 1


@pytest.mark.parametrize("needs", [False, True])
def test_agentic_budget_estimates_at_frame0_and_once_more_only_when_perception_asks(needs):
    env = _env()

    class AgenticPerception:
        def __init__(self):
            self.calls = []

        def _rec(self, dx=0.0, tag="b1"):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                                         "default_pos": [p[0] + dx, p[1], p[2]], "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]},
                    "_perception": {"agentic": {"result": "agent", "calls": 3, "tokens": 9000, "agent_s": 30.0,
                                                "model": "fake", "merge": [{"label": "block_0"}]}}, "_tag": tag}

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            self.calls.append("estimate")
            return self._rec()

        def agentic_estimate(self, frame, vocabulary, fold, b1_record=None):
            self.calls.append(("agentic", b1_record is not None))
            return self._rec(tag="agentic")

        def needs_agent(self, record):
            self.calls.append(("needs", record.get("_tag")))
            return needs, "not_found:['x']" if needs else "b1_confident"

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

    pc = AgenticPerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", agentic="budget"), assets=FakeAssets(),
                        perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    assert _stacked(env)
    assert pc.calls[0] == ("agentic", False)                          # frame 0: agentic (B1 inside the service)
    assert ("needs", "agentic") in pc.calls
    n_ag = sum(1 for c in pc.calls if isinstance(c, tuple) and c[0] == "agentic")
    assert n_ag == (2 if needs else 1) and len(rep["agentic"]) == n_ag
    assert rep["agentic"][0]["calls"] == 3 and rep["agentic"][0]["tokens"] == 9000
    if needs:
        assert pc.calls[pc.calls.index(("needs", "agentic")) + 1] == ("agentic", True)   # B1 hint = the current record
    # off (default): no agentic op is ever called
    pc2 = AgenticPerception()
    rep2 = run_episode(RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(),
                                     perception_client=pc2), FakeEpisode(_env()), "perception")
    assert rep2["agentic"] == [] and not any(isinstance(c, tuple) for c in pc2.calls)


def test_agentic_risk_trigger_on_reestimate_respects_the_budget():
    from dojo.runtime.policy import _Checker

    class PC:
        def __init__(self):
            self.n_ag = 0

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return {"_tag": "b1"}

        def needs_agent(self, record):
            return True, "risk:[('block_0', 0.4)]"

        def agentic_estimate(self, frame, vocabulary, fold, b1_record=None):
            self.n_ag += 1
            assert b1_record == {"_tag": "b1"}                          # the B1 re-estimate is the hint
            return {"_tag": "agentic", "_perception": {"agentic": {"result": "agent", "calls": 2}}}

    pc = PC()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", agentic="budget", agentic_max=3), perception_client=pc)
    pol.reset({"instruction": STACK, "step_lim": 550})
    pol.bridge.obs, pol.bridge.fresh = FakeEpisode(_env()).observe(), True
    pol.report["agentic"] = [{"tag": "frame0"}, {"tag": "after_looks"}]       # two of three already spent
    ch = _Checker(pc, "stack_blocks", 0)
    ch.agentic_hook = lambda obs, rec: pol._agentic_gate(obs, "stack_blocks", rec, "risk")
    assert ch.estimate({})["_tag"] == "agentic" and pc.n_ag == 1
    assert ch.estimate({})["_tag"] == "b1" and pc.n_ag == 1                 # budget spent: B1 stays
    gates = [e for e in pol.report["events"] if e["kind"] == "agentic_gate"]
    assert [g["trigger"] for g in gates] == ["risk", "risk"] and gates[-1]["why"] == "budget spent"
    assert [a["tag"] for a in pol.report["agentic"]] == ["frame0", "after_looks", "risk"]


@pytest.mark.parametrize("touched, conf, moves", [(False, 0.6, False), (False, 0.9, True), (True, 0.6, True)])
def test_tracker_keeps_an_untouched_object_against_an_unsure_reread(touched, conf, moves):
    """Divergence re-estimate through perception's tracker: an object the robot did not move keeps its belief pose
    unless the re-read is confident (>= 0.8); one it moved takes the re-read."""
    from dojo.runtime.belief import RuntimeProvider
    env = _env([(-0.30, -0.15, 0.78), (0.10, -0.10, 0.78), (0.30, -0.20, 0.78)])

    class Moved:
        def check(self, obs, objects):
            return {o.label: {"head": {"prec": 0.0 if o.label == "block_1" else 1.0, "n_vis": 500, "n_px": 600}}
                    for o in objects}

        def estimate(self, obs):
            return {"Rigid": {"block": [
                {"category": "block", "category_idx": 5, "label": "block_1", "default_pos": [0.12, -0.10, 0.78],
                 "default_ori": [1, 0, 0, 0], "_perception": {"confidence": conf}}]}}

    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets(), checker=Moved(), replan_on_divergence=False)
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.belief.objects["block_1"].confidence = 0.75
    prov.tracker_on = True
    if touched:
        prov.touched.add("block_1")
    prov.resync(b.obs)
    x = float(prov.belief.objects["block_1"].pos[0])
    assert (abs(x - 0.12) < 1e-9) == moves and (moves or abs(x - 0.10) < 1e-9)
    if not moves:
        assert any(e["kind"] == "tracker_kept" and e["labels"] == ["block_1"] for e in prov.events)


def test_held_look_drops_an_empty_grasp_from_the_jaws():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    calls = []
    prov.held_checker = lambda obs, arm, lab: calls.append((arm, lab)) or {"held": False, "confidence": 0.8, "source": "b1"}
    prov.held_confirm = lambda obs, arm, lab: {"held": False, "confidence": 0.9, "source": "agent"}   # agrees
    bel = prov.belief
    o = bel.objects["block_1"]
    pose0 = np.r_[o.pos, o.quat].copy()
    z0 = float(bel.kin.tcp("left", bel.q["left"])[2, 3])
    bel.attached["left"] = ("block_1", np.eye(4))
    bel.grasp_z["left"] = (z0 - 0.05, 1, pose0)                      # closed 5 cm below where the TCP is now
    o.pos = o.pos + np.array([0.0, 0.0, 0.05])                      # the kinematic belief lifted it
    prov._held_check(b.obs)
    assert calls == [("left", "block_1")] and "left" not in bel.attached
    assert np.allclose(bel.objects["block_1"].pos, pose0[:3], atol=1e-6)    # back where it was grasped
    assert prov.events[-1]["kind"] == "held_look" and prov.events[-1]["dropped"]
    prov._held_check(b.obs)
    assert len(calls) == 1                                          # once per grasp
    # a confident "held" keeps the grasp
    bel.attached["left"] = ("block_1", np.eye(4))
    bel.grasp_z["left"] = (z0 - 0.05, 2, pose0)
    prov.held_checker = lambda obs, arm, lab: {"held": True, "confidence": 0.9}
    prov._held_check(b.obs)
    assert "left" in bel.attached


def test_episode_with_held_look_checks_every_grasp_once():
    env = _env()

    class HeldPerception:
        def __init__(self):
            self.held = []

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                                         "default_pos": list(p), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]}}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

        def held_look(self, frame, arm, mode):
            self.held.append((arm, mode))
            return {"held": True, "confidence": 0.9, "calls": 1 if mode == "b1_then_agent" else 0, "source": "agent"}

    pc = HeldPerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", held_look="b1_then_agent", tracker="on"),
                        assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None and _stacked(env)
    hl = [e for e in rep["events"] if e["kind"] == "held_look"]
    n_picks = sum(1 for h in rep["history"] if h["skill"] == "pick")
    assert n_picks >= 2 and len(hl) == len(pc.held) == n_picks and all(e["held"] for e in hl)   # one check per grasp
    assert [a["tag"] for a in rep["agentic"]] == ["held"] * n_picks and sum(a["calls"] for a in rep["agentic"]) == n_picks


@pytest.mark.parametrize("mode, needs, step_lim, ran", [("always", False, 550, True), ("trigger", True, 550, True),
                                                       ("trigger", False, 550, False), ("trigger", True, 200, False)])
def test_scan_macro_parks_both_wrists_and_starts_from_the_scan_pose(mode, needs, step_lim, ran):
    env = _env(step_lim=step_lim)
    scan_q = {"left": [-0.9523, -0.24314, 1.28768, -1.77756, -0.63467, -0.14824],
              "right": [0.94768, -0.24334, 1.28759, -1.77684, 0.63018, 0.14684]}

    class ScanPerception:
        def __init__(self):
            self.calls = []

        def _rec(self):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                                         "default_pos": list(p), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]}}

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return self._rec()

        def scan_looks(self):
            return [{"arm": a, "joints": q, "steps_out": 36} for a, q in scan_q.items()]

        def scan_trigger(self, record):
            self.calls.append("trigger")
            return needs, "hidden:['block_1']" if needs else "frame0_clear", False

        def estimate_scan(self, frame, frame0, vocabulary, fold, rec0=None):
            self.calls.append(("scan", [round(float(v), 3) for v in frame["state"]["left_arm_joint_state"]]))
            return self._rec()

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

    pc = ScanPerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", scan=mode), assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    ev = next(e for e in rep["events"] if e["kind"] == "scan")
    assert ev["ran"] == ran and (mode == "always") == ("trigger" not in pc.calls)
    if mode == "trigger":                                            # both step gates logged per episode
        assert ev["frac_ok"] and ev["min_step_lim_ok"] == (step_lim >= 300)
    if ran:
        scans = [c for c in pc.calls if isinstance(c, tuple)]
        assert len(scans) == 1 and np.allclose(scans[0][1], scan_q["left"], atol=1e-3)      # observed at the scan pose
        assert ev["steps_used"] == ev["steps"] and 10 <= ev["steps"] <= 40 and rep["look_steps"] >= ev["steps"]
        assert "scan" in rep["records"]
        assert _stacked(env)                                         # the skills started from the scan pose
    else:
        assert not any(isinstance(c, tuple) for c in pc.calls)


def test_scan_trigger_rule_on_a_frame0_record():
    from dojo.runtime.perception_service import _Perception
    base = {"category": "block", "category_idx": 5, "default_pos": [0.1, -0.1, 0.78], "default_ori": [1, 0, 0, 0]}
    clear = {"Rigid": {"block": [dict(base, label="block_0", _perception={"confidence": 0.95, "residual": 0.05})]}}
    assert _Perception.scan_trigger({**clear, "_not_found": [{"label": "block_1"}]}) == \
        {"needs": True, "why": "not_found:['block_1']", "not_found": True}
    hidden = {"Rigid": {"block": [dict(base, label="block_0",
                                       _perception={"confidence": 0.95, "residual": 0.05, "hidden_frac": 0.3})]}}
    r = _Perception.scan_trigger(hidden)
    assert r["needs"] and r["why"].startswith("hidden") and not r["not_found"]


FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _vocab_rates(monkeypatch, data, **kw):
    import json as _json

    from dojo.runtime import vocab
    monkeypatch.setattr(vocab, "catalog_path", lambda: str(FIX / "catalog_task_label_categories.json"))
    ok = n = 0
    for task, ins in _json.loads((FIX / data).read_text())["instructions"].items():
        for i in ins:
            n += 1
            v = perception_vocabulary(i, **kw)
            ok += v is not None and v.name in (task, task.replace("_random", ""), task + "_random")
    return ok, n


def test_vocabulary_fallback_names_the_candidate_set_from_the_objects(monkeypatch):
    """The table is the fast path; the catalogue fallback (objects the instruction names) covers what it misses. All
    public templates and the paraphrase set resolve with both; the fallback alone gets most."""
    assert _vocab_rates(monkeypatch, "robodojo_instructions.json") == (55, 55)
    assert _vocab_rates(monkeypatch, "instruction_paraphrases.json") == (44, 44)
    ok, n = _vocab_rates(monkeypatch, "robodojo_instructions.json", table=False)
    assert ok >= 51, (ok, n)
    ok, n = _vocab_rates(monkeypatch, "instruction_paraphrases.json", table=False)
    assert ok >= 43, (ok, n)
    v = perception_vocabulary("Put the charger into the socket.")
    assert v.name == "plug_in_charger" and v.source == "catalog"
    assert perception_vocabulary("Wave hello.") is None                      # names no catalogue object


def test_lane_vocabularies_match_their_templates():
    import json as _json
    fx = _json.loads((FIX / "robodojo_instructions.json").read_text())["instructions"]
    lane = ["fold_clothes", "hang_mugs", "make_toast", "store_laptop_and_headphones", "sweep_blocks", "build_tower",
            "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",
            "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table", "play_stacking_toy", "play_tic_tac_toe",
            "imitate_sorting_sequence", "match_and_pick_from_conveyor", "pick_from_conveyor_by_image",
            "store_tools_in_toolbox", "stack_blocks", "press_by_number", "swap_blocks", "stack_blocks_by_language"]
    for t in lane:
        v = perception_vocabulary(fx[t][0], fallback=False)
        assert v is not None and v.name == t and v.source == "table", (t, v)


@pytest.mark.parametrize("via, far_m, accepted", [("full_fuse", 0.25, False), ("full_fuse", 0.05, True),
                                                  ("refine", 0.25, True)])
def test_lost_object_search_rejects_a_far_fuse_find(via, far_m, accepted):
    from types import SimpleNamespace

    from dojo.runtime.belief import LostObject, RuntimeProvider
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    pol = RuntimePolicy(RuntimeConfig(mode="perception", search_lost=True), assets=FakeAssets())
    pol.reset(FakeEpisode(env).info())
    pol.bridge = b
    o = prov.belief.objects["block_1"]
    p0 = o.pos.copy()
    found = [p0[0] + far_m, p0[1], p0[2]]

    def fake_search(sk, obs0, rec, scene, kin, seed=None, min_budget=None):
        pol.report.setdefault("search", []).append({"via": via, "found": ["block_1"]})
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": "block_1", "default_pos": found,
                                     "default_ori": [1, 0, 0, 0]}]}, "_not_found": []}
    pol._search = fake_search
    ctx = SimpleNamespace(prov=prov, sk=None, bridge=b, scene="stack_blocks", kin=None)
    ok = pol._search_lost(ctx, LostObject("block_1", o.pos, o.quat, o.asset.key))
    assert ok == accepted
    assert np.allclose(prov.belief.objects["block_1"].pos, found if accepted else p0)
    assert any(e["kind"] == "lost_fuse_rejected" for e in pol.report["events"]) == (not accepted)


def test_ambiguous_instruction_picks_the_candidate_set_that_explains_the_scene(monkeypatch):
    """'Cover the blocks ...': the table says cover_blocks, the catalogue fallback says build_tower (it names only
    "blocks"). Frame 0 is estimated with both; cover_blocks' set finds all its objects, build_tower's does not."""
    from dojo.runtime import vocab
    from dojo.runtime.supervisor import vocabulary_candidates
    monkeypatch.setattr(vocab, "catalog_path", lambda: str(FIX / "catalog_task_label_categories.json"))
    text = ("Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, "
            "and blue.")
    names = [v.name for v in vocabulary_candidates(text)]
    assert names[0] == "cover_blocks" and "build_tower" in names and len(names) <= 3
    assert [v.name for v in vocabulary_candidates("Plug the charger into the power strip.")] == ["plug_in_charger"]
    env = _env()

    class TwoSets:
        def __init__(self):
            self.vocabs = []

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            self.vocabs.append(vocabulary)
            recs = [{"category": "block", "category_idx": 5, "label": f"block_{i}", "default_pos": list(p),
                     "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]
            if vocabulary == "cover_blocks":
                return {"Rigid": {"block": recs}}
            return {"Rigid": {"block": recs[:1]}, "_not_found": [{"label": f"block{i}"} for i in range(1, 8)]}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

    pc = TwoSets()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(), perception_client=pc)
    pol.reset({"instruction": text, "step_lim": 550})
    pol.bridge.obs, pol.bridge.fresh = FakeEpisode(env).observe(), True
    v, rec = pol._explain_vocabulary(vocabulary_candidates(text), {"instruction": text})
    assert v.name == "cover_blocks" and len(rec["Rigid"]["block"]) == 3
    ev = pol.report["events"][-1]
    assert ev["kind"] == "vocab_explain" and ev["chosen"] == "cover_blocks" and len(ev["candidates"]) == len(names)


def test_vocabulary_fallback_treats_an_unreadable_catalogue_as_unavailable(monkeypatch):
    """CI runners: /root exists but is unreadable, so the catalogue probe raises PermissionError; the table still
    answers and the fallback is simply off."""
    from pathlib import Path as _P

    from dojo.runtime import vocab
    from dojo.runtime.supervisor import vocabulary_candidates
    vocab._load.cache_clear()

    def denied(self):
        raise PermissionError(13, "Permission denied", str(self))
    monkeypatch.setattr(_P, "is_file", denied)
    monkeypatch.delenv("DOJO_CATALOG", raising=False)
    assert vocab.catalog_path() is None
    assert perception_vocabulary(STACK).name == "stack_blocks"
    assert perception_vocabulary("Put the charger into the socket.") is None
    assert [v.name for v in vocabulary_candidates(STACK)] == ["stack_blocks"]
    monkeypatch.undo()
    monkeypatch.setattr(vocab, "catalog_path", lambda: "/root/no/tasks.json")
    monkeypatch.setattr(vocab, "_load", lambda p: (_ for _ in ()).throw(PermissionError(13, "denied")))
    assert vocab.catalog_vocabulary("Put the charger into the socket.")[0] is None


@pytest.mark.parametrize("needs", [True, False])
def test_agentic_with_scan_is_one_call_with_the_scan_as_second_observation(needs):
    env = _env()
    scan_q = {"left": [-0.9523, -0.24314, 1.28768, -1.77756, -0.63467, -0.14824],
              "right": [0.94768, -0.24334, 1.28759, -1.77684, 0.63018, 0.14684]}

    class P:
        def __init__(self):
            self.ag = []

        def _rec(self, tag):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                                         "default_pos": list(p), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]}, "_tag": tag}

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return self._rec("b1")

        def scan_looks(self):
            return [{"arm": a, "joints": q, "steps_out": 12} for a, q in scan_q.items()]

        def scan_trigger(self, record):
            return needs, "hidden" if needs else "clear", False

        def estimate_scan(self, frame, frame0, vocabulary, fold, rec0=None):
            return self._rec("scan")

        def agentic_estimate(self, frame, vocabulary, fold, b1_record=None, extra_frame=None):
            self.ag.append((b1_record.get("_tag"), extra_frame is not None and np.allclose(
                extra_frame["state"]["left_arm_joint_state"], scan_q["left"], atol=1e-3)))
            r = self._rec("agentic")
            r["_perception"] = {"agentic": {"result": "agent", "calls": 3}}
            return r

        def needs_agent(self, record):
            return False, "b1_confident"

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

    pc = P()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", scan="trigger", agentic="budget", agentic_max=3),
                        assets=FakeAssets(), perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None and _stacked(env)
    assert pc.ag == ([("scan", True)] if needs else [("b1", False)])     # ONE call: frame 0 (+ the scan when it ran)
    assert rep["agentic"][0]["tag"] == "frame0" and rep["agentic"][0]["with_scan"] == needs


def test_actors_moving_compares_with_the_frame_a_window_back():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    ep = FakeEpisode(env)
    b.obs, b.fresh = ep.observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    assert prov.actors_moving() is None                                   # no reader: the skill holds a fixed time
    pairs = []

    def reader(a, b_):
        pairs.append((a["steps_used"], b_["steps_used"]))
        return {"moving": b_["steps_used"] < 20}                          # the scripted arm stops at step 20
    prov.motion_reader = reader
    out = []
    for step in range(0, 40):
        b.obs, b.fresh = dict(ep.observe(), steps_used=step), True
        out.append(prov.actors_moving())
    assert out[0] is True                                                 # nothing to compare with yet
    assert all(pairs[k][1] - pairs[k][0] == min(12, pairs[k][1]) for k in range(len(pairs)))   # 12-step window
    assert out[19] is True and out[25] is False


def test_belt_tracking_fits_the_velocity_and_moves_the_belief():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    ep = FakeEpisode(env)
    b.obs, b.fresh = ep.observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    x0 = float(prov.belief.objects["block_0"].pos[0])
    v = 0.004                                                             # 4 mm / step along x
    rng = np.random.default_rng(1)

    def reader(obs):
        s_ = obs["steps_used"]
        recs = []
        for o in env.objects:
            p = list(o["pos"])
            if o["label"] == "block_0":
                p[0] = x0 + v * s_ + float(rng.normal(0, 0.0005))
            recs.append({"category": "block", "category_idx": 5, "label": o["label"], "default_pos": p,
                         "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}})
        return {"Rigid": {"block": recs}}
    prov.track_reader = reader
    for step in range(0, 44):
        b.obs, b.fresh = dict(ep.observe(), steps_used=step), True
        prov._track(b.obs)
    assert abs(float(prov.belt_velocity()[0]) - v) < 2e-4 and abs(float(prov.belt_velocity()[1])) < 2e-4
    assert abs(float(prov.belief.objects["block_0"].pos[0]) - (x0 + v * 43)) < 0.002     # the fit, at this step
    assert float(np.linalg.norm(prov.velocity.get("block_1", np.zeros(2)))) < 3e-4       # the others are at rest
    assert prov.stats["track_reads"] == 11                                              # every 4th step


def test_twin_is_started_for_the_scene_the_policy_chose():
    started = []
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="bundle"))
    pol._twin_factory = lambda scene: started.append(scene) or FakeTwinClient(task=scene)
    assert pol.twin("cover_blocks").info()["task"] == "cover_blocks" and started == ["cover_blocks"]
    assert pol.twin("build_tower").info()["task"] == "cover_blocks" and started == ["cover_blocks"]   # once


def test_refine_suggestion_onto_a_neighbour_is_not_followed():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    other = next(o for o in env.objects if o["label"] == "block_2")

    class Suggest:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            return {o["label"]: {"ok": False, "pos": list(other["pos"]), "quat": [1, 0, 0, 0], "iou": 0.9,
                                 "shift_mm": 90.0, "suggest_xy": list(other["pos"][:2]), "suggest_shift_mm": 90.0}}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=Suggest())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    assert pol._refine_hook(prov, None, "block_1", "left", "pick") is None
    assert "neighbour" in pol.report["events"][-1]["why"]


def test_twin_state_copy_keeps_unpicklable_handles():
    from dojo.simlab.twin import _safe_copy

    class Stage:                                                       # like pxr.Usd.Stage: refuses (deep)copy
        def __reduce_ex__(self, proto):
            raise RuntimeError('Pickling of "pxr.Usd.Stage" instances is not enabled')
    st = Stage()
    v = {"pre_state": {"bowl0": [1.0, 2.0], "stage": st}, "list": [st, {"a": [3]}]}
    c = _safe_copy(v)
    assert c["pre_state"]["stage"] is st and c["list"][0] is st          # the handle is kept
    assert c["pre_state"]["bowl0"] == [1.0, 2.0] and c["pre_state"]["bowl0"] is not v["pre_state"]["bowl0"]
    assert c["list"][1]["a"] is not v["list"][1]["a"]                   # everything else is still a deep copy


def test_refine_flip_keeps_the_prior_orientation():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class Flip:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            p = list(o["pos"])
            p[0] += 0.004
            return {o["label"]: {"ok": True, "pos": p, "quat": [0.0, 0.0, 0.0, 1.0], "shift_mm": 4.0, "iou": 0.97,
                                 "turned_deg": 178.0}}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=Flip())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    q0 = prov.belief.objects["block_1"].quat.copy()
    x0 = float(prov.belief.objects["block_1"].pos[0])
    out = pol._refine_hook(prov, None, "block_1", "left", "pick")
    o = prov.belief.objects["block_1"]
    assert out and np.allclose(o.quat, q0) and abs(float(o.pos[0]) - x0 - 0.004) < 1e-9
    assert "prior orientation kept" in pol.report["events"][-1]["why"]


@pytest.mark.parametrize("api", ["old", "new"])
def test_not_ok_refine_rehovers_over_the_fit_with_either_perception_api(api):
    """perception #36: a not-ok refine sends the prior as the pose and the raw fit as fit_pose; the runtime re-hovers
    over the fit either way, and a truncated object (larger than the wrist view) keeps its prior without re-hover."""
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class R:
        def __init__(self, why=None):
            self.why = why

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            fit = [o["pos"][0] + 0.02, o["pos"][1], o["pos"][2]]
            r = {"ok": False, "iou": 0.3, "p_why": self.why}
            if api == "new":
                r.update(pos=list(o["pos"]), quat=o["quat"], shift_mm=0.0, fit_pos=fit, fit_quat=o["quat"], fit_shift_mm=20.0)
            else:
                r.update(pos=fit, quat=o["quat"], shift_mm=20.0)
            return {o["label"]: r}

    def hook(pc):
        pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=pc)
        pol.reset(FakeEpisode(env).info())
        b = pol.bridge
        b.obs, b.fresh = FakeEpisode(env).observe(), True
        prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
        prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
        x0 = float(prov.belief.objects["block_1"].pos[0])
        return pol._refine_hook(prov, None, "block_1", "left", "pick"), x0, pol
    out, x0, _ = hook(R())
    assert out and abs(float(out["rehover"][0][0]) - (x0 + 0.02)) < 1e-9
    out, _, pol = hook(R("truncated"))
    assert out is None and "truncated" in pol.report["events"][-1]["why"]


def test_provider_watch_read_goes_to_perception_on_the_current_frame():
    env = _env()

    class W:
        def __init__(self):
            self.calls = []

        def estimate(self, frames, vocabulary, fold, prior_record=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                                         "default_pos": list(p), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]}}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

        def look_plan(self, frame, record, max_looks=2):
            return []

        def watch_read(self, frame, region, vocabulary, fold, prior=None):
            self.calls.append((region["kind"], vocabulary, frame["steps_used"], prior))
            return {"occluded": False, "visible_frac": 0.9, "stable_vs_prior": True, "new": [], "inside": []}

    seen = {}

    def program(ctx):
        seen["read"] = ctx.prov.watch_read({"kind": "container", "container": {"asset": "Geometry/basket/00005",
                                                                              "pos": [0, 0, 0.77], "quat": [1, 0, 0, 0]},
                                            "candidates": []}, prior={"inside": []})
        return {"ok": True}
    pc = W()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(), perception_client=pc,
                        program=program)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None
    assert pc.calls == [("container", "stack_blocks", 0, {"inside": []})] and seen["read"]["visible_frac"] == 0.9
    assert rep["watch_reads"] == 1


def test_every_table_vocabulary_is_a_catalogue_candidate_set():
    import json as _json

    from dojo.runtime.supervisor import VOCABULARIES
    sets = _json.loads((FIX / "catalog_task_label_categories.json").read_text())
    assert [v.name for v in VOCABULARIES if v.name not in sets] == []


def test_catalog_check_finds_and_syncs_missing_assets(tmp_path):
    import json as _json

    from dojo.runtime.catalog_check import missing_assets, sync
    cat, src = tmp_path / "catalog", tmp_path / "skills_catalog"
    for k, root in (("Rigid/cube/00003", cat), ("Rigid/mug/00000", src)):
        (root / k).mkdir(parents=True)
        (root / k / "geom.npz").write_bytes(b"x")
        (root / k / "info.json").write_text("{}")
    (cat / "tasks.json").write_text(_json.dumps({
        "swap_blocks": {"labels": {"a": {"candidates": ["Rigid/cube/00003"]}}},
        "hang_mugs": {"labels": {"m": {"candidates": ["Rigid/mug/00000", "Geometry/cup_holder/00000"]}}}}))
    (cat / "index.json").write_text(_json.dumps({"Rigid/cube/00003": {}}))
    miss = missing_assets(cat, ["swap_blocks", "hang_mugs", "make_toast"])
    assert miss == {"hang_mugs": ["Geometry/cup_holder/00000", "Rigid/mug/00000"], "make_toast": ["<not in tasks.json>"]}
    done = sync(cat, miss["hang_mugs"], [src], run_tools=False)
    assert [k for k, _ in done["copied"]] == ["Rigid/mug/00000"] and done["unresolved"] == ["Geometry/cup_holder/00000"]
    assert (cat / "Rigid/mug/00000/geom.npz").is_file()


@pytest.mark.skipif(not Path("~/dojo-perception-data/catalog/tasks.json").expanduser().is_file(),
                    reason="needs the perception catalogue (compute / GPU boxes)")
def test_the_installed_catalogue_holds_every_table_vocabulary():
    from dojo.runtime.catalog_check import missing_assets
    assert missing_assets(Path("~/dojo-perception-data/catalog").expanduser()) == {}



@pytest.mark.parametrize("confirm", [None, {"held": True, "confidence": 0.9}])
def test_held_look_never_drops_on_a_single_vision_negative(confirm):
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.held_checker = lambda obs, arm, lab: {"held": False, "confidence": 1.0, "source": "b1"}
    prov.held_confirm = None if confirm is None else (lambda obs, arm, lab: confirm)
    bel = prov.belief
    o = bel.objects["block_1"]
    z0 = float(bel.kin.tcp("left", bel.q["left"])[2, 3])
    bel.attached["left"] = ("block_1", np.eye(4))
    bel.grasp_z["left"] = (z0 - 0.05, 1, np.r_[o.pos, o.quat].copy())
    prov._held_check(b.obs)
    assert "left" in bel.attached and prov.events[-1]["kept"]                 # still in the jaws


def test_a_none_ok_prediction_is_not_adopted_and_drops_the_twin_chain():
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.verifier import BundleVerifier
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    tc = FakeTwinClient()
    ver = BundleVerifier(tc, "stack_blocks", prov, b)
    ver.chain = {"sid": 7, "n": 0}
    ver.kept = {0: 8}
    tc.snaps = {7: None, 8: None}
    info = {"idx": 2, "none_ok": True, "outcomes": [], "actions": []}
    prov.on_verified(info)
    ver.on_verified(info)
    assert prov.pred is None and prov.stats["pred_untrusted"] == 1
    assert ver.chain is None and ver.kept == {} and tc.snaps == {}


def test_verify_choose_runs_the_best_outcome_when_nothing_passes():
    from types import SimpleNamespace

    from dojo.skills.skills import Skills
    from dojo.skills.verify import Outcome

    outs = [Outcome(c, s_, {}, {}, False, False, 0.0, False) for c in range(3) for s_ in range(2)]

    def judge(c):
        j = lambda o: False                                             # nothing passes
        j.score = lambda o, c=c: [-0.05, -0.01, -0.03][c]               # candidate 1 comes closest
        return j
    seen = []
    fake = SimpleNamespace(ex=SimpleNamespace(actions_for=lambda c: [c]), verify_k=2,
                           verifier=SimpleNamespace(rollout=lambda acts, k, held_labels: outs),
                           vstats={"segments": 0, "first_ok": 0, "later_ok": 0, "none_ok": 0, "candidates": 0},
                           on_verified=seen.append, last_verified=None)
    idx, passes = Skills.verify_choose(fake, [{"a": 0}, {"a": 1}, {"a": 2}], [judge(0), judge(1), judge(2)], kind="place")
    assert idx == 1 and passes == [0, 0, 0] and seen[-1]["none_ok"] and fake.vstats["best_outcome"] == 1


@pytest.mark.parametrize("jaw, pick, held", [((0.1, 500), (0.9, 500), False), ((0.8, 500), (0.1, 500), True),
                                             ((0.1, 10), (0.9, 500), None), ((0.1, 500), (0.4, 500), None)])
def test_held_confirmation_uses_the_head_view_at_the_jaw_and_pick_poses(jaw, pick, held):
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class Head:
        def __init__(self):
            self.n = 0

        def check(self, frame, objects, fold):
            self.n += 1
            p, v = jaw if self.n == 1 else pick
            return {objects[0]["label"]: {"head": {"prec": p, "n_vis": v, "n_px": 600}}}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", held_look="b1"), assets=FakeAssets(), perception_client=Head())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    o = prov.belief.objects["block_1"]
    prov.belief.attached["left"] = ("block_1", np.eye(4))
    prov.belief.grasp_z["left"] = (0.8, 1, np.r_[o.pos, o.quat].copy())
    r = pol._held_head_confirm(prov, b.obs, "left", "block_1")
    assert r["held"] is held and r["source"] == "head" and r["calls"] == 0


def test_post_place_adopts_a_height_only_correction():
    """A released block the belief floats 56 mm high (a no-candidate place): perception re-fits it from its support
    at the same xy; the belief comes down although the xy shift is 0."""
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True

    class Z:
        def refine(self, obs, o, others, arm, held=None):
            return {"ok": True, "pos": [float(o.pos[0]), float(o.pos[1]), float(o.pos[2]) - 0.056], "quat": list(o.quat),
                    "shift_mm": 0.0, "iou": 0.84, "z_dropped_mm": 56.0}
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets(), checker=Z())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    o = prov.belief.objects["block_1"]
    o.pos = o.pos + np.array([0.0, 0.0, 0.056])
    prov.belief.released_by["block_1"] = "left"
    z_high = float(o.pos[2])
    ev = prov.post_place(b.obs, ["block_1"])
    assert abs(float(prov.belief.objects["block_1"].pos[2]) - (z_high - 0.056)) < 1e-9
    assert ev[0]["z_dropped_mm"] == 56.0 and ev[0].get("to")


def test_pick_refine_identity_is_restricted_to_the_labels_own_group(tmp_path, monkeypatch):
    import json as _json

    from dojo.runtime import vocab
    from dojo.runtime.belief import RuntimeProvider
    (tmp_path / "tasks.json").write_text(_json.dumps({"stack_blocks": {"labels": {
        "block_1": {"candidates": ["Rigid/block/00005"]}, "block_2": {"candidates": ["Rigid/block/00003"]}}}}))
    monkeypatch.setattr(vocab, "catalog_path", lambda: str(tmp_path / "tasks.json"))
    env = _env()
    seen = {}

    class R:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            seen["idv"] = identity_vocabulary
            o = objects[0]
            return {o["label"]: {"ok": True, "pos": list(o["pos"]), "quat": o["quat"], "shift_mm": 0.0, "iou": 0.9}}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True), assets=FakeAssets(), perception_client=R())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    pol._refine_hook(prov, None, "block_1", "left", "pick")
    assert seen["idv"] == ["Rigid/block/00005"]                       # its own group, not the whole vocabulary


def test_a_failed_recentre_reverts_the_correction():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class Seq:
        def __init__(self):
            self.n = 0

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            self.n += 1
            if self.n == 1:
                return {o["label"]: {"ok": True, "pos": [o["pos"][0] + 0.024, o["pos"][1], o["pos"][2]], "quat": o["quat"],
                                     "shift_mm": 24.0, "iou": 0.79, "visible_frac": 1.0}}
            return {o["label"]: {"ok": False, "pos": list(o["pos"]), "quat": o["quat"], "shift_mm": 0.0, "iou": 0.0,
                                 "visible_frac": 1.0}}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07), assets=FakeAssets(),
                        perception_client=Seq())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    x0 = float(prov.belief.objects["block_1"].pos[0])
    h = pol._refine_hook(prov, None, "block_1", "left", "pick")
    assert h["rehover"] is not None and abs(float(prov.belief.objects["block_1"].pos[0]) - (x0 + 0.024)) < 1e-9
    assert pol._refine_hook(prov, None, "block_1", "left", "pick") is None
    assert float(prov.belief.objects["block_1"].pos[0]) == x0 and "reverted" in pol.report["events"][-1]["why"]


def test_no_higher_rehover_over_a_fully_covered_support():
    env = _env()

    class Covered:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            return {o["label"]: {"ok": False, "pos": list(o["pos"]), "quat": o["quat"], "iou": 0.0, "visible_frac": 0.0}}
    pol, prov = _place_setup(RuntimeConfig(mode="perception", refine=True), Covered(), env)
    prov.belief.attached["left"] = ("block_2", np.eye(4))

    class Sk:
        refine_ctx = {"kind": "place", "label": "block_0", "arm": "left", "held": ("Rigid/block/00005", [0, 0, 0.9, 1, 0, 0, 0])}
    assert pol._refine_hook(prov, Sk(), "block_0", "left", "place") is None



def test_a_covered_recentre_keeps_the_correction():
    from dojo.runtime.belief import RuntimeProvider
    env = _env()

    class Seq:
        def __init__(self):
            self.n = 0

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            o = objects[0]
            self.n += 1
            if self.n == 1:
                return {o["label"]: {"ok": True, "pos": [o["pos"][0] + 0.02, o["pos"][1], o["pos"][2]], "quat": o["quat"],
                                     "shift_mm": 20.0, "iou": 0.8, "visible_frac": 1.0}}
            return {o["label"]: {"ok": False, "pos": list(o["pos"]), "quat": o["quat"], "iou": 0.0, "visible_frac": 0.1}}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07), assets=FakeAssets(),
                        perception_client=Seq())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    x0 = float(prov.belief.objects["block_1"].pos[0])
    pol._refine_hook(prov, None, "block_1", "left", "pick")
    pol._refine_hook(prov, None, "block_1", "left", "pick")
    assert abs(float(prov.belief.objects["block_1"].pos[0]) - (x0 + 0.02)) < 1e-9          # kept
    assert "correction kept" in pol.report["events"][-1]["why"]


@pytest.mark.parametrize("pred_prec, corr_prec, adopted", [(0.9, 0.7, False), (0.2, 0.9, True), (None, None, False)])
def test_a_large_post_place_correction_needs_the_head_view(pred_prec, corr_prec, adopted):
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True

    class C:
        def __init__(self):
            self.n = 0

        def refine(self, obs, o, others, arm, held=None):
            return {"ok": True, "pos": [float(o.pos[0]) + 0.041, float(o.pos[1]), float(o.pos[2])], "quat": list(o.quat),
                    "shift_mm": 41.0, "iou": 0.8}

        def check(self, obs, objects):
            self.n += 1
            p = pred_prec if self.n == 1 else corr_prec
            return {objects[0].label: {"head": {"prec": p, "n_vis": 500 if p is not None else 0, "n_px": 600}}}
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets(), checker=C())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.belief.released_by["block_1"] = "left"
    x0 = float(prov.belief.objects["block_1"].pos[0])
    prov.post_place(b.obs, ["block_1"])
    x1 = float(prov.belief.objects["block_1"].pos[0])
    assert (abs(x1 - (x0 + 0.041)) < 1e-9) == adopted and (adopted or x1 == x0)


class _HeldPosePerception:
    """B1 frame 0 from the env's truth; held_pose answers ``answer(prior)`` and records the calls."""

    def __init__(self, env, answer):
        self.env, self.answer, self.calls = env, answer, []

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                     "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                     "_perception": {"confidence": 0.9}} for o in self.env.objects]}}

    def check(self, frame, objects, fold):
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []

    def held_pose(self, frame, arm, asset_key, prior_T_tcp_obj, cfg=None):
        self.calls.append((arm, asset_key, frame["steps_used"], np.asarray(prior_T_tcp_obj, float)))
        return self.answer(np.asarray(prior_T_tcp_obj, float))


@pytest.mark.parametrize("held_pose", ["off", "auto"])
def test_held_pose_reads_the_grasp_at_every_place_hover_only_when_on(held_pose):
    env = _env()

    def ok(prior):                                        # the read: the object 2 mm off the prediction
        T = prior.copy()
        T[:3, 3] += [0.002, 0.0, 0.0]
        return {"ok": True, "why": "ok", "T_tcp_obj": T.tolist(), "shift_mm": 2.0, "turn_deg": 0.0,
                "sigma": {"pos_mm": [1.0, 1.0, 3.0], "rot_deg": [1.0, 1.0, 1.0]}}
    pc = _HeldPosePerception(env, ok)
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", held_pose=held_pose), assets=FakeAssets(),
                        perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    assert _stacked(env)
    ev = [e for e in rep["events"] if e["kind"] == "held_pose"]
    if held_pose == "off":
        assert pc.calls == [] and ev == []
        return
    places = [h for h in rep["history"] if h["skill"] == "place"]
    assert len(pc.calls) == len(places) == len(ev) == 3 and all(e["ok"] for e in ev)   # one read per place hover
    assert all(c[1] == "Rigid/block/00005" and c[3].shape == (4, 4) for c in pc.calls)
    assert rep["provider"]["held_pose"] == 3
    assert all("in-hand shift 2.0 mm, re-planned" in h["note"] for h in places)       # the tail used the read


def test_held_pose_adopts_only_a_confident_read_within_the_place_tolerance():
    from types import SimpleNamespace

    from dojo.skills.skills import Skills
    prior = np.eye(4)
    meas = np.eye(4)
    meas[:3, 3] = [0.003, 0.0, 0.0]
    answers = []

    def held_pose(arm, prior_T_tcp_obj, asset_key):
        return answers.pop(0)
    me = SimpleNamespace(provider=SimpleNamespace(held_pose=held_pose))
    h = SimpleNamespace(T_tcp_obj=prior)
    o = SimpleNamespace(asset=SimpleNamespace(key="Rigid/block/00005"))
    good = {"ok": True, "T_tcp_obj": meas.tolist(), "sigma": {"pos_mm": [1.0, 2.0, 9.0], "rot_deg": [1.0, 2.0, 2.0]}}
    answers += [good, {**good, "ok": False}, {**good, "sigma": {"pos_mm": [6.0, 1.0, 1.0], "rot_deg": [1, 1, 1]}},
                {**good, "sigma": {"pos_mm": [1.0, 1.0, 1.0], "rot_deg": [1.0, 4.0, 1.0]}}, {**good, "sigma": None}]
    pp = SimpleNamespace(arm="left")
    got = [Skills._measure_held(me, pp, h, o, 0.01, 6.0) for _ in range(5)]
    assert np.allclose(got[0], meas)                      # z sigma does not gate (the descent is guarded / verified)
    assert got[1:] == [None] * 4                          # not ok / lateral 6 mm > 5 / rot 4 deg > 3 / no sigma
    assert Skills._measure_held(SimpleNamespace(provider=SimpleNamespace()), pp, h, o, 0.01, 6.0) is None


def test_eval_rows_count_every_event_kind_and_keep_the_setup_events():
    from dojo.runtime.eval import event_counts
    ev = [{"kind": "scan", "ran": True}] + [{"kind": "resync"}] * 40 + [{"kind": "program_done"}, "junk"]
    assert event_counts(ev) == {"scan": 1, "resync": 40, "program_done": 1, "?": 1}
    assert event_counts(None) == {}


def test_twin_load_noise_is_a_fixed_offset_per_object_and_only_on_estimate_loads():
    from types import SimpleNamespace

    from dojo.runtime.verifier import BundleVerifier
    objs = [{"label": "block_0", "pos": [0.1, 0.0, 0.78], "quat": [1, 0, 0, 0]},
            {"label": "block_1", "pos": [0.2, 0.0, 0.81], "quat": [1, 0, 0, 0]}]
    prov = SimpleNamespace(checkpoint={"objects": objs, "robot": {}, "n": 0})
    seen = []
    v = BundleVerifier(None, "stack_blocks", prov, SimpleNamespace(n_executed=0), load_noise=(2.0, 1.5), seed=3,
                       layout_fn=lambda scene, o: seen.append(o) or {"objs": o})
    v._base(())
    v._base(())
    a, b = seen
    for o0, o1, o2 in zip(objs, a, b):
        d = np.asarray(o1["pos"]) - np.asarray(o0["pos"])
        assert 0 < np.linalg.norm(d) < 0.02 and o1["pos"] == o2["pos"] and o1["quat"] == o2["quat"]   # fixed per label
        assert abs(np.linalg.norm(o1["quat"]) - 1) < 1e-9 and o1["quat"] != o0["quat"]
    assert objs[0]["pos"] == [0.1, 0.0, 0.78]                                     # the provider's estimate untouched
    v2 = BundleVerifier(None, "stack_blocks", prov, SimpleNamespace(n_executed=0),
                        layout_fn=lambda scene, o: seen.append(o) or {"objs": o})
    v2._base(())
    assert seen[-1] is objs                                                        # off: the estimate as is


def test_consistent_twin_load_levels_and_seats_only_small_inconsistencies():
    from dojo.runtime.twin_load import consistent
    from dojo.skills.geometry import quat_to_R
    from dojo.skills.kinematics import TABLE_Z
    from dojo.skills.world import AssetInfo
    cube = AssetInfo("Rigid/block/00001", np.array([-0.015] * 3), np.array([0.015] * 3))
    tilt = [float(np.cos(np.radians(1))), float(np.sin(np.radians(1))), 0.0, 0.0]          # 2 deg about x
    objs = [{"label": "a", "pos": [0.0, 0.0, TABLE_Z + 0.015 - 0.003], "quat": [1, 0, 0, 0]},   # 3 mm into the table
            {"label": "b", "pos": [0.002, 0.0, TABLE_Z + 0.045 + 0.004], "quat": tilt},      # 4 mm above a, tilted
            {"label": "c", "pos": [0.0, 0.0, TABLE_Z + 0.075 - 0.020], "quat": [1, 0, 0, 0]},  # 20 mm into b: left
            {"label": "d", "pos": [0.2, 0.0, TABLE_Z + 0.015 + 0.002], "quat": [1, 0, 0, 0]}]  # fixed: left
    out, st = consistent(objs, {k: cube for k in "abcd"}, fixed={"d"})
    by = {o["label"]: o for o in out}
    assert abs(by["a"]["pos"][2] - (TABLE_Z + 0.015)) < 1e-6
    assert abs(by["b"]["pos"][2] - (TABLE_Z + 0.045)) < 1e-4 and abs(quat_to_R(by["b"]["quat"])[2, 2] - 1) < 1e-9
    assert by["c"]["pos"] == objs[2]["pos"] and by["d"]["pos"] == objs[3]["pos"] and by["a"]["quat"] == [1, 0, 0, 0]
    assert [x[0] for x in st["leveled"]] == ["b"] and sorted(x[0] for x in st["seated"]) == ["a", "b"]


def test_verifier_consistent_load_uses_the_belief_assets_and_counts_what_it_changed():
    from types import SimpleNamespace

    from dojo.runtime.verifier import BundleVerifier
    from dojo.skills.kinematics import TABLE_Z
    from dojo.skills.world import AssetInfo, ObjectState
    cube = AssetInfo("Rigid/block/00001", np.array([-0.015] * 3), np.array([0.015] * 3))
    bel = {"block_0": ObjectState("block_0", "Rigid", "block", 1, np.array([0.0, 0.0, TABLE_Z + 0.013]),
                                  np.array([1.0, 0, 0, 0]), cube)}
    prov = SimpleNamespace(checkpoint={"objects": [{"label": "block_0", "pos": [0.0, 0.0, TABLE_Z + 0.013],
                                                    "quat": [1, 0, 0, 0]}], "robot": {}, "n": 0},
                           belief=SimpleNamespace(objects=bel))
    seen = []
    v = BundleVerifier(None, "no_such_scene", prov, SimpleNamespace(n_executed=0), load="consistent",
                       layout_fn=lambda scene, o: seen.append(o) or {})
    v._base(())
    assert abs(seen[0][0]["pos"][2] - (TABLE_Z + 0.015)) < 1e-6 and v.stats["load_seated"] == 1
    assert prov.checkpoint["objects"][0]["pos"][2] == TABLE_Z + 0.013
