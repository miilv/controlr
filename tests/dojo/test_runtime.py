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
}


def test_instruction_selects_the_perception_vocabulary():
    for name, text in INSTR.items():
        assert perception_vocabulary(text).name == name
    assert perception_vocabulary("stack the blocks from bottom to top in the order of red, green, and blue, "
                                 "then reset the robot arm.") is None
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
            return {o["label"]: {"ok": False, "pos": None, "quat": None, "iou": 0.1}}

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
