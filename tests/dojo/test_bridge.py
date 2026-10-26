"""dojo.bridge: our runtime behind a robot-harness strict server, against a fake server speaking the same JSON-lines
protocol over a real Unix socket (kinematic env, CPU only): protocol, strict firewall, joint_steps, whole episodes
of RuntimePolicy and of AgentPolicy (scripted model) with the twin verifier and the server-perception route."""

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from controlr.llm import FakeLLM
from dojo.bridge.backend import RobotServerBackend, action_json
from dojo.bridge.client import RobotServerClient, ServerError
from dojo.bridge.contract import ContractViolation, check_reply, scan
from dojo.bridge.driver import run_episode
from dojo.bridge.fake_server import FakeStrictServer
from dojo.bridge.perception import ServerPerception, record_from_perceived
from dojo.bridge.twin import twin_scene
from dojo.runtime.firewall import OBS_KEYS, STATE_KEYS, check_obs
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from test_runtime import STACK, FakeAssets, FakeTwinClient, _env, _stacked

BRIDGE = Path(__file__).resolve().parents[2] / "dojo" / "bridge"


@pytest.fixture
def bench():
    d = tempfile.mkdtemp(prefix="rb", dir="/tmp")          # short: Unix socket paths are limited to ~107 bytes
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _home(env) -> bool:
    return all(np.abs(env.q[a]).max() < 0.05 for a in ("left", "right"))


def serve(bench, n=1, sid="t1", success=False, **kw):
    envs = []

    def mk():
        e = _env()
        envs.append(e)
        return e
    fn = (lambda e: _stacked(e) and _home(e)) if success else None
    srv = FakeStrictServer(sid, bench, [mk] * n, success_fn=fn, **kw).start()
    be = RobotServerBackend(RobotServerClient(sid, bench=str(bench)), frames_dir=bench / "frames" / sid, keep_every=5)
    return srv, be, envs


class EnvPerception:
    """The local perception route's stand-in: estimates from the env behind the server (as if perception were exact);
    asserts it only ever sees contract observations."""

    def __init__(self, envs):
        self.envs, self.calls = envs, []

    def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None):
        for f in frames:
            assert set(f) == OBS_KEYS and set(f["state"]) == set(STATE_KEYS)
        self.calls.append(("estimate", vocabulary, len(frames)))
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                     "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                     "_perception": {"confidence": 0.9}} for o in self.envs[-1].objects]}}

    def check(self, frame, objects, fold):
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []


# ------------------------------------------------------------------ protocol + firewall
def test_client_speaks_the_robot_client_v2_protocol(bench):
    srv, be, envs = serve(bench)
    c = be.c
    info = c.info()
    assert info["ok"] and info["cmd"] == "info" and "joint_steps" in info["commands"]
    assert c.reset()["instruction"] == STACK
    r = c.gripper(arm="left", cmd="close")                  # cmd= keyword -> "action" (robot_client_v2's fix)
    assert r["ok"] and r["grip_cmd"] == 0.0 and r["steps"] > 0
    bad = c.call("no_such_verb")
    assert not bad["ok"] and "unknown command" in bad["error"]
    with pytest.raises(ServerError):
        c.ok("no_such_verb")
    assert {"step", "steps_left", "ended", "t_wall"} <= set(r)
    srv.stop()


def test_observation_carries_only_the_contract(bench):
    srv, be, envs = serve(bench)
    be.connect()
    ep = be.reset()
    assert set(ep) == {"instruction", "step_lim"}                        # nothing about task / layout / seed
    obs = be.observe()
    assert set(obs) == OBS_KEYS and set(obs["state"]) == set(STATE_KEYS)
    assert set(obs["images"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    clean, gt = check_obs(obs, "perception")                            # the runtime's own firewall agrees
    assert gt is None
    st = envs[0].proprio()
    for k in STATE_KEYS:
        np.testing.assert_allclose(obs["state"][k], np.asarray(st[k], float).reshape(-1), atol=1e-5)
    # the frame is the PNG the server wrote for this step, decoded
    assert obs["images"]["cam_head"].dtype == np.uint8 and obs["images"]["cam_head"].shape == (48, 64, 3)
    srv.stop()


@pytest.mark.parametrize("leak", ["objects", "finger_q", "calib"])
def test_a_leaking_server_is_refused(bench, leak):
    srv, be, envs = serve(bench, leak=(leak,))
    be.connect()
    with pytest.raises(ContractViolation):
        be.reset()                                                      # the reset reply carries an obs
    srv.stop()


def test_non_strict_or_verbless_servers_are_refused(bench):
    srv, be, _ = serve(bench, tier="B-images+proprio")
    with pytest.raises(ContractViolation):
        be.connect()
    srv.stop()
    srv, be, _ = serve(bench, sid="t2", joint_steps=False)
    with pytest.raises(ServerError, match="joint_steps"):
        be.connect()
    srv.stop()


def test_scan_finds_banned_keys_at_any_depth():
    assert scan({"a": [{"b": {"finger_q": 1}}], "objects": {}}) == ["a[0].b.finger_q", "objects"]
    with pytest.raises(ContractViolation):
        check_reply("obs", {"eef": {"left": {"gap_m": 0.01}}})
    assert check_reply("obs", {"eef": {"left": {"q": [0] * 6}}, "images": {}})


# ------------------------------------------------------------------ joint_steps
def test_joint_steps_run_verbatim_and_refuse_before_stepping(bench):
    srv, be, envs = serve(bench)
    be.connect()
    be.reset()
    q0 = {a: envs[0].q[a].copy() for a in ("left", "right")}
    acts = []
    for i in range(1, 11):
        a = {}
        for arm in ("left", "right"):
            a[f"{arm}_arm_joint_state"] = (q0[arm] + 0.0137 * i * np.r_[1, -1, 0.5, 0, 0, 0.3]).astype(np.float32)
            a[f"{arm}_ee_joint_state"] = np.array([1.0 if i < 6 else 0.4], np.float32)
        acts.append(a)
    r = be.execute(acts)
    assert r == {"steps": 10, "ended": False} and be.steps_used == 10 and len(be.executed) == 10
    for arm in ("left", "right"):                          # bit-for-bit: float32 -> JSON -> float32
        assert np.array_equal(np.asarray(srv.hold_q[arm], np.float32), acts[-1][f"{arm}_arm_joint_state"])
        assert srv.hold_g[arm] == pytest.approx(0.4, abs=1e-7)
    jump = dict(acts[-1])
    jump["left_arm_joint_state"] = jump["left_arm_joint_state"] + np.float32(0.35)
    with pytest.raises(ServerError, match="refused: jump at entry 0"):
        be.execute([jump])
    assert envs[0].steps_used == 10                        # refused at 0 steps
    rep = be.c.call("joint_steps", actions=[action_json(acts[-1])] * (envs[0].step_lim - 15), reserve=10)
    assert rep["ok"] and rep["refused"] == "budget" and rep["steps"] == 0 and envs[0].steps_used == 10
    assert len(be.chunks) == 1 and be.chunks[0]["n"] == 10 and len(be.chunks[0]["md5"]) == 32   # md5 checked vs the reply
    assert sum(be.stats["grip_change_while_moving"].values()) == 2                     # i = 6: both grippers change while moving
    r = be.execute([acts[-1]] * (envs[0].step_lim + 50))  # past the limit: runs to the limit, the episode ends
    assert r["ended"] and envs[0].steps_used == envs[0].step_lim == be.steps_used
    with pytest.raises(ValueError):
        action_json({"left_arm_joint_state": [0] * 6})
    srv.stop()


def test_buffered_steps_flush_before_any_foreign_verb(bench):
    """robot-harness skills get the client as ``robot``: our buffered actions run before their first verb."""
    srv, be, envs = serve(bench)
    be.connect()
    obs = be.reset()
    st = be.proprio()
    hold = {k: np.asarray(v) for k, v in st.items() if k.endswith("joint_state")}
    for _ in range(4):
        be.step(hold)
    assert be.steps_used == 4 and envs[0].steps_used == 0
    rep = be.robot.hold(steps=2)                          # a foreign skill's verb
    assert rep["ok"] and envs[0].steps_used == 6 and [r["cmd"] for r in srv.requests][-2:] == ["joint_steps", "hold"]
    assert be.fresh is False and obs
    srv.stop()


# ------------------------------------------------------------------ whole episodes
def test_runtime_episode_through_the_server(bench, monkeypatch):
    """RuntimePolicy in mode perception behind the fake strict server: the info has no task, GT is never read, the
    skills' joint actions run through joint_steps, the blocks end stacked and both arms home."""
    import dojo.runtime.belief as B
    monkeypatch.setattr(B.RuntimeProvider, "_gt_objects", lambda self: (_ for _ in ()).throw(AssertionError("GT read")))
    srv, be, envs = serve(bench)
    be.connect()
    pc = EnvPerception(envs)
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(), perception_client=pc)
    res = run_episode(pol, be)
    s, rep = res["summary"], res["report"]
    assert s["end"] == "policy_done" and rep["error"] is None, (s, rep.get("error"))
    assert pol.log_task is None and rep["info_dropped"] == []
    assert pc.calls[0] == ("estimate", "stack_blocks", 1)
    assert _stacked(envs[0]) and _home(envs[0])
    assert s["steps_used"] == envs[0].steps_used == len(res["executed"])
    cmds = {r["cmd"] for r in srv.requests}
    assert cmds <= {"info", "reset", "obs", "joint_steps"}, cmds   # the runtime moves only through joint_steps
    srv.stop()


def test_full_system_agent_twin_and_server_perception(bench, monkeypatch, tmp_path):
    """The whole stack through the server: the J1 ``perceive`` verb -> record -> WorldState; the twin (our side)
    loaded from that estimate + the measured proprio; the LLM agent (scripted) runs a library skill whose segments
    the twin verifies; RoboDojo's latch (stacked + home) ends the episode."""
    from dojo.agent.loop import AgentConfig
    from dojo.agent.policy import AgentPolicy
    import dojo.runtime.estimate as E
    monkeypatch.setattr(E, "estimate_to_layout", lambda scene, objs: {"Rigid": {"block": [
        {"label": o["label"], "category_idx": o["category_idx"], "default_pos": o["pos"], "default_ori": o["quat"]}
        for o in objs]}})

    def model(messages):
        if len(messages) == 2:
            ids = re.findall(r"^- (block_\d+):", messages[1]["content"][0]["text"], re.M)
            return ("```python\nr = robo.skill('stack_objects', objects=%r)\nprint(r)\nif r.ok:\n    finish()\n```" % ids)
        return "```python\nfinish('failed', 'skill failed')\n```"

    srv, be, envs = serve(bench, success=True, perceive_noise_m=0.0005)
    be.connect()

    class RecordingTwin(FakeTwinClient):
        robots = []

        def load_from_estimate(self, layout, robot=None, settle_steps=50):
            self.robots.append(robot)
            return super().load_from_estimate(layout, robot, settle_steps)
    tc = RecordingTwin()
    scenes = []

    def twin_factory(scene):
        scenes.append(scene)
        return tc
    pol = AgentPolicy(RuntimeConfig(mode="perception", verify="bundle", n_cands=2, verify_k=2),
                      agent_cfg=AgentConfig(model="fake", max_calls=3), client=FakeLLM(model), assets=FakeAssets(),
                      log_root=tmp_path)
    pol._pc = ServerPerception(be)
    res = run_episode(pol, be, twin_factory=twin_factory)
    s, rep = res["summary"], res["report"]
    assert rep["error"] is None, rep["error"]
    assert scenes == ["stack_blocks"] and s["scene"] == "stack_blocks"   # twin scene from the instruction
    assert srv.success and s["end"] in ("ended", "policy_done")
    assert _stacked(envs[0])
    # one perceive call (no flags, no looks); the twin's first load = its poses + the measured proprio at that step
    assert srv.n_perceive == 1 and pol._pc.calls[0]["prior_n"] is None
    per = {o["label"]: o for o in pol._pc.last["perceived"]}
    for r in tc.loads[0]["Rigid"]["block"]:
        np.testing.assert_allclose(r["default_pos"], per[r["label"]]["pos"], atol=1e-6)
    rb = tc.robots[0]
    assert set(rb) == {"left_q", "right_q", "left_grip", "right_grip"}
    np.testing.assert_allclose(rb["left_q"], np.zeros(6), atol=1e-6)        # the kinematic env starts at q = 0
    v = rep["verify"]
    assert isinstance(v, dict) and v["segments"] >= 4 and v["from_estimate"] >= 2, v
    # RoboDojo's latch (stacked + home) ends the episode inside the skill's finish(), before the cell returns
    assert s["end"] == "ended" and rep["agent"]["model"] == "fake"
    assert [e.get("skill") for e in rep["events"] if e.get("kind") == "agent_skill"] == ["stack_objects"]
    srv.stop()


def test_record_from_perceived_and_look_plan():
    per = [{"label": "block_0", "category": "block", "asset": "Rigid/block/00005", "pos": [0.1, -0.1, 0.78],
            "quat": [1, 0, 0, 0], "confidence": 0.8, "source": "head"},
           {"label": "t", "category": "T", "pos": [0.0, 0.0, 0.77], "confidence": 0.9}]          # no asset: unusable
    rec = record_from_perceived(per, [{"reason": "not_found", "label": "block_2"}, {"reason": "occluded", "label": "x"}])
    r = rec["Rigid"]["block"][0]
    assert r["category_idx"] == 5 and r["default_pos"] == [0.1, -0.1, 0.78] and r["_perception"]["confidence"] == 0.8
    assert rec["_not_found"] == [{"label": "block_2", "candidates": []}] and rec["_unusable"][0]["label"] == "t"

    class B:
        def perceive(self, want=None, prior_n=None):
            self.prior = prior_n
            return {"perceived": per, "needs_look": [{"cam": "cam_left_wrist", "reason": "occluded", "label": "block_0",
                                                     "target": [0.1, -0.1, 0.78], "flange": [0, 0, 1, 1, 0, 0, 0]},
                                                    {"cam": None, "reason": "not_found", "label": "block_2"}], "n": 3}
    b = B()
    sp = ServerPerception(b)
    sp.estimate([{}], "stack_blocks", 0)
    assert b.prior is None
    sp.estimate([{}, {}], "stack_blocks", 0)                  # a fuse after a look chains the previous call
    assert b.prior == 3
    plan = sp.look_plan({}, rec)
    assert plan == [{"arm": "left", "purpose": "occluded", "label": "block_0", "target": [0.1, -0.1, 0.78],
                     "flange_pose": [0, 0, 1, 1, 0, 0, 0]}]
    assert sp.refine({}, [], 0) == {} and sp.check({}, [], 0) == {} and sp.search_regions({}, {}, "x") == []


def test_twin_scene_follows_the_instruction():
    assert twin_scene(STACK) == "stack_blocks"
    assert twin_scene("Press the red buttons as many times as the number cards show, then the blue button.") \
        == "press_by_number"
    assert twin_scene("Do a dance.") is None


def test_bridge_sources_never_read_gt_or_grader_state():
    # GT accessors, eval files, and the grader verbs (``success`` / ``end``: never called in a camera-only episode)
    forbidden = re.compile(r"objects_gt|robot_gt|gt_objects|\.gt\(|layout_file|Eval_Layout|Assets/Traj|reward\(|"
                           r"(?:call|ok)\(\s*[\"'](?:success|end)[\"']|\.(?:success|end)\(")
    for name in ("backend.py", "client.py", "contract.py", "driver.py", "perception.py", "twin.py"):
        src = (BRIDGE / name).read_text()
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        code = re.sub(r'"""[\s\S]*?"""', "", code)            # docstrings may name what is forbidden
        hits = [m.group(0) for m in forbidden.finditer(code)]
        assert not hits, (name, hits)


def test_joint_steps_smoke_runs_one_chunk_per_phase_and_keeps_the_tracking(bench):
    from dojo.bridge.smoke import JointStepsSmoke
    srv, be, envs = serve(bench)
    be.connect()
    pol = JointStepsSmoke()
    res = run_episode(pol, be)
    s, rep = res["summary"], res["report"]
    assert s["end"] == "policy_done", s
    chunks = s["joint_steps_chunks"]
    assert len(chunks) == len(rep["phases"]) == 1 + 2 * 2 * 2 + 2 * 2
    assert all(c["ran"] == c["n"] and "tracking_max_rad" in c for c in chunks)
    assert s["steps_used"] == sum(p["n"] for p in rep["phases"]) < 100
    assert max(max(c["max_step_dq"].values()) for c in chunks) == pytest.approx(0.18, abs=1e-6)   # per arm, per chunk
    assert _home(envs[0])                                             # out and back: the arms end where they began
    srv.stop()


def test_env_notes_record_run_settings_but_never_keys():
    from dojo.bridge.driver import env_notes
    env = {"DOJO_BG_FOLDS": "/d/bg_folds_51", "SIMLAB_USD_SYNC_LINKS": "1", "AGENTIC_MODEL": "claude/claude-sonnet-5-5",
           "OMNIROUTE_API_KEY": "sk-secret", "OMNIROUTE_BASE_URL": "https://x", "HOME": "/h"}
    out = env_notes(env)
    assert out == {"DOJO_BG_FOLDS": "/d/bg_folds_51", "SIMLAB_USD_SYNC_LINKS": "1",
                   "AGENTIC_MODEL": "claude/claude-sonnet-5-5"}
    assert not any(k.startswith("OMNIROUTE") for k in out)
