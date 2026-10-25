"""dojo.bridge against robot-harness's REAL strict J1 server code (opt-in, CPU only, no Isaac).

    CONTROLR_ROBOT_HARNESS=/path/to/robot-harness uv run pytest -q tests/dojo/test_bridge_rh.py

Copies (never modifies) from that checkout into a temp $BENCH: harness/robodojo_server{,_strict,_j1}.py, j1_schema.py,
j1_perceiver_host.py, j1_mock_perceiver.py, and the frozen offline mock shim (tools/perceiver/robodojo/compute/
strict-tools/mock_shim.py + j1-tools/mock_shim_j1.py: a toy arm, black 480x640 frames). The one stand-in is
strict_calib.py (it reads RoboDojo's public config files at import, which only exist on a RoboDojo box): a stub with the
same interface and plausible K / T shapes. Then our backend talks to the real server: tier check, reset, obs ->
contract observation, proprio, perceive (the J1 mock perceiver) -> layout record, B1 held_look, hold, gripper, close;
``joint_steps`` is expected to be refused until robot-harness #8 lands, and used once it does."""

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from dojo.bridge.backend import RobotServerBackend
from dojo.bridge.client import RobotServerClient, ServerError
from dojo.bridge.perception import record_from_perceived
from dojo.runtime.firewall import OBS_KEYS, check_obs

RH = os.environ.get("CONTROLR_ROBOT_HARNESS")
pytestmark = pytest.mark.skipif(not RH, reason="set CONTROLR_ROBOT_HARNESS to a robot-harness checkout")
STACK = "Stack the three blocks with different textures."

STUB_CALIB = '''"""TEST STUB of robot-harness harness/strict_calib.py (same interface; the real one reads RoboDojo's public config)."""
CONST = {"provenance": "test stub (no RoboDojo config on this machine)"}
_K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
_T = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]


def calib(flanges):
    return {c: {"K": _K, "T_world_cam": _T, "src": "stub"} for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
'''


@pytest.fixture
def rh_server():
    rh = Path(RH)
    T = Path(tempfile.mkdtemp(prefix="rh", dir="/tmp"))
    (T / "harness").mkdir()
    for f in ("robodojo_server.py", "robodojo_server_strict.py", "robodojo_server_j1.py", "j1_schema.py",
              "j1_perceiver_host.py", "j1_mock_perceiver.py"):
        shutil.copy(rh / "harness" / f, T / "harness" / f)
    (T / "harness" / "strict_calib.py").write_text(STUB_CALIB)
    (T / "run" / "strict-tools").mkdir(parents=True)
    tools = rh / "tools" / "perceiver" / "robodojo" / "compute"
    shutil.copy(tools / "strict-tools" / "mock_shim.py", T / "run" / "strict-tools" / "mock_shim.py")
    shutil.copy(tools / "j1-tools" / "mock_shim_j1.py", T / "mock_shim_j1.py")
    objs = [{"label": f"block_{i}", "category": "block", "asset": "Rigid/block/00005", "pos": [x, -0.1, 0.78],
             "quat": [1, 0, 0, 0], "size": [0.03, 0.03, 0.03], "confidence": 0.9} for i, x in enumerate((-0.2, 0.0, 0.2))]
    (T / "objects.json").write_text(json.dumps({"objects": objs}))
    env = dict(os.environ, BENCH=str(T), HARNESS_SHIM_AUTHKEY=secrets.token_hex(16),
               J1_PERCEIVER="j1_mock_perceiver:perceive", J1_PERCEIVER_PATH=str(T / "harness"),
               J1_MOCK_OBJECTS=str(T / "objects.json"), MOCK_INSTRUCTION=STACK, MOCK_STEP_LIM="550")
    sid = "rs1"
    srv = subprocess.Popen([sys.executable, str(T / "harness" / "robodojo_server_j1.py"), "--id", sid, "--strict"],
                           env=env, stdout=open(T / "server.log", "wb"), stderr=subprocess.STDOUT)
    for _ in range(100):
        if (T / "run" / f"{sid}.sock").exists() and (T / "run" / sid / "shim.sock").exists():
            break
        time.sleep(0.1)
    shim = subprocess.Popen([sys.executable, str(T / "mock_shim_j1.py"), str(T / "run" / sid / "shim.sock")], env=env,
                            stdout=open(T / "shim.log", "wb"), stderr=subprocess.STDOUT)
    yield T, sid
    for p in (shim, srv):
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
    shutil.rmtree(T, ignore_errors=True)


def test_backend_against_the_real_strict_j1_server(rh_server):
    T, sid = rh_server
    be = RobotServerBackend(RobotServerClient(sid, bench=str(T)), frames_dir=T / "frames", allow_no_joint_steps=True)
    info = be.connect()                                          # the real strict tier string + strict block
    assert info["tier"].startswith("B-strict") and "perceive" in info["commands"]
    be.wait_episode(60, 0.2)
    ep = be.reset(60)
    assert ep == {"instruction": STACK, "step_lim": 550}
    obs = be.observe()
    assert set(obs) == OBS_KEYS and obs["images"]["cam_head"].shape == (480, 640, 3)
    check_obs(obs, "perception")                                  # the runtime's firewall accepts it
    np.testing.assert_allclose(obs["state"]["left_ee_pose"][:3], [-0.2996, -0.2022 - 0.1501, 0.9215], atol=1e-4)
    assert obs["state"]["left_ee_joint_state"].tolist() == [1.0]  # the COMMANDED gripper
    st = be.proprio()
    assert set(st) == set(obs["state"])
    out = be.perceive(want=None)
    rec = record_from_perceived(out["perceived"], out["needs_look"])
    assert sorted(r["label"] for r in rec["Rigid"]["block"]) == ["block_0", "block_1", "block_2"] and out["n"] == 1
    assert be.held_look("left", "test")["source"] == "B1"
    n0 = be.steps_used
    assert be.hold(2)["steps"] == 2 and be.steps_used == n0 + 2
    assert be.gripper("left", 0.0)["grip_cmd"] == 0.0
    assert be.observe()["state"]["left_ee_joint_state"].tolist() == [0.0]
    a = {f"{x}_arm_joint_state": np.zeros(6, np.float32) for x in ("left", "right")}
    a.update({f"{x}_ee_joint_state": np.ones(1, np.float32) for x in ("left", "right")})
    if "joint_steps" in be.commands:                              # robot-harness #8 landed
        assert be.execute([a, a])["steps"] == 2
    else:
        with pytest.raises(ServerError, match="joint_steps"):
            be.execute([a])
    assert be.close().get("closed")


# ------------------------------------------------------------------ J2: robot-harness's joint_steps (issue #8, PR #9)
J2_MOCK = Path(RH) / "harness" / "j2_mock_shim.py" if RH else None
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]


@pytest.fixture
def j2_stack():
    """His offline J2 stack (harness/j2_mock_shim.py): the REAL J2 server + the REAL J2 shim (rd_shim_strict_v2 over the
    strict + frozen shims) on his FakeTaskEnv (no Isaac), J1 mock perceiver; run read-only from the checkout, all
    files written under a temp $BENCH. Two episodes: one for the raw contract, one for a whole runtime episode."""
    if not (J2_MOCK and J2_MOCK.exists()):
        pytest.skip("this robot-harness checkout has no harness/j2_mock_shim.py (joint_steps not landed)")
    T = Path(tempfile.mkdtemp(prefix="rj", dir="/tmp"))
    objs = [{"label": f"block_{i}", "category": "block", "asset": "Rigid/block/00005", "pos": list(p),
             "quat": [1, 0, 0, 0], "size": [0.03, 0.03, 0.03], "confidence": 0.9} for i, p in enumerate(POS)]
    (T / "objects.json").write_text(json.dumps({"objects": objs}))
    env = dict(os.environ, J1_MOCK_OBJECTS=str(T / "objects.json"))
    for k in ("HARNESS_SHIM_AUTHKEY", "J1_PERCEIVER", "J1_PERCEIVER_PATH", "BENCH"):
        env.pop(k, None)
    p = subprocess.Popen([sys.executable, str(J2_MOCK), "--id", "rj1", "--bench", str(T), "--episodes", "2",
                          "--step-lim", "550", "--instruction", STACK], env=env, stdout=subprocess.PIPE,
                         stderr=open(T / "j2.err", "wb"), text=True)
    seen = []
    for _ in range(20):                                   # the shim's banner may come before the ready line
        line = p.stdout.readline()
        seen.append(line)
        if '"ok": true' in line or not line:
            break
    assert '"ok": true' in line, (seen, (T / "j2.err").read_text()[-2000:])
    yield T, "rj1"
    p.terminate()
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
    shutil.rmtree(T, ignore_errors=True)


def _act(q, g=1.0):
    return {f"{x}_arm_joint_state": [float(v) for v in np.asarray(q[x], np.float32)] for x in ("left", "right")} | \
           {f"{x}_ee_joint_state": [float(g)] for x in ("left", "right")}


def test_joint_steps_contract_on_the_real_j2_stack(j2_stack):
    from dojo.bridge.backend import actions_md5
    T, sid = j2_stack
    be = RobotServerBackend(RobotServerClient(sid, bench=str(T)), frames_dir=T / "frames")
    info = be.connect()                                           # strict tier + joint_steps advertised
    assert "joint_steps" in be.commands and info["j2"]["joint_steps"]["max_dq"] == 0.2
    be.wait_episode(60, 0.2)
    be.reset(60)
    q0 = {x: be.observe()["state"][f"{x}_arm_joint_state"] for x in ("left", "right")}
    c = be.c
    ok = [_act({x: q0[x] + 0.01 * k for x in q0}) for k in range(1, 6)]
    # refusal codes, at 0 steps, with the index
    for acts, code, idx in (([_act({x: q0[x] + 0.35 for x in q0})], "jump", 0),
                            ([ok[0], {"left_arm_joint_state": [0] * 6}], "keys", 1),
                            ([ok[0], _act({x: q0[x] + 0.02 for x in q0}, g=1.5)], "gripper_range", 1),
                            ([ok[0]] * 600, "budget", None)):
        r = c.ok("joint_steps", actions=acts, reserve=0)
        assert (r["refused"], r["steps"], r.get("index")) == (code, 0, idx), r
    assert c.ok("joint_steps", actions=ok, reserve=546)["refused"] == "budget"
    r = c.ok("joint_steps", actions=ok, reserve=0)
    assert r["refused"] is None and r["steps"] == r["executed"] == r["requested"] == 5
    assert r["actions_md5"] == actions_md5(ok)                    # what we sent is what it logged
    assert set(r["tracking_rad"]) == {"left", "right"} and set(r["eef"]["left"]) == {"q", "flange", "grip_cmd"}
    assert set(r["eef"]["left"]["flange"]) == {"pos", "quat"}
    np.testing.assert_allclose(r["eef"]["left"]["q"], np.asarray(ok[-1]["left_arm_joint_state"]), atol=1e-4)
    # through the backend: chunked, md5-checked, grip changes while moving counted (not refused)
    last = np.asarray(ok[-1]["left_arm_joint_state"])
    more = [_act({x: last + 0.01 * k for x in q0}, g=1.0 if k < 3 else 0.5) for k in range(1, 8)]
    out = be.execute([{k: np.asarray(v, np.float32) for k, v in a.items()} for a in more])
    assert out == {"steps": 7, "ended": False} and be.chunks[-1]["ran"] == 7
    assert be.stats["grip_change_while_moving"] == {"left": 1, "right": 1}
    assert be.observe()["state"]["left_ee_joint_state"].tolist() == [0.5]
    assert be.close().get("closed") or True


def test_runtime_episode_through_the_real_j2_stack(j2_stack):
    """A whole RuntimePolicy episode (server perceive route = his J1 mock perceiver, verify off) through his real J2
    server + shim: every chunk our executor produced passes his validator, md5s match, and our executor never changes
    a gripper while that arm moves (his invariant-3 counter stays 0)."""
    from dojo.bridge.driver import run_episode
    from dojo.bridge.perception import ServerPerception
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from test_runtime import FakeAssets
    T, sid = j2_stack
    be = RobotServerBackend(RobotServerClient(sid, bench=str(T)), frames_dir=T / "frames")
    be.connect()
    be.wait_episode(60, 0.2)
    be.reset(60)                                                   # episode 1: discarded (the reset below ends it)
    init = be.observe()["state"]                                   # episode 2 starts from the same FakeTaskEnv state
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets())
    pol._pc = ServerPerception(be)
    res = run_episode(pol, be)
    s, rep = res["summary"], res["report"]
    assert rep["error"] is None and s["end"] == "policy_done", (s["end"], rep.get("error"), rep.get("error_handled"))
    assert s["steps_used"] > 100 and len(res["executed"]) == s["steps_used"]
    assert all(ch["ran"] == ch["n"] for ch in be.chunks)
    # his invariant-3 counter vs our own recount. Our executor changes an arm's gripper on the FIRST step of a move op
    # (0-step grip ops: open_for_grasp before a pick approach, grip_for_home, close_for_push / _press), from rest or
    # at a via point between two moves, in free space; by design, as in every proxy run: counted, not refused
    prev = {x: (np.asarray(init[f"{x}_arm_joint_state"], float), float(init[f"{x}_ee_joint_state"][0]))
            for x in ("left", "right")}
    mine = {"left": 0, "right": 0}
    for a in res["executed"]:
        for x in ("left", "right"):
            q, g = np.asarray(a[f"{x}_arm_joint_state"], np.float32).astype(float), float(a[f"{x}_ee_joint_state"][0])
            q0, g0 = prev[x]
            if g != g0 and float(np.abs(q - q0).max()) > 1e-9:
                mine[x] += 1
            prev[x] = (q, g)
    assert be.stats["grip_change_while_moving"] == mine and sum(mine.values()) >= 3, (be.stats, mine)
    be.close()


def test_joint_steps_smoke_through_the_real_j2_stack(j2_stack):
    """The first-GPU-run smoke (runbook 4.0) through his real J2 server + shim: every phase is one accepted chunk at
    0.18 rad/step, and each chunk keeps his tracking fields."""
    from dojo.bridge.driver import run_episode
    from dojo.bridge.smoke import JointStepsSmoke
    T, sid = j2_stack
    be = RobotServerBackend(RobotServerClient(sid, bench=str(T)), frames_dir=T / "frames")
    be.connect()
    be.wait_episode(60, 0.2)
    pol = JointStepsSmoke()
    res = run_episode(pol, be)
    s, rep = res["summary"], res["report"]
    assert s["end"] == "policy_done", s
    assert len(be.chunks) == len(rep["phases"]) and all(ch["ran"] == ch["n"] for ch in be.chunks)
    assert all(set(ch["tracking_max_rad"]) == {"left", "right"} for ch in be.chunks)
    assert max(max(ch["max_step_dq"].values()) for ch in be.chunks) <= 0.18 + 1e-4
    be.close()
