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
