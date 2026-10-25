"""scripts/bootstrap_gpu_box.sh + scripts/infra/: dry runs, the preflight's driver gate, pins, the replay policy.

No network, no GPU, no ssh: the dry run executes nothing; the preflight runs locally against a stub nvidia-smi."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "bootstrap_gpu_box.sh"
LIB = ROOT / "scripts" / "infra" / "box_steps.sh"
PINS = ROOT / "scripts" / "infra" / "pins"


def run(args, env=None, cwd=None, timeout=120):
    e = {**os.environ, **(env or {})}
    return subprocess.run(args, capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout)


def dry(*args):
    r = run(["bash", str(DRIVER), "testbox-dry", "--dry-run", *args])
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_dry_run_prints_every_step_and_the_pins():
    out = dry("--role", "full")
    for step in ("preflight", "conventions", "code", "catalog_post", "smoke_perception", "assets_post", "smoke_twin",
                 "smoke_render", "smoke_dev_eval", "ready"):
        assert f"### {step}" in out, step
    pins = (PINS / "robodojo_commit.txt").read_text().split()
    shas = [s.lstrip("+- ") for s in pins if re.fullmatch(r"[+ -]?[0-9a-f]{40}", s)]
    assert len(shas) == 4
    for sha in shas:                                      # RoboDojo + IsaacLab + cuRobo + XPolicyLab at compute3's commits
        assert f"checkout -q --detach {sha}" in out, sha
    assert "269457a3b42d8d0ae69898a601f2c1487c2f826e" in out or "checkout -q main" in out
    assert "lfs pull --include" in out
    assert "--python-preference only-managed" in out      # never the system 3.11.0rc1
    assert "scripts/simlab/dev_eval.sh --task stack_blocks --policy dojo_replay --eval-num 1" in out
    assert "-m dojo.simlab.twin --task stack_blocks --num-envs 1 --steps 40 --physics-only" in out
    assert "-m dojo.simlab.runner --task stack_blocks" in out and "--images" in out
    assert "dojo.perception.eval_g1" in out and "PYOPENGL_PLATFORM=egl" in out
    assert "update_embodiment_config_path.py" in out      # curobo.yml regeneration
    assert "dojo-gpu.lock" in out
    assert "ssh testbox-dry" in out and "ssh compute3" in out


def test_dry_run_touches_nothing(tmp_path):
    before = set((ROOT / "runs").glob("bootstrap/*")) if (ROOT / "runs").exists() else set()
    dry()
    after = set((ROOT / "runs").glob("bootstrap/*")) if (ROOT / "runs").exists() else set()
    assert after == before                                # no local log dir for a dry run
    assert not any("testbox-dry" in str(p) for p in after)


def test_physics_role_skips_the_render_smokes():
    out = dry("--role", "physics")
    assert "### smoke_render" not in out and "### smoke_dev_eval" not in out
    assert "### smoke_twin" in out and "### smoke_perception" in out


def test_assets_all_never_fetches_eval_data():
    out = dry("--assets", "all")
    assert "--exclude 'Assets/Eval_Layout/**,Assets/Traj/RoboDojo/imitate_sorting_sequence/**'" in out
    assert "!/Assets/Eval_Layout/" in out


def test_only_preflight_runs_just_the_probes():
    out = dry("--only", "preflight")
    assert "### preflight" in out and "nvidia-smi" in out
    assert "### conventions" not in out and "push scripts/infra" not in out


def test_pins_files():
    lines = [x for x in (PINS / "robodojo_assets.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
    assert len(lines) > 400 and all(x.startswith("/Assets/") for x in lines)
    assert not any("Eval_Layout" in x for x in lines)
    assert not any("imitate_sorting_sequence" in x for x in lines)    # eval-paired demos (firewall)
    venv = (PINS / "controlr-venv.txt").read_text()
    assert "pyopengl==3.1.9" in venv and "pyrender==0.1.45" in venv
    freeze = (PINS / "robodojo-env.freeze.txt").read_text()
    assert "isaacsim==5.1.0.0" in freeze and "\n-e " not in freeze


def stub_nvidia_smi(bindir: Path, driver: str) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    s = bindir / "nvidia-smi"
    s.write_text(f"""#!/usr/bin/env bash
case "$*" in
  *query-compute-apps*) ;;
  -L) echo "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-0)" ;;
  *) echo "NVIDIA GeForce RTX 5090, {driver}, 32607, 300" ;;
esac
""")
    s.chmod(0o755)


@pytest.mark.parametrize("driver, role, level", [("595.91.07", "full", "FAIL"), ("580.178.04", "full", "PASS"),
                                                  ("570.86.10", "full", "PASS"), ("595.91.07", "physics", "WARN")])
def test_preflight_driver_gate(tmp_path, driver, role, level):
    stub_nvidia_smi(tmp_path / "bin", driver)
    r = run(["bash", str(LIB), "preflight"], cwd=tmp_path,
            env={"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "ROLE": role, "PREFLIGHT_NET": "0",
                 "HOME": str(tmp_path), "MIN_DISK_GB": "1"})
    line = next(x for x in r.stdout.splitlines() if " driver " in f" {x} ")
    assert line.startswith(level), r.stdout
    if level == "FAIL":
        assert r.returncode == 1 and "NO-GO" in r.stdout
    assert "GPU" in r.stdout and not (tmp_path / "dojo-bootstrap").exists()      # read-only


def test_preflight_disk_gate(tmp_path):
    stub_nvidia_smi(tmp_path / "bin", "580.1")
    r = run(["bash", str(LIB), "preflight"], cwd=tmp_path,
            env={"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "PREFLIGHT_NET": "0", "HOME": str(tmp_path),
                 "MIN_DISK_GB": "999999"})
    assert any(x.startswith("FAIL  disk") for x in r.stdout.splitlines()), r.stdout
    assert r.returncode == 1


def load_replay_model():
    p = ROOT / "scripts" / "infra" / "xpl_dojo_replay" / "model.py"
    spec = importlib.util.spec_from_file_location("dojo_replay_model", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_replay_model_serves_segments_then_empty(tmp_path, monkeypatch):
    act = lambda k: {"left_arm_joint_state": [0.01 * k] * 6, "left_ee_joint_state": [1.0],
                     "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [0.5]}
    traj = {"actions": [act(k) for k in range(5)], "segments": [{"name": "a", "steps": 2}, {"name": "b", "steps": 3}]}
    f = tmp_path / "traj.json"
    f.write_text(json.dumps(traj))
    monkeypatch.setenv("DOJO_REPLAY_TRAJ", str(f))
    m = load_replay_model().Model({"policy_name": "dojo_replay"})
    for _ in range(2):                                   # reset replays from the start
        m.reset()
        a, b = m.get_action(), m.get_action()
        assert [len(a), len(b)] == [2, 3]
        assert a[0]["left_arm_joint_state"].dtype == np.float32 and a[0]["left_arm_joint_state"].shape == (6,)
        np.testing.assert_allclose(b[2]["left_arm_joint_state"], [0.04] * 6, rtol=1e-6)
        assert m.get_action() == []


def test_replay_deploy_marks_unfinished_episode_failed():
    p = ROOT / "scripts" / "infra" / "xpl_dojo_replay" / "deploy.py"
    spec = importlib.util.spec_from_file_location("dojo_replay_deploy", p)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)

    class Env:                                           # RoboDojo's latch semantics: success starts True, ends the episode when False
        num_envs, success, end_flag, take_action_cnt = 1, [True], [False], [0]

        def is_episode_end(self):
            if not self.success[0]:
                self.end_flag[0] = True
            return all(self.end_flag)

        def get_obs(self):
            return {}

        def take_action(self, a):
            self.take_action_cnt[0] += 1

    class Client:
        def __init__(self):
            self.chunks = [[{"a": 1}] * 3, []]

        def call(self, func_name, **kw):
            return self.chunks.pop(0) if func_name == "get_action" else None

    env = Env()
    d.eval_one_episode(env, Client())
    assert env.take_action_cnt == [3] and env.success == [False] and env.end_flag == [True]


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    r = run(["shellcheck", "-x", str(DRIVER), str(LIB)])
    assert r.returncode == 0, r.stdout
