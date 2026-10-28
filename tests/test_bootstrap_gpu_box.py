"""scripts/bootstrap_gpu_box.sh + scripts/infra/: dry runs, the preflight's driver / role gate, the basic probe, secrets,
pins, dojo-box.json, the replay policy.

No network, no GPU, no ssh: the dry run executes nothing; the preflight runs locally against a stub nvidia-smi."""

from __future__ import annotations

import hashlib
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
    for step in ("preflight", "conventions", "secrets", "code", "catalog_post", "smoke_perception", "smoke_bridge",
                 "assets_post", "smoke_twin", "smoke_render", "smoke_dev_eval", "ready"):
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
    assert "dojo-gpu.lock" in out and "dojo-gpu-light4.lock" in out    # heavy + four light-lane slots
    assert "box_json.py > ~/dojo-box.json" in out
    assert "bridge_episode.sh --id fk-smoke --fake --policy dojo.runtime.policy:RuntimePolicy" in out
    assert "ssh testbox-dry" in out and "ssh compute3" in out
    assert "isaac6" not in out.split("### preflight")[1].split("### ready")[0].replace("isaac6.sh", "")  # role full: no 6.0


def test_smokes_never_run_reserved_layouts():
    """D-C10a: layouts >= 100100 are reserved for joint runs. smoke_twin generates 100000 / 100001; the G2 replays on
    110007 (contact restore, dev_eval, proxy_eval) run only with SMOKE_RESERVED_OK=1."""
    out = dry("--role", "full", "--with-isaac6")
    twin = next(x for x in out.splitlines() if "-m dojo.simlab.twin" in x)
    assert "--seeds 100000,100001" in twin and "--g2-dir" not in twin
    assert out.count("only with SMOKE_RESERVED_OK=1: it replays G2 on layout 110007") == 2     # dev_eval + proxy_eval
    lib = LIB.read_text()
    assert "reserved_gate && return 0" in lib.split("step_smoke_dev_eval()")[1].split("\n}")[0]
    assert "reserved_gate && return 0" in lib.split("step_smoke_proxy_eval()")[1].split("\n}")[0]


def test_shared_home_dry_run():
    out = dry("--role", "full", "--shared-home")
    assert "SHARED_HOME=1" in out and "UV_UNMANAGED_INSTALL=~/dojo-bootstrap/tools/uv" in out
    conv = out.split("### conventions")[1].split("### secrets")[0]
    assert "enable-linger" not in conv and "uv.toml <<" not in conv and ">> ~/.config/uv/uv.toml" not in conv
    out = dry("--role", "full")
    conv = out.split("### conventions")[1].split("### secrets")[0]
    assert "loginctl enable-linger" in conv and ">> ~/.config/uv/uv.toml" in conv and "UV_NO_MODIFY_PATH=1" in conv


def test_dry_run_touches_nothing(tmp_path):
    before = set((ROOT / "runs").glob("bootstrap/*")) if (ROOT / "runs").exists() else set()
    dry()
    after = set((ROOT / "runs").glob("bootstrap/*")) if (ROOT / "runs").exists() else set()
    assert after == before                                # no local log dir for a dry run
    assert not any("testbox-dry" in str(p) for p in after)


def test_physics_role_skips_the_render_smokes_and_adds_isaac6():
    out = dry("--role", "physics")
    assert "### smoke_render" not in out and "### smoke_dev_eval" not in out
    assert "### smoke_twin" in out and "### smoke_perception" in out and "### smoke_proxy_eval" in out
    # compute3's Isaac Sim 6.0 build, without its shader caches, then a VERSION + python check
    assert "cd /home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0 && find . -path ./kit/cache -prune" in out
    assert "6.0.0-rc.59+release.41464.5f2772bc.gl" in out
    assert "ISAAC_SIM_ROOT=~/dojo-isaac6/isaac-sim-6.0" in out and "scripts/simlab/proxy_eval.sh --task stack_blocks" in out


def test_auto_role_dry_run_and_with_isaac6():
    out = dry()                                           # default --role auto: the dry run shows full
    assert "role auto: the preflight decides" in out and "### smoke_render" in out
    out = dry("--role", "full", "--with-isaac6")
    assert "### smoke_render" in out and "### smoke_proxy_eval" in out and "# isaac6" in out


def test_basic_probe_runs_only_the_four_commands():
    out = dry("--only", "preflight", "--probe", "basic")
    line = next(x for x in out.splitlines() if x.strip().startswith("+ ssh testbox-dry"))
    remote = line.split("'", 1)[1].rsplit("'", 1)[0]
    assert {c.split()[0] for c in remote.split(";")} == {"nvidia-smi", "echo", "df", "free", "nproc"}
    assert "bash -s" not in out and "curl" not in out    # nothing streamed, no network probes


def test_secrets_dry_run_names_keys_never_values(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OMNIROUTE_BASE_URL=https://x.invalid/v1\nOMNIROUTE_API_KEY=fake-SECRET-1234\n"
                   "ANTHROPIC_AUTH_TOKEN=decoy-SECRET\nexport OMNIROUTE_EXTRA='also SECRET'\n")
    out = dry("--env-file", str(env))
    assert "SECRET" not in out and "x.invalid" not in out
    assert "keys: OMNIROUTE_API_KEY OMNIROUTE_BASE_URL OMNIROUTE_EXTRA" in out and "ANTHROPIC" not in out
    assert "chmod 600 controlr/.env.new" in out
    assert "### secrets" not in dry("--no-secrets")


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


@pytest.mark.parametrize("driver, role, level, picked", [
    ("595.91.07", "full", "FAIL", "full"), ("580.178.04", "full", "PASS", "full"), ("570.86.10", "full", "PASS", "full"),
    ("595.91.07", "physics", "WARN", "physics"), ("580.178.04", "auto", "PASS", "full"),
    ("595.91.07", "auto", "WARN", "physics")])
def test_preflight_driver_gate(tmp_path, driver, role, level, picked):
    stub_nvidia_smi(tmp_path / "bin", driver)
    r = run(["bash", str(LIB), "preflight"], cwd=tmp_path,
            env={"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "ROLE": role, "PREFLIGHT_NET": "0",
                 "HOME": str(tmp_path), "MIN_DISK_GB": "1"})
    line = next(x for x in r.stdout.splitlines() if re.match(r"(PASS|WARN|FAIL)\s+driver\s", x))
    assert line.startswith(level), r.stdout
    assert f"@ role {picked}" in r.stdout.splitlines() and "@ driver " + driver in r.stdout
    assert "sudo -n" not in LIB.read_text()             # the preflight never tries privileges
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


def test_box_json_capabilities_follow_the_smokes(tmp_path):
    st = tmp_path / "box" / "state"
    st.mkdir(parents=True)
    for s, rc, res in (("smoke_twin", 0, "boot 15 s"), ("smoke_proxy_eval", 0, "success 1.0"),
                       ("smoke_render", 0, "skipped (role physics)"), ("smoke_perception", 1, "T 9 mm [FAILED]")):
        (st / f"{s}.rc").write_text(f"{rc}\n")
        (st / f"{s}.result").write_text(res + "\n")
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"], "BOXJ_ROLE": "physics", "BOXJ_BOX": str(tmp_path / "box"),
           "BOXJ_LOCKS": "dojo-gpu-light.lock dojo-gpu-light2.lock", "BOXJ_SB": "-p MemoryMax=24G -p CPUQuota=1400%",
           "BOXJ_MIN": "25", "BOXJ_I6": "/x/isaac-sim-6.0", "BOXJ_SCOPES": "1", "BOXJ_ROOT": str(tmp_path)}
    r = subprocess.run(["python3", str(ROOT / "scripts" / "infra" / "box_json.py")], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["schema"] == "dojo-box/1" and d["role"] == "physics"
    c = d["capabilities"]
    assert c["isaac51_physics"] and c["isaac60_render"] and c["systemd_scopes"]
    assert not c["isaac51_render"] and not c["official_dev_eval"] and not c["perception_egl"] and not c["llm_env"]
    assert d["scopes"]["bundle"] == {"MemoryMax": "24G", "CPUQuota": "1400%"} and len(d["locks"]["light"]) == 2
    assert d["paths"]["isaac_sim_6"] == "/x/isaac-sim-6.0"


def test_manifest_matches_a_reference_and_lands_in_box_json(tmp_path):
    d = tmp_path / "perc"
    (d / "bg_folds_51" / "fold0" / "scene").mkdir(parents=True)
    (d / "bg_folds_51" / "fold0" / "scene" / "cam head.png").write_bytes(b"png")
    (d / "catalog").mkdir()
    (d / "catalog" / "index.json").write_text("{}")
    env = {"BOX": str(tmp_path / "box"), "HOME": str(tmp_path)}
    listing = run(["bash", str(LIB), "manifest", str(d)], env=env)
    assert listing.returncode == 0 and not (tmp_path / "box").exists()           # read-only without NAME
    assert [x.split("  ", 1)[1] for x in listing.stdout.splitlines()] == ["bg_folds_51/fold0/scene/cam head.png",
                                                                        "catalog/index.json"]
    ref = hashlib.md5(listing.stdout.encode()).hexdigest()                     # what the reference box prints
    r = run(["bash", str(LIB), "manifest", str(d), "perception", ref, "compute3"], env=env)
    m = json.loads(r.stdout.splitlines()[-1])
    assert m["files"] == 2 and m["bytes"] == 5 and m["md5"] == ref and m["ref"] == {"host": "compute3", "md5": ref, "equal": True}
    (d / "catalog" / "index.json").write_text("{ }")
    r = run(["bash", str(LIB), "manifest", str(d), "perception", ref, "compute3"], env=env)
    assert json.loads(r.stdout.splitlines()[-1])["ref"]["equal"] is False
    bj = subprocess.run(["python3", str(ROOT / "scripts" / "infra" / "box_json.py")], capture_output=True, text=True,
                        env={"HOME": str(tmp_path), "PATH": os.environ["PATH"], "BOXJ_BOX": str(tmp_path / "box"),
                             "BOXJ_ROLE": "full"}, timeout=60)
    assert json.loads(bj.stdout)["manifests"]["perception"]["ref"]["equal"] is False


def shellcheck_cmd():
    if shutil.which("shellcheck"):
        return ["shellcheck"]
    return None


@pytest.mark.skipif(shellcheck_cmd() is None, reason="shellcheck not installed (uvx --from shellcheck-py shellcheck)")
def test_shellcheck_clean():
    r = run([*shellcheck_cmd(), "-x", str(DRIVER), str(LIB), str(ROOT / "scripts" / "bootstrap_fleet.sh"),
             str(ROOT / "scripts" / "infra" / "dev" / "ssh")])
    assert r.returncode == 0, r.stdout
