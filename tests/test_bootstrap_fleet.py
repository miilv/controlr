"""scripts/bootstrap_fleet.sh + the fleet half of scripts/bootstrap_gpu_box.sh: the dry-run plan, and real runs on fake
boxes (sandbox HOMEs through scripts/infra/dev/ssh) with a fake source box: peer-to-peer copies, secrets, conventions.

No network, no GPU, no real ssh: every host is "fake*" (the shim runs it locally); nvidia-smi / loginctl / systemd-run
are stubs; logs go to tmp_path (BOOT_RUNS_DIR)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "scripts" / "bootstrap_fleet.sh"
BOX = ROOT / "scripts" / "bootstrap_gpu_box.sh"
SHIM = ROOT / "scripts" / "infra" / "dev"


def run(args, env=None, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=timeout)


def test_fleet_dry_run_plans_seeds_p2p_and_the_key():
    r = run(["bash", str(FLEET), "--dry-run", "boxa", "boxb:physics", "boxc"])
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "seed: boxa fetches Assets from HF" in out and "boxb fetches Isaac Sim 6.0 from compute3" in out
    assert out.count("ssh-keygen -q -t ed25519") == 1 and "the private key: stdin only, never printed" in out
    for h in ("boxa", "boxb", "boxc"):
        assert f"########## {h}: scripts/bootstrap_gpu_box.sh {h}" in out
        assert f"ssh {h} \"sed -i '/ dojo-fleet-key$/d' .ssh/authorized_keys" in out     # removed at the end
    # the seed fetches from the origins; the others wait for a holder and copy box-to-box (rsync), else relay
    assert "fleet: boxa is the seed for assets" in out and "fleet: boxb is the seed for isaac6" in out
    assert "fleet: wait until a fleet box has assets (seed boxa" in out
    assert "rsync -a --partial --stats --exclude /.git/lfs --exclude /Assets/Eval_Layout" in out
    assert "# else (no direct path between the two boxes): relayed through this box" in out
    assert "--role physics --fleet-dir" in out and "boxc --role auto --fleet-dir" in out


def test_fleet_keep_key_and_relay_mode():
    out = run(["bash", str(FLEET), "--dry-run", "--keep-fleet-key", "boxa", "boxb"]).stdout
    assert "fleet key removal" not in out
    out = run(["bash", str(FLEET), "--dry-run", "--p2p", "relay", "boxa", "boxb"]).stdout
    assert "ssh-keygen" not in out and "rsync -a --partial --stats" not in out and "tar --null -T -" in out


@pytest.fixture()
def sandbox(tmp_path):
    """Fake boxes fakesrc (the source, like compute3), fakea (driver 580), fakeb (driver 595)."""
    sbx, stub = tmp_path / "sbx", tmp_path / "stub"
    stub.mkdir()
    (stub / "nvidia-smi").write_text("""#!/usr/bin/env bash
drv="$(cat "$HOME/.stub_driver" 2>/dev/null || echo 580.95.05)"
case "$*" in
  *query-compute-apps*) ;;
  -L) echo "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-0)" ;;
  *memory.used*) echo "NVIDIA GeForce RTX 5090, $drv, 32607, 300" ;;
  *) echo "NVIDIA GeForce RTX 5090, $drv, 32607" ;;
esac
""")
    (stub / "loginctl").write_text("#!/bin/sh\necho stub\n")
    (stub / "systemd-run").write_text("#!/bin/sh\nexit 0\n")
    for f in stub.iterdir():
        f.chmod(0o755)
    for h in ("fakea", "fakeb"):
        b = sbx / h / ".local" / "bin"
        b.mkdir(parents=True)
        (b / "uv").write_text('#!/bin/sh\necho "uv 0.12.1"\n')
        (b / "uv").chmod(0o755)
    (sbx / "fakeb" / ".stub_driver").write_text("595.91.07\n")
    src = sbx / "fakesrc"
    for d, n in (("dojo-data/simlab/g2_c3b", 5), ("dojo-perception-data/task_config", 3)):
        (src / d / "sub").mkdir(parents=True)
        for i in range(n):
            (src / d / "sub" / f"f{i}.json").write_text(json.dumps({"i": i, "d": d}))
    (src / "dojo-data/simlab/g2_c3b/empty_dir").mkdir()
    env = {"PATH": f"{SHIM}:{stub}:{os.environ['PATH']}", "BOOT_SANDBOX": str(sbx), "BOOT_RUNS_DIR": str(tmp_path / "runs"),
           "PREFLIGHT_NET": "0", "FLEET_POLL_S": "1", "FLEET_TICK_S": "1",
           "BOOT_DATA_DIRS": "dojo-data/simlab/g2_c3b", "BOOT_CATALOG_DIRS": "dojo-perception-data/task_config"}
    return sbx, tmp_path / "runs", env


def test_fleet_copies_peer_to_peer(sandbox):
    sbx, runs, env = sandbox
    r = run(["bash", str(FLEET), "--p2p", "relay", "--ref", "HEAD", "--source", "fakesrc", "--only", "data,catalog", "--no-secrets",
             "--min-disk-gb", "1", "--ignore-preflight", "fakea", "fakeb"], env=env)
    fd = next(runs.glob("fleet-*"))
    la, lb = (fd / "logs" / "fakea.log").read_text(), (fd / "logs" / "fakeb.log").read_text()
    assert "relay fakesrc:dojo-data/simlab/g2_c3b -> fakea" in la, la            # the seed: from the source
    assert "copying data from fakea" in lb and "relay fakea:dojo-data/simlab/g2_c3b -> fakeb" in lb, lb
    assert "copying catalog from fakea" in lb and "relay fakesrc:" not in lb       # never from the source itself
    for d in ("dojo-data/simlab/g2_c3b", "dojo-perception-data/task_config"):
        a = sorted(p.relative_to(sbx / "fakesrc" / d) for p in (sbx / "fakesrc" / d).rglob("*"))
        b = sorted(p.relative_to(sbx / "fakeb" / d) for p in (sbx / "fakeb" / d).rglob("*"))
        assert a == b and (sbx / "fakeb" / d / "sub" / "f0.json").read_text() == (sbx / "fakesrc" / d / "sub" / "f0.json").read_text()
    assert (sbx / "fakeb" / "dojo-data/simlab/g2_c3b/empty_dir").is_dir()         # empty dirs too
    table = (fd / "table.txt").read_text()
    assert "fakea" in table and "fakeb" in table and "physics" in table          # fakeb: driver 595 -> role physics
    sb = dict(x.split("\t", 1) for x in (fd / "summary" / "fakeb.tsv").read_text().splitlines() if not x.startswith("step"))
    assert sb["role"] == "physics" and sb["driver"] == "595.91.07" and "auto" in sb["role_reason"]
    assert r.returncode == 1                                                     # a partial run is not "ready"


def test_secrets_and_conventions_on_a_box(sandbox, tmp_path):
    sbx, runs, env = sandbox
    dotenv = tmp_path / "dotenv"
    dotenv.write_text("OMNIROUTE_BASE_URL=https://x.invalid/v1\nOMNIROUTE_API_KEY=fake-SECRET-42\n"
                      "ANTHROPIC_AUTH_TOKEN=decoy-SECRET\n")
    r = run(["bash", str(BOX), "fakeb", "--only", "conventions,secrets", "--env-file", str(dotenv), "--min-disk-gb", "1"],
            env=env)
    assert r.returncode == 1, r.stdout                                           # not ready (only two steps), but both OK
    assert "conventions: done" in r.stdout or "conventions       done" in r.stdout, r.stdout
    h = sbx / "fakeb"
    e = h / "controlr" / ".env"
    assert stat.S_IMODE(e.stat().st_mode) == 0o600
    lines = e.read_text().splitlines()
    assert [x.split("=")[0] for x in lines] == ["OMNIROUTE_BASE_URL", "OMNIROUTE_API_KEY"]   # no other key
    for f in runs.rglob("*"):                                                    # values never logged here
        if f.is_file():
            assert "SECRET" not in f.read_text(errors="replace"), f
    for f in (h / "dojo-bootstrap" / "logs").glob("*.log"):
        assert "SECRET" not in f.read_text(errors="replace"), f
    envsh = (h / "dojo-env.sh").read_text()
    assert 'DOJO_BOX_ROLE="physics"' in envsh and "ISAAC_SIM_ROOT=" in envsh and "DOJO_GPU_LIGHT_LOCKS=" in envsh
    assert 'DOJO_SCOPE_BUNDLE="-p MemoryMax=24G -p CPUQuota=1400%"' in envsh
    for lk in ("dojo-gpu.lock", "dojo-gpu-light.lock", "dojo-gpu-light2.lock", "dojo-gpu-light3.lock",
               "dojo-gpu-light4.lock"):
        assert (h / lk).exists(), lk
    box = json.loads((h / "dojo-box.json").read_text())
    assert box["schema"] == "dojo-box/1" and box["role"] == "physics" and box["ssh_alias"] == "fakeb"
    assert box["gpus"][0]["driver"] == "595.91.07" and box["scopes"]["light"]["MemoryMax"] == "8G"


def test_fleet_preflight_only_writes_nothing(sandbox):
    sbx, runs, env = sandbox
    r = run(["bash", str(FLEET), "--only", "preflight", "--probe", "basic", "--min-disk-gb", "1", "fakea", "fakeb"],
            env=env)
    assert "fakea" in r.stdout and "580.95.05" in r.stdout and "595.91.07" in r.stdout, r.stdout
    assert not (sbx / "fakea" / "dojo-bootstrap").exists() and not (sbx / "fakeb" / "dojo-bootstrap").exists()
    assert not (runs / "fleet_key").exists()                                     # no fleet key for a preflight
