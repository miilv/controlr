"""Sealed joint block (robot-harness D-C19): only manifest ids, md5-verified, staged into the run only, no dev path.
SYNTHETIC fixtures only: hand-written records with ids >= 100100 in a tmp dir (the generator never runs on 100100+)."""

import hashlib
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dojo.simlab import sealed

ROOT = Path(__file__).resolve().parents[2]
SECRET = "SYNTHETIC-RECORD-CONTENT-MARKER"


def _block(tmp_path, settled=True, task="stack_blocks", ids=(100100, 100101)):
    d = tmp_path / "sealed"
    (d / task).mkdir(parents=True)
    recs = {}
    for i in ids:
        data = json.dumps({"layout_id": i, "note": SECRET}).encode()
        (d / task / sealed.record_name(i, settled)).write_bytes(data)
        recs[str(i)] = hashlib.md5(data).hexdigest()
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"block": "synthetic", "tasks": {task: {"settled": settled, "records": recs}}}))
    return d, m


def test_manifest_rules():
    with pytest.raises(SystemExit, match="joint-block id"):
        sealed.load_manifest(_write({"tasks": {"t": {"settled": True, "records": {"100005": "0" * 32}}}}))
    with pytest.raises(SystemExit, match="bad entry"):
        sealed.load_manifest(_write({"tasks": {"t": {"settled": True, "records": {"100100": "xyz"}}}}))
    with pytest.raises(SystemExit, match="settled"):
        sealed.load_manifest(_write({"tasks": {"t": {"records": {"100100": "0" * 32}}}}))


def _write(d):
    import tempfile
    f = Path(tempfile.mkdtemp()) / "m.json"
    f.write_text(json.dumps(d))
    return f


@pytest.mark.parametrize("settled", [True, False])
def test_stage_verifies_md5_and_writes_read_only_copies_into_the_run_only(tmp_path, settled, capsys):
    d, m = _block(tmp_path, settled=settled)
    dest = tmp_path / "run" / "layouts"
    assert sealed.main(["stage", "--dir", str(d), "--manifest", str(m), "--task", "stack_blocks",
                        "--ids", "100101 100100", "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and out.count(" pass") == 2                     # id + md5 + pass only
    staged = sorted(p.name for p in (dest / "stack_blocks").iterdir())
    suffix = ".settled.json" if settled else ".json"
    assert staged == [f"100100{suffix}", f"100101{suffix}"]
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o400 for p in (dest / "stack_blocks").iterdir())
    assert sealed.main(["settled", "--manifest", str(m), "--task", "stack_blocks"]) == 0
    assert capsys.readouterr().out.strip() == ("1" if settled else "0")


def test_stage_refuses_a_mismatch_an_unlisted_id_or_a_missing_record_and_stages_nothing(tmp_path):
    d, m = _block(tmp_path)
    man = sealed.load_manifest(m)
    dest = tmp_path / "run"
    with pytest.raises(SystemExit, match="100102 unlisted"):
        sealed.stage(d, man, "stack_blocks", [100100, 100102], dest)
    (d / "stack_blocks" / "100101.settled.json").write_bytes(b'{"tampered": true}')
    with pytest.raises(SystemExit, match="100101 md5") as e:
        sealed.stage(d, man, "stack_blocks", [100100, 100101], dest)
    assert SECRET not in str(e.value) and not (dest / "stack_blocks").exists()
    with pytest.raises(SystemExit, match="not in the manifest"):
        sealed.stage(d, man, "make_toast", [100100], dest)


def _patched_seed_manager(monkeypatch):
    class SeedManager:
        task_name, config = "stack_blocks", {}
    mods = {"env": types.ModuleType("env"), "env.seed_manager": types.ModuleType("env.seed_manager"),
            "env.seed_manager.seed_manager": types.ModuleType("env.seed_manager.seed_manager"),
            "utils": types.ModuleType("utils"), "utils.load_file": types.ModuleType("utils.load_file")}
    mods["env.seed_manager.seed_manager"].SeedManager = SeedManager
    mods["env.seed_manager"].seed_manager = mods["env.seed_manager.seed_manager"]
    mods["utils.load_file"].load_json = lambda p: json.loads(Path(p).read_text())
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return SeedManager


def test_dev_eval_runs_joint_ids_only_under_the_manifest(tmp_path, monkeypatch):
    from dojo.simlab import dev_eval
    d, m = _block(tmp_path)
    dest = tmp_path / "run" / "layouts"
    sealed.stage(d, sealed.load_manifest(m), "stack_blocks", [100100, 100101], dest)
    SM = _patched_seed_manager(monkeypatch)
    monkeypatch.setenv("SIMLAB_DEV_LAYOUT_SUFFIX", ".settled")
    monkeypatch.delenv("SIMLAB_SEALED_MANIFEST", raising=False)
    dev_eval.patch_seed_manager(dest)
    with pytest.raises(SystemExit, match="not ours"):                       # without the manifest: refused as ever
        SM().init_eval()
    monkeypatch.setenv("SIMLAB_SEALED_MANIFEST", str(m))
    sm = SM()
    sm.init_eval()
    assert [v["layout_id"] for v in sm.seed_info.values()] == [100100, 100101]
    p = dest / "stack_blocks" / "100101.settled.json"
    p.chmod(0o600)
    p.write_bytes(b"{}")                                                     # tampered after staging
    with pytest.raises(SystemExit, match="refused"):
        SM().init_eval()


def _env(tmp_path):
    return {**{k: v for k, v in os.environ.items() if not k.startswith(("OMNIROUTE_", "HARNESS_"))},
            "DOJO_DATA": str(tmp_path / "data")}


def test_official_eval_sealed_mode_needs_a_lock_refuses_dev_and_bad_records(tmp_path):
    d, m = _block(tmp_path)
    sh = str(ROOT / "scripts/dojo/official_eval.sh")
    base = ["--task", "stack_blocks", "--layouts", "100100", "--sealed-block", str(d), "--manifest", str(m)]
    r = subprocess.run(["bash", sh] + base, capture_output=True, text=True, env=_env(tmp_path))
    assert r.returncode == 2 and "explicit --lock" in r.stderr                # no default lock in sealed mode
    lock = ["--lock", str(tmp_path / "x.lock")]
    r = subprocess.run(["bash", sh, "--inner"] + base + lock, capture_output=True, text=True,
                       env={**_env(tmp_path), "HARNESS_DEV": "1"})
    assert r.returncode == 2 and "refuses HARNESS_DEV" in r.stderr
    (d / "stack_blocks" / "100100.settled.json").write_bytes(b'{"tampered": 1}')
    r = subprocess.run(["bash", sh, "--inner"] + base + lock + ["--id", "t1"], capture_output=True, text=True,
                       env=_env(tmp_path))
    assert r.returncode == 2 and "refused" in r.stderr and SECRET not in r.stderr + r.stdout
    assert not list((tmp_path / "data").rglob("*.settled.json"))            # nothing staged
    r = subprocess.run(["bash", sh, "--inner", "--task", "stack_blocks", "--layouts", "100100"], capture_output=True,
                       text=True, env=_env(tmp_path))
    assert r.returncode == 2 and "reserved" in r.stderr                      # the dev path still refuses >= 100100


def test_route_probe_reports_ok_and_rate_limits_without_keys():
    sys.path.insert(0, str(ROOT / "scripts" / "dojo"))
    import route_probe
    from dojo.perception.agentic.llm import ChatReply

    ok = route_probe.probe("cx/x", chat=lambda model, msgs, max_tokens=16: ChatReply(text="OK", completion_tokens=1))
    assert ok["ok"] and not ok["rate_limited"] and ok["model"] == "cx/x"
    lim = route_probe.probe("claude/x", chat=lambda model, msgs, max_tokens=16: ChatReply(
        text="", error="HTTP 429: weekly limit reached, resets in 61 h"))
    assert not lim["ok"] and lim["rate_limited"] and "429" in lim["error"]
    assert route_probe.default_model()
