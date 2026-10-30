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


def _man(tasks: dict) -> dict:
    return {"schema": "dc19-sealed/1", "block": "D-C19-synthetic",
            "generator": {"repo": "controlr", "commit": "abc", "GENERATOR": "simlab/v15", "settler": "s", "robodojo": "f57e163",
                          "isaacsim": "5.1.0"},
            "seed_salt_sha256": "0" * 64, "created_utc": "2026-10-30T00:00:00Z", "tasks": tasks}


def _block(tmp_path, settled=True, task="stack_blocks", ids=(100100, 100101), b_ids=()):
    d = tmp_path / "sealed"
    (d / task).mkdir(parents=True)
    recs = {}
    for i in list(ids) + list(b_ids):
        data = json.dumps({"layout_id": i, "note": SECRET}).encode()
        (d / task / sealed.record_name(i, settled)).write_bytes(data)
        recs[str(i)] = hashlib.md5(data).hexdigest()
    n = len(ids) + len(b_ids)
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(_man({task: {"settled": settled, "records": recs, "tranche_a": list(ids),
                                         "tranche_b": list(b_ids), "unstable_count": 0, "shortfall": 10 - n}})))
    return d, m


def test_manifest_rules():
    def t(**kw):
        e = {"settled": True, "records": {"100100": "0" * 32}, "tranche_a": [100100], "tranche_b": [],
             "unstable_count": 0, "shortfall": 9}
        e.update(kw)
        return _write(_man({"t": e}))
    sealed.load_manifest(t())
    with pytest.raises(SystemExit, match="outside the block"):
        sealed.load_manifest(t(records={"100120": "0" * 32}, tranche_a=[100120]))
    with pytest.raises(SystemExit, match="bad entry"):
        sealed.load_manifest(t(records={"100100": "xyz"}))
    with pytest.raises(SystemExit, match="settled"):
        sealed.load_manifest(t(settled=None))
    with pytest.raises(SystemExit, match="equal the records"):
        sealed.load_manifest(t(tranche_a=[]))
    with pytest.raises(SystemExit, match="shortfall"):
        sealed.load_manifest(t(shortfall=0))                         # 1 id but no explicit shortfall: refused
    recs = {str(i): "0" * 32 for i in range(100100, 100110)}
    with pytest.raises(SystemExit, match="ascending"):
        sealed.load_manifest(t(records=recs, tranche_a=[100101, 100100, 100102, 100103, 100104],
                               tranche_b=list(range(100105, 100110)), shortfall=0))
    with pytest.raises(SystemExit, match="below every B"):
        sealed.load_manifest(t(records=recs, tranche_a=[100100, 100101, 100102, 100103, 100109],
                               tranche_b=list(range(100104, 100109)), shortfall=0))
    ok = sealed.load_manifest(t(records=recs, tranche_a=list(range(100100, 100105)), tranche_b=list(range(100105, 100110)),
                                shortfall=0))
    assert sealed.position_id(ok, "t", "b", 0) == 100105
    for bad in ({"schema": "x"}, {"seed_salt_sha256": "abc"}, {"generator": {"repo": "controlr"}}):
        m = {**_man({"t": {"settled": True, "records": {"100100": "0" * 32}, "tranche_a": [100100], "tranche_b": [],
                            "unstable_count": 0, "shortfall": 9}}), **bad}
        with pytest.raises(SystemExit):
            sealed.load_manifest(_write(m))


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


def test_tranche_b_runs_only_once_preregistered(tmp_path):
    a_ids, b_ids = list(range(100100, 100105)), list(range(100106, 100111))
    d, m = _block(tmp_path, ids=a_ids, b_ids=b_ids)
    man = sealed.load_manifest(m)
    assert sealed.position_id(man, "stack_blocks", "a", 4) == 100104 and sealed.position_id(man, "stack_blocks", "b", 0) == 100106
    with pytest.raises(SystemExit, match="tranche B not pre-registered"):
        sealed.stage(d, man, "stack_blocks", [100106], tmp_path / "run")
    (m.parent / sealed.PREREGISTERED).write_text("sha256 of the B list\n")
    assert sealed.stage(d, man, "stack_blocks", [100106], tmp_path / "run")[0][0] == 100106


def test_verify_prints_hashes_only_and_fails_on_strays(tmp_path, capsys):
    d, m = _block(tmp_path)
    (d / ".salt").write_text("s")
    assert sealed.main(["verify", "--dir", str(d), "--manifest", str(m)]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and sum(ln.endswith(" pass") for ln in out.splitlines()) == 2
    assert f"verifier md5 {sealed.self_md5()}" in out
    (d / "stack_blocks" / "100109.settled.json").write_text("{}")             # an unlisted file
    (d / "other_task").mkdir()                                                 # an unlisted task folder
    assert sealed.main(["verify", "--dir", str(d), "--manifest", str(m)]) == 1
    out = capsys.readouterr().out
    assert "unlisted file stack_blocks/100109.settled.json" in out and "unlisted task folder other_task" in out
    (d / "stack_blocks" / "100109.settled.json").unlink()
    (d / "other_task").rmdir()
    p = d / "stack_blocks" / "100101.settled.json"
    p.write_bytes(b"")                                                         # an empty record
    assert sealed.main(["verify", "--dir", str(d), "--manifest", str(m)]) == 1
    assert "100101 d41d8cd98f00b204e9800998ecf8427e fail" in capsys.readouterr().out


def test_claim_is_run_once_with_one_disclosed_rerun(tmp_path, monkeypatch):
    d, m = _block(tmp_path)
    man = sealed.load_manifest(m)
    coord = tmp_path / "coord"
    made = sealed.claim(coord, man, "stack_blocks", [100100], "run1")
    p = coord / "scored" / "D-C19-synthetic" / "stack_blocks-100100.json"
    assert made == [str(p)] and json.loads(p.read_text())["run_id"] == "run1"
    with pytest.raises(SystemExit, match="already claimed"):
        sealed.claim(coord, man, "stack_blocks", [100100], "run2")
    monkeypatch.setenv("DC19_INFRA_RERUN", "Kit froze at step 0")
    sealed.claim(coord, man, "stack_blocks", [100100], "run2")
    assert json.loads(Path(str(p) + ".rerun").read_text())["infra_rerun_reason"] == "Kit froze at step 0"
    with pytest.raises(SystemExit, match="re-run is spent"):
        sealed.claim(coord, man, "stack_blocks", [100100], "run3")
    with pytest.raises(SystemExit, match="unlisted"):
        sealed.claim(coord, man, "stack_blocks", [100105], "run4")


def test_official_eval_sealed_mode_claims_before_anything_starts(tmp_path):
    d, m = _block(tmp_path)
    sh = str(ROOT / "scripts/dojo/official_eval.sh")
    base = ["--inner", "--task", "stack_blocks", "--layouts", "100100", "--sealed-block", str(d), "--manifest", str(m),
            "--lock", str(tmp_path / "x.lock")]
    env = {**_env(tmp_path), "DOJO_PERCEPTION_DATA": str(tmp_path / "pd")}
    r = subprocess.run(["bash", sh] + base + ["--id", "c1"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "needs COORD_DIR" in r.stderr
    env["COORD_DIR"] = str(tmp_path / "coord")
    r = subprocess.run(["bash", sh] + base + ["--id", "c2"], capture_output=True, text=True, env=env)
    assert "claimed stack_blocks-100100.json" in r.stderr and SECRET not in r.stderr + r.stdout   # (stops later: no calib)
    r = subprocess.run(["bash", sh] + base + ["--id", "c3"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "already claimed" in r.stderr


def test_generation_script_makes_a_valid_block_from_synthetic_draws(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts" / "dojo"))
    import dc19_generate as GEN

    def gen(task, i, salt, v6):                     # synthetic: no generator, never 100100+ for real
        return None if i == 100102 else {"layout_id": i, "salted": hashlib.sha256(salt.encode()).hexdigest()[:4],
                                         "note": SECRET}

    def settle(task, work, lock, log):              # synthetic: every raw record but 100107 "settles"
        dd = work / "simlab" / "layouts" / task
        for p in dd.glob("*.json"):
            if p.stem != "100107" and ".settled" not in p.name:
                p.with_name(f"{p.stem}.settled.json").write_bytes(p.read_bytes() + b" ")

    prov = lambda: {"repo": "controlr", "commit": "abc", "GENERATOR": "simlab/v15", "settler": "s",   # noqa: E731
                    "robodojo": "f57e163", "isaacsim": "5.1.0"}
    out, man_p = tmp_path / "block", tmp_path / "manifest.json"
    res = GEN.run(out, man_p, "lock", ["stack_blocks", "make_toast"], None, gen=gen, settle=settle, prov=prov)
    man = sealed.load_manifest(man_p)
    sb, mt = man["tasks"]["stack_blocks"], man["tasks"]["make_toast"]
    assert sb["tranche_a"] == [100100, 100101, 100103, 100104, 100105] and sb["tranche_b"][0] == 100106
    assert 100107 not in sb["tranche_b"] and sb["unstable_count"] == 2 and sb["shortfall"] == 0
    assert mt["settled"] is False and (out / "make_toast" / "100100.json").is_file()      # raw tasks keep the raw record
    assert stat.S_IMODE((out / ".salt").stat().st_mode) == 0o400
    assert man["seed_salt_sha256"] == hashlib.sha256((out / ".salt").read_text().encode()).hexdigest()
    rows, problems = sealed.verify_dir(out, man)
    assert all(r[3] for r in rows) and not problems                                   # only records + .salt in DIR
    assert not list(tmp_path.glob(".block.work-*")) and res["counts"]["stack_blocks"] == (18, 2)
    assert res["sha256"] == hashlib.sha256(man_p.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="--go"):
        GEN.main(["--out", str(tmp_path / "b2"), "--manifest", str(tmp_path / "m2.json"), "--lock", "l", "--v6-checkout", "v"])
    with pytest.raises(SystemExit, match="exists"):
        GEN.main(["--out", str(out), "--manifest", str(tmp_path / "m3.json"), "--lock", "l", "--v6-checkout", "v", "--go"])
