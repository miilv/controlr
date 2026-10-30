"""The official path on our dev layouts (scripts/dojo/official_eval.sh, dojo.bridge.official): RoboDojo's result rows
mapped back to our layout ids, the pairing with integ_eval rows and the first action divergence, episodes back to back
through ONE strict server (the next reset ends the previous episode), and the RoboDojo farm. CPU only."""

import gzip
import json
import os
import subprocess
from pathlib import Path

import pytest

from dojo.bridge import official as O
from dojo.bridge.backend import RobotServerBackend
from dojo.bridge.client import RobotServerClient
from dojo.bridge.driver import run_episode
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from test_bridge import EnvPerception, bench, serve  # noqa: F401  (bench: the short-path tmp fixture)
from test_runtime import FakeAssets, _stacked

ROOT = Path(__file__).resolve().parents[2]


def _act(x: float) -> dict:
    return {"left_arm_joint_state": [x] * 6, "left_ee_joint_state": [1.0],
            "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [1.0]}


def _gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def _official_run(tmp: Path, task: str, ids: list[int], verdicts: dict, actions: dict, ends: dict) -> Path:
    out = tmp / "off"
    for lid in ids:
        p = out / "layouts" / task / f"{lid}.settled.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    # RoboDojo's details: episode index -> layout_id = index of the dev layout in id order (dev_eval.py)
    det = {str(k): {"layout_id": ids.index(lid), "success": v[0], "score": v[1]}
           for k, (lid, v) in enumerate(sorted(verdicts.items()))}
    (out / "_result.json").write_text(json.dumps({"details": det, "success_rate": 0.5}))
    for lid in ids:
        d = out / "eps" / str(lid) / "off-1"
        d.mkdir(parents=True)
        (d / "bridge.json").write_text(json.dumps({"end": ends[lid], "steps_used": 100 + lid % 10, "step_lim": 700,
                                                   "wall_s": 12.0, "backend": {"tracking_max_rad": 0.01}}))
        _gz(d / "report.json.gz", {"skill": "press_sequence", "history": [], "events": [], "result": {"ok": True}})
        _gz(d / "actions.json.gz", {"actions": actions[lid]})
    return out


def _integ_run(tmp: Path, task: str, rows: dict, actions: dict) -> Path:
    d = tmp / "integ"
    d.mkdir()
    tag = f"{task}_perception_verify-bundle"
    with (d / f"{tag}_episodes.jsonl").open("w") as f:
        for lid, (succ, score, steps, cat) in rows.items():
            f.write(json.dumps({"task": task, "layout_id": lid, "success": succ, "score": score, "steps_used": steps,
                                "category": cat}) + "\n")
            _gz(d / f"{tag}_{lid}_actions.json.gz", {"actions": actions[lid]})
    return d


def test_rows_map_robodojo_details_to_our_layouts_and_pair_with_integ(tmp_path):
    task, ids = "press_by_number", [100000, 100001, 100002]
    a0 = [_act(0.01 * i) for i in range(5)]
    a1 = [_act(0.01 * i) for i in range(5)]
    a1_div = a1[:3] + [_act(0.5)] + a1[4:]
    out = _official_run(tmp_path, task, ids, {100000: (True, 1.0), 100001: (False, 0.4), 100002: (True, 1.0)},
                        {100000: a0, 100001: a1, 100002: a0}, {100000: "ended", 100001: "policy_done", 100002: "ended"})
    assert O.staged(out) == (task, ids)
    rs = O.rows(out)
    assert [r["layout_id"] for r in rs] == ids
    assert [r["success"] for r in rs] == [True, False, True] and rs[1]["score"] == 0.4
    assert rs[0]["category"] is None and rs[1]["category"] is not None
    integ = _integ_run(tmp_path, task, {100000: (True, 1.0, 100, None), 100001: (True, 1.0, 101, None),
                                         100002: (True, 1.0, 102, None)},
                       {100000: a0, 100001: a1_div, 100002: a0[:4]})
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    ps = O.pair(O.load_official([str(out)]), O.load_integ([str(integ)]))
    by = {p["layout_id"]: p for p in ps}
    assert by[100000]["agree"] and by[100000]["first_divergence"] is None
    assert not by[100001]["agree"] and by[100001]["first_divergence"] == 3
    assert by[100002]["first_divergence"] == 4                       # one list is a prefix of the other
    md = O.table_md(ps)
    assert "agree 2/3" in md and "step 3" in md


def test_first_divergence():
    a = [_act(0.0), _act(0.1)]
    assert O.first_divergence(a, [dict(x) for x in a]) is None
    assert O.first_divergence(a, [a[0], _act(0.1 + 1e-7)]) is None         # within tol
    assert O.first_divergence(a, [a[0], _act(0.2)]) == 1
    assert O.first_divergence(a, None) == -1


def test_episodes_back_to_back_through_one_server(bench):  # noqa: F811
    """official_eval.sh's sequence: one server, one driver per episode; the next driver's reset ends the previous
    episode on the server (the shim books it), the last one closes."""
    srv, be, envs = serve(bench, n=2, sid="b2")
    for k in range(2):
        b = be if k == 0 else RobotServerBackend(RobotServerClient("b2", bench=str(bench)))
        b.connect()
        pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(),
                            perception_client=EnvPerception(envs))
        res = run_episode(pol, b)
        assert res["summary"]["end"] == "policy_done", res["summary"]
        assert _stacked(envs[k])
    b.close()
    assert [e["outcome"] for e in srv.episodes] == ["reset", "close"]
    srv.stop()


def test_rd_farm_keeps_the_shared_install_read_only(tmp_path):
    src = tmp_path / "rd"
    for p in ("RoboDojo/XPolicyLab/policy/demo", "RoboDojo/src", "RoboDojo/eval_result", "miniconda3", "policy_venv"):
        (src / p).mkdir(parents=True)
    (src / "RoboDojo/XPolicyLab/setup_policy_server.py").write_text("print('x')\n")
    (src / "RoboDojo/XPolicyLab/policy/__init__.py").write_text("")
    (src / "robodojo_commit.txt").write_text("f57e163\n")
    farm = tmp_path / "farm"
    sh = ROOT / "scripts" / "dojo" / "rd_farm.sh"
    r = subprocess.run(["bash", str(sh), str(src), str(farm)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (farm / "RoboDojo/src").is_symlink() and (farm / "miniconda3").is_symlink()
    for ours in ("RoboDojo/eval_result", "RoboDojo/XPolicyLab/policy"):
        assert (farm / ours).is_dir() and not (farm / ours).is_symlink()
    assert not (farm / "RoboDojo/XPolicyLab/setup_policy_server.py").is_symlink()     # a byte copy
    assert (farm / "RoboDojo/XPolicyLab/policy/__init__.py").is_symlink()
    assert not (farm / "RoboDojo/XPolicyLab/policy/demo").exists()                     # the shared policies stay out
    r = subprocess.run(["bash", str(sh), str(src), str(farm), "--check"], capture_output=True, text=True)
    assert r.returncode == 0 and "farm ok" in r.stdout
    r2 = subprocess.run(["bash", str(sh), str(src), str(farm)], capture_output=True, text=True)
    assert r2.returncode == 0 and "link" not in r2.stdout                              # idempotent


def test_launchers_parse():
    for s in ("scripts/dojo/official_eval.sh", "scripts/dojo/rd_farm.sh", "scripts/simlab/dev_eval.sh"):
        r = subprocess.run(["bash", "-n", str(ROOT / s)], capture_output=True, text=True)
        assert r.returncode == 0, (s, r.stderr)


def test_official_eval_never_reads_eval_layouts_or_verdicts_before_the_close():
    src = (ROOT / "scripts" / "dojo" / "official_eval.sh").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "Eval_Layout" not in code.replace("*Eval_Layout*", "")         # only the refusal pattern names it
    assert "-ge 100000" in src                                           # dev layouts only
    # the result is copied after the eval client exited, never read inside the episode loop
    loop = src[src.index("# 5. one driver per episode"):src.index("# 6. the eval client writes")]
    assert "_result.json" not in loop and "dev.jsonl" not in loop


def test_reserved_layout_ids_are_refused(tmp_path, monkeypatch):
    """D-C10: ids >= 100100 are the joint paired runs' layouts: the official path never loads them."""
    import sys
    import types

    from dojo.simlab import dev_eval, paths
    assert paths.is_reserved_id(100100) and paths.is_reserved_id("100250") and not paths.is_reserved_id(100099)
    d = tmp_path / "layouts" / "stack_blocks"
    d.mkdir(parents=True)
    for lid in (100000, 100001):
        (d / f"{lid}.settled.json").write_text("{}")

    class SeedManager:                                   # RoboDojo's class, as dev_eval.py patches it
        task_name, config = "stack_blocks", {}
    mods = {"env": types.ModuleType("env"), "env.seed_manager": types.ModuleType("env.seed_manager"),
            "env.seed_manager.seed_manager": types.ModuleType("env.seed_manager.seed_manager"),
            "utils": types.ModuleType("utils"), "utils.load_file": types.ModuleType("utils.load_file")}
    mods["env.seed_manager.seed_manager"].SeedManager = SeedManager
    mods["env.seed_manager"].seed_manager = mods["env.seed_manager.seed_manager"]
    mods["utils.load_file"].load_json = lambda p: json.loads(Path(p).read_text())
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setenv("SIMLAB_DEV_LAYOUT_SUFFIX", ".settled")
    dev_eval.patch_seed_manager(tmp_path / "layouts")
    sm = SeedManager()
    sm.init_eval()
    assert [v["layout_id"] for v in sm.seed_info.values()] == [100000, 100001]
    (d / "100100.settled.json").write_text("{}")
    with pytest.raises(SystemExit):
        SeedManager().init_eval()
    r = subprocess.run(["bash", str(ROOT / "scripts" / "dojo" / "official_eval.sh"), "--inner", "--task", "stack_blocks",
                        "--layouts", "100100"], capture_output=True, text=True, env={**os.environ, "DOJO_DATA": str(tmp_path)})
    assert r.returncode == 2 and "reserved" in r.stderr


def test_unstable_episodes_are_not_samples_on_either_path(tmp_path):
    """RoboDojo's eval client drops an episode the task marked unstable (make_kong's scripted discard failed, ...):
    no result row. integ_eval now records it (``unstable_episode``) and leaves it out of n, and the pairing agrees."""
    import types

    from dojo.runtime import eval as EV
    from dojo.runtime.policy import RuntimeConfig as RC
    from dojo.simlab.runner import RoboDojoRunner

    env = types.SimpleNamespace(success=[True], end_flag=[True], take_action_cnt=[300], step_lim=600, unstable_envs={0},
                                reward_manager=types.SimpleNamespace(check_list=[[]], final_check_list=[[]], query_list=[[]]))
    rw = RoboDojoRunner.reward(types.SimpleNamespace(env=env), 0)
    assert rw["unstable"] is True
    assert EV.classify({"success": True}, rw, {}, None, "perception")[0] == "unstable_episode"
    rows = [{"success": False, "score": 0.0, "steps_used": 300, "step_lim": 600, "layout_id": 100003, "unstable_episode": True},
            {"success": True, "score": 1.0, "steps_used": 400, "step_lim": 600, "layout_id": 100002}]
    summ = EV.summarize(rows, "make_kong", RC(mode="perception"))
    assert summ["n"] == 1 and summ["success"] == 1 and summ["unstable"] == 1

    task, ids = "make_kong", [100002, 100003]
    a = [_act(0.0)]
    out = _official_run(tmp_path, task, ids, {100002: (True, 1.0)}, {100002: a, 100003: a},
                        {100002: "ended", 100003: "policy_done"})
    rs = O.rows(out)
    assert rs[1]["success"] is None and rs[1]["category"] == "unstable_episode"
    integ = _integ_run(tmp_path, task, {100002: (True, 1.0, 400, None), 100003: (False, 0.0, 300, "unstable_episode")},
                       {100002: a, 100003: a})
    lines = (integ / f"{task}_perception_verify-bundle_episodes.jsonl").read_text().splitlines()
    lines[1] = json.dumps({**json.loads(lines[1]), "unstable_episode": True})
    (integ / f"{task}_perception_verify-bundle_episodes.jsonl").write_text("\n".join(lines) + "\n")
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    ps = {p["layout_id"]: p for p in O.pair(O.load_official([str(out)]), O.load_integ([str(integ)]))}
    assert ps[100002]["agree"] and ps[100003]["agree"]


CAND = """# Integration: the candidate
--preset submit_v3 --setenv DOJO_GRIP_SOURCE=v2 --seg v2b --glyph digit_v2 --glyph-kong kong_v1b
--setenv DOJO_PERCEPTION_CFG={"series_count":"seg","glyph_read":"on"}
--config {"grip_check":"on","ported_classify":"p2","kong_draw_read":"place"}
"""


def _models(tmp_path):
    m = tmp_path / "models"
    for d, files in (("v2b", ("best.pt", "classes.json")), ("digit_v2", ("model.pt",)), ("kong_v1b", ("model.pt",))):
        (m / d).mkdir(parents=True)
        for f in files:
            (m / d / f).write_text(d + f)
    return m


def test_official_eval_takes_the_candidate_file_verbatim(tmp_path):
    """D-C19: the official path runs the SAME system as integ_eval: CANDIDATE.txt verbatim (preset, setenv, side
    servers, config), RuntimePolicy by default with a candidate; --dry-run resolves it without starting anything."""
    (tmp_path / "CANDIDATE.txt").write_text(CAND)
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("OMNIROUTE_")}, "DOJO_SEG_MODELS": str(_models(tmp_path))}
    r = subprocess.run(["bash", str(ROOT / "scripts/dojo/official_eval.sh"), "--candidate", str(tmp_path / "CANDIDATE.txt"),
                        "--dry-run"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["policy"] == "dojo.runtime.policy:RuntimePolicy" and d["preset"] == "submit_v3"
    assert d["setenv"] == {"DOJO_GRIP_SOURCE": "v2", "DOJO_PERCEPTION_CFG": '{"series_count":"seg","glyph_read":"on"}'}
    assert {k: v["env"] for k, v in d["side_servers"].items()} == {
        "seg": ["DOJO_SEG_ADDR"], "glyph": ["DOJO_GLYPH_ADDR", "DOJO_GLYPH_ADDR_DIGIT"], "glyphk": ["DOJO_GLYPH_ADDR_KONG"]}
    assert all(f["md5"] for v in d["side_servers"].values() for f in v["files"].values())
    rc = d["runtime_config"]
    assert rc["mode"] == "perception" and rc["verify"] == "bundle" and rc["ported_classify"] == "p2"
    r = subprocess.run(["bash", str(ROOT / "scripts/dojo/official_eval.sh"), "--inner", "--setenv", "OMNIROUTE_API_KEY=x",
                        "--dry-run"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "credentials never travel" in r.stderr


def test_candidate_parity_is_zero_and_catches_a_difference(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "dojo"))
    import candidate_parity as CP

    (tmp_path / "CANDIDATE.txt").write_text(CAND)
    models = _models(tmp_path)
    for k in ("LANE_BG_FOLDS", "DOJO_BG_FOLDS", "DOJO_COLOR_CALIB", "DOJO_CALIB_SET"):
        monkeypatch.delenv(k, raising=False)
    integ = CP.integ_side(CP.candidate_tokens(CAND), models)
    official = CP.official_side(tmp_path / "CANDIDATE.txt", models, ROOT / "scripts/dojo/official_eval.sh")
    assert set(CP.diff(integ, official)) == {"perception_env"}       # lane2.sh runs bg_folds_51_ilia, official the default
    monkeypatch.setenv("DOJO_BG_FOLDS", integ["perception_env"]["DOJO_BG_FOLDS"])
    official = CP.official_side(tmp_path / "CANDIDATE.txt", models, ROOT / "scripts/dojo/official_eval.sh")
    assert CP.diff(integ, official) == {}
    assert integ["policy"] == official["policy"] == "dojo.runtime.policy:RuntimePolicy"
    bad = json.loads(json.dumps(official))
    bad["runtime_config"]["kong_draw_read"] = "off"
    del bad["side_servers"]["glyphk"]
    bad["setenv"].pop("DOJO_PERCEPTION_CFG")
    d = CP.diff(integ, bad)
    assert set(d) == {"runtime_config", "side_servers", "setenv"} and d["runtime_config"] == {"kong_draw_read": ["place", "off"]}


def test_freeze_manifest_records_names_and_md5s_never_keys(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "dojo"))
    import freeze_manifest as FM

    (tmp_path / "CANDIDATE.txt").write_text(CAND)
    for k in [k for k in os.environ if k.startswith("OMNIROUTE_")] + ["LANE_BG_FOLDS", "DOJO_CALIB_SET", "DOJO_COLOR_CALIB"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DOJO_BG_FOLDS", str(Path.home() / "dojo-data/bg_folds_51_ilia"))
    monkeypatch.setenv("AGENTIC_API_KEY_FAKE", "sk-SYNTHETIC-never-recorded")
    out = tmp_path / "m.json"
    models = _models(tmp_path)
    rc = FM.main(["--candidate", str(tmp_path / "CANDIDATE.txt"), "--models", str(models), "--allow-mismatch",
                  "--out", str(out)])
    text = out.read_text()
    m = json.loads(text)
    assert rc == 0 and m["parity"]["zero_diff"] and m["candidate"]["text"] == CAND
    assert m["launcher"]["policy"] == "dojo.runtime.policy:RuntimePolicy" and m["side_servers"]["seg"]["files"]["best.pt"]["sha256"]
    assert m["routes"]["agentic_model"] and "ee_joint_state_official" in m["grip"]
    assert "color_calib_sha256" in m["perception"]                    # (None here: no catalogue on the CI box)
    assert "SYNTHETIC" not in text
    assert m["launcher"]["lock_slots"] is None and "--lock-slots" not in m["launcher"]["line"]
    rc = FM.main(["--candidate", str(tmp_path / "CANDIDATE.txt"), "--models", str(models), "--allow-mismatch",
                  "--lock-slots", "/c/dc19.lock.1 /c/dc19.lock.2", "--out", str(out)])
    m = json.loads(out.read_text())
    assert rc == 0 and m["launcher"]["lock_slots"] == ["/c/dc19.lock.1", "/c/dc19.lock.2"]
    assert '--sealed-block <DIR> --manifest <MANIFEST> --lock-slots "/c/dc19.lock.1 /c/dc19.lock.2"' in m["launcher"]["line"]
    # robot-harness's server env in a sealed call: unattended, never waits for an operator (DR3)
    assert m["launcher"]["server_env_sealed"] == {"J1_HELD_FALLBACK": "no", "J1_PERCEIVER_CPU_ONLY": "1", "J1_NO_OPERATOR": "1"}


def test_raw_ok_is_off_by_default_and_never_mixes(tmp_path):
    """--raw-ok (dev only): a dev id with only a raw record runs from it; without the flag it is refused as before."""
    d = tmp_path / "data" / "simlab" / "layouts" / "make_toast"
    d.mkdir(parents=True)
    (d / "100000.json").write_text("{}")
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("OMNIROUTE_")}, "DOJO_DATA": str(tmp_path / "data"),
           "DOJO_PERCEPTION_DATA": str(tmp_path / "pd")}
    sh = str(ROOT / "scripts" / "dojo" / "official_eval.sh")
    r = subprocess.run(["bash", sh, "--inner", "--task", "make_toast", "--layouts", "100000", "--id", "t1"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no settled dev layout" in r.stderr
    r = subprocess.run(["bash", sh, "--inner", "--task", "make_toast", "--layouts", "100000", "--id", "t2", "--raw-ok"],
                       capture_output=True, text=True, env=env)
    assert "--raw-ok: raw dev records" in r.stderr                       # staged raw (it stops later: no calibration set)
    assert (tmp_path / "data" / "official" / "runs" / "t2" / "layouts" / "make_toast" / "100000.json").is_file()
    (d / "100001.settled.json").write_text("{}")
    r = subprocess.run(["bash", sh, "--inner", "--task", "make_toast", "--layouts", "100000 100001", "--id", "t3",
                        "--raw-ok"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "all settled or all raw" in r.stderr
