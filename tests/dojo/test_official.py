"""The official path on our dev layouts (scripts/dojo/official_eval.sh, dojo.bridge.official): RoboDojo's result rows
mapped back to our layout ids, the pairing with integ_eval rows and the first action divergence, episodes back to back
through ONE strict server (the next reset ends the previous episode), and the RoboDojo farm. CPU only."""

import gzip
import json
import subprocess
from pathlib import Path

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
