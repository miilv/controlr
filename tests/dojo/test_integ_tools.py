"""Integration tooling. The queue runner survives its own script being rewritten in place while it runs (a deploy
into the runner's checkout; 2026-10-26 the runner re-ran a finished job with mangled args and died on a syntax
error). dojo.runtime.report's breadth table and ledger rows."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs util-linux flock")
def test_runner_finishes_the_queue_when_its_script_is_rewritten_in_place(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/dojo").mkdir(parents=True)
    runner = repo / "scripts/dojo/queue_runner.sh"
    shutil.copy(ROOT / "scripts/dojo/queue_runner.sh", runner)
    # a fake eval: the first job rewrites the runner IN PLACE (same inode, other content), as cp / an editor would
    (repo / "scripts/dojo/integ_eval.sh").write_text(
        "#!/usr/bin/env bash\n"
        'echo "$@" >> "$DOJO_DATA/ran.txt"\n'
        'if [[ ! -f "$DOJO_DATA/rewritten" ]]; then touch "$DOJO_DATA/rewritten"\n'
        '  python3 -c "import sys; open(sys.argv[1], \'r+\').write(\'echo BROKEN; done; fi; ((\' * 400)" "$RUNNER"; fi\n')
    data = tmp_path / "data"
    (data / "integ").mkdir(parents=True)
    (data / "integ/queue.txt").write_text("job_a --task a\njob_b --task b\njob_c --task c\n")
    env = {**os.environ, "DOJO_DATA": str(data), "RUNNER": str(runner)}
    out = subprocess.run(["bash", str(runner)], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    ran = (data / "ran.txt").read_text().splitlines()
    assert [ln.split()[1].rsplit("/", 1)[-1] for ln in ran] == ["job_a", "job_b", "job_c"], out.stdout + out.stderr
    assert "queue empty" in out.stdout and "BROKEN" not in out.stdout and "syntax error" not in out.stderr


def test_breadth_table_and_ledger_rows():
    """dojo.runtime.report --breadth / --ledger: one row per task, failures by category with readable reasons."""
    import json

    from dojo.runtime.report import breadth_table, ledger_rows

    def row(task, lid, ok, cat=None, why=None, calls=3):
        return {"task": task, "layout_id": lid, "success": ok, "score": 1.0 if ok else 0.25, "steps_used": 100 + lid % 10,
                "step_lim": 400, "wall_s": 60.0, "category": cat, "why": why, "agentic": [{"calls": calls}],
                "preset": "submit_v1", "renderer": "isaac51", "layout": f"/x/{lid}.settled.json", "code": {"head": "abc1234"}}
    rows = [row("deposit_coin", 100000, True), row("deposit_coin", 100001, False, "perception_gross",
                                                    json.dumps({"coin0": {"found": False}})),
            row("deposit_coin", 100002, False, "plan", "lift ik_fail: no rim grasp"),
            row("hang_mugs", 100000, False, "perception_gross",
                json.dumps({"mug0": {"found": True, "est_label": "mug1", "xy_mm": 571.3}}), calls=5)]
    t = breadth_table(rows, {"deposit_coin": "G3 A 0.9"})
    assert "| deposit_coin | 3 | 1/3 | 50.0 |" in t and "coin0 not found" in t and "plan 1: 02; lift ik_fail" in t
    assert "mug0 read as mug1 571 mm" in t and "G3 A 0.9" in t and "4 episodes, 1 successes, 14 LLM calls" in t
    led = {r["task"]: r for r in ledger_rows(rows, "dev_5.1_rendered", "2026-10-27")}
    d = led["deposit_coin"]
    assert (d["n"], d["success"], d["score"], d["calls_per_ep"], d["tier"]) == (3, 1, 50.0, 3.0, "dev_5.1_rendered")
    assert d["layouts"] == "100000-100002 settled" and d["commit"] == "miilv/controlr@abc1234"


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs util-linux flock")
def test_runner_retries_a_job_whose_services_never_got_ready_once(tmp_path):
    """integ_eval.sh exits 3 when a service never got ready (no episode ran): the runner keeps that dir as
    NAME__services_not_ready and runs the job once more, first in the queue; a second 3 is not retried again."""
    repo = tmp_path / "repo"
    (repo / "scripts/dojo").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/dojo/queue_runner.sh", repo / "scripts/dojo/queue_runner.sh")
    (repo / "scripts/dojo/integ_eval.sh").write_text(
        "#!/usr/bin/env bash\n"
        'out=$2; mkdir -p "$out"; echo "$(basename "$out") $3 $4" >> "$DOJO_DATA/ran.txt"\n'
        '[[ "$(basename "$out")" == job_a* || "$(basename "$out")" == job_c* ]] && exit 3\nexit 0\n')
    data = tmp_path / "data"
    (data / "integ").mkdir(parents=True)
    (data / "integ/queue.txt").write_text("job_a --task a\njob_b --task b\njob_c --task c\n")
    out = subprocess.run(["bash", str(repo / "scripts/dojo/queue_runner.sh")], cwd=repo,
                         env={**os.environ, "DOJO_DATA": str(data)}, capture_output=True, text=True, timeout=60)
    ran = (data / "ran.txt").read_text().splitlines()
    assert ran == ["job_a --task a", "job_a_retry --task a", "job_b --task b", "job_c --task c",
                   "job_c_retry --task c"], out.stdout + out.stderr
    runs = data / "integ/runs"
    assert (runs / "job_a__services_not_ready").is_dir() and (runs / "job_a_retry").is_dir()
    assert not (runs / "job_c_retry__services_not_ready").exists() and "queue empty" in out.stdout


def test_perception_eval_scores_the_record_the_skills_planned_with():
    """breadth2 scored perception on the fused record, else the initial one: an object the scan / agentic / search record
    found (insert_tubes 100002, play_Xylophone 100000 / 01, deposit_coin 100001) read as perception_gross."""
    from dojo.runtime.estimate import planning_record
    from dojo.runtime.eval import perception_eval

    def ent(lab, x):
        return {"category": "tube", "category_idx": 0, "label": lab, "default_pos": [x, 0.0, 0.8], "default_ori": [1, 0, 0, 0]}
    initial = {"Rigid": {"tube": [ent("tube0", 0.1)]}, "_not_found": [{"label": "tube1"}]}
    agentic0 = {"Rigid": {"tube": [ent("tube0", 0.1), ent("tube1", 0.2)]}}
    gt = [{"label": "tube0", "type": "Rigid", "category": "tube", "category_idx": 0, "pos": [0.1, 0, 0.8], "quat": [1, 0, 0, 0]},
          {"label": "tube1", "type": "Rigid", "category": "tube", "category_idx": 0, "pos": [0.2, 0, 0.8], "quat": [1, 0, 0, 0]}]
    rep = {"records": {"initial": initial, "agentic0": agentic0}}
    assert planning_record(rep["records"])[0] == "agentic0"
    out = perception_eval(rep, gt, None)
    assert out["tube1"]["found"] and out["tube1"]["xy_mm"] < 1.0 and out["tube0"]["xy_mm_first"] < 1.0
    assert not perception_eval({"records": {"initial": initial}}, gt, None)["tube1"]["found"]
