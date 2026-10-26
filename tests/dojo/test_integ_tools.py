"""Integration tooling: the queue runner survives its own script being rewritten in place while it runs (a deploy into the runner's
checkout; 2026-10-26 the runner re-ran a finished job with mangled args and died on a syntax error)."""
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
