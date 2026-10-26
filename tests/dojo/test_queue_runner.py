"""The integ queue runner survives its own script being rewritten in place while it runs (a deploy into the runner's
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
