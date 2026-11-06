"""A stopped lane can initialize either box profile without consuming its queue."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("compute6", [False, True])
def test_stopped_lane_profile(tmp_path, compute6):
    repo = tmp_path / "repo"
    gate = repo / "scripts/dojo/catalog_gate.sh"
    gate.parent.mkdir(parents=True)
    gate.write_text("echo catalog_md5 test\n")
    data = tmp_path / "data"
    bg = tmp_path / "dojo-data/bg_folds_51_ilia/fold0/scene"
    bg.mkdir(parents=True)
    (tmp_path / "dojo-env.sh").write_text(f'export DOJO_DATA="{data}"\n')
    state = tmp_path / ("c6-state" if compute6 else "dojo-integ-c4-state")
    state.mkdir()
    (state / "STOP").touch()
    (state / "queue2.txt").write_text("sentinel\n")
    lock = tmp_path / ("dojo-gpu.lock" if compute6 else "dojo-integ-c4-heavy1.lock")
    lock.touch()
    env = {**os.environ, "HOME": str(tmp_path), "LANE_REPO": str(repo)}
    if compute6:
        env.update(LANE_STREAM="integ-c6", LANE_STATE=str(state), LANE_RUNS=str(tmp_path / "runs"), LANE_DATA=str(data))
    script = Path(__file__).parents[2] / "scripts/dojo/queues/lane2.sh"
    r = subprocess.run(["bash", str(script), "heavy1", str(lock)], env=env, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert "STOP file: exiting" in r.stdout
    assert (state / "queue2.txt").read_text() == "sentinel\n"
    assert not (tmp_path / "coord").exists()
