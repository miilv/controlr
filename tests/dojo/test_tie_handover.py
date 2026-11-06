"""Migration keeps an entire pair on compute4 when one arm has left the queue."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("started", [False, True])
def test_handover_preserves_started_pair(tmp_path, started):
    state = tmp_path / "dojo-integ-c4-state"
    wave = tmp_path / "dojo-data/wave-1106"
    state.mkdir()
    wave.mkdir(parents=True)
    names = [f"c4b_imitatesg_tie{arm}r1_{i}" for i in range(54) for arm in "AB"]
    (wave / "tie-queue.txt").write_text("\n".join(names) + "\n")
    queue = ["unrelated args"] + names[:]
    if started:
        queue.remove(names[2])  # second pair has started; its B arm must stay on compute4
    (state / "queue2.txt").write_text("\n".join(queue) + "\n")
    script = Path(__file__).parents[2] / "scripts/dojo/queues/tie_handover.py"
    r = subprocess.run([os.sys.executable, str(script)], env={**os.environ, "HOME": str(tmp_path)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (state / "queue2.txt").read_text() == "unrelated args\n" + (names[3] + "\n" if started else "")
    archive = (wave / "tie-moved-to-c6.txt").read_text().splitlines()
    assert len(archive) == (106 if started else 108)
    assert names[3] not in archive if started else names[3] in archive
