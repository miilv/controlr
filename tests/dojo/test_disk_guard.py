"""scripts/dojo/disk_guard.sh: a job never starts below DOJO_MIN_FREE_GB free on the $HOME / $DOJO_DATA filesystem."""
import os
import subprocess
from pathlib import Path

GUARD = Path(__file__).resolve().parents[2] / "scripts" / "dojo" / "disk_guard.sh"


def run(need: str | None, tmp_path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(tmp_path), "DOJO_DATA": str(tmp_path)}
    env.pop("DOJO_MIN_FREE_GB", None)
    if need is not None:
        env["DOJO_MIN_FREE_GB"] = need
    return subprocess.run(["bash", "-c", f'source "{GUARD}"; disk_guard "[t]"'], env=env, capture_output=True, text=True)


def test_refuses_below_the_floor_and_says_so(tmp_path):
    r = run("100000000", tmp_path)            # no test box has 100 PB free
    assert r.returncode == 1 and "DISK GUARD" in r.stderr and "not starting" in r.stderr


def test_passes_above_the_floor_and_when_disabled(tmp_path):
    assert run("1", tmp_path).returncode == 0
    assert run("0", tmp_path).returncode == 0


def test_runners_call_it_before_any_work():
    integ = (GUARD.parent / "integ_eval.sh").read_text()
    rdpy = (GUARD.parents[1] / "simlab" / "rdpy.sh").read_text()
    assert 'disk_guard "[integ]" || exit 6' in integ
    assert integ.index("disk_guard \"[integ]\"") < integ.index("wait_res || {")
    assert 'disk_guard "[rdpy]" || exit 6' in rdpy and rdpy.index("disk_guard") < rdpy.index("flock -n 9")
