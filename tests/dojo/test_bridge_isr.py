"""isr-lab helpers (scripts/dojo/isr_prune.py, isr_summary.py, isr_job.sh): CPU only, temp dirs."""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dojo"))

import isr_prune  # noqa: E402
import isr_summary  # noqa: E402


def test_prune_keeps_first_last_every_kth_and_perception_steps(tmp_path):
    bench = tmp_path / "bench"
    rd = bench / "run" / "x"
    obs = rd / "obs"
    obs.mkdir(parents=True)
    for s in range(0, 100, 2):
        for c in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            (obs / f"s{s:04d}_{c}.png").write_bytes(b"x" * 10)
    runs = tmp_path / "runs"
    (runs / "x").mkdir(parents=True)
    with gzip.open(runs / "x" / "report.json.gz", "wt") as f:
        json.dump({"events": [{"kind": "look", "step": 34}, {"nested": {"steps_used": 77}}]}, f)
    isr_prune.main(["x", "--bench", str(bench), "--runs", str(runs)])
    assert len(list(obs.iterdir())) == 50 * 3                  # not closed: untouched
    (rd / "launch_done.json").write_text("{}")
    isr_prune.main(["x", "--bench", str(bench), "--runs", str(runs), "--every", "20"])
    left = sorted(int(p.name[1:5]) for p in obs.iterdir() if p.name.endswith("cam_head.png"))
    assert left == [0, 34, 40, 80, 98]
    assert len(list(obs.iterdir())) == 5 * 3


def test_prune_refuses_mikhails_bench():
    with pytest.raises(SystemExit):
        isr_prune.main(["x", "--bench", "/home/anywherevla/data2/bench"])


def test_summary_perception_errors_match_by_category_and_flag_gross():
    pytest.importorskip("scipy")                                # the perception extra (CI installs dev only)
    hello = {"objects": [{"type": "Rigid", "inst": "block_17_1", "label": "block_0", "category": "block"},
                         {"type": "Rigid", "inst": "block_14_2", "label": "block_1", "category": "block"},
                         {"type": "Geometry", "inst": "camera_stand_0_4", "label": "camera_stand", "category": "camera_stand"},
                         {"type": "Rigid", "inst": "apple_3_9", "category": "apple"}],          # clutter: unlabelled
             "objects_pos": {"block_17_1": [0.0, 0.0, 0.78], "block_14_2": [0.2, 0.0, 0.78],
                             "camera_stand_0_4": [0.0, -0.47, 0.765], "apple_3_9": [0.1, 0.1, 0.78]}}
    rec = {"Rigid": {"block": [
        {"category": "block", "category_idx": 14, "label": "block_1", "default_pos": [0.201, 0.0, 0.78]},
        {"category": "block", "category_idx": 12, "label": "block_0", "default_pos": [0.0, 0.04, 0.78]}]}}
    truth = isr_summary.truth_objects(hello, {"block"})
    assert [t["label"] for t in truth] == ["block_0", "block_1"]     # the stand is not estimated: not scored
    e = isr_summary.perception_errors(rec, truth)
    by = {o["label"]: o for o in e["objects"]}
    assert by["block_1"]["gross"] is False and by["block_1"]["err_mm"] == pytest.approx(1.0, abs=0.1)
    assert by["block_0"]["gross"] is True and by["block_0"]["identity_ok"] is False     # 40 mm off, asset 12 vs 17
    assert e["gross"] == 1 and e["n_true"] == 2


def _fake_coord(tmp_path):
    src = (ROOT / "scripts" / "dojo" / "isr_job.sh").read_text()
    src = src.replace("COORD=/home/anywherevla/data2/coord", f"COORD={tmp_path}/coord").replace(
        "CLOG=/home/anywherevla/data2/bench/COORDINATION.md RUN=/home/anywherevla/data2/ilia/run",
        f"CLOG={tmp_path}/C.md RUN={tmp_path}/run")
    (tmp_path / "coord" / "inbox" / "ilia" / "done").mkdir(parents=True)
    (tmp_path / "coord" / "gpu.lock").touch()
    (tmp_path / "coord" / "gpu.owner").touch()
    j = tmp_path / "job.sh"
    j.write_text(src)
    return j


def test_isr_job_owner_log_and_gates(tmp_path):
    j = _fake_coord(tmp_path)
    env = dict(os.environ, ISR_YIELD_S="0")
    r = subprocess.run(["bash", str(j), "t1", "5", "bash", "-c", f"cat {tmp_path}/coord/gpu.owner; exit 3"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "who: ilia bridge-isr" in r.stdout and "what: t1" in r.stdout
    assert (tmp_path / "coord" / "gpu.owner").read_text() == ""
    log = (tmp_path / "C.md").read_text()
    assert "START t1" in log and "DONE t1 (rc 3)" in log
    (tmp_path / "coord" / "inbox" / "ilia" / "m.md").write_text("x")
    assert subprocess.run(["bash", str(j), "t2", "5", "true"], env=env, capture_output=True).returncode == 76
    (tmp_path / "coord" / "inbox" / "ilia" / "m.md").unlink()
    (tmp_path / "coord" / "gpu.reserve").write_text("until: 2999-01-01T00:00Z\n")
    assert subprocess.run(["bash", str(j), "t3", "5", "true"], env=env, capture_output=True).returncode == 76
    assert "START t2" not in (tmp_path / "C.md").read_text()


def test_perception_extra_carries_robodojo_generator_deps():
    """The bridge runs the policy in OUR venv; the twin-layout template (dojo.runtime.estimate._template ->
    gen_layouts -> RoboDojo's utils.cluttered_generator) imports tqdm and transforms3d at module level. Without them
    every twin verify of a bridge episode fails (isr-lab 2026-10-26: ModuleNotFoundError, verify disabled)."""
    try:
        import tomllib
    except ModuleNotFoundError:                                   # python 3.10
        tomllib = pytest.importorskip("tomli")
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["perception"]
    names = {d.split(">")[0].split("=")[0].split("<")[0].strip().lower() for d in extra}
    assert {"tqdm", "transforms3d", "shapely", "rtree", "scipy"} <= names


def test_twin_requests_are_numpy_free_across_interpreters():
    """Behind the bridge the twin client runs in our venv (numpy 2.x) and the service in RoboDojo's python (numpy
    1.26): a request must unpickle without numpy (isr-lab 2026-10-26: "No module named 'numpy._core'")."""
    import pickle
    import pickletools

    import numpy as np

    from dojo.simlab.twin_service import plain

    req = {"op": "verify", "world": {"layout": {"Rigid": {"block": [{"default_pos": np.array([0.1, 0.2, 0.78]),
                                                                       "scale": (np.float64(1.0),) * 3}]}},
                                     "robot": {"left_q": np.zeros(6, np.float32), "left_grip": np.float32(1.0)}},
           "candidates": [{"actions": [{"left_arm_joint_state": np.ones(6)}], "meta": {"i": np.int64(0)}}],
           "k": np.int32(2)}
    out = plain(req)
    blob = pickle.dumps(out)
    assert b"numpy" not in blob and "numpy" not in str(list(pickletools.genops(blob)))
    assert out["world"]["robot"]["left_q"] == [0.0] * 6 and out["k"] == 2 and isinstance(out["k"], int)
    assert out["world"]["layout"]["Rigid"]["block"][0]["default_pos"] == [0.1, 0.2, 0.78]
