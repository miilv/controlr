"""isr-lab helpers (scripts/dojo/isr_prune.py, isr_summary.py, isr_job.sh): CPU only, temp dirs."""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    env = dict(os.environ, ISR_YIELD_S="0", ISR_SKIP_PREFLIGHT="1")
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


def test_fleet_venv_pins_carry_every_perception_extra():
    """The GPU boxes' venvs are built from ``scripts/infra/pins/controlr-venv.txt`` (box_steps.sh ``venv``), not from
    the extra. That freeze lacked tqdm / transforms3d, so every bridge / official-path episode on compute5-8 ran with
    twin verify disabled (2026-10-27, compute7: "No module named 'tqdm'", none_ok on every segment)."""
    try:
        import tomllib
    except ModuleNotFoundError:                                   # python 3.10
        tomllib = pytest.importorskip("tomli")
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    norm = lambda d: d.split(">")[0].split("=")[0].split("<")[0].split("[")[0].strip().lower().replace("_", "-")  # noqa: E731
    want = {norm(d) for d in proj["dependencies"] + proj["optional-dependencies"]["perception"]}
    pins = (ROOT / "scripts" / "infra" / "pins" / "controlr-venv.txt").read_text().splitlines()
    have = {norm(ln) for ln in pins if ln.strip() and not ln.lstrip().startswith("#")}
    assert want <= have, f"missing from the fleet venv pins: {sorted(want - have)}"


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


def test_summary_identity_is_position_only_when_estimates_name_no_assets():
    pytest.importorskip("scipy")
    hello = {"objects": [{"type": "Rigid", "inst": f"mahjong_{k}_{i}", "label": f"mahjong{i}", "category": "mahjong"}
                         for i, k in enumerate((17, 30))],
             "objects_pos": {"mahjong_17_0": [0.0, 0.0, 0.78], "mahjong_30_1": [0.1, 0.0, 0.78]}}
    rec = {"Rigid": {"mahjong": [{"category": "mahjong", "category_idx": 0, "label": f"mahjong{i}",
                                  "default_pos": [0.1 * i, 0.0, 0.78]} for i in range(2)]}}
    e = isr_summary.perception_errors(rec, isr_summary.truth_objects(hello))
    assert e["gross"] == 0


def test_runtime_eval_isaac51_uses_the_rendered_episode_and_no_render_service(tmp_path, monkeypatch):
    """``dojo.runtime.eval --renderer isaac51`` (isr-lab's rendered 5.1 dev eval): episodes come from
    ``RenderedEpisode`` (one 5.1 EvalEnv with RTX); the 6.0 render client is never built; rows and the summary say
    which renderer ran."""
    import types

    import dojo.runtime.eval as EV
    from dojo.simlab.proxy_eval import RENDERED_TAG

    made = []

    class FakeRun:
        def objects_gt(self, i):
            return []

        def reward(self, i):
            return {"success": True, "ended": True, "steps_used": 3, "step_lim": 10}

    class FakeRendered:
        def __init__(self, task, workdir):
            made.append((task, Path(workdir)))
            self.run = FakeRun()
            self.run.close = lambda: made.append("closed")

        def reset(self, layout):
            return {"stable": True, "task": "stack_blocks", "layout_id": 100000, "instruction": "x", "step_lim": 10,
                    "reset_s": 0.0, "render_stage_s": 0.0}

        def observe(self):
            return {"images": {}, "state": {}, "instruction": "x", "steps_used": 0, "step_lim": 10,
                    "renderer": "isaac-5.1"}

        def step(self, a):
            return False

        def result(self):
            return {"success": True, "score": 1.0, "steps_used": 3, "step_lim": 10, "observations": 1}

    class Pol:
        def __init__(self, cfg):
            self.bridge = types.SimpleNamespace(executed=[])

        def reset(self, info):
            pass

        def next(self, obs):
            return {"done": True}

        def episode_report(self):
            return {"timings": {}, "looks": [], "events": []}

    def no_render(*a, **k):
        raise AssertionError("the 6.0 render client must not be built for isaac51")

    monkeypatch.setattr(EV, "RenderedEpisode", FakeRendered)
    monkeypatch.setattr(EV, "RenderClient", no_render)
    lay = tmp_path / "100000.settled.json"
    lay.write_text(json.dumps({"_simlab": {"id": 100000}}))
    cfg = EV.RuntimeConfig(mode="gt", verify="off")
    summ = EV.run("stack_blocks", [str(lay)], cfg, tmp_path / "out", policy_factory=Pol, renderer="isaac51")
    assert made and made[0][0] == "stack_blocks" and made[-1] == "closed"   # the RTX app is closed by run()
    assert summ["renderer"] == "isaac51" and summ["proxy"] == RENDERED_TAG and summ["success"] == 1
    row = json.loads((tmp_path / "out" / "stack_blocks_gt_verify-off_episodes.jsonl").read_text().splitlines()[0])
    assert row["renderer"] == "isaac51"


def test_runtime_eval_gate_skips_the_remaining_layouts(tmp_path, monkeypatch):
    import dojo.runtime.eval as EV

    calls = []

    class Ep:
        def __init__(self, task, workdir):
            self.run = type("R", (), {"objects_gt": lambda s, i: [], "reward": lambda s, i: {"success": True},
                                      "close": lambda s: None})()

        def reset(self, layout):
            calls.append(layout["_simlab"]["id"])
            return {"stable": True, "layout_id": layout["_simlab"]["id"], "instruction": "x", "step_lim": 10,
                    "reset_s": 0.0}

        def observe(self):
            return {}

        def step(self, a):
            return False

        def result(self):
            return {"success": True, "score": 1.0, "steps_used": 1, "step_lim": 10, "observations": 1}

    class Pol:
        def __init__(self, cfg):
            self.bridge = type("B", (), {"executed": []})()

        def reset(self, info):
            pass

        def next(self, obs):
            return {"done": True}

        def episode_report(self):
            return {"timings": {}}

    monkeypatch.setattr(EV, "RenderedEpisode", Ep)
    flag = tmp_path / "stop"
    monkeypatch.setenv("DOJO_EVAL_GATE", f"test ! -e {flag} || {{ echo disk low; exit 77; }}")
    lays = []
    for i in range(3):
        p = tmp_path / f"{100000 + i}.json"
        p.write_text(json.dumps({"_simlab": {"id": 100000 + i}}))
        lays.append(str(p))

    class Gated(Pol):
        def episode_report(self):
            flag.touch()                                   # the gate refuses from the second episode on
            return {"timings": {}}

    summ = EV.run("stack_blocks", lays, EV.RuntimeConfig(mode="gt", verify="off"), tmp_path / "o",
                  policy_factory=Gated, renderer="isaac51")
    assert calls == [100000] and summ["n"] == 1 and summ["skipped"] == 2


def test_isr_job_preflight_refuses_without_or_against_the_pin(tmp_path):
    j = _fake_coord(tmp_path)
    env = dict(os.environ, ISR_YIELD_S="0", ISR_BENCH_PIN=str(tmp_path / "missing_pin.txt"))
    r = subprocess.run(["bash", str(j), "t1", "5", "true"], env=env, capture_output=True, text=True)
    assert r.returncode == 78
    assert not (tmp_path / "C.md").exists() or "START t1" not in (tmp_path / "C.md").read_text()


def test_isr_job_watchdog_stops_a_runaway_tree(tmp_path):
    j = _fake_coord(tmp_path)
    env = dict(os.environ, ISR_YIELD_S="0", ISR_SKIP_PREFLIGHT="1", ISR_CPU_MAX="-1", ISR_WD_HOLD_S="1",
               ISR_WD_POLL_S="0.5", DOJO_ILIA="1")
    r = subprocess.run(["bash", str(j), "wd", "5", "bash", "-c", "sleep 60 & wait"], env=env, capture_output=True,
                       text=True, timeout=40)
    assert r.returncode != 0
    assert "over the limit" in (tmp_path / "run" / "wd.watchdog").read_text()
    assert "WATCHDOG wd" in (tmp_path / "C.md").read_text() and "DONE wd" in (tmp_path / "C.md").read_text()


def test_frames_plan_takes_negatives_first_then_equal_positives_within_budget(tmp_path):
    import frames_plan
    d = tmp_path / "ev"
    d.mkdir()
    ev = [{"kind": "held_look", "label": "block_0", "arm": "left", "step": s, "held": s % 3 != 0} for s in range(1, 31)]
    ev += [{"kind": "post_place", "label": "block_0", "arm": "left", "step": 100 + s, "ok": False} for s in range(4)]
    with gzip.open(d / "stack_blocks_perception_verify-bundle_100000_report.json.gz", "wt") as f:
        json.dump({"events": ev}, f)
    out = tmp_path / "plan.json"
    frames_plan.main(["plan", str(d), "--budget-mb", "18", "--out", str(out)])   # 20 sets at 0.9 MB
    p = json.loads(out.read_text())
    assert p["n_negatives_total"] == 14 and p["n_chosen_negatives"] == 14 and p["n_chosen_positives"] == 6
    assert all(c["negative"] for c in p["checks"][:14]) and len(p["episodes"][0]["steps"]) == 20


def test_runtime_eval_actions_file_holds_what_the_env_ran(tmp_path, monkeypatch):
    """``<tag>_<lid>_actions.json.gz`` = every action the env stepped, including the batch RoboDojo's latch ended (the
    policy's own ``Bridge.executed`` never gets that batch: the worker waits for an observation that never comes). The
    kit replays (scripts/dojo/replay_kits.py) replay this file."""
    import types

    import dojo.runtime.eval as EV

    stepped = []

    class FakeRun:
        def objects_gt(self, i):
            return []

        def reward(self, i):
            return {"success": True, "ended": True, "steps_used": len(stepped), "step_lim": 10}

        def close(self):
            pass

    class FakeRendered:
        def __init__(self, task, workdir):
            self.run = FakeRun()

        def reset(self, layout):
            return {"stable": True, "task": "stack_blocks", "layout_id": 100000, "instruction": "x", "step_lim": 10,
                    "reset_s": 0.0, "render_stage_s": 0.0}

        def observe(self):
            return {"images": {}, "state": {}, "instruction": "x", "steps_used": len(stepped), "step_lim": 10}

        def step(self, a):
            stepped.append(a)
            return len(stepped) == 5                       # the success latch fires on the 5th step

        def result(self):
            return {"success": True, "score": 1.0, "steps_used": len(stepped), "step_lim": 10, "observations": 2}

    act = lambda x: {"left_arm_joint_state": np.full(6, x, np.float32), "left_ee_joint_state": np.ones(1, np.float32),  # noqa: E731
                     "right_arm_joint_state": np.zeros(6, np.float32), "right_ee_joint_state": np.ones(1, np.float32)}
    batches = [[act(0.1), act(0.2)], [act(0.3), act(0.4), act(0.5), act(0.6)]]

    class Pol:
        def __init__(self, cfg):
            self.bridge = types.SimpleNamespace(executed=[])
            self.k = 0

        def reset(self, info):
            pass

        def next(self, obs):
            if self.k:                                     # the runtime's Bridge records a batch at the NEXT observation
                self.bridge.executed.extend(batches[self.k - 1])
            if self.k >= len(batches):
                return {"done": True}
            self.k += 1
            return {"actions": batches[self.k - 1]}

        def episode_report(self):
            return {"timings": {}, "looks": [], "events": []}

    monkeypatch.setattr(EV, "RenderedEpisode", FakeRendered)
    lay = tmp_path / "100000.settled.json"
    lay.write_text(json.dumps({"_simlab": {"id": 100000}}))
    EV.run("stack_blocks", [str(lay)], EV.RuntimeConfig(mode="gt", verify="off"), tmp_path / "out", policy_factory=Pol,
           renderer="isaac51")
    with gzip.open(tmp_path / "out" / "stack_blocks_gt_verify-off_100000_actions.json.gz", "rt") as f:
        got = json.load(f)["actions"]
    assert len(stepped) == 5 and len(got) == 5           # the latch-ended batch is in the file, the unrun action not
    assert [round(a["left_arm_joint_state"][0], 3) for a in got] == [0.1, 0.2, 0.3, 0.4, 0.5]
