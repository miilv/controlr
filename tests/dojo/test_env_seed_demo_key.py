"""The env seed vs our layout key (SIMLAB_ENV_SEED; dev_eval's upstream seeding) and imitate's routed demo loader.

RoboDojo keys imitate_sorting_sequence's demo file by the ENV seed. Upstream SeedManager (and dev_eval) seeds with
the layout's position in the call, so the routed loader must map that position back to our layout id
(the D-C19 rehearsal crashed on "seed 0 is not a dev seed"). Synthetic fixtures only.
"""

import json
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dojo.simlab import paths, runner

ROOT = Path(__file__).resolve().parents[2]
TASK = "imitate_sorting_sequence"
EVAL_DEMO = f"/rd/Assets/Traj/RoboDojo/{TASK}/0/0.pkl"     # what RoboDojo asks for: <eval_seed>/<env seed>.pkl


@pytest.fixture(autouse=True)
def _clean_maps():
    runner.DEMO_KEY.clear()
    runner.DEMO_FILE.clear()
    yield
    runner.DEMO_KEY.clear()
    runner.DEMO_FILE.clear()


def _demo(path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"joint_path": [], "target_label": [], "tag": tag}))


def _never(path):
    raise AssertionError(f"the eval-paired demo was opened: {path}")


def test_routed_loader_maps_the_env_seed_to_our_layout_id(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    _demo(tmp_path / "simlab" / "traj" / TASK / "100001.pkl", "dev")
    load = runner._routed_load_pkl(_never)
    with pytest.raises(RuntimeError, match="seed 0 is not a dev seed"):
        load(EVAL_DEMO)                                   # unmapped: refused, as before
    runner.DEMO_KEY[0] = 100001
    assert load(EVAL_DEMO)["tag"] == "dev"
    staged = tmp_path / "staged" / "100001.demo.pkl"
    _demo(staged, "staged")
    runner.DEMO_FILE[100001] = staged                     # a staged demo replaces DATA/simlab/traj
    assert load(EVAL_DEMO)["tag"] == "staged"
    runner.DEMO_FILE[100001] = tmp_path / "missing.pkl"   # and never falls back to the stub
    with pytest.raises(FileNotFoundError):
        load(EVAL_DEMO)


def test_env_seed_keys_default_is_the_layout_key(monkeypatch):
    monkeypatch.delenv("SIMLAB_ENV_SEED", raising=False)
    assert runner.env_seed_keys([100000, 100001]) == [100000, 100001] and runner.DEMO_KEY == {}
    monkeypatch.setenv("SIMLAB_ENV_SEED", "layout_id")
    assert runner.env_seed_keys([100003]) == [100003] and runner.DEMO_KEY == {}


def test_env_seed_index_seeds_like_upstream_and_maps_back(monkeypatch):
    monkeypatch.setenv("SIMLAB_ENV_SEED", "index")
    assert runner.env_seed_keys([100003, 100001]) == [0, 1]
    assert runner.DEMO_KEY == {0: 100003, 1: 100001}
    monkeypatch.setenv("SIMLAB_ENV_SEED", "layout_id")
    runner.env_seed_keys([100003])
    assert runner.DEMO_KEY == {}                          # a later layout_id load leaves no stale mapping
    monkeypatch.setenv("SIMLAB_ENV_SEED", "position")
    with pytest.raises(ValueError, match="SIMLAB_ENV_SEED"):
        runner.env_seed_keys([100003])


def test_dev_eval_maps_each_position_to_its_layout_id(tmp_path, monkeypatch):
    from dojo.simlab import dev_eval
    d = tmp_path / "layouts" / TASK
    d.mkdir(parents=True)
    for lid in (100001, 100003):
        (d / f"{lid}.settled.json").write_text("{}")

    class SeedManager:                                    # RoboDojo's class, as dev_eval.py patches it
        task_name, config = TASK, {}
    mods = {"env": types.ModuleType("env"), "env.seed_manager": types.ModuleType("env.seed_manager"),
            "env.seed_manager.seed_manager": types.ModuleType("env.seed_manager.seed_manager"),
            "utils": types.ModuleType("utils"), "utils.load_file": types.ModuleType("utils.load_file")}
    mods["env.seed_manager.seed_manager"].SeedManager = SeedManager
    mods["env.seed_manager"].seed_manager = mods["env.seed_manager.seed_manager"]
    mods["utils.load_file"].load_json = lambda p: json.loads(Path(p).read_text())
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setenv("SIMLAB_DEV_LAYOUT_SUFFIX", ".settled")
    monkeypatch.delenv("SIMLAB_SEALED_MANIFEST", raising=False)
    dev_eval.patch_seed_manager(tmp_path / "layouts")
    sm = SeedManager()
    sm.init_eval()
    assert sm.seed_list == [0, 1]                         # the env seeds stay upstream's positions
    assert runner.DEMO_KEY == {0: 100001, 1: 100003}


def test_official_eval_refuses_imitate_without_its_demo(tmp_path):
    d = tmp_path / "data" / "simlab" / "layouts" / TASK
    d.mkdir(parents=True)
    (d / "100001.settled.json").write_text("{}")
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("OMNIROUTE_")}, "DOJO_DATA": str(tmp_path / "data"),
           "DOJO_PERCEPTION_DATA": str(tmp_path / "pd")}
    sh = str(ROOT / "scripts" / "dojo" / "official_eval.sh")
    r = subprocess.run(["bash", sh, "--inner", "--task", TASK, "--layouts", "100001", "--id", "t1"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no demo" in r.stderr
    _demo(tmp_path / "data" / "simlab" / "traj" / TASK / "100001.pkl", "dev")
    r = subprocess.run(["bash", sh, "--inner", "--task", TASK, "--layouts", "100001", "--id", "t2"],
                       capture_output=True, text=True, env=env)
    assert "no demo" not in r.stderr and f"task {TASK}, 1 layouts: 100001" in r.stderr   # (stops later: no calib set)
