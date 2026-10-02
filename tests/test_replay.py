"""ReplayRobot: frames from a run directory or a plain image directory."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from controlr.config import Config
from controlr.robot import make_robot
from controlr.robot.replay import ReplayRobot, frames_from_run
from controlr.types import Action, ActionMode


def _img(path, value):
    Image.fromarray(np.full((48, 64, 3), value, np.uint8)).save(path)


def _fake_run(tmp_path):
    run = tmp_path / "run"
    (run / "images").mkdir(parents=True)
    shas = ["aaa", "bbb", "ccc"]
    for i, s in enumerate(shas):
        _img(run / "images" / f"{s}.jpg", 40 + 80 * i)
    _img(run / "images" / "ddd.jpg", 7)          # a derived (diff) image in turn 1
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "TASK"},
                                         {"type": "image", "image_sha": "aaa", "label": "scene"}]},
            {"role": "assistant", "content": "MOVE ee_delta 1 0 0\nSTATUS OK"},
            {"role": "user", "content": [{"type": "image", "image_sha": "bbb", "label": "scene"},
                                         {"type": "image", "image_sha": "ddd", "label": "diff"}]},
            {"role": "assistant", "content": "STATUS OK"},
            {"role": "user", "content": [{"type": "image", "image_sha": "ccc", "label": "scene"}]}]
    (run / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n{trunc")
    state = {"t": 1.0, "q": [0.1] * 6, "tcp_pos": [-0.3, -0.1, 0.12], "tcp_rotvec": [3.14, 0, 0],
             "gripper_mm": 40.0, "gripper_closed": True, "holding": None}
    (run / "turns.jsonl").write_text(json.dumps({"turn": 0, "state": state}) + "\n")
    (run / "setup.json").write_text(json.dumps({"instruction": "pick the cube", "state0": None}))
    return run


def test_run_dir_frames_in_turn_order(tmp_path):
    run = _fake_run(tmp_path)
    assert [f.stem for f in frames_from_run(run)] == ["aaa", "bbb", "ccc"]
    assert [f.stem for f in frames_from_run(run, label="diff")] == ["aaa", "ddd", "ccc"]
    r = ReplayRobot({"dir": str(run)})
    assert r.n_frames == 3 and r.task_instruction == "pick the cube"
    obs = r.reset({"name": "x"})
    means = [int(obs.images["scene"].mean())]
    assert obs.cameras["scene"].width == 64
    assert np.allclose(obs.state.tcp_pos, r.kin.fk(r.spec.home_q)[0])      # state0 unknown -> home
    for _ in range(3):
        rep = r.execute([Action(ActionMode.EE_DELTA, (0.1, 0, 0))])
        assert rep.executed == rep.requested and rep.events[0].kind == "replay"
        means.append(int(r.observe().images["scene"].mean()))
    assert means[0] < means[1] < means[2] == means[3]                       # held at the end
    r.reset({})
    r.execute([])
    assert r.state().gripper_closed and np.allclose(r.state().tcp_pos, [-0.3, -0.1, 0.12])
    assert not r.check_goal().success


def test_plain_dir_natural_order_and_loop(tmp_path):
    for i, v in [(10, 200), (2, 100), (1, 50)]:
        _img(tmp_path / f"frame_{i}.png", v)
    (tmp_path / "notes.txt").write_text("ignore me")
    r = ReplayRobot({"dir": str(tmp_path), "loop": True, "camera": "cam0"})
    assert [f.name for f in r.frames] == ["frame_1.png", "frame_2.png", "frame_10.png"]
    seen = [int(r.reset({}).images["cam0"].mean())]
    for _ in range(3):
        r.execute([])
        seen.append(int(r.observe().images["cam0"].mean()))
    assert seen == [50, 100, 200, 50]


def test_factory_and_errors(tmp_path):
    _img(tmp_path / "a.jpg", 1)
    cfg = Config()
    cfg.robot.backend = "replay"
    cfg.robot.params = {"dir": str(tmp_path)}
    assert isinstance(make_robot(cfg), ReplayRobot)
    with pytest.raises(ValueError):
        ReplayRobot({})
    with pytest.raises(FileNotFoundError):
        ReplayRobot({"dir": str(tmp_path / "missing")})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ReplayRobot({"dir": str(empty)})
