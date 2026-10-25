"""Twin pure helpers: which object is held."""

from collections import deque

import numpy as np
import pytest

from dojo.simlab.twin import held_objects


def frames(tcp_path, objs_path):
    h = deque(maxlen=10)
    for tcp, objs in zip(tcp_path, objs_path):
        h.append(({"left": np.array(tcp), "right": np.array([0.3, -0.2, 0.9])},
                  {k: np.array(v) for k, v in objs.items()}))
    return h


def test_object_moving_with_tcp_is_held():
    tcp = [[0, 0, 0.80 + 0.01 * i] for i in range(10)]
    objs = [{"block_1": [0, 0, 0.79 + 0.01 * i], "block_0": [0.05, 0, 0.78]} for i in range(10)]
    out = held_objects(frames(tcp, objs), {"left": 0.0, "right": 1.0}, {"left": True, "right": False})
    assert out == {"left": "block_1", "right": None}


def test_object_left_behind_is_not_held():
    tcp = [[0, 0, 0.80 + 0.01 * i] for i in range(10)]
    objs = [{"block_1": [0, 0, 0.79]} for _ in range(10)]          # did not follow the TCP
    assert held_objects(frames(tcp, objs), {"left": 0.0, "right": 1.0}, {})["left"] is None


def test_static_tcp_uses_closest_object_when_closed():
    tcp = [[0, 0, 0.80]] * 10
    objs = [{"a": [0.0, 0.0, 0.79], "b": [0.03, 0.0, 0.79]}] * 10
    assert held_objects(frames(tcp, objs), {"left": 0.2, "right": 1.0}, {})["left"] == "a"


def test_open_gripper_holds_nothing():
    tcp = [[0, 0, 0.80 + 0.01 * i] for i in range(10)]
    objs = [{"a": [0, 0, 0.79 + 0.01 * i]} for i in range(10)]
    assert held_objects(frames(tcp, objs), {"left": 1.0, "right": 1.0}, {"left": False})["left"] is None


def test_required_labels_and_incomplete_world(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    from dojo.simlab import gen_layouts as G
    from dojo.simlab import paths
    from dojo.simlab.twin import missing_labels
    (tmp_path / "toy.yml").write_text("""
Rigid:
  - common: {xlim: [0, 0], ylim: [0, 0]}
    category: [{name: block, index: [1, 2, 3]}]
    select_mode: {nums: [3], mode: unique, label: [block_0, block_1, block_2]}
  - common: {xlim: [0, 0], ylim: [0, 0]}
    category: [{name: toy, index: [1]}, {name: car, index: [2]}]
    select_mode: {mode: hierarchical, select_category_nums: [1, 2], select_instance_nums: [1, 1], label: [t0, t1]}
  - common: {xlim: [0, 0], ylim: [0, 0]}
    category: [{name: number}]
    select_mode: {mode: hierarchical, label_prefix: [digit], select_instance_nums: [2, 4]}
Geometry:
  - common: {xlim: [0, 0], ylim: [0, 0]}
    category: [{name: basket, index: [5]}]
    select_mode: {nums: [1], mode: allow_duplicate, label: [basket0]}
""")
    monkeypatch.setattr(paths, "TASK_CONFIGS", tmp_path)
    req = G.required_labels("toy")
    assert req == {"block_0", "block_1", "block_2", "t0", "basket0"}   # t1 optional, digits prefix-only
    lay = {"Rigid": {"block": [{"label": "block_0"}, {"label": "block_1"}], "toy": [{"label": "t0"}]},
           "Geometry": {"basket": [{"label": "basket0"}]}}
    assert missing_labels(lay, req) == ["block_2"]                    # perception missed one block
