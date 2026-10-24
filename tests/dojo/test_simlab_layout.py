"""simlab pure-Python parts: layout schema, perturbation, sampler helpers, motion math (no Isaac)."""

import math

import numpy as np
import pytest

from dojo.simlab import layout as L
from dojo.simlab import motion as M
from dojo.simlab import paths

SCENE = {
    "Room": {"default": "Simple_Room_nolight", "default_pos": [0, 0, 0], "default_ori": [1, 0, 0, 0],
             "scale": [0.5, 0.5, 0.5]},
    "Table": {"default": "material_0122", "scale": [1.4, 1.1, 0.05], "default_pos": [0.0, -0.05, 0.74],
              "default_ori": [1, 0, 0, 0]},
    "Background": {"intensity": 1000, "default": "brown_photostudio_02_4k.hdr"},
    "Ground": {"geometry": "cube", "thickness": 0.1, "axis": "Z", "default_pos": [0, 0, 0],
               "materials": {"type": "mdl", "default": "material_0564"}, "physics_material": {}},
}


def task_inst(label, pos, ptype="rigid"):
    return {"category": "block", "category_idx": 5, "group": None, "xlim": [-0.4, 0.4], "ylim": [-0.2, 0.0],
            "zlim": None, "qpos": [1, 0, 0, 0], "rotate_deg": 0, "rotate_rand": False, "relative_plane": "Table",
            "place_tag": None, "margin": 0.01, "check_mode": "bbox", "need_check_stable": True, "label": label,
            "default_pos": list(pos), "default_ori": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
            "physics": {"mass": 0.2, "friction": 0.0, "type": ptype}, "visual": {}}


def record():
    lay = {"Rigid": {"block": [task_inst("block_0", (0.1, -0.1, 0.78)), task_inst("block_1", (0.2, -0.1, 0.78))]}}
    lay["Rigid"]["pen"] = [{"category_idx": 0, "physics": {"type": "rigid"}, "scale": [1.0, 1.0, 1.0],
                            "type": "cluttered", "clutter_idx": 0, "yaml_path": "Clutter/clutter.yml",
                            "relative_plane": "Table", "default_pos": [0.5, 0.3, 0.77],
                            "default_ori": [1.0, 0.0, 0.0, 0.0]}]
    lay.update(L.fixtures_from_scene(SCENE))
    return lay


def test_fixtures_match_eval_schema_keys():
    f = L.fixtures_from_scene(SCENE)
    assert set(f["Room"]) == {"default", "default_pos", "default_rot", "scale"}
    assert set(f["Table"]) == {"default", "scale", "default_pos", "default_ori"}
    assert set(f["Background"]) == {"intensity", "category_name"}
    assert f["Ground"]["materials"]["random"] is False
    assert "random" not in SCENE["Ground"]["materials"]       # the scene config is not mutated


def test_valid_record_passes_and_errors_are_specific():
    lay = record()
    assert L.validate(lay) == []
    bad = record()
    bad["Rigid"]["block"][1]["label"] = "block_0"
    bad["Rigid"]["block"][0]["physics"]["type"] = "geometry"
    del bad["Rigid"]["pen"][0]["clutter_idx"]
    del bad["Table"]
    errs = L.validate(bad)
    assert any("duplicate label block_0" in e for e in errs)
    assert any("physics.type" in e for e in errs)
    assert any("clutter_idx" in e for e in errs)
    assert not any("Table" in e for e in errs)          # Table is optional (conveyor scene)
    del bad["Room"]
    assert any("Room" in e for e in L.validate(bad))


def test_object_table_and_clutter_flag():
    rows = L.object_table(record())
    assert [r["label"] for r in rows] == ["block_0", "block_1", None]
    assert [r["clutter"] for r in rows] == [False, False, True]


def test_perturb_moves_one_object_only_and_keeps_unit_quaternion():
    lay = record()
    p = L.perturb(lay, "block_1", dxyz=(0.002, 0.0, 0.0), dyaw_deg=2.0)
    a, b = L.find(lay, "block_1"), L.find(p, "block_1")
    assert b["default_pos"][0] == pytest.approx(a["default_pos"][0] + 0.002)
    assert L.quat_angle_deg(a["default_ori"], b["default_ori"]) == pytest.approx(2.0, abs=1e-6)
    assert L.yaw_deg(b["default_ori"]) == pytest.approx(2.0, abs=1e-6)
    assert np.linalg.norm(b["default_ori"]) == pytest.approx(1.0)
    assert L.find(p, "block_0") == L.find(lay, "block_0")
    assert L.find(lay, "block_1")["default_pos"] == a["default_pos"]          # original untouched


def test_quaternion_helpers():
    q = L.quat_mul(L.quat_z(30), L.quat_z(60))
    assert L.yaw_deg(q) == pytest.approx(90.0)
    assert L.quat_angle_deg([1, 0, 0, 0], L.quat_z(-170)) == pytest.approx(170.0)


def test_eval_guard_refuses_eval_layout_paths(tmp_path):
    with pytest.raises(PermissionError):
        paths.assert_not_eval(paths.EVAL_LAYOUT_DIR / "RoboDojo" / "arx_x5" / "0" / "x_0.json")
    with pytest.raises(PermissionError):
        paths.assert_not_eval(tmp_path / "Eval_Layout" / "a.json")
    paths.assert_not_eval(tmp_path / "layouts" / "a.json")


def test_tcp_offset_round_trip():
    q = L.quat_mul(L.quat_z(37), [math.cos(0.3), math.sin(0.3), 0, 0])
    fl = M.flange_for_tcp([0.1, -0.2, 0.9], q)
    assert M.tcp_of(fl) == pytest.approx([0.1, -0.2, 0.9])
    assert np.linalg.norm(fl[:3] - [0.1, -0.2, 0.9]) == pytest.approx(0.1501)


def test_resample_and_step_count_respect_step_limit():
    path = np.array([[0.0] * 6, [0.3] + [0.0] * 5, [0.2] + [0.0] * 5])     # doubles back: travel 0.4
    n = M.n_steps(path, 0.05)
    assert n == 8
    w = M.resample(path, n)
    assert w.shape == (8, 6) and w[-1] == pytest.approx(path[-1])


def test_trajectory_json_round_trip_is_bit_exact():
    rng = np.random.default_rng(0)
    q = rng.uniform(-3, 3, 6).astype(np.float32)
    t = M.Trajectory(actions=[{"left_arm_joint_state": q}])
    import json
    u = M.Trajectory.from_json(json.loads(json.dumps(t.to_json())))
    assert u.actions[0]["left_arm_joint_state"].tobytes() == q.tobytes()


def test_trajectory_json_round_trip():
    t = M.Trajectory(actions=[{"left_arm_joint_state": np.arange(6, dtype=np.float32),
                               "left_ee_joint_state": np.array([1.0], np.float32)}], segments=[{"name": "x"}])
    u = M.Trajectory.from_json(t.to_json())
    assert u.segments == [{"name": "x"}]
    assert u.actions[0]["left_arm_joint_state"] == pytest.approx(np.arange(6))


def test_sampler_helpers():
    shapely = pytest.importorskip("shapely")
    del shapely
    import random

    from dojo.simlab import gen_layouts as G
    assert G._ranges([0.3, -0.2]) == [(-0.2, 0.3)]
    assert G._ranges([[-0.45, -0.3], [0.3, 0.45]]) == [(-0.45, -0.3), (0.3, 0.45)]
    assert G.region([0.0, 0.0], [-0.2, -0.2]).geom_type == "Point"
    assert G.region([[-0.45, -0.3], [0.3, 0.45]], [-0.2, 0.0]).geom_type == "MultiPolygon"
    rng = random.Random(0)
    group = {"category": [{"name": "block", "index": [5, 8], "group": "a"}, {"name": "block", "index": [6, 9],
                                                                             "group": "b"}],
             "select_mode": {"mode": "hierarchical", "select_category_nums": [1, 1],
                             "select_instance_nums": [2, 2], "instance_sample_mode": "unique"}}
    sel = G.select_instances("Rigid", group, rng)
    assert len(sel) == 2 and len({e["group"] for e, _ in sel}) == 1 and len({i for _, i in sel}) == 2
    same = G.select_instances("Rigid", {"category": [{"name": "bowl", "index": [1, 2, 3]}],
                                        "select_mode": {"mode": "same", "nums": 3}}, rng)
    assert len({i for _, i in same}) == 1 and len(same) == 3
    uniq = G.select_instances("Rigid", {"category": [{"name": "b", "index": [0, 2]}, {"name": "b", "index": [0, 2]}],
                                        "select_mode": {"mode": "unique", "nums": [2]}}, rng)
    assert sorted(i for _, i in uniq) == [0, 2]
    with pytest.raises(G.LayoutError):
        G.select_instances("Rigid", {"category": [{"name": "b", "index": [0]}],
                                     "select_mode": {"mode": "unique", "nums": 2}}, rng)
