"""``RuntimeConfig.series_labels``: perception keeps a series with per-category prefixes under its config label
(classify_objects' ``rigid0`` x 6); keyed by label the belief kept one of them and the classify route saw one item."""

from __future__ import annotations

import numpy as np

from dojo.runtime.estimate import _label_entry, objects_from_record, series_names
from dojo.skills import planner
from dojo.skills.world import WorldState

Z = 0.765
INSTR = "Sort the objects by category into the three baskets."


def _rec(cat: str, idx: int, label: str, x: float, y: float, z: float = Z + 0.02) -> dict:
    return {"category": cat, "category_idx": idx, "label": label, "default_pos": [x, y, z],
            "default_ori": [1.0, 0.0, 0.0, 0.0], "_perception": {"confidence": 0.5}}


def camera_record() -> dict:
    """A frame-0 record as B1 writes it for classify_objects: six items of three categories, all labelled ``rigid0``."""
    items = [("pen", 0, 0.40, -0.18), ("pen", 0, -0.37, -0.10), ("action_camera", 1, -0.27, -0.22),
             ("action_camera", 1, 0.01, -0.17), ("garage", 2, -0.14, -0.18), ("garage", 2, 0.27, -0.15)]
    rec: dict = {"Rigid": {}, "Geometry": {}}
    for cat, idx, x, y in items:
        rec["Rigid"].setdefault(cat, []).append(_rec(cat, idx, "rigid0", x, y))
    for i, x in enumerate((-0.28, 0.0, 0.28)):
        rec["Geometry"].setdefault("basket", []).append(_rec("basket", i, f"basket{i}", x, 0.04, Z + 0.04))
    return rec


def _world(objs: dict) -> WorldState:
    return WorldState("", INSTR, objs, {}, {})


def test_keep_collapses_a_repeated_label_and_routes_nothing():
    objs = objects_from_record(camera_record(), None)
    assert sorted(objs) == ["basket0", "basket1", "basket2", "rigid0"]
    assert planner.ported_plan(INSTR, _world(objs)) is None


def test_unique_names_each_item_and_the_classify_route_runs():
    rec = camera_record()
    objs = objects_from_record(rec, None, unique=True)
    items = sorted(k for k in objs if not k.startswith("basket"))
    assert items == [f"rigid0_{k}" for k in range(6)]
    # numbered by x then y; categories and poses are perception's
    xs = [float(objs[k].pos[0]) for k in items]
    assert xs == sorted(xs) and objs["rigid0_0"].category == "pen"
    assert sorted(o.category for o in objs.values() if o.label in items) == sorted(
        ["pen", "pen", "action_camera", "action_camera", "garage", "garage"])
    assert {k: objs[k].category for k in ("basket0", "basket1", "basket2")} == {f"basket{i}": "basket" for i in range(3)}
    p = planner.ported_plan(INSTR, _world(objs))
    assert p is not None and p.args["skill"] == "classify_into_bins_v3b"
    # deterministic, and the record itself is not relabelled
    assert series_names(rec) == series_names(rec)
    assert {r["label"] for r in rec["Rigid"]["pen"]} == {"rigid0"}


def test_unique_leaves_single_labels_alone():
    rec = {"Rigid": {"cube": [_rec("cube", 0, f"cube_{k}", 0.1 * k, -0.2) for k in range(3)]}}
    assert series_names(rec) == {}
    assert sorted(objects_from_record(rec, None, unique=True)) == ["cube_0", "cube_1", "cube_2"]


def test_label_entry_of_a_numbered_series_is_its_config_label():
    ent, cube = {"per_category": {"pen": {}}}, {"per_category": {"cube": {}}}
    labels = {"rigid0": ent, ("prefix", "cube"): cube, "basket0": {}}
    assert _label_entry(labels, "rigid0_4") is ent
    assert _label_entry(labels, "cube_2") is cube
    assert _label_entry(labels, "rigid0") is ent
    assert _label_entry(labels, "basket0_x") is None and _label_entry(labels, "mystery_1") is None


def test_series_labels_default_keep():
    from dojo.runtime.policy import RuntimeConfig
    assert RuntimeConfig().series_labels == "keep"
    np.testing.assert_equal(len(series_names(camera_record())), 6)


def _three() -> dict:
    return {"Rigid": {"pen": [_rec("pen", 0, "rigid0", x, -0.15) for x in (0.2, -0.2, 0.0)]}}


def test_three_repeated_labels_three_belief_objects_and_keep_is_today():
    """Through the runtime's belief (``RuntimeProvider.init_from_record``): "numbered" -> 3 objects; "keep" -> today's
    dict exactly (the last entry under the repeated label)."""
    import pytest

    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from test_runtime import STACK, FakeAssets, FakeEpisode, _env

    env = _env()
    pol = RuntimePolicy(RuntimeConfig(), assets=FakeAssets())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "classify_objects", STACK, "perception", FakeAssets())
    assert prov.unique_labels is False
    prov.init_from_record(_three())
    keep = prov.belief.objects
    assert list(keep) == ["rigid0"] and np.allclose(keep["rigid0"].pos, [0.0, -0.15, Z + 0.02])
    # "keep": the same objects as before the flag existed (one label -> the last entry)
    ref = {r["label"]: r for r in _three()["Rigid"]["pen"]}
    assert {k: (o.category, list(o.pos)) for k, o in objects_from_record(_three(), None).items()} == {
        k: ("pen", r["default_pos"]) for k, r in ref.items()}
    prov.unique_labels = True
    prov.init_from_record(_three())
    uni = prov.belief.objects
    assert sorted(uni) == ["rigid0_0", "rigid0_1", "rigid0_2"]
    assert [round(float(uni[k].pos[0]), 2) for k in sorted(uni)] == [-0.2, 0.0, 0.2]
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(series_labels="maybe"))
