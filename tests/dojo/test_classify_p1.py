"""``RuntimeConfig.ported_classify`` "p1" (``classify_into_bins_p1``): v3b's grounding raised when the scene's categories
were not the instruction's three (camera: a misread item, a phantom) or outnumbered the baskets, and nothing was sorted."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dojo.skills import planner
from dojo.skills.ported import TIER_A, ported

from test_port_adapter import Z, gt_skills, rows_classify

P1 = Path(__file__).resolve().parents[2] / "dojo/skills/ported/skills/classify_into_bins_p1/skill_robodojo.py"
INSTR = ("Put watch objects into the left basket, pen objects into the middle basket, and toy_car objects into the right "
         "basket, then reset the robot arm.")


def _p1():
    spec = importlib.util.spec_from_file_location("cib_p1_test", P1)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _objs():
    def box(x0):
        return {"aabb": [[x0, 0.0, Z], [x0 + 0.1, 0.1, Z + 0.1]], "category": "basket", "pos": [x0 + 0.05, 0.05, Z]}
    return {"b0": box(-0.3), "b1": box(-0.05), "b2": box(0.2)}


def test_fixed_map_skips_what_it_cannot_ground_instead_of_raising():
    m = _p1()
    objs = _objs()
    groups = {"watch": ["o1", "o2"], "pen": ["o3"], "garage": ["o4"]}       # toy_car unseen, garage unnamed
    skipped = []
    gmap = m.fixed_map(objs, ["b0", "b1", "b2"], groups, INSTR, skipped=skipped)
    assert gmap == {"watch": "b0", "pen": "b1"}
    assert [s["phrase"] for s in skipped] == ["toy_car"]
    with pytest.raises(ValueError):                                         # nothing named at all
        m.fixed_map(objs, ["b0", "b1", "b2"], {"garage": ["o4"]}, INSTR)
    assert m.fixed_map(objs, ["b0"], groups, "Sort the objects by category into the baskets.") is None


def test_free_sort_keeps_one_group_per_container():
    m = _p1()
    kept, left = m.keep_groups({"a": [1, 2], "b": [3], "c": [4, 5, 6], "d": [7]}, 3)
    assert sorted(kept) == ["a", "b", "c"] and left == ["d"]
    kept, left = m.keep_groups({"a": [1], "b": [2]}, 3)
    assert sorted(kept) == ["a", "b"] and left == []


def test_route_takes_p1_only_when_switched():
    env, sk = gt_skills(rows_classify(), INSTR)
    w = sk.refresh()
    assert planner.plan(INSTR, w, ported=True).args["skill"] == "classify_into_bins_v3b"
    assert planner.plan(INSTR, w, ported={"classify": "p1"}).args["skill"] == "classify_into_bins_p1"
    assert "classify_into_bins_p1" in TIER_A


def test_ported_classify_defaults_v3b():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().ported_classify == "v3b" and SKILL_ATTRS["ported_classify"][0] == "v3b"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(ported_classify="v9"))


def test_an_unnamed_category_leaves_its_item_and_sorts_the_rest():
    """One item of a category the instruction does not name: v3b fails grounding before any motion; p1 sorts and leaves it."""
    rows = rows_classify() + [dict(label="extra", category="garage", category_idx=9, pos=[-0.1, -0.25, Z + 0.012],
                                   quat=[1, 0, 0, 0], half=[0.03, 0.012, 0.012])]
    env, sk = gt_skills(rows, INSTR, step_lim=1100)
    r3 = ported(sk, "classify_into_bins_v3b")
    assert not r3["ok"] and "grounding" in r3["error"] and r3["steps"] == 0
    env, sk = gt_skills(rows, INSTR, step_lim=1100)
    r1 = ported(sk, "classify_into_bins_p1")
    d = json.loads(sk.ported_runs[-1]["details"])
    assert r1["ok"] and r1["steps"] > 0, r1
    assert d["left_groups"] == ["garage"] and len(d["left"]) == 1 and d["skipped"] == []


def test_free_sort_with_more_groups_than_baskets_sorts_three():
    """Four categories for three baskets: v3b's plan2 finds no permutation (nothing moves); p1 leaves one group."""
    free = "Sort the objects by category into the three baskets."
    rows = rows_classify() + [dict(label="extra", category="garage", category_idx=9, pos=[-0.1, -0.25, Z + 0.012],
                                   quat=[1, 0, 0, 0], half=[0.03, 0.012, 0.012])]
    env, sk = gt_skills(rows, free, step_lim=1100)
    assert planner.plan(free, sk.refresh(), ported=True).args["skill"] == "classify_into_bins_v3b"
    r3 = ported(sk, "classify_into_bins_v3b")
    assert not r3["verified"] and r3["steps"] == 0
    env, sk = gt_skills(rows, free, step_lim=1100)
    r1 = ported(sk, "classify_into_bins_p1")
    d = json.loads(sk.ported_runs[-1]["details"])
    assert r1["ok"] and r1["steps"] > 0 and len(d["left"]) == 1 and len(d["left_groups"]) == 1
