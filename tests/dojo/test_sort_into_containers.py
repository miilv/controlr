"""``sort_into_containers``: the vendored sort mechanics (classify_into_bins_p2) with two general options, selected by
``RuntimeConfig.sort_contain`` / ``sort_park`` (default off: the sort route runs the selected variant as before).
``contain="bottom"`` judges "in the container" on the object's lowest point: a believed object let go into a container
keeps its in-hand orientation, so its root can sit above the rim while it rests on the floor (journal
2026-11-07-dojo-classify3). ``park_other="home"`` homes the idle arm before a one-arm carry. CPU only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dojo.skills import planner
from dojo.skills.ported import TIER_A, ported

from test_port_adapter import Z, gt_skills, rows_classify

SKILL = Path(__file__).resolve().parents[2] / "dojo/skills/ported/skills/sort_into_containers/skill_robodojo.py"
INSTR = ("Put watch objects into the left basket, pen objects into the middle basket, and toy_car objects into the right "
         "basket, then reset the robot arm.")


def _sort():
    spec = importlib.util.spec_from_file_location("sort_into_containers_test", SKILL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _objs(root_z, bottom_z, x=0.05):
    return {"b": {"aabb": [[0.0, 0.0, Z], [0.1, 0.1, Z + 0.077]], "category": "basket", "pos": [0.05, 0.05, Z]},
            "o": {"aabb": [[x - 0.02, 0.03, bottom_z], [x + 0.02, 0.07, bottom_z + 0.1]], "pos": [x, 0.05, root_z]}}


def test_bottom_test_takes_an_upright_object_resting_on_the_floor_and_the_root_test_does_not():
    m = _sort()
    p2 = m.load_p2()
    o = _objs(root_z=Z + 0.12, bottom_z=Z + 0.01)          # standing on the floor, its root 43 mm over the rim
    assert m.in_bin_bottom(o, "o", "b") and not p2.in_bin(o, "o", "b")
    o = _objs(root_z=Z + 0.05, bottom_z=Z + 0.01)          # lying: both agree
    assert m.in_bin_bottom(o, "o", "b") and p2.in_bin(o, "o", "b")
    assert not m.in_bin_bottom(_objs(Z + 0.15, Z + 0.095), "o", "b")    # held / hanging above the rim
    assert not m.in_bin_bottom(_objs(Z + 0.05, Z + 0.01, x=0.2), "o", "b")  # beside the container
    o = _objs(Z + 0.05, Z + 0.01)
    del o["o"]["aabb"]                                       # no box: the root decides, as p2
    assert m.in_bin_bottom(o, "o", "b") == p2.in_bin(o, "o", "b")


def test_options_are_checked_and_p2_stays_untouched():
    m = _sort()
    assert m.run(object(), contain="centre")["error"].startswith("contain must be")
    assert m.run(object(), park_other="survey")["error"].startswith("park_other must be")
    a, b = m.load_p2(), m.load_p2()
    a.in_bin = m.in_bin_bottom                               # a private instance: another one keeps p2's test
    assert b.in_bin is not m.in_bin_bottom


def test_the_sort_route_takes_the_general_skill_only_with_an_option_set():
    env, sk = gt_skills(rows_classify(), INSTR)
    w = sk.refresh()
    assert planner.plan(INSTR, w, ported={"classify": "p2"}).args["skill"] == "classify_into_bins_p2"
    p = planner.plan(INSTR, w, ported={"classify": "p2", "sort_contain": "bottom"})
    assert p.args == {"skill": "sort_into_containers", "args": {"contain": "bottom"}}
    p = planner.plan(INSTR, w, ported={"classify": "p2", "sort_contain": "bottom", "sort_park": "home"})
    assert p.args["args"] == {"contain": "bottom", "park_other": "home"}
    p = planner.plan(INSTR, w, ported={"classify": "p2", "sort_contain": "root", "sort_park": "off"})
    assert p.args["skill"] == "classify_into_bins_p2"
    assert "sort_into_containers" in TIER_A


def test_defaults_unchanged_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    c = RuntimeConfig()
    assert (c.ported_classify, c.sort_contain, c.sort_park) == ("v3b", "root", "off")
    assert SKILL_ATTRS["sort_contain"][0] == "root" and SKILL_ATTRS["sort_park"][0] == "off"
    for bad in ({"sort_contain": "centre"}, {"sort_park": "survey"}, {"ported_classify": "p3"}):
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**bad))


def test_runs_through_the_ported_loader_with_its_options():
    env, sk = gt_skills(rows_classify(), INSTR, step_lim=1100)
    r = ported(sk, "sort_into_containers", {"contain": "bottom"})
    d = json.loads(sk.ported_runs[-1]["details"])
    assert r["ok"] and r["steps"] > 0, r
    assert d["contain"] == "bottom" and d["park_other"] == "off" and d["skipped"] == []


class _Robot:
    def __init__(self, other_home, grip):
        self.e = {"left": {"at_home": True, "grip_cmd": 1.0}, "right": {"at_home": other_home, "grip_cmd": grip}}

    def obs(self, cameras=None):
        return {"eef": self.e}


@pytest.mark.parametrize("home,grip,stage,parks", [(False, 1.0, "direct", 1), (True, 1.0, "direct", 0),
                                                   (False, 0.0, "direct", 0), (False, 1.0, "relay", 0),
                                                   (False, 1.0, "nudge", 1)])
def test_parking_homes_only_an_idle_away_arm_before_a_one_arm_carry(home, grip, stage, parks):
    m = _sort()
    calls = []

    class V3:
        @staticmethod
        def run(robot, stage, **kw):
            calls.append(stage)
            return {"ok": True}

        @staticmethod
        def home_safe(robot, arms, mode, margin_m, margin_deg):
            calls.append(("home", tuple(arms)))
            return {"ok": True}
    log: list = []
    V = m.parking(lambda: V3, log)()
    V.run(_Robot(home, grip), stage, arm="left", obj="o")
    assert len(log) == parks and calls[-1] == stage
    if parks:
        assert calls[0] == ("home", ("right",)) and log[0]["arm"] == "right"
