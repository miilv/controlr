"""``RuntimeConfig.ported_classify`` "p3" (``classify_into_bins_p3``): p2 with its container test on the object's lowest
point. On the camera a believed object let go into a basket keeps its in-hand orientation, so its root sits above the rim
while it rests on the floor, and p2's root test re-did placed objects (journal 2026-11-07-dojo-classify3). CPU only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from dojo.skills import planner
from dojo.skills.ported import TIER_A

from test_port_adapter import Z, gt_skills, rows_classify

P3 = Path(__file__).resolve().parents[2] / "dojo/skills/ported/skills/classify_into_bins_p3/skill_robodojo.py"
INSTR = ("Put watch objects into the left basket, pen objects into the middle basket, and toy_car objects into the right "
         "basket, then reset the robot arm.")


def _p3():
    spec = importlib.util.spec_from_file_location("cib_p3_test", P3)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _objs(root_z, bottom_z, x=0.05):
    return {"b": {"aabb": [[0.0, 0.0, Z], [0.1, 0.1, Z + 0.077]], "category": "basket", "pos": [0.05, 0.05, Z]},
            "o": {"aabb": [[x - 0.02, 0.03, bottom_z], [x + 0.02, 0.07, bottom_z + 0.1]], "pos": [x, 0.05, root_z]}}


def test_bottom_test_takes_an_upright_object_resting_on_the_floor_and_p2s_root_test_does_not():
    m = _p3()
    p2 = m.load_p2()
    o = _objs(root_z=Z + 0.12, bottom_z=Z + 0.01)          # standing on the floor, its root 43 mm over the rim
    assert m.in_bin_bottom(o, "o", "b") and not p2.in_bin(o, "o", "b")
    o = _objs(root_z=Z + 0.05, bottom_z=Z + 0.01)          # lying: both agree
    assert m.in_bin_bottom(o, "o", "b") and p2.in_bin(o, "o", "b")
    assert not m.in_bin_bottom(_objs(Z + 0.15, Z + 0.095), "o", "b")    # held / hanging above the rim
    assert not m.in_bin_bottom(_objs(Z + 0.05, Z + 0.01, x=0.2), "o", "b")  # beside the basket
    o = _objs(Z + 0.05, Z + 0.01)
    del o["o"]["aabb"]                                       # no box: the root decides, as p2
    assert m.in_bin_bottom(o, "o", "b") == p2.in_bin(o, "o", "b")


def test_contain_switch_and_p2_untouched():
    m = _p3()
    assert m.run(object(), contain="centre")["error"].startswith("contain must be")
    a, b = m.load_p2(), m.load_p2()
    a.in_bin = m.in_bin_bottom                               # a private instance: another one keeps p2's test
    assert b.in_bin is not m.in_bin_bottom


def test_route_takes_p3_only_when_switched():
    env, sk = gt_skills(rows_classify(), INSTR)
    w = sk.refresh()
    assert planner.plan(INSTR, w, ported={"classify": "p2"}).args["skill"] == "classify_into_bins_p2"
    assert planner.plan(INSTR, w, ported={"classify": "p3"}).args["skill"] == "classify_into_bins_p3"
    assert "classify_into_bins_p3" in TIER_A


def test_defaults_unchanged_and_p3_accepted():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().ported_classify == "v3b" and SKILL_ATTRS["ported_classify"][0] == "v3b"
    assert "p3" in SKILL_ATTRS["ported_classify"]
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(ported_classify="p4"))


def test_p3_runs_through_the_ported_loader_with_its_contain_mode():
    import json

    from dojo.skills.ported import ported
    env, sk = gt_skills(rows_classify(), INSTR, step_lim=1100)
    r = ported(sk, "classify_into_bins_p3")
    d = json.loads(sk.ported_runs[-1]["details"])
    assert r["ok"] and r["steps"] > 0, r
    assert d["contain"] == "bottom" and d["skipped"] == []


def test_park_other_is_off_by_default_and_reaches_p3_through_the_planner():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().ported_classify_park == "off" and SKILL_ATTRS["ported_classify_park"][0] == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(ported_classify_park="survey"))
    env, sk = gt_skills(rows_classify(), INSTR)
    w = sk.refresh()
    assert planner.plan(INSTR, w, ported={"classify": "p3"}).args["args"] == {}
    p = planner.plan(INSTR, w, ported={"classify": "p3", "classify_park": "home"})
    assert p.args["skill"] == "classify_into_bins_p3" and p.args["args"] == {"park_other": "home"}
    p = planner.plan(INSTR, w, ported={"classify": "p2", "classify_park": "home"})   # p2 takes no such argument
    assert p.args["args"] == {}
    assert _p3().run(object(), park_other="survey")["error"].startswith("park_other must be")


class _Robot:
    def __init__(self, other_home, grip):
        self.e = {"left": {"at_home": True, "grip_cmd": 1.0}, "right": {"at_home": other_home, "grip_cmd": grip}}

    def obs(self, cameras=None):
        return {"eef": self.e}


@pytest.mark.parametrize("home,grip,stage,parks", [(False, 1.0, "direct", 1), (True, 1.0, "direct", 0),
                                                   (False, 0.0, "direct", 0), (False, 1.0, "relay", 0),
                                                   (False, 1.0, "nudge", 1)])
def test_parking_homes_only_an_idle_away_arm_before_a_one_arm_carry(home, grip, stage, parks):
    m = _p3()
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
