from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dojo.skills.motion import Collision, Motion, MoveResult, Segment
from dojo.skills.world import WorldState


def fixture():
    kin = SimpleNamespace(tcp=lambda *a: np.eye(4))
    coll = Collision(kin, WorldState("test", "", {}, {}, {}))
    return Motion(kin, coll)


def test_default_never_calls_object_planner(monkeypatch):
    from dojo.skills import world_plan
    m = fixture()
    legacy = MoveResult("ik_fail", "left")
    m._legacy_to_pose = Mock(return_value=legacy)
    monkeypatch.setattr(world_plan, "plan_objects", Mock(side_effect=AssertionError))
    assert m.to_pose("left", np.zeros(6), np.eye(4)) is legacy


def test_attachment_and_ignore_are_passed_and_easing_kept(monkeypatch):
    from dojo.skills import world_plan
    m = fixture()
    held = object()
    seen = []
    result = MoveResult("reached", "left", [Segment(np.zeros((2, 6)), .05)])
    def line(self, arm, q0, target, **kw):
        seen.append((self.coll, kw))
        return result
    monkeypatch.setattr(Motion, "line", line)
    got = world_plan.plan_objects(m, "left", np.zeros(6), np.eye(4), .05,
                                 "carry", 0., ("tool",), held, 2)
    assert got is result
    assert seen[0][1]["held"] is held and seen[0][1]["ignore"] == ("tool",)
    assert result.segments[0].ease == 2
    assert not m.world_plans[0]["fallback"]
    assert seen[0][0] is not m.coll


def test_budget_falls_back_without_executing_partial(monkeypatch):
    from dojo.skills import world_plan
    m = fixture()
    m.plan_world = "objects"
    partial = MoveResult("partial", "left", [Segment(np.zeros((2, 6)), .05)])
    monkeypatch.setattr(Motion, "line", Mock(return_value=partial))
    monkeypatch.setattr(world_plan, "BUDGET_S", 0.)
    expected = MoveResult("collision", "left")
    m._legacy_to_pose = Mock(return_value=expected)
    assert m.to_pose("left", np.zeros(6), np.eye(4)) is expected
    assert m.world_plans[0]["fallback"]
    assert m.world_plans[0]["routes"] == 1
    m._legacy_to_pose.assert_called_once()


def test_runtime_config_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().plan_world == "table"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(plan_world="oops"))
