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


def test_failed_line_uses_checked_via(monkeypatch):
    from dojo.skills import world_plan
    m = fixture()
    outcomes = [MoveResult("collision", "left"),
                MoveResult("reached", "left", [Segment(np.ones((2, 6)), .05)]),
                MoveResult("reached", "left", [Segment(np.ones((2, 6))*2, .05)])]
    checked = Mock(side_effect=outcomes)
    monkeypatch.setattr(Motion, "line", checked)
    monkeypatch.setattr(world_plan, "BUDGET_S", 100.)
    result = world_plan.plan_objects(m, "left", np.zeros(6), np.eye(4), .05,
                                    "carry", 0., ("tool",), None, 2)
    assert result.ok and len(result.segments) == 2
    assert checked.call_count == 3
    assert m.world_plans[0]["route"].startswith("via:")


def test_belief_geometry_added_without_mutating_caller(monkeypatch):
    from dojo.skills import world_plan
    m = fixture()
    m.coll.world.objects["basket"] = SimpleNamespace(label="basket", obj_type="Geometry")
    def line(self, *a, **kw):
        assert "basket" in self.coll.include
        assert self.coll.obj_margin == .006
        return MoveResult("reached", "left")
    monkeypatch.setattr(Motion, "line", line)
    world_plan.plan_objects(m, "left", np.zeros(6), np.eye(4), .05, "x", 1., (), None, 2)
    assert not m.coll.include


def test_port_attachment_comes_only_from_belief(monkeypatch):
    from dojo.skills import world_plan
    from dojo.skills.ported.harness_robot import HarnessRobot
    from dojo.skills.world import AssetInfo
    robot = object.__new__(HarnessRobot)
    obj = SimpleNamespace(asset=AssetInfo("tool", np.zeros(3), np.ones(3)*.01))
    rel = np.eye(4)
    belief = SimpleNamespace(attached={"left": ("tool", rel)})
    robot.sk = SimpleNamespace(refresh=lambda: None, held={"left": None},
                               provider=SimpleNamespace(belief=belief),
                               world=SimpleNamespace(objects={"tool": obj}),
                               motion=SimpleNamespace(kin=SimpleNamespace(tcp_offset=.145)))
    robot.ex = SimpleNamespace(g={"left": 0.})
    robot.env = SimpleNamespace()  # no truth accessors exist
    def plan(*args):
        assert args[7] == ("tool",)
        assert np.allclose(args[3][:3, 3], [.145, 0., 1.])
        assert np.array_equal(args[8].T_tcp_obj, rel)
        return None
    monkeypatch.setattr(world_plan, "plan_objects", plan)
    assert robot.object_path("left", [0, 0, 1, 1, 0, 0, 0], np.zeros(6)) is None


def test_port_unknown_attachment_falls_back_without_empty_hand_plan(monkeypatch):
    from dojo.skills import world_plan
    from dojo.skills.ported.harness_robot import HarnessRobot
    robot = object.__new__(HarnessRobot)
    robot.sk = SimpleNamespace(refresh=lambda: None, held={"left": None},
                               provider=SimpleNamespace(), motion=SimpleNamespace(world_plans=[]))
    robot.ex = SimpleNamespace(g={"left": 0.})
    monkeypatch.setattr(world_plan, "plan_objects", Mock(side_effect=AssertionError))
    assert robot.object_path("left", [0, 0, 1, 1, 0, 0, 0], np.zeros(6)) is None
    assert robot.sk.motion.world_plans[0]["reason"] == "unknown_attachment"
