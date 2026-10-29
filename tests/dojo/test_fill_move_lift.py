"""Stream H: ``RuntimeConfig.fill_move_lift`` / ``Skills.fill_move_lift`` ("off" = today | "clear"): after
``fill_container`` moves the container, a hand whose fingertips are still below its top goes straight up clear of it
(fill_pen_holder: the 120 mm holder dragged over by the first pen's pick approach). CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library
from dojo.skills.kinematics import TIP_BEYOND_TCP


def _down(xyz):
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])     # approach axis (x) pointing down
    T[:3, 3] = xyz
    return T


class _Plan:
    def __init__(self, arm, q, g):
        self.arm, self.moves = arm, []

    def move(self, m, name):
        self.moves.append((m, name))
        return self


def _sk(tips):
    holder = SimpleNamespace(aabb=lambda: (np.array([-0.28, -0.305, 0.765]), np.array([-0.20, -0.235, 0.885])))
    lines, ran = [], []
    tcp = {a: _down(np.array(p) + [0, 0, TIP_BEYOND_TCP]) for a, p in tips.items()}

    def line(arm, q0, T_goal, step, name, gripper=1.0, check=True):
        lines.append((arm, T_goal.copy(), check))
        return SimpleNamespace(ok=True, note="")
    sk = SimpleNamespace(
        refresh=lambda: SimpleNamespace(obj=lambda l: holder),
        kin=SimpleNamespace(tcp=lambda arm, q: tcp[arm]),
        ex=SimpleNamespace(q={"left": np.zeros(6), "right": np.zeros(6)}, g={"left": 1.0, "right": 1.0}),
        motion=SimpleNamespace(line=line), log=lambda m: None,
        parallel=lambda plans, name="": ran.extend(plans))
    return sk, lines, ran


def test_a_hand_below_the_rim_goes_up_clear(monkeypatch):
    monkeypatch.setattr(library, "ArmPlan", _Plan)
    # the left hand let go of the holder across its walls (tips 15 mm under the rim); the right one is far away
    sk, lines, ran = _sk({"left": (-0.24, -0.27, 0.870), "right": (0.25, -0.2, 0.80)})
    library._lift_clear_of(sk, "pen_holder")
    assert [a for a, _, _ in lines] == ["left"] and [p.arm for p in ran] == ["left"]
    T_goal = lines[0][1]
    tip_goal = T_goal[2, 3] - TIP_BEYOND_TCP
    assert tip_goal == pytest.approx(0.885 + library.FILL_LIFT_CLEAR)
    assert T_goal[:2, 3] == pytest.approx([-0.24, -0.27])


def test_a_hand_already_clear_does_not_move(monkeypatch):
    monkeypatch.setattr(library, "ArmPlan", _Plan)
    sk, lines, ran = _sk({"left": (-0.24, -0.27, 0.95), "right": (0.25, -0.2, 0.80)})
    library._lift_clear_of(sk, "pen_holder")
    assert lines == [] and ran == []


def test_default_and_validation():
    assert RuntimeConfig().fill_move_lift == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", fill_move_lift="up"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["fill_move_lift"] == ("off", "clear")
