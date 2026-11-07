"""Lift-first switches and native-envelope guards; no simulator or model calls."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from dojo.skills import interactive as I
from dojo.skills.basket_first import profile


def test_switches_are_independent_and_default_off():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS

    cfg = RuntimeConfig()
    assert cfg.belt_first == cfg.belt_target == "off"
    assert SKILL_ATTRS["belt_first"] == ("off", "on")
    assert SKILL_ATTRS["belt_target"] == ("off", "match")
    assert I.target_lift(NS()) == 0.15
    assert I.target_lift(NS(belt_target="match")) == 0.125
    for field in ("belt_first", "belt_target"):
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**{field: "typo"}))


def test_lift_first_only_dispatches_when_enabled(monkeypatch):
    from dojo.skills import basket_first

    objects = {
        k: NS(pos=np.zeros(3), asset=NS(key=k)) for k in ("basket", "target", "photo")
    }
    w = NS(objects=objects, obj=objects.__getitem__)
    sk = NS(refresh=lambda: w, log=lambda _: None, belt_deposit="on")
    monkeypatch.setattr(I, "belt_velocity", lambda *args: np.array([0.004, 0.0]))
    monkeypatch.setattr(I, "depicted_identity", lambda _: "target")
    calls = []
    monkeypatch.setattr(
        I, "_deposit_then_lift", lambda *args: calls.append("legacy") or {"ok": False}
    )
    monkeypatch.setattr(
        basket_first, "lift_first", lambda *args: calls.append("first") or {"ok": True}
    )
    I.pick_into_held(sk, ["target"], "basket", "photo", "conveyor")
    assert calls == ["legacy"]
    sk.belt_first = "on"
    I.pick_into_held(sk, ["target"], "basket", "photo", "conveyor")
    assert calls == ["legacy", "first"]


def test_profile_restricts_and_restores_custom_bounds_on_exception():
    from dojo.skills.kinematics import X5Kin, JOINT_LIMITS_MEASURED
    from dojo.skills.skills import Skills

    sk = Skills.__new__(Skills)
    sk.kin = X5Kin()
    sk._joint_limits = "wide"
    sk.belt_container, sk.belt_deposit = "off", "off"
    sk.kin.limits[0] = [-0.2, 0.2]
    before = sk.kin.limits.copy()
    sk.kin._ik_cache = {"old": object()}
    with pytest.raises(RuntimeError, match="abort"):
        with profile(sk):
            np.testing.assert_allclose(
                sk.kin.limits[:, 0],
                np.maximum(before[:, 0], JOINT_LIMITS_MEASURED[:, 0]),
            )
            np.testing.assert_allclose(
                sk.kin.limits[:, 1],
                np.minimum(before[:, 1], JOINT_LIMITS_MEASURED[:, 1]),
            )
            assert sk.belt_container == sk.belt_deposit == "on"
            assert "_ik_cache" not in sk.kin.__dict__
            sk.kin._ik_cache = {"scoped": object()}
            raise RuntimeError("abort")
    np.testing.assert_array_equal(sk.kin.limits, before)
    assert sk.joint_limits == "wide"
    assert sk.belt_container == sk.belt_deposit == "off"
    assert "_ik_cache" not in sk.kin.__dict__


def test_legacy_pick_keeps_its_original_planner_arguments(monkeypatch):
    # Refuse before touching hardware; this exercises the public default path.
    sk = NS(refresh=lambda: NS(obj=lambda _: NS(pos=np.zeros(3))))
    calls = []
    monkeypatch.setattr(
        I, "plan_intercept", lambda *args, **kwargs: calls.append(kwargs) or None
    )
    result = I.pick_from_belt(sk, "target", np.array([0.004, 0.0]))
    assert result.status == "ik_fail"
    assert calls == [dict(lift=0.125, ignore=(), arms=None, pull=0.0, skip_methods=())]


@pytest.mark.parametrize("reachable", [False, True])
def test_failed_or_lost_carry_never_opens_the_gripper(monkeypatch, reachable):
    from dojo.skills import basket_first as B
    from dojo.skills.motion import MoveResult
    from dojo.skills.skills import Held

    vertices = np.array(
        [[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.02, 0.02)]
    )
    asset = NS(
        bbox_min=np.array([-0.1, -0.1, -0.1]),
        bbox_max=np.array([0.1, 0.1, 0.1]),
        mesh=(vertices, []),
    )
    container = NS(
        asset=asset, pos=np.zeros(3), R=np.eye(3), center_w=np.zeros(3), bottom_z=0.0
    )
    target = NS(
        asset=asset,
        pos=np.array([0.3, 0.0, 0.2]),
        center_w=np.array([0.3, 0.0, 0.2]),
        bottom_z=0.18,
    )
    world = NS(obj=lambda name: {"basket": container, "target": target}[name])
    h = Held("target", np.eye(4), None)
    executed = []
    sk = NS(
        refresh=lambda: world,
        held={"left": h},
        log=lambda _: None,
        kin=NS(base=lambda _: np.eye(4), tcp=lambda *args: np.eye(4)),
        ex=NS(
            q={"left": np.zeros(6)},
            run=lambda plans, name: executed.append((name, plans)),
        ),
        motion=NS(
            to_pose=lambda *args, **kwargs: MoveResult(
                "reached" if reachable else "ik_fail", "left"
            )
        ),
    )
    monkeypatch.setattr(B.G, "box_rim_z", lambda _: 0.1)
    result = B.drop_center(sk, "left", "target", "basket", ("conveyor",))
    assert not result.ok
    assert result.status == ("grasp_slip" if reachable else "no_plan")
    assert sk.held["left"] is h
    # The only allowed executed operation is transit; opening must remain unreachable.
    assert [name for name, _ in executed] == (["basket drop"] if reachable else [])
    for _, plans in executed:
        assert all(op[0] != "grip" for plan in plans.values() for op in plan.ops)
