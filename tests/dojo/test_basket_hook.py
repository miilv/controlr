"""CPU-only guards for the opt-in basket experiment."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from dojo.skills.basket import hook_pose, route


def test_config_defaults_and_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS

    assert RuntimeConfig().belt_lift == "off"
    assert RuntimeConfig().new_routes == []
    assert SKILL_ATTRS["belt_lift"] == ("off", "braced_hook")
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(belt_lift="typo"))
    RuntimePolicy(RuntimeConfig(belt_lift="braced_hook", new_routes=["basket_image"]))


def obj(label, category, typ="Rigid"):
    return NS(label=label, category=category, obj_type=typ, clutter=False)


def test_route_needs_opt_in_instruction_and_observed_roles():
    objects = [
        obj("b", "basket_grasp"),
        obj("p", "photo", "Rigid"),
        obj("t", "apple"),
        obj("c", "conveyor", "Dynamic"),
    ]
    w = NS(task_objects=lambda: objects)
    text = "lift the basket and pick the target on the conveyor according to the image"
    assert route(text, w, []) is None
    r = route(text, w, ["basket_image"])
    assert r[:2] == (
        "pick_into_held",
        dict(objects=["t"], container="b", picture="p", conveyor="c"),
    )
    assert (
        route("remember the first object on the conveyor", w, ["basket_image"]) is None
    )
    objects.pop(1)
    assert route(text, w, ["basket_image"]) is None
    objects.append(obj("p", "photo"))
    objects.append(obj("b2", "basket_grasp"))
    assert route(text, w, ["basket_image"]) is None


def test_hook_geometry_and_missing_mesh():
    x, y = np.meshgrid(np.linspace(-0.07, 0.07, 10), [-0.005, 0.005])
    v = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.2)]
    o = NS(asset=NS(mesh=(v, [])), R=np.eye(3), pos=np.array([0.1, 0.0, 0.8]))
    t = hook_pose(o, np.array([0.1, -0.4, 0.8]))
    np.testing.assert_allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-9)
    assert np.linalg.det(t[:3, :3]) == pytest.approx(1)
    assert t[2, 3] == pytest.approx(0.964)
    assert t[1, 0] > 0  # approach away from the base, across the bar
    o.asset.mesh = None
    assert hook_pose(o, np.zeros(3)) is None


def test_hook_refuses_unreachable_transit_without_releasing_brace():
    from dojo.skills.basket import braced_hook
    from dojo.skills.motion import MoveResult

    x = np.linspace(-0.07, 0.07, 12)
    v = np.c_[x, np.zeros(12), np.full(12, 0.2)]
    o = NS(asset=NS(mesh=(v, [])), R=np.eye(3), pos=np.array([0.0, 0.0, 0.8]))

    def unexpected(*args, **kwargs):
        pytest.fail("unreachable hook must execute nothing")

    sk = NS(
        refresh=lambda: NS(obj=lambda _: o),
        kin=NS(base=lambda _: np.eye(4)),
        ex=NS(q={"left": np.zeros(6)}, run=unexpected),
        motion=NS(to_pose=lambda *args, **kwargs: MoveResult("ik_fail", "left")),
        held={"right": "brace"},
    )
    r = braced_hook(sk, "basket", "left", "right", ("conveyor",))
    assert not r.ok and r.status == "ik_fail"
    assert sk.held["right"] == "brace"


def test_legacy_deposit_dispatch_does_not_call_hook(monkeypatch):
    from dojo.skills import interactive as i

    # Reach dispatch with no actual physics: the original lift must remain the default.
    from dojo.skills.skills import SkillResult

    calls = []
    o = NS(pos=np.zeros(3))
    w = NS(obj=lambda _: o)
    sk = NS(kin=NS(base=lambda _: np.eye(4)), refresh=lambda: w, log=lambda _: None)
    ok = SkillResult(True, "mock", "ok", 0, "left")
    for name in ("_stop_on_belt", "pick_from_belt", "drop_into_held"):
        monkeypatch.setattr(i, name, lambda *args, **kwargs: ok)
    monkeypatch.setattr(i, "inside_xy", lambda *args: True)
    monkeypatch.setattr(i, "_handle_lift", lambda *args: calls.append("handle") or ok)
    from dojo.skills import basket

    monkeypatch.setattr(basket, "braced_hook", lambda *args: calls.append("hook") or ok)
    assert i._deposit_then_lift(sk, "basket", "target", np.array([0.004, 0]), ())["ok"]
    assert calls == ["handle"]
    sk.belt_lift = "braced_hook"
    assert i._deposit_then_lift(sk, "basket", "target", np.array([0.004, 0]), ())["ok"]
    assert calls == ["handle", "hook"]


@pytest.mark.parametrize("reject_second_load", [False, True])
def test_brace_release_requires_both_load_plans(reject_second_load):
    from dojo.skills.basket import braced_hook
    from dojo.skills.motion import MoveResult

    x = np.linspace(-0.07, 0.07, 12)
    o = NS(
        asset=NS(mesh=(np.c_[x, np.zeros(12), np.full(12, 0.2)], [])),
        R=np.eye(3),
        pos=np.array([0.0, 0.0, 0.8]),
        bottom_z=0.8,
    )
    executions = []

    def run(plans, name):
        executions.append((name, set(plans)))
        if name == "hook lift":
            o.bottom_z += 0.15

    def line(arm, q, t, rate, name, **kwargs):
        status = (
            "ik_fail"
            if reject_second_load and arm == "right" and name == "hook_load"
            else "reached"
        )
        return MoveResult(status, arm)

    sk = NS(
        refresh=lambda: NS(obj=lambda _: o),
        kin=NS(base=lambda _: np.eye(4), tcp=lambda *args: np.eye(4)),
        ex=NS(q={a: np.zeros(6) for a in ("left", "right")}, run=run),
        motion=NS(
            to_pose=lambda *args, **kwargs: MoveResult("reached", "left"), line=line
        ),
        held={"right": "brace"},
        log=lambda _: None,
        _record=lambda r: r,
    )
    r = braced_hook(sk, "basket", "left", "right", ("conveyor",))
    if reject_second_load:
        assert not r.ok and sk.held["right"] == "brace"
        assert [n for n, _ in executions] == ["hook transit", "hook contact"]
    else:
        assert r.ok and sk.held["right"] is None
        assert executions[2:4] == [
            ("load hook with brace", {"left", "right"}),
            ("release brace", {"right"}),
        ]
