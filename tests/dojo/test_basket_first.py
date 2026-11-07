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
    sk = NS(refresh=lambda: w, log=lambda _: None, belt_deposit="on",
            belt_first_recover="on", belt_first_phase="on")
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


def test_round3_switches_default_off_and_mirrored():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.env import KinematicEnv

    cfg = RuntimeConfig()
    skills = Skills(KinematicEnv([]), NS(refresh=lambda *args: NS(objects={})))
    for field in ("belt_first_recover", "belt_first_phase"):
        assert getattr(cfg, field) == getattr(skills, field) == "off"
        assert SKILL_ATTRS[field] == ("off", "on")
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**{field: "typo"}))


@pytest.mark.parametrize("enabled", [False, True])
def test_round3_is_scoped_to_lift_first_and_preserves_baseline_calls(monkeypatch, enabled):
    from contextlib import nullcontext
    from dojo.skills import basket_first as B, basket_transfer as T

    obj = NS(asset=NS(mesh=([], [])), R=np.eye(3), bottom_z=0.7)
    calls = []
    sk = NS(
        refresh=lambda: NS(obj=lambda _: obj),
        kin=NS(base=lambda _: np.eye(4)),
        belt_first_recover="on" if enabled else "off",
        belt_first_phase="on" if enabled else "off",
    )
    monkeypatch.setattr(B, "profile", lambda _: nullcontext())
    monkeypatch.setattr(B, "end_grasps", lambda *args, **kwargs: [])
    monkeypatch.setattr(I, "_intercept_search", lambda *args: "plan")
    monkeypatch.setattr(I, "target_lift", lambda _: 0.125)
    monkeypatch.setattr(I, "pick_from_belt", lambda *args, **kwargs:
                        calls.append(("pick", args[1], kwargs)) or NS(ok=True))
    monkeypatch.setattr(B, "level_hold", lambda *args: (True, 0.9, 0.0))
    monkeypatch.setattr(B, "translate", lambda *args, **kwargs:
                        calls.append(("translate", args[3])) or True)
    monkeypatch.setattr(B, "drop_center", lambda *args:
                        calls.append(("legacy_drop",)) or NS(ok=True))
    monkeypatch.setattr(T, "choose_arms", lambda *args:
                        calls.append(("phase",)) or ("right", "left"))
    monkeypatch.setattr(T, "pick_target", lambda *args:
                        calls.append(("retry",)) or NS(ok=True))
    monkeypatch.setattr(T, "drop_inset", lambda *args:
                        calls.append(("inset",)) or NS(ok=True))
    monkeypatch.setattr(T, "lift_loaded", lambda *args:
                        calls.append(("dual",)) or True)
    assert B.lift_first(sk, "basket", "target", np.array([0.004, 0.0]), ("conveyor",))["ok"]
    kinds = [c[0] for c in calls]
    if enabled:
        assert kinds == ["phase", "pick", "retry", "translate", "inset", "dual"]
    else:
        assert kinds == ["pick", "pick", "translate", "legacy_drop", "translate"]
        assert calls[1][2] == dict(lift=0.125, arms=["right"], ignore=("conveyor",))
        assert calls[-1] == ("translate", "loaded basket lift")


def test_retry_keeps_basket_arm_closed_and_restores_absent_flag_on_error(monkeypatch):
    from dojo.skills import basket_transfer as T

    calls = []
    sk = NS(notes_belt_method="waist", home=lambda **kwargs: calls.append(kwargs),
            refresh=lambda: NS(obj=lambda _: NS(top_z=0.1, bottom_z=0.0)),
            ex=NS(steps_used=0))
    def pick(*args, **kwargs):
        if "skip_methods" in kwargs:
            assert sk.belt_waist_first == "on"
            assert kwargs["skip_methods"] == ("waist",)
            raise RuntimeError("abort retry")
        assert kwargs["ignore"] == ("conveyor", "container")
        return NS(ok=False, status="grasp_slip")
    monkeypatch.setattr(I, "pick_from_belt", pick)
    monkeypatch.setattr(I, "target_lift", lambda _: 0.125)
    with pytest.raises(RuntimeError, match="abort retry"):
        T.pick_target(sk, "object", "container", np.array([0.004, 0.0]), "left", ("conveyor",))
    assert calls == [dict(arms=["left"], gripper=1.0)]
    assert not hasattr(sk, "belt_waist_first")


def test_phase_prediction_reserves_the_reachable_arm(monkeypatch):
    from dojo.skills import basket_transfer as T

    calls = []
    sk = NS()
    monkeypatch.setattr(I, "target_lift", lambda _: 0.125)
    monkeypatch.setattr(I, "plan_intercept", lambda *args, **kwargs:
                        calls.append(kwargs) or ("right",))
    assert T.choose_arms(sk, "object", "container", np.array([0.004, 0.0]), ()) == ("left", "right")
    assert calls == [dict(t_min=75, t_max=225, lift=0.125, ignore=("container",))]


@pytest.mark.parametrize("slip_stage", ["squeeze check", "fixed wrist inset carry"])
def test_inset_attachment_guards_never_open_after_slip(monkeypatch, slip_stage):
    from dojo.skills import basket_transfer as T
    from dojo.skills.motion import MoveResult

    vertices = np.array([[x, y, z] for x in (-0.01, 0.01)
                         for y in (-0.01, 0.01) for z in (-0.01, 0.01)])
    target = NS(T=np.eye(4), asset=NS(mesh=(vertices, [])), center_w=np.zeros(3))
    c = NS(T=np.eye(4), R=np.eye(3), pos=np.zeros(3),
           asset=NS(mesh=(vertices, []), bbox_min=np.full(3, -0.2), bbox_max=np.full(3, 0.2)))
    executed = []
    def run(plans, name):
        executed.append((name, plans))
        if name == slip_stage:
            target.T[0, 3] += 0.06
    sk = NS(
        refresh=lambda: NS(obj=lambda name: {"box": c, "object": target}[name],
                           arms={"left": NS(q=np.zeros(6))}),
        held={"left": NS(label="object")}, log=lambda _: None,
        kin=NS(tcp=lambda *args: np.eye(4), base=lambda _: np.eye(4)),
        ex=NS(q={"left": np.zeros(6)}, run=run),
        motion=NS(line=lambda *args, **kwargs: MoveResult("reached", "left")),
    )
    monkeypatch.setattr(T.G, "box_rim_z", lambda _: 0.1)
    monkeypatch.setattr(T, "held_box", lambda *args: None)
    result = T.drop_inset(sk, "left", "object", "box", ())
    assert not result.ok and result.status == "grasp_slip"
    assert all(op[0] != "grip" for _, plans in executed
               for plan in plans.values() for op in plan.ops)


@pytest.mark.parametrize("ik_ok", [False, True])
def test_dual_lift_queues_are_synchronous_and_refuse_failed_ik(monkeypatch, ik_ok):
    from dojo.skills import basket_transfer as T
    from dojo.skills.motion import MoveResult

    c = NS(T=np.eye(4), R=np.eye(3), bottom_z=0.9)
    grasp = NS(T=np.eye(4))
    q = {arm: np.zeros(6) for arm in ("left", "right")}
    queues = []
    def run_lists(lists, **kwargs):
        queues.append(lists)
        c.bottom_z += 0.1
        c.T[2, 3] += 0.1
    sk = NS(log=lambda _: None,
        refresh=lambda: NS(obj=lambda _: c, arms={a: NS(q=v) for a, v in q.items()}),
        kin=NS(base=lambda _: np.eye(4), tcp=lambda *args: np.eye(4),
               ik=lambda *args: NS(ok=ik_ok, q=np.zeros(6))),
        ex=NS(q=q, run=lambda *args, **kwargs: None,
              run_lists=run_lists, hold=lambda _: None),
        motion=NS(
            to_pose=lambda *args, **kwargs: MoveResult("reached", "left"),
            line=lambda *args, **kwargs: MoveResult("reached", "left"),
        ),
    )
    monkeypatch.setattr(T.G, "box_rim_grasps", lambda *args, **kwargs: [grasp])
    assert T.lift_loaded(sk, "container", "object", "right", "left", ()) is ik_ok
    if ik_ok:
        assert len(queues) == 1
        assert len(queues[0]["left"]) == len(queues[0]["right"]) == 40
        assert all(g == 0.0 for arm in queues[0].values() for _, g in arm)
    else:
        assert queues == []


def test_end_wall_selection_preserves_default_and_can_mirror(monkeypatch):
    from dojo.skills import basket_first as B

    grasps = []
    for x in (-0.15, 0.15):
        t = np.eye(4)
        t[0, 3] = x
        grasps.append(NS(T=t))
    monkeypatch.setattr(B.G, "box_rim_grasps", lambda *args, **kwargs: grasps)
    sk = NS(kin=object())
    obj = NS(center_w=np.zeros(3))
    assert B.end_grasps(sk, obj, "right", np.array([1., 0., 0.])) == [grasps[1]]
    assert B.end_grasps(sk, obj, "left", np.array([1., 0., 0.]), side=-1) == [grasps[0]]


@pytest.mark.parametrize("lead", [0, 3])
def test_approach_settle_preserves_contact_phase_and_default_schedule(lead):
    from dojo.skills.executor import EpisodeEnded

    events = []
    steps = [0]
    def hold(n):
        events.append(("hold", n))
        steps[0] += n
        ex.steps_used = steps[0]
    def run_lists(plans, name):
        events.append((name, len(plans["left"])))
        if name == "belt pick object":
            raise EpisodeEnded("stop before contact")
        steps[0] += len(plans["left"])
        ex.steps_used = steps[0]
    ex = NS(steps_used=0, hold=hold, run_lists=run_lists)
    # Zero belt speed isolates scheduling from tracking correction.
    obj = NS(pos=np.zeros(3))
    sk = NS(refresh=lambda: NS(obj=lambda _: obj), ex=ex, log=lambda _: None)
    g = NS(method="top", note="mock")
    plan = ("left", 20, [(np.zeros(6), 1.0)] * 5, [], g, [], 1.0)
    result = I.pick_from_belt(sk, "object", np.zeros(2), planned_intercept=plan,
                              approach_settle=lead)
    assert result.status == "ended"
    assert ex.steps_used == 12  # Contact minus the unchanged eight-step descent.
    assert events == ([("hold", 7), ("belt approach", 5), ("belt pick object", 0)]
                      if lead == 0 else
                      [("hold", 4), ("belt approach", 5), ("hold", 3),
                       ("belt pick object", 0)])


@pytest.mark.parametrize("height,expected", [(0.024, 24), (0.05, 0)])
def test_recovery_limits_timing_to_thin_riders_and_pull_to_unexecuted_misses(monkeypatch, height, expected):
    from dojo.skills import basket_transfer as T

    calls = []
    sk = NS(refresh=lambda: NS(obj=lambda _: NS(top_z=height, bottom_z=0.0)),
            ex=NS(steps_used=0))
    def pick(*args, **kw):
        calls.append(kw)
        return NS(ok=False, status="ik_fail")
    monkeypatch.setattr(I, "pick_from_belt", pick)
    monkeypatch.setattr(I, "target_lift", lambda _: 0.125)
    T.pick_target(sk, "object", "box", np.zeros(2), "left", ())
    assert len(calls) == 2
    assert calls[0].get("approach_settle", 0) == expected
    assert calls[1]["pull"] == 0.10
    calls.clear()
    def executed(*args, **kw):
        calls.append(kw)
        sk.ex.steps_used += 1
        return NS(ok=False, status="ik_fail")
    monkeypatch.setattr(I, "pick_from_belt", executed)
    T.pick_target(sk, "object", "box", np.zeros(2), "left", ())
    assert len(calls) == 1
