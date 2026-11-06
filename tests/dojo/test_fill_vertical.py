"""Vertical pen pinches: rolled widths, narrow entry, table clearance."""
import math

import numpy as np
import pytest

from dojo.skills.geometry import R_to_quat, axis_angle_R
from dojo.skills.kinematics import X5Kin
from dojo.skills.library import _fill_vertical_grasps
from dojo.skills.world import AssetInfo, ObjectState


@pytest.mark.parametrize("roll", [0, 26, -26])
def test_rolled_pen_pinches_are_horizontal_and_cover_projected_width(roll):
    half = np.array([0.079, 0.008, 0.010])
    asset = AssetInfo(key="Rigid/pen/00000", bbox_min=-half, bbox_max=half)
    R = axis_angle_R(np.array([1., 0, 0]), math.radians(roll))
    o = ObjectState("p", "Rigid", "pen", 0, np.array([-.2, -.2, .779]), R_to_quat(R), asset)
    kin = X5Kin()
    gs = _fill_vertical_grasps(o, kin, "left")
    assert len(gs) >= 10
    assert len({round(g.T[0, 3], 4) for g in gs}) >= 5
    for g in gs:
        assert g.approach[2] == pytest.approx(-1.)
        assert g.T[2, 1] == pytest.approx(0.)
        projection = o.world_bbox_corners() @ g.T[:3, 1]
        assert g.width == pytest.approx(np.ptp(projection))
        assert g.open_gap() == pytest.approx(g.width + .008)
        assert kin.finger_points(kin.flange_from_tcp(g.T), 0.)[:, 2].min() >= o.bottom_z + .0039


def test_vertical_switch_defaults_and_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().fill_grasp == "off"
    assert SKILL_ATTRS["fill_grasp"] == ("off", "vertical", "rolled", "rolled_tilt", "rolled_relay", "rolled_stage")
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(fill_grasp="typo"), assets=None)


@pytest.mark.parametrize("field,value", [("fill_holder", "stay"), ("fill_drop", "center"), ("fill_insert", "safe")])
def test_other_fill_switches_validate(field, value):
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert getattr(RuntimeConfig(), field) == "off"
    assert SKILL_ATTRS[field][0] == "off" and value in SKILL_ATTRS[field]
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(**{field: "typo"}), assets=None)


@pytest.mark.parametrize("fail", [False, True])
def test_safe_insert_restores_joint_range(monkeypatch, fail):
    from types import SimpleNamespace
    from dojo.skills import library
    sk = SimpleNamespace(fill_insert="safe", fill_joint_limits="inherit", joint_limits="wide")

    def inner(*args):
        assert sk.joint_limits == "measured"
        if fail:
            raise RuntimeError("test")
        return {"ok": True}

    monkeypatch.setattr(library, "_fill_container", inner)
    if fail:
        with pytest.raises(RuntimeError):
            library.fill_container(sk, ["p"], "h")
    else:
        assert library.fill_container(sk, ["p"], "h")["ok"]
    assert sk.joint_limits == "wide"


def test_center_drop_uses_maximum_clearance_without_relaxing_occupied_space(monkeypatch):
    from types import SimpleNamespace
    from dojo.skills import insertion, library
    calls = []

    def fit(*args, **kw):
        calls.append(kw)
        return None

    sk = SimpleNamespace(fill_drop="center", world=SimpleNamespace(objects={"p": 1, "h": 2, "done": 3},
                                                                 obj=lambda l: l))
    monkeypatch.setattr(insertion, "fit_insertion", fit)
    assert library._fill_drop_target(sk, "p", "h", ["done"]) == (None, None)
    assert len(calls) == 1
    assert calls[0]["others"] == ["done"]
    assert calls[0]["pack"] is False
    assert calls[0]["others_pad"] == .004


def test_vertical_preference_retains_original_candidates_when_none_are_available(monkeypatch):
    from types import SimpleNamespace
    from dojo.skills import library
    half = np.array([.079, .008, .01])
    o = ObjectState("p", "Rigid", "pen", 0, np.array([-.2, -.2, .779]), np.array([1., 0, 0, 0]),
                    AssetInfo(key="Rigid/pen/00000", bbox_min=-half, bbox_max=half))
    sk = SimpleNamespace(fill_grasp="off", world=SimpleNamespace(obj=lambda _: o), kin=X5Kin())
    original = library._fingers_above(sk, "p", np.eye(4), -np.inf)("left")
    assert original
    sk.fill_grasp = "vertical"
    monkeypatch.setattr(library, "_fill_vertical_grasps", lambda *args: [])
    fallback = library._fingers_above(sk, "p", np.eye(4), -np.inf)("left")
    assert [g.note for g in fallback] == [g.note for g in original]
    assert [g.score for g in fallback] == pytest.approx([g.score + 2 for g in original])


def test_stay_holder_bypasses_container_transfer(monkeypatch):
    from types import SimpleNamespace
    from dojo.skills import library
    pen = SimpleNamespace(center_w=np.array([-.2, -.2, .78]), asset=SimpleNamespace(extents=np.array([.15, .01, .01])))
    holder = SimpleNamespace(obj_type="Rigid", center_w=np.array([2., 2., .8]))
    world = SimpleNamespace(objects={"p": pen, "h": holder}, obj=lambda l: {"p": pen, "h": holder}[l], notes={})
    sk = SimpleNamespace(fill_holder="stay", refresh=lambda: world, kin=X5Kin(), coll=SimpleNamespace(include=set()))
    monkeypatch.setattr(library, "free_spots", lambda *a, **k: pytest.fail("holder moved"))
    monkeypatch.setattr(library, "in_tip_down", lambda *a: True)
    assert library._fill_container(sk, ["p"], "h")["ok"]


@pytest.mark.parametrize("roll,active", [(0, False), (26, True), (-26, True)])
def test_rolled_mode_preserves_flat_grasps_and_targets_rolled_pen(roll, active):
    from types import SimpleNamespace
    from dojo.skills import library
    half = np.array([.079, .008, .010])
    R = axis_angle_R(np.array([1., 0, 0]), math.radians(roll))
    o = ObjectState("p", "Rigid", "pen", 0, np.array([-.2, -.2, .779]), R_to_quat(R),
                    AssetInfo(key="Rigid/pen/00000", bbox_min=-half, bbox_max=half))
    sk = SimpleNamespace(fill_grasp="off", world=SimpleNamespace(obj=lambda _: o), kin=X5Kin())
    original = library._fingers_above(sk, "p", np.eye(4), -np.inf)("left")
    sk.fill_grasp = "rolled"
    gs = library._fingers_above(sk, "p", np.eye(4), -np.inf)("left")
    assert any(g.note.startswith("vertical") for g in gs) is active
    if not active:
        assert [g.note for g in gs] == [g.note for g in original]
        assert [g.score for g in gs] == [g.score for g in original]


def test_tilted_approaches_keep_horizontal_jaws_and_narrow_opening():
    half = np.array([.079, .008, .010])
    R = axis_angle_R(np.array([1., 0, 0]), math.radians(26))
    o = ObjectState("p", "Rigid", "pen", 0, np.array([-.2, -.2, .779]), R_to_quat(R),
                    AssetInfo(key="Rigid/pen/00000", bbox_min=-half, bbox_max=half))
    gs = _fill_vertical_grasps(o, X5Kin(), "left", tilts=(0., 25., 40.))
    assert len(gs) == 30
    assert {round(math.degrees(math.acos(-g.approach[2]))) for g in gs} == {0, 25, 40}
    for g in gs:
        assert g.T[2, 1] == pytest.approx(0.)
        assert g.open_gap() == pytest.approx(g.width + .008)


def test_inset_drop_keeps_occupied_space_in_both_fit_attempts(monkeypatch):
    from types import SimpleNamespace
    from dojo.skills import insertion, library
    calls = []
    monkeypatch.setattr(insertion, "fit_insertion", lambda *a, **kw: calls.append(kw))
    sk = SimpleNamespace(fill_drop="inset", world=SimpleNamespace(objects={"done": 3}, obj=lambda l: l))
    assert library._fill_drop_target(sk, "p", "h", ["done"]) == (None, None)
    assert len(calls) == 2
    assert all(kw["others"] == ["done"] for kw in calls)
    assert calls[0]["min_clear"] == .007
    assert calls[0]["pack"] is True
    assert calls[1]["pack"] is False


def test_rolled_relay_excludes_original_pinches_only_for_rolled_objects():
    from types import SimpleNamespace
    from dojo.skills import library
    half = np.array([.079, .008, .010])
    o = ObjectState("p", "Rigid", "pen", 0, np.array([-.2, -.2, .779]),
                    R_to_quat(axis_angle_R(np.array([1., 0, 0]), math.radians(26))),
                    AssetInfo(key="Rigid/pen/00000", bbox_min=-half, bbox_max=half))
    sk = SimpleNamespace(fill_grasp="rolled_relay", world=SimpleNamespace(obj=lambda _: o), kin=X5Kin())
    gs = library._fingers_above(sk, "p", np.eye(4), -np.inf)("left")
    assert len(gs) == 30 and all(g.note.startswith("vertical") for g in gs)
    o.quat = np.array([1., 0, 0, 0])
    gs = library._fingers_above(sk, "p", np.eye(4), -np.inf)("left")
    assert gs and any(g.method == "box" for g in gs)


@pytest.mark.parametrize("relay_kw", [None, {"release_gap": .020, "drop_ok": True}])
def test_intermediate_relay_options_reach_planning_and_execution_only(monkeypatch, relay_kw):
    from types import SimpleNamespace
    from dojo.skills import library
    from dojo.skills.skills import SkillResult
    o = SimpleNamespace(R=np.eye(3), asset=SimpleNamespace(extents=np.array([.15, .01, .01])))
    sk = SimpleNamespace(refresh=lambda: None, plan_transfer=lambda *a, **k: SkillResult(False, "transfer", "ik_fail"),
                         world=SimpleNamespace(obj=lambda _: o), log=lambda _: None, bank=lambda: None)
    monkeypatch.setattr(library, "rests_stably", lambda *a: True)
    monkeypatch.setattr(library, "free_spots", lambda *a, **k: [(0., 0.)])
    monkeypatch.setattr(library, "rest_with_R", lambda *a: np.eye(4))
    seen_plan, seen_run = [], []
    plan = (SimpleNamespace(arm="left", grasp=SimpleNamespace(note="test")), SimpleNamespace(T_obj_target=np.eye(4)))

    def plan_leg(*a, **kw):
        seen_plan.append(kw.get("place_kw") if len(a) <= 5 else a[5])
        return plan

    def run_leg(*a, **kw):
        # Final leg passes place_kw positionally; intermediate leg uses the new option.
        seen_run.append(kw.get("place_kw") if len(a) <= 5 else a[5])
        return SkillResult(True, "place", "ok")

    monkeypatch.setattr(library, "_plan_leg", plan_leg)
    monkeypatch.setattr(library, "_run_leg", run_leg)
    final_kw = {"servo": True}
    assert library.place_oriented(sk, "p", [np.eye(4)], place_kw=final_kw, relay_place_kw=relay_kw).ok
    assert seen_plan == [relay_kw, final_kw]
    assert seen_run == [relay_kw, final_kw]
