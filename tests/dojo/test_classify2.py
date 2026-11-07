"""Native classify grounding and default parity; CPU kinematic/mocked executor only."""
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.classify import assignment, classify
from dojo.skills.g3 import SKILL_ATTRS
from dojo.skills.planner import plan
from test_port_adapter import gt_skills, rows_classify

FIXED = "Put watch objects into the left basket, pen objects into the middle basket, and toy_car objects into the right basket."
FREE = "Sort the objects by category into the three baskets."


def test_defaults_and_validation():
    _, sk = gt_skills(rows_classify(), FIXED)
    for name in ("classify_route", "classify_carry", "classify_repair"):
        assert getattr(RuntimeConfig(), name) == getattr(sk, name) == SKILL_ATTRS[name][0] == "off"
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**{name: "invalid"}))
    w = sk.refresh()
    assert plan(FIXED, w, ported=True).args["skill"] == "classify_into_bins_v3b"
    assert plan(FIXED, w, ported={"classify_route": "native"}).skill == "classify_native"
    assert plan("Stack the objects.", w, ported={"classify_route": "native"}).skill != "classify_native"


def test_fixed_mapping_and_no_unknown_assignment():
    _, sk = gt_skills(rows_classify(), FIXED)
    w = sk.refresh()
    w.obj("cat1").identity_known = False
    mapping, groups, cost = assignment(w, sk.kin)
    assert mapping == {"watch": "basket0", "toy_car": "basket2"}
    assert "pen" not in groups and cost is None
    w.instruction = "Put unknown objects into the left basket."
    assert assignment(w, sk.kin)[0] == {}


def test_free_assignment_nearest_and_fixed_never_overridden():
    _, sk = gt_skills(rows_classify(), FREE)
    w = sk.refresh()
    mapping, _, cost = assignment(w, sk.kin)
    assert set(mapping) == {"watch", "pen", "toy_car"} and len(set(mapping.values())) == 3
    assert cost > 0
    w.instruction = FIXED.replace("left", "right").replace("toy_car objects into the right", "toy_car objects into the left")
    assert assignment(w, sk.kin)[0]["watch"] == "basket2"


def test_pick_refusals_bounded_and_reserve_restored(monkeypatch):
    from dojo.skills.skills import SkillResult
    _, sk = gt_skills(rows_classify(), FIXED, step_lim=1100)
    sk.classify_repair = "budget"
    seen = []
    def refused(label, **kw):
        seen.append((label, kw))
        return SkillResult(False, "pick", "no_plan", 0)
    monkeypatch.setattr(sk, "plan_pick", refused)
    reserve = sk.ex.reserve
    result = classify(sk)
    assert not result["ok"] and len(seen) == 6
    assert all(kw["method"] == "top" for _, kw in seen)
    assert sk.ex.reserve == reserve
    assert all(e["status"] == "pick_plan" for e in result["events"])


def test_conflicting_language_does_not_assign_two_categories_to_one_basket():
    _, sk = gt_skills(rows_classify(), FIXED.replace("middle", "left"))
    assert assignment(sk.refresh(), sk.kin)[0] == {}


def test_drop_refuses_without_executing_unchecked_legs(monkeypatch):
    from dojo.skills.classify import central_drop
    from dojo.skills.motion import MoveResult
    from dojo.skills.skills import Held
    from dojo.skills.geometry import inv_T
    from dojo.skills.motion import held_box
    _, sk = gt_skills(rows_classify(), FREE, step_lim=1100)
    w = sk.refresh()
    arm = "left"
    o = w.obj("cat0")
    rel = inv_T(sk.kin.tcp(arm, sk.ex.q[arm])) @ o.T
    sk.held[arm] = Held(o.label, rel, held_box(o, rel))
    calls = []
    def refuse(*args, **kwargs):
        calls.append(kwargs)
        return MoveResult("collision", arm)
    monkeypatch.setattr(sk.motion, "line", refuse)
    def forbidden(*args, **kwargs):
        pytest.fail("a refused central drop must execute no actions")
    monkeypatch.setattr(sk.ex, "run", forbidden)
    assert central_drop(sk, arm, "basket0").status == "no_plan"
    assert calls and all(c["held"] is sk.held[arm].box for c in calls)
