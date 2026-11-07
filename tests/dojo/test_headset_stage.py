"""CPU-only contracts for optional checked headset staging."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import headset, library
from dojo.skills.geometry import T_from, rot_z
from dojo.skills.skills import SkillResult


def test_cup_bank_uses_bridge_coordinates_and_requires_broad_deep_contact(monkeypatch):
    from dojo.skills import grasps as G

    wide = object()
    obj = SimpleNamespace(T=T_from([.2, -.1, .9], rot_z(.7)), asset=SimpleNamespace(mesh=object()))
    bridge = T_from([0, 0, .1])
    monkeypatch.setattr(headset, "dense_grasps", lambda *a: [wide])
    monkeypatch.setattr(library, "functional_frames", lambda *a, **kw: [bridge])

    def grasp(width=.025, z=.02, patch=1., depth=.02):
        return SimpleNamespace(width=width, T=obj.T @ T_from([.03, 0, z]), note="test",
                               patch=patch, depth=depth)

    good = grasp()
    candidates = [grasp(width=.017), grasp(width=.035), grasp(z=.06),
                  grasp(patch=.4), grasp(depth=.005), good]
    seen = []

    def sample(*a, **kw):
        seen.append(kw)
        return candidates

    def contact(o, g):
        # pad_contact's public contract accepts a TCP matrix; associate the fixture.
        c = next(c for c in candidates if c.T is g)
        contact.depth = c.depth
        return c.patch, None

    monkeypatch.setattr(G, "antipodal_grasps", sample)
    monkeypatch.setattr(G, "pad_contact", contact)
    assert headset.cup_grasps(SimpleNamespace(kin=object()), obj, "right") == [wide, good]
    assert good.note == "cup test"
    assert seen == [{"n_samples": 800, "n_approach": 13}]


def stage_fixture(monkeypatch, *, obj_x=.2, hook_x=-.3, distance_y=.1, outcomes=()):
    obj = SimpleNamespace(pos=np.array([obj_x, -.2, .8]), R=np.eye(3))
    world = SimpleNamespace(objects={"headset": obj}, obj=lambda label: obj)
    H = T_from([hook_x, distance_y, 1.])
    monkeypatch.setattr(library, "hook_frames", lambda *a: [(0, H, None)])
    monkeypatch.setattr(library, "on_hook", lambda *a: None)
    monkeypatch.setattr(library, "free_spots", lambda *a, **kw: [(0., -.2)] * 12)
    monkeypatch.setattr(library, "rest_with_R", lambda sk, label, xy, R: T_from([*xy, .8], R))
    bases = {"left": T_from([-.3, -.45, .765]), "right": T_from([.3, -.45, .765])}
    calls = []
    plans = iter(outcomes)

    def plan(*a, **kw):
        calls.append(("plan", kw))
        return next(plans)

    sk = SimpleNamespace(world=world, refresh=lambda: world, kin=SimpleNamespace(base=bases.__getitem__),
                         plan_transfer=plan, log=lambda m: None,
                         run_transfer=lambda *a, **kw: calls.append(("run", kw)) or SkillResult(True, "place"),
                         bank=lambda: calls.append(("bank", None)))
    return sk, calls


@pytest.mark.parametrize("obj_x,hook_x,dy", [(-.2, -.3, .1), (.2, .15, .1), (.1, -.1, -.1)])
def test_stage_skips_same_arm_or_short_transfer(monkeypatch, obj_x, hook_x, dy):
    sk, calls = stage_fixture(monkeypatch, obj_x=obj_x, hook_x=hook_x, distance_y=dy)
    assert headset.stage_for_cradle(sk, ["headset"], "stand", "cradle", "bridge") == set()
    assert not calls


def test_stage_checks_plans_and_executes_only_one_transfer(monkeypatch):
    sk, calls = stage_fixture(monkeypatch, outcomes=[SkillResult(False, "plan"), ("pick", "place")])
    assert headset.stage_for_cradle(sk, ["headset"], "stand", "cradle", "bridge") == {"headset"}
    assert [c[0] for c in calls] == ["plan", "plan", "run", "bank"]
    kw = calls[1][1]
    assert kw["arms"] == ["right"] and kw["max_full"] == 24
    assert kw["pick_kw"] == {"squeeze": 8}
    assert kw["place_kw"]["approach_line"] is True
    assert calls[2][1] == {"tol_xy": .015, "tol_tilt": 25., "tol_z": .05}


def test_stage_limits_plan_refusals_and_preserves_physical_failure(monkeypatch):
    sk, calls = stage_fixture(monkeypatch, outcomes=[SkillResult(False, "plan")] * 8)
    assert not headset.stage_for_cradle(sk, ["headset"], "stand", "cradle", "bridge")
    assert len(calls) == 8 and all(c[0] == "plan" for c in calls)
    sk, calls = stage_fixture(monkeypatch, outcomes=[("pick", "place")])
    sk.run_transfer = lambda *a, **kw: SkillResult(False, "pick", "grasp_slip")
    assert not headset.stage_for_cradle(sk, ["headset"], "stand", "cradle", "bridge")
    assert [c[0] for c in calls] == ["plan", "bank"]


def test_staging_defaults_and_threaded_hooks_remain_unchanged(monkeypatch):
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.env import KinematicEnv
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.world import WorldState

    assert RuntimeConfig().headset_stage == "off"
    assert SKILL_ATTRS["headset_stage"] == ("off", "center")
    assert Skills(KinematicEnv([]), SimpleNamespace(refresh=lambda: WorldState("", "", {}, {}, {}))).headset_stage == "off"
    with pytest.raises(ValueError, match="bad runtime config"):
        RuntimePolicy(RuntimeConfig(headset_stage="invalid"), assets=object())
    monkeypatch.setattr(headset, "stage_for_cradle", lambda *a: pytest.fail("unexpected staging"))
    monkeypatch.setattr(library, "_hang_on_hooks", lambda *a, **kw: kw)
    for flag, tag in [("off", "headband_bridge"), ("center", "handle")]:
        sk = SimpleNamespace(headset_stage=flag, coll=SimpleNamespace(held_skip={}))
        assert library.hang_on_hooks(sk, [], "stand", "cradle", tag) == {}


def test_stage_result_scopes_cup_bank_and_mesh_guard(monkeypatch):
    monkeypatch.setattr(headset, "stage_for_cradle", lambda *a: {"headset"})
    seen = []
    def hang(sk, *a, **kw):
        seen.append((sk.coll.held_table, kw))
        return {"ok": True}
    monkeypatch.setattr(library, "_hang_on_hooks", hang)
    sk = SimpleNamespace(headset_stage="center", headset_table="mesh",
                         coll=SimpleNamespace(held_table="bbox", held_skip={}))
    assert library.hang_on_hooks(sk, ["headset"], "stand", "cradle", "headband_bridge") == {"ok": True}
    assert seen == [("mesh", {"cup_labels": {"headset"}})]
    assert sk.coll.held_table == "bbox"
