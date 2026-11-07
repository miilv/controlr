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
    assert SKILL_ATTRS["headset_stage"] == ("off", "center", "mesh_center", "mesh_clear")
    assert Skills(KinematicEnv([]), SimpleNamespace(refresh=lambda: WorldState("", "", {}, {}, {}))).headset_stage == "off"
    with pytest.raises(ValueError, match="bad runtime config"):
        RuntimePolicy(RuntimeConfig(headset_stage="invalid"), assets=object())
    monkeypatch.setattr(headset, "stage_for_cradle", lambda *a: pytest.fail("unexpected staging"))
    monkeypatch.setattr(library, "_hang_on_hooks", lambda *a, **kw: kw)
    for flag, tag in [("off", "headband_bridge"), ("center", "handle")]:
        sk = SimpleNamespace(headset_stage=flag, coll=SimpleNamespace(held_skip={}))
        assert library.hang_on_hooks(sk, [], "stand", "cradle", tag) == {}


@pytest.mark.parametrize("mode", ["center", "mesh_center", "mesh_clear"])
def test_stage_result_scopes_cup_bank_and_mesh_guard(monkeypatch, mode):
    monkeypatch.setattr(headset, "stage_for_cradle", lambda *a: {"headset"})
    seen = []
    def hang(sk, *a, **kw):
        seen.append((sk.coll.held_table, kw))
        return {"ok": True}
    monkeypatch.setattr(library, "_hang_on_hooks", hang)
    sk = SimpleNamespace(headset_stage=mode, headset_table="mesh",
                         coll=SimpleNamespace(held_table="bbox", held_skip={}))
    assert library.hang_on_hooks(sk, ["headset"], "stand", "cradle", "headband_bridge") == {"ok": True}
    assert seen == [("mesh", {"cup_labels": {"headset"}})]
    assert sk.coll.held_table == "bbox"


def test_failed_sites_follow_object_motion_and_preserve_other_arm_and_axes():
    from dojo.skills.geometry import inv_T
    from dojo.skills.grasps import Grasp

    sites = headset.GraspSites()
    local = T_from([.04, 0, -.05])
    pick = SimpleNamespace(label="headset", arm="right", T_tcp_obj=inv_T(local))
    sites.remember_pick(SimpleNamespace(), pick)
    obj = SimpleNamespace(label="headset", T=T_from([.2, -.1, .8], rot_z(.6)))
    same = Grasp("antipodal", obj.T @ local, .025)
    near = Grasp("antipodal", obj.T @ T_from([.045, 0, -.05]), .025)
    far = Grasp("antipodal", obj.T @ T_from([.06, 0, -.05]), .025)
    different_axis = Grasp("antipodal", obj.T @ local @ T_from([0, 0, 0], rot_z(.8)), .025)
    finger_flip = same.T.copy()
    finger_flip[:3, 1:3] *= -1
    flip = Grasp("antipodal", finger_flip, .025)
    assert sites.filter(obj, "right", [same, near, far, different_axis, flip]) == [far, different_axis]
    assert sites.filter(obj, "left", [same]) == [same]


def test_sites_record_the_actual_refined_pick_attachment():
    from dojo.skills.geometry import inv_T

    sites = headset.GraspSites()
    planned = SimpleNamespace(label="headset", arm="left", T_tcp_obj=np.eye(4))
    actual = SimpleNamespace(label="headset", arm="left", T_tcp_obj=T_from([.03, 0, .02]))
    sites.remember_pick(SimpleNamespace(last_pick=actual), planned)
    assert np.allclose(sites.rejected[("headset", "left")][0], inv_T(actual.T_tcp_obj))


def test_site_staging_caps_physical_attempts_without_relaxing_verifier(monkeypatch):
    pick = SimpleNamespace(label="headset", arm="right", T_tcp_obj=np.eye(4))
    sk, calls = stage_fixture(monkeypatch, outcomes=[(pick, "place")] * 4)
    sk.headset_retry = "sites"
    def fail(*a, **kw):
        calls.append(("run", kw))
        return SkillResult(False, "transfer", "loose")
    sk.run_transfer = fail
    assert not headset.stage_for_cradle(sk, ["headset"], "stand", "cradle", "bridge")
    assert [c[0] for c in calls] == ["plan", "run", "bank"] * 4
    assert all(c[1]["tol_xy"] == .015 for c in calls if c[0] == "run")


@pytest.mark.parametrize("status", ["place_miss", "step_budget", "ended"])
def test_site_staging_does_not_retry_other_failures(monkeypatch, status):
    sk, calls = stage_fixture(monkeypatch, outcomes=[("pick", "place")])
    sk.headset_retry = "sites"
    sk.run_transfer = lambda *a, **kw: SkillResult(False, "transfer", status)
    assert not headset.stage_for_cradle(sk, ["headset"], "stand", "cradle", "bridge")
    assert [c[0] for c in calls] == ["plan", "bank"]


def test_retry_defaults_and_cup_mode_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.env import KinematicEnv
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.world import WorldState

    assert RuntimeConfig().headset_retry == "legacy"
    assert SKILL_ATTRS["headset_retry"] == ("legacy", "sites")
    sk = Skills(KinematicEnv([]), SimpleNamespace(refresh=lambda: WorldState("", "", {}, {}, {})))
    assert sk.headset_retry == "legacy" and sk.headset_grasp == "legacy"
    RuntimePolicy(RuntimeConfig(headset_grasp="cups", headset_retry="sites"), assets=object())
    with pytest.raises(ValueError, match="bad runtime config"):
        RuntimePolicy(RuntimeConfig(headset_retry="invalid"), assets=object())


@pytest.mark.parametrize('retry,category,expected', [
    ('legacy', 'headset', 'place'), ('sites', 'other', 'place'),
    ('sites', 'headset', 'grasp_unusable'),
])
def test_measured_replan_failure_never_uses_nominal_carry_in_headset_site_mode(retry, category, expected):
    from dojo.skills.skills import Skills
    events = []
    obj = SimpleNamespace(category=category, asset=SimpleNamespace(extents=np.ones(3)*.1))
    nominal = SimpleNamespace(T_obj_target=np.eye(4), transfer_kw={}, plan=object())
    pp = SimpleNamespace(label='item', arm='right')
    def carry(plan, *a, **kw):
        assert plan is nominal
        events.append('carry')
        return SkillResult(True, 'place')
    def put_down(arm):
        events.append('put_down')
        return SkillResult(True, 'regrasp')
    sk = SimpleNamespace(headset_retry=retry, _record=lambda r:r,
                         run_pick=lambda p:SkillResult(True,'pick'), refine=None,
                         refresh=lambda:None, world=SimpleNamespace(obj=lambda l:obj),
                         plan_place=lambda *a,**kw:SkillResult(False,'place','ik_fail',note='blocked'),
                         regrasp=put_down, held={'left':None,'right':None},
                         _park_if_blocking=lambda *a,**kw:None, run_place=carry)
    r = Skills.run_transfer(sk, pp, nominal)
    assert (r.status if expected=='grasp_unusable' else r.skill) == expected
    assert events == (['put_down'] if expected=='grasp_unusable' else ['carry'])


def test_cup_opening_encloses_physical_envelope_without_changing_legacy(monkeypatch):
    from dojo.skills import grasps as G
    monkeypatch.setattr(library, 'functional_frames', lambda *a,**kw:[])
    obj = SimpleNamespace(asset=SimpleNamespace(mesh=None))
    for profile,expected in [('dense',.048),('cups',.088)]:
        grasp = G.Grasp('antipodal',np.eye(4),.026)
        monkeypatch.setattr(headset,'dense_grasps',lambda *a:[grasp])
        out = headset.cup_grasps(SimpleNamespace(headset_grasp=profile),obj,'right')
        assert out[0] is grasp and grasp.width == .026
        assert grasp.open_gap() == pytest.approx(expected)


def test_mesh_rest_seats_real_geometry_without_changing_xy_or_orientation():
    from dojo.skills.kinematics import TABLE_Z
    target = T_from([.1,-.2,.85],rot_z(.5))
    vertices = np.array([[-.05,-.02,-.04],[.05,.02,.08]])
    obj = SimpleNamespace(asset=SimpleNamespace(mesh=(vertices, np.empty((0,3)))))
    out = headset.mesh_support_target(obj,target)
    world = vertices @ out[:3,:3].T + out[:3,3]
    assert world[:,2].min() == pytest.approx(TABLE_Z+.003)
    assert np.array_equal(out[:2,3],target[:2,3]) and np.array_equal(out[:3,:3],target[:3,:3])
    assert target[2,3] == .85


@pytest.mark.parametrize('mesh',[None,([],[]),([[0,0,float('nan')]],[]),([[0,0]],[]),('bad',[])])
def test_invalid_support_mesh_retains_bbox_target(mesh):
    target=np.eye(4)
    assert headset.mesh_support_target(SimpleNamespace(asset=SimpleNamespace(mesh=mesh)),target) is target


def test_slow_lift_is_opt_in_and_preserves_pick_verifier_defaults():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.env import KinematicEnv
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.world import WorldState
    sk = Skills(KinematicEnv([]), SimpleNamespace(refresh=lambda:WorldState("", "", {}, {}, {})))
    assert sk.headset_lift == RuntimeConfig().headset_lift == "legacy"
    assert SKILL_ATTRS["headset_lift"] == ("legacy", "slow")
    assert headset.cradle_pick_kwargs(sk, seat=False) == {}
    assert headset.cradle_pick_kwargs(sk, seat=True) == {"squeeze":8}
    sk.headset_lift = "slow"
    assert headset.cradle_pick_kwargs(sk, seat=False) == {"lift_rad":.03}
    assert headset.cradle_pick_kwargs(sk, seat=True) == {"squeeze":8, "lift_rad":.03}
    RuntimePolicy(RuntimeConfig(headset_lift="slow"), assets=object())
    with pytest.raises(ValueError, match="bad runtime config"):
        RuntimePolicy(RuntimeConfig(headset_lift="invalid"), assets=object())


@pytest.mark.parametrize("mode", ["off", "center", "mesh_center"])
def test_existing_stage_modes_preserve_the_original_transfer_target(mode):
    target = T_from([.1, -.2, .8])
    seen = []
    result = object()
    sk = SimpleNamespace(headset_stage=mode,
                         plan_transfer=lambda *a, **kw: seen.append((a, kw)) or result)
    assert headset.plan_stage(sk, "headset", target, arms=["right"]) is result
    assert seen[0][0][1][0] is target
    assert seen[0][1] == {"arms": ["right"]}


def clear_stage_fixture(monkeypatch, failed_leg=None):
    from dojo.skills import motion
    from dojo.skills.skills import PickPlan, PlacePlan

    monkeypatch.setattr(motion, "held_box", lambda *a: "box")
    ground = T_from([.1, -.2, .8], rot_z(.4))
    failure = SkillResult(False, "transfer", "ik_fail", note="native refusal")
    calls, picks = [], []

    def screen(label, targets, **kw):
        calls.append(("screen", targets[0].copy(), kw))
        if failed_leg == "screen":
            return failure
        pick = PickPlan("right", SimpleNamespace(q_end=np.ones(6)), object(),
                        T_from([0, 0, -.04]), 40, label)
        picks.append(pick)
        return pick, PlacePlan("right", object(), targets[0], 30, label)

    def place(arm, target, **kw):
        calls.append(("place", target.copy(), kw))
        return failure if failed_leg == "place" else PlacePlan(arm, object(), target, 30, "headset")

    sk = SimpleNamespace(headset_stage="mesh_clear", plan_transfer=screen, plan_place=place,
                         world=SimpleNamespace(obj=lambda l: object()), held={"right": None})
    return sk, ground, calls, picks, failure


def test_clear_stage_checks_release_height_but_repairs_and_verifies_ground(monkeypatch):
    sk, ground, calls, picks, _ = clear_stage_fixture(monkeypatch)
    original = ground.copy()
    pp, pl = headset.plan_stage(sk, "headset", ground, arms=["right"], ignore=("stand",),
                                pick_kw={"squeeze": 8}, place_kw={"approach_line": True})
    assert np.array_equal(ground, original)
    assert calls[0][1][2, 3] == pytest.approx(ground[2, 3] + .025)
    assert calls[0][2]["place_kw"] == {"approach_line": True, "release_gap": 0.0}
    assert np.array_equal(calls[1][1], ground) and pl.T_obj_target is ground
    assert calls[1][2]["held"].T_tcp_obj is pp.T_tcp_obj
    assert pl.transfer_kw == {"ignore": ("stand",), "approach_line": True, "release_gap": .025}
    assert sk.held == {"right": None}  # planning never manufactures a physical hold
    q = {"right": np.arange(6)}
    new_pick, new_place = pp.repair(q)
    assert new_pick is not pp and new_pick is picks[-1]
    assert np.array_equal(new_place.T_obj_target, ground)
    assert new_place.transfer_kw["release_gap"] == .025
    assert calls[-2][2]["q_start"] is q
    assert calls[-2][2]["arms"] == ["right"]
    assert calls[-2][1][2, 3] == pytest.approx(ground[2, 3] + .025)


@pytest.mark.parametrize("failed_leg", ["screen", "place"])
def test_clear_stage_propagates_native_refusal_without_a_raised_goal_fallback(monkeypatch, failed_leg):
    sk, ground, calls, picks, failure = clear_stage_fixture(monkeypatch, failed_leg)
    assert headset.plan_stage(sk, "headset", ground) is failure
    assert len(calls) == (1 if failed_leg == "screen" else 2)
    assert all(p.repair is None for p in picks)
    assert sk.held == {"right": None}


def test_clear_stage_is_an_explicit_runtime_choice():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy

    assert RuntimeConfig().headset_stage == "off"
    RuntimePolicy(RuntimeConfig(headset_stage="mesh_clear"), assets=object())
