"""CPU contract for opt-in multi-bottle profiles (no simulator or model calls)."""
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.g3 import SKILL_ATTRS
from dojo.skills.planner import pour_bottle_call


def test_profile_default_preserves_legacy_dispatch():
    assert RuntimeConfig().ported_pour_profile == "off"
    assert SKILL_ATTRS["ported_pour_profile"] == ("off", "sparse", "via", "checked")
    for aim in ("v1", "b10", "b20"):
        for pace in ("v1", "lean", "overlap"):
            assert pour_bottle_call(aim, pace) == pour_bottle_call(aim, pace, "off")


@pytest.mark.parametrize("profile,via", [("sparse", False), ("via", True)])
def test_new_profile_dispatch_and_aim(profile, via):
    assert pour_bottle_call("b10", "v1", profile) == (
        "pour_bottle_v2", {"via": via, "spout_back": .01})


def test_bad_profile_is_rejected():
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(ported_pour_profile="bad"), assets=object())


def test_checked_uses_strict_wrappers_underlying_profile():
    assert pour_bottle_call("v1", "v1", "checked") == ("pour_bottle_v5", {})
    assert pour_bottle_call("b20", "v1", "checked") == ("pour_bottle_v5", {"spout_back": .02})


def test_diagnostic_boxes_follow_rotated_checker_frame():
    from dojo.skills.pour_diagnostics import box_mask
    q = [2**-.5, 0, 0, 2**-.5]
    vertices = [[-.1, -.02, 0], [.1, .02, .1]]
    assert box_mask([[1, 2.09, .05], [1.09, 2, .05], [1, 2, 0]],
                    [1, 2, 0], q, vertices, .001).tolist() == [True, False, False]


@pytest.mark.parametrize("profile,skill", [("sparse", "pour_bottle_v2"), ("via", "pour_bottle_v2")])
def test_ordered_route_forwards_profile_and_only_final_pour_homes(monkeypatch, profile, skill):
    import dojo.skills.ported as adapter
    from dojo.skills.pour_order import pour_in_order
    calls = []

    def fake(sk, name, args, reserve=None):
        calls.append((name, args))
        return {"ok": True, "verified": True}

    monkeypatch.setattr(adapter, "ported", fake)

    class Sk:
        ported_pour_profile = profile
        ported_pour_aim = "b10"

        def refresh(self):
            return None

        def log(self, message):
            pass

    assert pour_in_order(Sk(), [["a", "x"], ["b", "y"], ["c", "z"]])["ok"]
    assert [name for name, _ in calls] == [skill] * 3
    assert [args.get("home", True) for _, args in calls] == [False, False, True]
    assert [args["container"] for _, args in calls] == ["@a", "@b", "@c"]
    assert all(args["spout_back"] == .01 for _, args in calls)


def test_checked_order_uses_original_budget_and_refusal_composite(monkeypatch):
    import dojo.skills.ported as adapter
    from dojo.skills.pour_order import pour_in_order
    calls = []
    monkeypatch.setattr(adapter, "ported", lambda sk, name, args, reserve=None:
                        calls.append((name, args)) or {"ok": True, "verified": True})
    class Sk:
        ported_pour_profile, ported_pour_aim = "checked", "b10"
    assert pour_in_order(Sk(), [["a", "x"], ["b", "y"], ["c", "z"]])["verified"]
    assert calls == [("pour_in_order_v5", {"pairs": [["@a", "@x"], ["@b", "@y"], ["@c", "@z"]],
                                          "pour_args": {"spout_back": .01}})]


@pytest.mark.parametrize("profile,homes", [("off", True), ("checked", False)])
def test_checked_held_return_suppresses_generic_open_without_changing_default(monkeypatch, profile, homes):
    from types import SimpleNamespace
    from dojo.skills import planner
    monkeypatch.setattr(planner, "plan", lambda *a, **kw: SimpleNamespace(skill="pour_in_order", args={}, why="test"))
    monkeypatch.setattr(planner.L, "skill_fn", lambda name: lambda sk: {"ok": True, "holding": {"arm": "left"}})
    finished = []
    sk = SimpleNamespace(ported_pour_profile=profile, refresh=lambda: SimpleNamespace(instruction="pour", notes={}),
                         log=lambda m: None, finish=lambda: finished.append(True) or SimpleNamespace(status="ok"))
    result = planner.solve(sk)
    assert bool(finished) is homes
    if not homes:
        assert result["stage"] == "holding"


def test_snapshot_queries_checker_without_changing_episode():
    from types import SimpleNamespace
    import numpy as np
    from dojo.skills.pour_diagnostics import snapshot
    calls = []
    fluid = SimpleNamespace(init_ori=[1, 0, 0, 0], init_pos=[0, 0, 0],
                            get_particle_positions=lambda: (np.array([[0, 0, .05]]), None, None))
    lm = SimpleNamespace(get_instance_name=lambda label, env_idx: label,
                         get_scene_object=lambda **kw: fluid,
                         get_instance_pose=lambda inst_name, env_idx: ([0 if inst_name == 'cup' else 1, 0, 0], [1, 0, 0, 0]),
                         get_instance_bbox_vertices=lambda **kw: [[-.1, -.1, 0], [.1, .1, .1]])
    pending = [([("is_axis_up", {"label": "bottle"})], [], "rising_edge", True)]
    parser = SimpleNamespace(is_A_fluid_in_B=lambda args: calls.append(args) or 1.)
    run = SimpleNamespace(env=SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=lm),
                                              reward_manager=SimpleNamespace(trigger_check_list=[pending], func_parser=parser)))
    r = snapshot(run, 'pour_liquid_into_cup')
    assert r['pending_upright_checks'] == 1 and len(pending) == 1
    assert r['pending_upright_labels'] == ['bottle']
    assert r['fluids'][0]['predicate_now'] and r['fluids'][0]['in_target'] == 1
    assert calls[0]['percentage_threshold'] == .97 and calls[0]['C_residual_threshold'] == .15


@pytest.mark.parametrize('skill,commit', [('pour_bottle', 'a21dd6f'), ('pour_bottle_v2', '6769a6c'),
                                         ('pour_bottle_v5', '6769a6c'), ('pour_in_order_v5', '6769a6c')])
def test_later_originals_report_their_actual_source(monkeypatch, skill, commit):
    from types import SimpleNamespace
    import dojo.skills.ported as adapter
    class Robot:
        def __init__(self, *a, **kw):
            self.calls, self.truth = [], True
        def run_skill(self, *a):
            return {'ok': True, 'verified': True}
        def grip_source(self):
            return 'mock'
    monkeypatch.setattr(adapter, 'HarnessRobot', Robot)
    sk = SimpleNamespace(ex=SimpleNamespace(steps_used=0, ended=False, steps_left=800),
                         provider=SimpleNamespace(events=[]), _record=lambda r: None)
    assert adapter.ported(sk, skill)['source'] == f'robot-harness {commit} skills/{skill}'


def test_budget_pace_keeps_flow_and_angles_and_spends_only_last_profile_reserve(monkeypatch):
    import dojo.skills.ported as adapter
    from dojo.skills import planner
    from dojo.skills.pour_order import pour_in_order
    assert RuntimeConfig().ported_pour_pace == 'v1'
    assert 'budget' in SKILL_ATTRS['ported_pour_pace']
    args = planner.pour_bottle_call('b10', 'budget')[1]
    assert args == {'tilt_speed': .8, 'untilt_speed': 1., 'lift_speed': 'normal', 'drain_steps': 12, 'spout_back': .01}
    assert not {'phis', 'flow_speed', 'untilt_phis'} & args.keys()
    calls = []
    monkeypatch.setattr(adapter, 'ported', lambda sk, name, args, reserve=None:
                        calls.append(args) or {'ok': True, 'verified': True, 'pour': {'max_phi': 150, 'stop': 'profile end', 'final_tilt': .5}})
    class Sk:
        ported_pour_pace, ported_pour_aim = 'budget', 'b10'
        def refresh(self):
            return None
        def log(self, message):
            pass
    assert pour_in_order(Sk(), [['a', 'x'], ['b', 'y'], ['c', 'z']])['ok']
    assert [c.get('profile_reserve') for c in calls] == [None, None, 0]
    assert [c.get('untilt_reserve') for c in calls] == [None, None, 0]
    assert [c.get('companion') for c in calls] == [['@b', '@y'], ['@c', '@z'], None]


@pytest.mark.parametrize('phi,stop,tilt,complete', [(120, 'step budget', .5, False), (150, 'step budget', .5, False),
                                                  (150, 'profile end', 47.4, False), (150, 'profile end', 29., True)])
def test_budget_never_counts_partial_or_untilted_pour_as_complete(phi, stop, tilt, complete):
    from dojo.skills.pour_order import poured
    r = {'ok': True, 'verified': True, 'pour': {'max_phi': phi, 'stop': stop, 'final_tilt': tilt}}
    assert poured(r, full=True) is complete
    assert poured(r)  # legacy path remains unchanged
