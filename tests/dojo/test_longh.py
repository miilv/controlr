"""Long-Horizon stream: a non-grasp action's arm leaves the head view before the provider's confirm (mode B only)."""

from __future__ import annotations

import numpy as np

from dojo.skills.env import KinematicEnv
from dojo.skills.g3 import OffsetBeliefProvider
from dojo.skills.skills import GTProvider, Skills


def _kong_scene(confirm_clear: bool):
    from dojo.skills.heldsim import HeldBeliefProvider
    from tests.dojo.test_skills2 import kong_rows
    env = KinematicEnv(kong_rows(0)[:3], step_lim=400)
    base = OffsetBeliefProvider(GTProvider(env, None), {"mahjong0_1": (0.002, 0.0)})
    prov = HeldBeliefProvider(base, env, mode=None, freeze=True, confirm=True)
    logs: list = []
    sk = Skills(env, prov, remeasure=False, refine=lambda *a: None, log=logs.append)
    sk.logs = logs
    sk.confirm_clear = confirm_clear
    seen = []
    orig = prov.confirm_pose

    def spy(label, pos, quat):
        # where the arms are when the provider looks (the runtime: the head view of this frame); an arm away from home
        # hides the outcome (proxy B kong: the topple's arm over the tile), so it is not confirmed then
        seen.append({a: float(np.abs(np.asarray(sk.ex.q[a]) - sk.world.home_q[a]).max()) for a in ("left", "right")})
        if max(seen[-1].values()) > 0.02:
            return {"adopted": False, "why": "arm in view"}
        return orig(label, pos, quat)
    prov.confirm_pose = spy
    return env, prov, sk, seen


def test_not_confirmed_over_the_tile_the_arm_goes_home_and_asks_again():
    from dojo.skills.library import topple, up_body_axis
    env, prov, sk, seen = _kong_scene(confirm_clear=True)
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    assert r.ok, r.note
    assert len(seen) == 2 and max(seen[0].values()) > 0.1 and max(seen[1].values()) < 0.02, seen
    assert [e["adopted"] for e in sk.confirm_events] == [False, True], sk.confirm_events
    assert sk.confirm_events[1]["why"] == "topple, arm clear"
    assert any("home for a second look" in m for m in sk.logs), sk.logs[-5:]


def test_without_confirm_clear_the_topple_retries_blind():
    from dojo.skills.library import topple, up_body_axis
    env, prov, sk, seen = _kong_scene(confirm_clear=False)
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    assert not r.ok and all(not e["adopted"] for e in sk.confirm_events), sk.confirm_events   # the old behaviour


def test_a_confirmed_topple_does_not_move_the_arm_home():
    from dojo.skills.library import topple, up_body_axis
    env, prov, sk, seen = _kong_scene(confirm_clear=True)
    prov.confirm_pose = prov._confirm_pose                       # GT confirm wherever the arm is (G3 mode B)
    n0 = sk.ex.steps_used
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    assert r.ok and [e["adopted"] for e in sk.confirm_events] == [True]
    assert not any("second look" in m for m in sk.logs) and sk.ex.steps_used - n0 < 40


def test_no_confirm_provider_no_clearing_move():
    from dojo.skills.library import topple, up_body_axis
    from tests.dojo.test_skills2 import kong_rows
    env = KinematicEnv(kong_rows(0)[:3], step_lim=300)
    logs: list = []
    sk = Skills(env, GTProvider(env, None), log=logs.append)
    sk.logs = logs
    n0 = sk.ex.steps_used
    assert topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1"))).ok
    assert sk.confirm_events == [] and not any("second look" in m for m in sk.logs)
    assert sk.ex.steps_used - n0 < 40                             # G3 mode A: the topple alone


def test_a_rejected_push_confirm_gets_a_second_look_with_the_arm_home():
    from dojo.skills.kinematics import TABLE_Z
    from dojo.skills.library import push_to
    env = KinematicEnv([{"label": "kb", "category": "keyboard", "pos": [-0.05, -0.2, TABLE_Z + 0.0065],
                         "quat": [1, 0, 0, 0], "half": [0.18, 0.075, 0.0065]}], step_lim=400)
    logs: list = []
    sk = Skills(env, GTProvider(env, None), remeasure=False, refine=lambda *a: None, log=logs.append)
    seen = []

    def confirm_pose(label, pos, quat):              # never confirmed: a stroke the head view cannot see
        seen.append(max(float(np.abs(np.asarray(sk.ex.q[a]) - sk.world.home_q[a]).max()) for a in ("left", "right")))
        return {"adopted": False}
    sk.provider.confirm_pose = confirm_pose
    push_to(sk, "kb", np.array([-0.05, -0.12]), max_iter=1)
    assert len(seen) == 2 and seen[0] > 0.1 and seen[1] < 0.02, seen
    assert [e["why"] for e in sk.confirm_events] == ["push", "push, arm clear"]
