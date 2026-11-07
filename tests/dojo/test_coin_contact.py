"""COIN2 (``dojo.skills.coin_contact``): the thin-disc contact switches default off, the wall-squeeze fit, the pick ops."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import coin_contact as CC
from dojo.skills import grasps as G
from dojo.skills.env import KinematicEnv
from dojo.skills.geometry import R_to_quat, rot_z
from dojo.skills.kinematics import X5Kin
from dojo.skills.skills import GTProvider, Skills
from dojo.skills.world import AssetInfo, ObjectState

SWITCHES = {"disc_align": ("off", "walls"), "disc_snap": ("off", "on"), "disc_lift": ("off", "shallow"),
            "disc_seat": ("off", "table"), "disc_release": ("off", "staged"), "disc_reserve": ("off", "on")}


def _reads(e, d_mm, thetas):
    return {t: CC.WALL_A + CC.WALL_B * abs(e + t) + CC.WALL_C * d_mm for t in thetas}


@pytest.mark.parametrize("e,d", [(0.0, 0.0), (4.0, 0.0), (-3.0, 1.2), (6.5, -1.5), (-7.5, 0.4)])
def test_wall_fit_recovers_the_yaw_error_and_the_offset_when_bracketed(e, d):
    f = CC.wall_fit(_reads(e, d, (0.0, CC.WALL_DELTA, -CC.WALL_DELTA)))
    assert f["bracketed"]
    assert abs(f["e"] - e) <= 0.1 and abs(f["apex"] + e) <= 0.1 and abs(f["d_mm"] - d) <= 0.05 and f["resid"] < 1e-4


def test_wall_fit_outside_the_first_three_needs_another_squeeze_on_the_low_side():
    th = (0.0, CC.WALL_DELTA, -CC.WALL_DELTA)
    f = CC.wall_fit(_reads(12.0, 0.5, th))         # the apex is at -12: all three on one branch
    assert not f["bracketed"] and f["low_side"] == -1
    f2 = CC.wall_fit(_reads(12.0, 0.5, th + (-2 * CC.WALL_DELTA,)))
    assert f2["bracketed"] and abs(f2["e"] - 12.0) <= 0.1 and abs(f2["d_mm"] - 0.5) <= 0.05


def test_switches_default_off_mirrored_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    cfg = RuntimeConfig()
    for name, vals in SWITCHES.items():
        assert getattr(cfg, name) == "off" and SKILL_ATTRS[name] == vals
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**{name: "maybe"}))
    env = KinematicEnv([], step_lim=50)
    sk = Skills(env, GTProvider(env, None))
    assert all(getattr(sk, n) == "off" for n in SWITCHES)


def test_off_switches_leave_the_disc_arguments_as_they_are():
    sk = SimpleNamespace()                          # no attribute at all reads as "off"
    base_pick = {"squeeze": 4, "lift_rad": 0.03}
    base_place = {"approach_rad": 0.09, "servo": True}
    assert CC.pick_kw(sk, base_pick) == base_pick and CC.place_kw(sk, base_place) == base_place
    assert CC.edge_floor(sk, 0.782, 0.785) == 0.785
    assert CC.reserve_ok(sk, cross=True) == (True, -1)
    sk = SimpleNamespace(disc_snap="on", disc_lift="shallow", disc_release="staged", disc_reserve="on",
                         ex=SimpleNamespace(steps_left=100))
    kw = CC.pick_kw(sk, base_pick)
    assert kw["snap_in"] == pytest.approx(CC.SNAP_IN + CC.SNAP_CLEAR) and kw["lift_step"] == CC.LIFT_STEP_M
    assert CC.edge_floor(sk, 0.782, 0.785) == pytest.approx(0.782 + CC.SNAP_CLEAR)
    pk = CC.place_kw(sk, base_place)
    assert pk["approach_rad"] == CC.CARRY_RAD and pk["held_read"] and pk["release_stages"] == CC.RELEASE_STAGES
    assert CC.reserve_ok(sk, cross=False) == (True, 100) and CC.reserve_ok(sk, cross=True) == (False, 100)


def _skills_with_block():
    half = np.array([0.012, 0.012, 0.04])
    rows = [{"label": "cube", "pos": [0.25, -0.15, 0.765 + 0.04], "quat": R_to_quat(rot_z(0.0)), "half": half,
             "category": "block"}]
    env = KinematicEnv(rows, step_lim=400)
    return Skills(env, GTProvider(env, None)), env


def _ops(pp):
    return [op[-1] if op[0] in ("move", "hold") else op[3] for op in pp.plan.ops if op[0] != "mark"]


def test_plan_pick_default_ops_unchanged_and_snap_adds_the_rise_from_deeper():
    sk, _ = _skills_with_block()
    o = sk.refresh().obj("cube")
    g = [x for x in G.top_grasps(o, X5Kin(), "right")][:1]
    base = sk.plan_pick("cube", "right", grasps=g, lift=0.03, squeeze=4)
    assert _ops(base) == ["open_for_grasp", "approach", "descend", "settle", "close", "squeeze", "lift"]
    snap = sk.plan_pick("cube", "right", grasps=g, lift=0.03, squeeze=4, snap_in=0.006, open_g=0.4)
    assert _ops(snap) == ["open_for_grasp", "approach", "descend", "settle", "close", "snap_rise", "squeeze", "lift"]
    kin = sk.kin
    end_descend = kin.tcp("right", snap.plan.ops[2][1][-1].q[-1])
    end_rise = kin.tcp("right", snap.plan.ops[5][1][-1].q[-1])
    assert end_descend[2, 3] == pytest.approx(g[0].T[2, 3] - 0.006, abs=5e-4)   # straight down: deeper by snap_in
    assert end_rise[2, 3] == pytest.approx(g[0].T[2, 3], abs=5e-4)
    assert np.allclose(snap.T_tcp_obj, base.T_tcp_obj)                        # the hold is the planned grasp
    assert snap.plan_kw["snap_in"] == 0.006                                    # a refined re-plan keeps it


def test_plan_pick_lift_step_moves_a_few_mm_per_env_step():
    sk, _ = _skills_with_block()
    o = sk.refresh().obj("cube")
    g = G.top_grasps(o, X5Kin(), "right")[:1]
    pp = sk.plan_pick("cube", "right", grasps=g, lift=0.03, lift_step=0.004)
    lift = [op for op in pp.plan.ops if op[0] == "move" and op[2] == "lift"][0][1][0]
    zs = [sk.kin.tcp("right", q)[2, 3] for q in lift.q]
    assert len(zs) >= 8 and max(np.diff(zs)) <= 0.0045
    assert lift.n_steps() == len(zs) - 1                                       # one waypoint per env step


def test_post_lift_ops_and_in_hand_pose_replace_the_planned_ones():
    sk, _ = _skills_with_block()
    o = sk.refresh().obj("cube")
    g = G.top_grasps(o, X5Kin(), "right")[:1]
    seen = {}

    def post(arm, q, T_tcp_obj, obj):
        seen["q"] = q
        T2 = T_tcp_obj.copy()
        T2[0, 3] -= 0.004
        return [("hold", 2), ("grip", 0.0, 2)], T2
    pp = sk.plan_pick("cube", "right", grasps=g, lift=0.03, post_lift=post)
    assert _ops(pp)[-2:] == ["post_lift", "post_lift"] and "q" in seen
    base = sk.plan_pick("cube", "right", grasps=g, lift=0.03)
    assert pp.T_tcp_obj[0, 3] == pytest.approx(base.T_tcp_obj[0, 3] - 0.004)


def test_offset_belief_provider_keeps_a_written_pose():
    from dojo.skills.g3 import OffsetBeliefProvider
    sk, env = _skills_with_block()
    prov = OffsetBeliefProvider(GTProvider(env, None), {"cube": [0.0, 0.0, 10.0]})
    o = prov.refresh({}).obj("cube")
    R = rot_z(math.radians(3.0))
    pos = o.pos + np.array([0.002, -0.001, 0.0])
    assert prov.set_pose("cube", pos, R_to_quat(R), writer="wall_probe")
    o2 = prov.refresh({}).obj("cube")
    assert np.allclose(o2.pos[:2], pos[:2], atol=1e-6) and np.allclose(o2.R, R, atol=1e-6)
    assert prov.writes[-1]["writer"] == "wall_probe"


def test_disc_contact_argument_scopes_the_profile_to_one_call_and_restores():
    sk = SimpleNamespace(disc_seat="off", disc_snap="on")
    with CC.scoped(sk, "on"):
        assert all(getattr(sk, k) == CC.PROFILE.get(k, "off") for k in CC.SWITCHES)
    assert sk.disc_seat == "off" and sk.disc_snap == "on" and sk.disc_align == "off"
    with pytest.raises(RuntimeError):
        with CC.scoped(sk, "off"):
            assert sk.disc_snap == "off"
            raise RuntimeError("x")
    assert sk.disc_snap == "on"
    with CC.scoped(sk, None):
        assert sk.disc_snap == "on"
    with pytest.raises(ValueError):
        CC.scoped(sk, "maybe")


def test_insert_into_takes_disc_contact_and_no_task_name_reaches_the_switches():
    import inspect

    from dojo.skills import library
    assert inspect.signature(library.insert_into).parameters["disc_contact"].default is None
    src = inspect.getsource(CC)
    assert "deposit_coin" not in src.split('"""', 2)[2]          # no task-name key below the module docstring
