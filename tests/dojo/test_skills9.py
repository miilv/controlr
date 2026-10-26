"""skills9: a refined pick keeps the grasp its place was planned with; the mode-B held-pose protocol at the pre-insert
hover (read, look / show, servo on the measured transform, safe fallback); the G3 held-belief / held-pose test
provider (CPU only)."""

from __future__ import annotations

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.env import KinematicEnv
from dojo.skills.g3 import OffsetBeliefProvider
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, PickPlan, PlacePlan, Skills

HALF = [0.03, 0.012, 0.01]


def _scene(offset=(0.003, -0.002), refine_fixes=True):
    """A block whose believed pose is ``offset`` off until the pick's refine re-measures it (then the truth)."""
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = OffsetBeliefProvider(GTProvider(env, None), {"b": offset})

    def refine(sk_, label, arm, kind):
        if kind == "pick" and refine_fixes:
            prov.offsets.pop(label, None)
    sk = Skills(env, prov, remeasure=False, refine=refine)
    return env, prov, sk


def _tilted(sk):
    """Only 25-deg tilted box grasps (the transfer's pairing; the default pick would come straight down)."""
    return lambda a: [g for g in G.box_grasps(sk.world.obj("b"), sk.kin, a, tilts=(25.0,)) if -g.T[2, 0] < 0.95]


def _target(sk):
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:2, 3] = (-0.17, -0.22)
    return T


def test_a_refined_pick_keeps_the_planned_grasp_in_the_object_frame():
    env, prov, sk = _scene()
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    assert callable(pp.repair)
    default = sk.plan_pick("b", pp.arm)
    assert abs(float(default.grasp.T[2, 0])) > 0.99          # the method's own grasp: straight down (the old re-plan)
    r = sk.run_transfer(pp, pl)
    assert r.ok, r.note
    ex = sk.last_pick
    assert ex is not pp and np.allclose(ex.T_tcp_obj, pp.T_tcp_obj, atol=1e-9)   # same grasp in the object frame
    assert -float(ex.grasp.T[2, 0]) < 0.95                    # still the tilted pinch
    # ... at the refined (true) pose: the grasp moved with the object by the 3 mm belief error
    assert np.allclose(ex.grasp.T[:2, 3] - pp.grasp.T[:2, 3], [-0.003, 0.002], atol=1e-6)
    assert np.linalg.norm(sk.refresh().obj("b").pos[:2] - [-0.17, -0.22]) < 0.002


def test_an_unmoved_refine_runs_the_original_descent():
    env, prov, sk = _scene(offset=(0.0, 0.0), refine_fixes=False)
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    assert sk.run_pick(pp).ok
    assert sk.last_pick is pp


def test_an_infeasible_grasp_re_pairs_pick_and_place(monkeypatch):
    env, prov, sk = _scene()
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    monkeypatch.setattr(sk, "_same_grasp_pick", lambda p, kw: None)
    calls = []
    orig = pp.repair

    def repair(q_s):
        calls.append(q_s)
        return orig(q_s)
    pp.repair = repair
    r = sk.run_transfer(pp, pl)
    assert calls and r.ok, r.note
    ex = sk.last_pick
    assert ex is not pp and isinstance(ex.paired_place, PlacePlan)
    assert np.allclose(calls[0][pp.arm], ex.plan.q0)           # re-planned from the hover
    assert np.linalg.norm(sk.refresh().obj("b").pos[:2] - [-0.17, -0.22]) < 0.002


def test_a_plain_pick_falls_back_to_the_method_grasps(monkeypatch):
    env, prov, sk = _scene()
    pp = sk.plan_pick("b", "left", grasps=_tilted(sk)("left"))
    assert isinstance(pp, PickPlan) and pp.repair is None
    monkeypatch.setattr(sk, "_same_grasp_pick", lambda p, kw: None)
    assert sk.run_pick(pp).ok
    assert abs(float(sk.last_pick.grasp.T[2, 0])) > 0.99       # no place to keep: the method's best grasp, as before


def test_a_pick_without_refine_is_unchanged():
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    sk = Skills(env, GTProvider(env, None))
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    assert sk.run_pick(pp).ok and sk.last_pick is pp
