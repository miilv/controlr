"""Precision stream (2026-10-27): mode B's held-pose protocol falls back to its own show poses when none of the
provider's candidates plans (rendered 5.1 plug: perception's looks / show poses were all infeasible for our planner)."""

import numpy as np

from dojo.skills import skills as S
from dojo.skills.env import KinematicEnv
from dojo.skills.g3 import OffsetBeliefProvider
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills

HALF = [0.03, 0.012, 0.01]
AMB = {"ok": 0, "why": "ambiguous"}
GOOD = {"ok": 1, "why": "", "err": [0.0] * 6, "sig": [0.1, 0.1, 0.1, 0.1, 0.3]}


def _scene():
    from dojo.skills.heldsim import HeldBeliefProvider
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = HeldBeliefProvider(OffsetBeliefProvider(GTProvider(env, None), {"b": (0.003, 0.0)}), env, mode="noise",
                              table={"block": {"hover": [AMB], "look": [GOOD], "show": [GOOD]}})
    prov.show_pose = lambda *a, **k: {"poses": []}            # nothing of the provider's plans
    sk = Skills(env, prov, remeasure=False, refine=lambda *a: None)
    assert sk.run_pick(sk.plan_pick("b", "left")).ok
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:3, 3] = (-0.18, -0.22, TABLE_Z + 0.01)
    return env, prov, sk, T


def test_no_provider_candidate_plans_so_a_generic_show_pose_reads_the_held_object():
    env, prov, sk, T = _scene()
    logs = []
    sk.log = logs.append
    q_obs = sk.ex.q["right"].copy()
    r = sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04))
    views = [e["view"] for e in sk.held_events]
    assert views == ["hover", "show"], (views, logs)
    assert sk.held_events[1]["adopted"] and sk.held_events[1]["steps"] > 0
    assert any("generic show" in m for m in logs) and "held read adopted" in r.note
    assert np.abs(sk.ex.q["right"] - q_obs).max() < 1e-9          # the observer is back where it was


def test_without_the_fallback_only_the_hover_is_read(monkeypatch):
    monkeypatch.setattr(S, "HELD_GENERIC_SHOW", False)
    env, prov, sk, T = _scene()
    sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04))
    assert [e["view"] for e in sk.held_events] == ["hover"]
