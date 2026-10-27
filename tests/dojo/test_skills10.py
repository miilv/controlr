"""skills10: mode B places that do not probe (a non-probing servo place runs the servo tail, the guarded tail honours
``probe`` False); the G3 test provider's ``drop`` emulation of the runtime belief's release (CPU only)."""

from __future__ import annotations

import numpy as np

from dojo.skills.env import KinematicEnv
from dojo.skills.g3 import OffsetBeliefProvider
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills

HALF = [0.03, 0.012, 0.01]


def _held_scene(mode_b=True):
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = OffsetBeliefProvider(GTProvider(env, None), {})
    sk = Skills(env, prov, remeasure=not mode_b, refine=(lambda *a: None) if mode_b else None)
    assert sk.run_pick(sk.plan_pick("b", "left")).ok
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:3, 3] = (-0.18, -0.22, TABLE_Z + 0.01)
    return env, prov, sk, T


def test_mode_b_place_that_does_not_probe_runs_the_servo_tail():
    env, prov, sk, T = _held_scene()
    r = sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04, probe=False, seat_tol=0.3))
    assert "servo:" in r.note and "probing tail" not in r.note
    env, prov, sk, T = _held_scene()                       # an insertion (probes) keeps the contact-probing tail
    r = sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04))
    assert "probing tail" in r.note


def test_guarded_descent_does_not_probe_when_the_place_does_not():
    env, prov, sk, T = _held_scene()
    h = sk.held["left"]
    T_low = T.copy()
    T_low[2, 3] -= 0.05                                    # 5 cm into the table
    calls = []
    real = sk._guarded_line

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)
    sk._guarded_line = spy
    sk._lag = lambda arm: 0.01                             # the arm is held off its command at once (contact)
    note, off, exc = sk._guarded_descent("left", T_low, h, {"probe": False})
    assert exc is None and off is None and "no probing" in note and len(calls) == 1
    note, off, exc = sk._guarded_descent("left", T_low, h, {})
    assert "probes" in note and len(calls) > 2


def test_heldsim_drop_keeps_the_runtime_prediction_where_perception_cannot_fit_it():
    """``drop``: a release onto something the post-place fit cannot explain (here: the truth hangs 30 mm over the
    table, as on a hook) keeps the belief's prediction; one resting on the table is re-measured."""
    from dojo.skills.heldsim import HeldBeliefProvider
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = HeldBeliefProvider(GTProvider(env, None), env, mode=None, drop="runtime")
    w = prov.refresh()
    o = w.obj("b")
    T_rel = np.linalg.inv(prov.kin.tcp("left", np.zeros(6))) @ o.T
    prov._pending["b"] = prov._belief_rest("b", o.T, fixtures=False)
    prov.refresh()
    assert "b" not in prov._frozen and prov.stats["remeasured"] == 1       # on the table: measured
    row = next(r for r in env.objects if r["label"] == "b")
    row["pos"] = [-0.22, -0.15, TABLE_Z + 0.04]                             # the truth hangs 30 mm up
    T_up = o.T.copy()
    T_up[2, 3] += 0.03
    prov._pending["b"] = prov._belief_rest("b", T_up, fixtures=False)
    w = prov.refresh()
    assert "b" in prov._frozen and abs(w.obj("b").bottom_z - TABLE_Z) < 1e-6   # the belief dropped it to the table
    assert T_rel.shape == (4, 4)


def test_heldsim_fixtures_look_reestimates_a_fixture_held_prediction_that_is_far_off():
    """``drop="fixtures_look"``: a release the fixture rule kept where it was let go ("held by a fixture") whose truth is
    > RESYNC_TOL away (a mug that fell off its hook) is re-estimated, as the runtime's release resync would; with
    ``drop="fixtures"`` the prediction stays."""
    from dojo.skills.heldsim import RESYNC_TOL, HeldBeliefProvider
    for drop, kept in (("fixtures", True), ("fixtures_look", False)):
        env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.04],
                             "quat": [1, 0, 0, 0], "half": HALF}])           # the truth: 30 mm over the table (no fit)
        prov = HeldBeliefProvider(GTProvider(env, None), env, mode=None, drop=drop)
        w = prov.refresh()
        o = w.obj("b")
        pred = (o.pos + np.array([0.0, 0.0, 0.03 + RESYNC_TOL]), o.quat.copy())
        prov._pending["b"] = (pred[0], pred[1], "held by a fixture")
        w = prov.refresh()
        assert ("b" in prov._frozen) == kept, drop
        if not kept:
            assert np.allclose(w.obj("b").pos, o.pos) and prov.stats["resynced"] == 1
            assert any(e["kind"] == "release_resynced" for e in prov.events)
