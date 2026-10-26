"""skills10: mode B against a belief that cannot measure what rests on a static fixture (integration-2's breadth run): a
non-probing servo place runs the servo tail, the guarded tail honours ``probe`` False, screw_on tracks its nut, the
tic-tac-toe game counts a piece believed under the board as on its cell; the G3 test provider's ``drop`` emulation of
the runtime belief's release (CPU only)."""

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


def test_screw_on_tracks_a_nut_the_belief_dropped_through_its_bolt():
    from dojo.skills.precision import _screw_seen
    env = KinematicEnv([{"label": "bolt", "type": "Geometry", "category": "factory_bolt", "pos": [0.2, -0.15, TABLE_Z + 0.025],
                         "quat": [1, 0, 0, 0], "half": [0.012, 0.012, 0.025]},
                        {"label": "nut", "category": "factory_nut", "pos": [0.2, -0.15, TABLE_Z + 0.0105],
                         "quat": [1, 0, 0, 0], "half": [0.018, 0.018, 0.0105]}])
    sk = Skills(env, GTProvider(env, None), remeasure=False, refine=lambda *a: None)
    top = sk.world.obj("bolt").top_z
    o, b = _screw_seen(sk, "nut", "bolt", top - 0.0026)       # believed on the table, inside the bolt
    assert abs(o.bottom_z - (top - 0.0026)) < 1e-6 and abs(sk.world.obj("nut").bottom_z - TABLE_Z) < 1e-6
    sk.remeasure = True                                        # G3: the world as measured
    o, _ = _screw_seen(sk, "nut", "bolt", top - 0.0026)
    assert abs(o.bottom_z - TABLE_Z) < 1e-6


def test_board_occupancy_in_mode_b_counts_a_piece_believed_inside_the_board():
    from dojo.skills.interactive import board_state
    env = KinematicEnv([{"label": "board", "type": "Geometry", "category": "checkerboard", "pos": [0.0, -0.1, TABLE_Z],
                         "quat": [1, 0, 0, 0], "half": [0.15, 0.15, 0.0085]},
                        {"label": "p", "category": "chessman", "pos": [0.05, -0.05, TABLE_Z],
                         "quat": [1, 0, 0, 0], "half": [0.016, 0.016, 0.006]}])
    w = GTProvider(env, None).refresh()
    cells = [("cell/0", np.array([0.05, -0.05, TABLE_Z + 0.017]))]
    assert board_state(w, "board", cells, ["p"], []) == {}
    assert board_state(w, "board", cells, ["p"], [], dz=(-0.02, 0.03)) == {0: ("own", "p")}


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
