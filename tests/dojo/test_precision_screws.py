"""Precision stream (2026-10-27): screw_on in mode B, where the belief cannot measure a nut on its bolt."""

import numpy as np

from dojo.skills import precision as P
from dojo.skills.env import KinematicEnv
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills


def _env():
    return KinematicEnv([
        {"label": "nut0", "category": "factory_nut", "pos": [0.20, -0.16, TABLE_Z + 0.0095], "quat": [1, 0, 0, 0],
         "half": [0.0176, 0.0203, 0.0095]},
        {"label": "bolt0", "category": "factory_bolt", "type": "Geometry", "pos": [0.26, -0.22, TABLE_Z + 0.03],
         "quat": [1, 0, 0, 0], "half": [0.012, 0.012, 0.03]},
    ], step_lim=1900)


def _sweeps(sk):
    return [h for h in sk.history if h.skill == "screw"]


def test_mode_b_screw_counts_the_modelled_descent_not_the_belief():
    env = _env()
    sk = Skills(env, GTProvider(env, None), remeasure=False)
    r = P.screw_on(sk, [["nut0", "bolt0"]])
    sw = _sweeps(sk)
    # the kinematic belief lets the nut fall through the bolt (53+ mm "below the top"): mode A would stop at once;
    # the model seats it at most SCREW_SEAT_MAX and adds what each sweep can give
    seat = P.SCREW_SEAT_MAX
    per = P.SCREW_PITCH * np.radians(300.0) / (2 * np.pi)
    need = 1 + int(np.ceil((P.SCREW_DEPTH - seat - per * P.SCREW_FIRST_EFF) / (per * P.SCREW_EFF)))
    assert len(sw) == need, [h.note for h in sw]
    assert all("(model; belief" in h.note for h in sw)
    assert r["ok"], r


def test_mode_a_screw_keeps_the_measured_depth():
    env = _env()
    sk = Skills(env, GTProvider(env, None))            # remeasure: the world is measured (GT)
    P.screw_on(sk, [["nut0", "bolt0"]])
    assert not _sweeps(sk)                             # the kinematic nut fell through: measured as deep enough


def test_screw_model_puts_the_nut_on_the_bolt_axis_at_the_depth():
    env = _env()
    sk = Skills(env, GTProvider(env, None))
    w = sk.refresh()
    o, b = w.obj("nut0"), w.obj("bolt0")
    m = P._screw_model(o, b, 0.006)
    assert np.allclose(m.center_w[:2], b.center_w[:2]) and abs((b.top_z - m.bottom_z) - 0.006) < 1e-9
    assert np.allclose(o.pos, [0.20, -0.16, TABLE_Z + 0.0095])        # the belief's object is not touched
