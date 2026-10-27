"""Generalization-New: a garment folded with pinches (planner route, frozen-read fold in the kinematic env), the
camera garment read's template fit, an object let go onto a cradle (mesh clearance, the way on), plan_pick's opt-in
pull-out lift (CPU only)."""

from __future__ import annotations

import math

import numpy as np

from tests.dojo.test_skills2 import _box_mesh, _merge, _obj, _world

FOLD_KP = {"left_sleeve": [-0.29, -0.14, 0.775], "right_sleeve": [0.21, -0.28, 0.774],
           "left_chest": [-0.07, -0.045, 0.78], "right_chest": [0.087, -0.089, 0.79],
           "left_hem": [-0.14, -0.25, 0.774], "right_hem": [0.02, -0.29, 0.774],
           "left_shoulder": [-0.044, 0.015, 0.774], "right_shoulder": [0.077, -0.024, 0.774]}


def test_planner_routes_fold_to_the_garment():
    from dojo.skills import planner
    g = _obj("target", "Top_Long", 4, (0.0, -0.12, 0.78), (0.28, 0.17, 0.08), typ="Garment")
    p = planner.plan("Fold the clothes neatly.", _world([g]))
    assert p.skill == "fold_garment" and p.args == {"garment": "target"}
    # no garment: not this rule
    b = _obj("block", "cube", 1, (0.0, -0.12, 0.78), (0.02, 0.02, 0.02))
    assert planner.plan("Fold the clothes neatly.", _world([b])).skill != "fold_garment"


def test_garment_axes_follow_the_root_yaw():
    from dojo.skills.garment import garment_axes
    th = math.radians(-15.7)
    xa, ya = garment_axes({"quat": [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]})
    assert np.allclose(xa, [math.cos(th), math.sin(th)], atol=1e-9)
    assert np.allclose(ya, [-math.sin(th), math.cos(th)], atol=1e-9)


def test_a_garment_is_not_an_obstacle():
    g = _obj("target", "Top_Long", 4, (0.0, -0.12, 0.78), (0.28, 0.17, 0.08), typ="Garment")
    b = _obj("block", "cube", 1, (0.2, -0.12, 0.78), (0.02, 0.02, 0.02))
    assert [o.label for o in _world([g, b]).obstacles()] == ["block"]


def test_fold_from_one_frozen_read_runs_every_fold(monkeypatch):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.garment import fold_garment
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv([], step_lim=500, instruction="Fold the clothes neatly.")
    prov = GTProvider(env, None, instruction="Fold the clothes neatly.")
    reads = []

    def garment(label=None):
        reads.append(label)
        return {"keypoints": FOLD_KP, "quat": [0.99, 0.0, 0.0, -0.137], "pos": [0.0, -0.12, 0.78], "live": False}
    prov.garment = garment
    sk = Skills(env, prov, reserve=30, log=lambda m: None)
    res = fold_garment(sk, "target")
    assert res["ok"], res
    assert res["done"] == ["left sleeve", "right sleeve", "body"]
    assert len(reads) == 1                               # planned from the one read
    assert 150 < env.steps_used < 500


def test_template_fit_recovers_yaw_and_shift():
    import pytest
    pytest.importorskip("scipy")
    from dojo.perception.garment import fit_template
    rng = np.random.default_rng(0)
    # a T-shirt-like footprint: body 30 x 34 cm, sleeves out to +-28 cm, 8 cm tall, drooping
    body = rng.uniform([-0.15, -0.2], [0.15, 0.14], (3000, 2))
    sl = rng.uniform([0.15, 0.02], [0.28, 0.1], (800, 2))
    sr = sl * [-1, 1]
    T = np.vstack([body, sl, sr])
    th, t = math.radians(12), np.array([0.03, -0.13])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    obs = T @ R.T + t + rng.normal(0, 0.002, T.shape)
    f = fit_template(obs, {"xy": T})
    assert abs(math.degrees(f["yaw"] - th)) < 2.5
    assert np.linalg.norm(f["t"] - t) < 0.01
    assert f["iou"] > 0.6


def test_rest_clearance_and_way_on_for_a_saddle():
    from dojo.skills.library import rest_clearance, way_out
    # support: a 10 x 10 cm block with a 1 cm lip front and back (a trough along x), top at 0.80
    sup = _merge(_box_mesh((-0.05, -0.05, 0.0), (0.05, 0.05, 0.03)), _box_mesh((-0.05, -0.05, 0.03), (0.05, -0.04, 0.04)),
                 _box_mesh((-0.05, 0.04, 0.03), (0.05, 0.05, 0.04)))
    S = _obj("stand", "stand", 1, (0.0, 0.0, 0.77), (0.05, 0.05, 0.02), typ="Geometry", mesh=sup)
    # object: a 3 cm cube whose bottom is put 2 mm into the trough floor
    O = _obj("thing", "thing", 1, (0.0, 0.0, 0.0), (0.015, 0.015, 0.015),
             mesh=_box_mesh((-0.015, -0.015, -0.015), (0.015, 0.015, 0.015)))
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.8 + 0.015 - 0.002]
    up = rest_clearance(O, S, T, gap=0.004)
    assert up is not None and 0.005 < up < 0.02
    T[2, 3] += up
    d, h = way_out(O, S, T, (0.0, -0.45), dist=0.08, gap=0.003)
    # out along the trough (x) needs no lift; towards the robot (-y) it must clear the lip
    assert abs(d[2]) < 1e-9 and (abs(d[0]) > 0.9 or h >= 0.01)


def test_plan_pick_lift_pulls_are_opt_in():
    import inspect

    from dojo.skills.skills import Skills
    sig = inspect.signature(Skills.plan_pick)
    assert sig.parameters["lift_pulls"].default == ()
    assert sig.parameters["held_mesh_table"].default is False
