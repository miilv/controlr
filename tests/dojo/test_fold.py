"""fold_garment: the planner's route, the garment's axes, a garment is no obstacle, and a fold planned from one
frozen read (the camera's at eval) runs every fold in the kinematic env (CPU only)."""

from __future__ import annotations

import math

import numpy as np



def _obj(label, cat, idx, xyz, half, typ="Rigid", quat=(1.0, 0, 0, 0)):
    from dojo.skills.world import AssetInfo, ObjectState
    info = AssetInfo(key=f"{typ}/{cat}/{idx:05d}", bbox_min=-np.asarray(half, float), bbox_max=np.asarray(half, float))
    return ObjectState(label, typ, cat, idx, np.asarray(xyz, float), np.asarray(quat, float), info)


def _world(objs):
    from dojo.skills.world import WorldState
    return WorldState("", "", {o.label: o for o in objs}, {}, {})

FOLD_KP = {"left_sleeve": [-0.29, -0.14, 0.775], "right_sleeve": [0.21, -0.28, 0.774],
           "left_chest": [-0.07, -0.045, 0.78], "right_chest": [0.087, -0.089, 0.79],
           "left_hem": [-0.14, -0.25, 0.774], "right_hem": [0.02, -0.29, 0.774],
           "left_shoulder": [-0.044, 0.015, 0.774], "right_shoulder": [0.077, -0.024, 0.774]}


def test_planner_routes_fold_to_the_garment():
    from dojo.skills import planner
    g = _obj("target", "Top_Long", 4, (0.0, -0.12, 0.78), (0.28, 0.17, 0.08), typ="Garment")
    p = planner.plan("Fold the clothes neatly.", _world([g]))
    assert p.skill == "fold_garment" and p.args == {"garment": "target"}
    b = _obj("block", "cube", 1, (0.0, -0.12, 0.78), (0.02, 0.02, 0.02))
    # no garment in the belief: still routed, the skill reads the garment through the provider
    p = planner.plan("Fold the clothes neatly.", _world([b]))
    assert p.skill == "fold_garment" and p.args == {"garment": None}
    assert planner.plan("Fold the paper in half and stack the blocks.", _world([b])).skill != "fold_garment"


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


def test_blown_up_cloth_is_detected():
    from dojo.skills.garment import blown_up
    assert not blown_up({"keypoints": FOLD_KP})
    assert blown_up({"keypoints": {**FOLD_KP, "left_hem": [709.9, 668.0, 448.9]}})


def test_fold_from_one_frozen_read_runs_every_fold():
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


def test_g3_mode_b_reads_the_garment_once_with_noise():
    from dojo.skills.g3 import FrozenGarmentProvider

    class Base:
        n = 0

        def garment(self, label=None):
            Base.n += 1
            return {"label": "target", "keypoints": FOLD_KP, "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "points": None}
    p = FrozenGarmentProvider(Base(), noise_mm=5.0, seed=1)
    a, b = p.garment("target"), p.garment("target")
    assert a is b and Base.n == 1 and a["live"] is False
    d = [np.linalg.norm(np.asarray(a["keypoints"][k])[:2] - np.asarray(FOLD_KP[k])[:2]) for k in FOLD_KP]
    assert 0 < max(d) < 0.03
