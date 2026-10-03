"""Hollow box obstacle geometry (controlr.robot.obstacles)."""

from __future__ import annotations

import numpy as np
import pytest

from controlr.robot.kinematics import UR3Kinematics
from controlr.robot.obstacles import BoxObstacle, body_clearance, body_points, box_from_bin_info

BIN = {"lower": [-0.5705, -0.0651, -0.0055], "upper": [-0.2105, 0.1949, 0.1805],
       "center": [-0.3905, 0.0649, -0.0095], "yaw": 0.0, "wall": 0.02, "dynamic": True}


def test_box_from_bin_info_matches_phantom_geometry():
    b = box_from_bin_info(BIN)
    assert b.outer == pytest.approx((0.40, 0.30, 0.19)) and b.wall_xy == pytest.approx((0.02, 0.02))
    assert b.floor == pytest.approx(0.004) and b.movable
    assert b.center == pytest.approx((-0.3905, 0.0649, -0.0095))
    assert BoxObstacle.from_dict(b.to_dict()) == b


def test_inside_the_opening_is_free_walls_are_not():
    b = box_from_bin_info(BIN)
    c = np.array([-0.3905, 0.0649, 0.10])
    d, part = b.clearance([c])
    assert d == pytest.approx(0.10 - 0.0055 + 0.004 - 0.004, abs=0.01) or d > 0.09     # far from every wall
    near_wall_inner = np.array([-0.3905, -0.065 + 0.01, 0.10])                         # 10 mm inside the -y wall
    d, part = b.clearance([near_wall_inner])
    assert d == pytest.approx(0.01, abs=1e-3) and part == "near (-y) wall"
    assert b.clearance([[-0.3905, -0.075, 0.10]])[0] == 0.0                            # in the wall
    above_rim = [-0.3905, -0.075, 0.25]
    assert b.clearance([above_rim])[0] == pytest.approx(0.25 - 0.1805, abs=1e-3)


def test_yawed_box_and_body_points():
    b = box_from_bin_info({**BIN, "yaw": np.radians(30)})
    loc = b.to_local([b.center])[0]
    np.testing.assert_allclose(loc, 0.0, atol=1e-12)
    kin = UR3Kinematics()
    q = np.array([0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406])
    pts = body_points(kin, q)
    tcp = kin.fk(q)[0]
    assert np.all(np.linalg.norm(pts - tcp, axis=1) > 0.06) and len(pts) > 30
    d, name, part = body_clearance(kin, q, [box_from_bin_info(BIN)])
    assert d > 0.05 and name == "the blue box"
    assert body_clearance(kin, q, [])[0] == float("inf")
