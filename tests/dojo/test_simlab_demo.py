"""Support-arm demo planning helpers (pure)."""

import math

import numpy as np
import pytest

from dojo.simlab import demo_gen as D
from dojo.simlab.motion import quat_to_mat
from dojo.simlab.runner import demo_order


def box(dx, dy, dz):
    return np.array([[x, y, z] for x in (-dx / 2, dx / 2) for y in (-dy / 2, dy / 2) for z in (0.0, dz)])


def test_grasp_closes_across_the_narrow_side():
    meta = {"geometry": {"oriented_bbox": {"vertices": box(0.10, 0.04, 0.05).tolist()}}}
    yaw = math.radians(30)
    q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    v = D.world_vertices(meta, [0.2, 0.3, 0.765], q)
    g = D.top_down_grasp(v)
    assert g["width"] == pytest.approx(0.04, abs=0.002)
    d = np.array([math.cos(g["yaw"]), math.sin(g["yaw"])])
    narrow = np.array([-math.sin(yaw), math.cos(yaw)])          # the object's local y in the world
    assert abs(float(d @ narrow)) > 0.99
    assert g["xy"] == pytest.approx([0.2, 0.3], abs=1e-6)
    assert g["top"] == pytest.approx(0.815) and g["bottom"] == pytest.approx(0.765)
    wide = D.top_down_grasp(D.world_vertices({"geometry": {"oriented_bbox": {"vertices": box(0.2, 0.2, 0.2).tolist()}}},
                                             [0, 0, 0.765], [1, 0, 0, 0]))
    assert wide["too_wide"]                                        # wider than the fingers


def test_grasp_band_uses_the_top_of_the_shape():
    base = box(0.12, 0.10, 0.05)                                   # wide body ...
    top = box(0.05, 0.03, 0.02) + [0.0, 0.0, 0.05]                 # ... with a narrow knob on top
    pts = np.concatenate([base, top]) + [0.0, 0.0, 0.765]
    assert D.top_down_grasp(pts)["too_wide"]
    g = D.top_down_grasp(pts, band=0.015)
    assert not g.get("too_wide") and g["width"] == pytest.approx(0.03, abs=0.002)
    assert g["top"] == pytest.approx(0.835) and g["bottom"] == pytest.approx(0.765)


def test_hand_points_down_with_fingers_along_yaw():
    for yaw in (0.0, 0.7, -2.0):
        R = quat_to_mat(D.hand_quat(yaw))
        assert R[:, 2] == pytest.approx([0, 0, -1], abs=1e-9)
        assert R[:, 1] == pytest.approx([math.cos(yaw), math.sin(yaw), 0], abs=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
    p = D.hand_pose([0.1, 0.2, 0.8], 0.0)
    assert p[:3] == pytest.approx([0.1, 0.2, 0.8 + D.HAND_TCP])


def test_step_format_matches_robodojo_reader():
    s = D.step_rec(np.arange(7) * 0.1, np.zeros(7), 0.02)
    assert set(s) == {"support_arm0_joint_state", "support_ee0_joint_state"}
    assert len(s["support_arm0_joint_state"]["position"]) == 7 and s["support_ee0_joint_state"]["position"] == [0.02, 0.02]


def test_demo_order_is_seed_fixed_permutation():
    assert demo_order(100003) == demo_order(100003) and sorted(demo_order(100003)) == list(range(5))
    assert len({tuple(demo_order(s)) for s in range(100000, 100025)}) > 10


def test_retime_paces_like_robodojo_demos():
    path = np.linspace(np.zeros(7), np.r_[0.6, np.zeros(6)], 300)        # cuRobo's slow 300-step path
    p, v = D.retime(path, 0.006)
    dq = np.abs(np.diff(np.vstack([path[:1], p]), axis=0)).max()
    assert dq <= 0.006 + 1e-9 and len(p) < 300
    assert p[-1] == pytest.approx(path[-1]) and abs(v[0, 0]) < abs(v[len(v) // 2, 0])   # eases in
    assert len(D.retime(path[:2] * 0.0, 0.006)[0]) == 40                 # a no-move segment keeps a minimum
