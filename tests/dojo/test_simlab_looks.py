"""look_render geometry (pure)."""

import math

import numpy as np
import pytest

from dojo.simlab import look_render as LR
from dojo.simlab.layout import quat_mul, quat_z


def test_quat_matrix_round_trip():
    for q in ([1, 0, 0, 0], quat_z(170), quat_mul(quat_z(30), [math.cos(1.2), math.sin(1.2), 0, 0])):
        q = np.asarray(q, float) / np.linalg.norm(q)
        q2 = LR.mat_to_quat(LR.quat_to_mat(q))
        assert min(np.abs(q2 - q).max(), np.abs(q2 + q).max()) < 1e-9


def test_roll_keeps_centre_and_optical_axis():
    T = LR.pose_to_T([0.1, -0.2, 0.9, *quat_mul(quat_z(40), [math.cos(0.4), math.sin(0.4), 0, 0])])
    for deg in (30, -90, 180):
        Tr = LR.roll_cv(T, deg)
        cm, ax = LR.cam_error(T, Tr)
        assert cm == pytest.approx(0.0, abs=1e-9) and ax == pytest.approx(0.0, abs=1e-6)


def test_backoff_looks_at_target_from_45_deg():
    T_req = np.eye(4)
    T_req[:3, 3] = [-0.3, -0.4, 1.0]
    la = np.array([-0.1, -0.1, 0.78])
    Tb = LR.backoff_cam(T_req, la)
    z = Tb[:3, 2]
    assert np.linalg.norm(la - Tb[:3, 3]) == pytest.approx(0.18)
    assert math.degrees(math.asin(-z[2])) == pytest.approx(45.0)
    assert np.allclose(Tb[:3, 3] + 0.18 * z, la)
    assert Tb[2, 1] < 0                                     # image y points down
    assert np.allclose(Tb[:3, :3] @ Tb[:3, :3].T, np.eye(3))
