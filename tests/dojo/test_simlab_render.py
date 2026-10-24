"""Proxy renderer pure helpers: RoboDojo's camera Euler convention, PNG writer."""

import math

import numpy as np
import pytest

from dojo.simlab.motion import quat_to_mat
from dojo.simlab.render_d0_proxy import euler_xyz_intrinsic_deg_to_quat, write_png


def rx(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize("e", [(30, 0, 0), (0, -80, -90), (12, 34, -56)])
def test_euler_is_robodojo_intrinsic_xyz(e):
    # RoboDojo utils.rotations.euler_angles_to_quat(degrees=True): scipy "XYZ" (intrinsic) = Rx Ry Rz
    R = quat_to_mat(euler_xyz_intrinsic_deg_to_quat(e))
    assert R == pytest.approx(rx(e[0]) @ ry(e[1]) @ rz(e[2]), abs=1e-12)


def test_head_camera_looks_forward_and_down():
    R = quat_to_mat(euler_xyz_intrinsic_deg_to_quat((30, 0, 0)))
    view = R @ np.array([0.0, 0.0, -1.0])          # USD camera looks along -z
    assert view == pytest.approx([0.0, 0.5, -math.sqrt(3) / 2])


def test_png_writer_round_trip(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    img = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    write_png(tmp_path / "a.png", img)
    assert np.asarray(PIL.open(tmp_path / "a.png")).tolist() == img.tolist()
