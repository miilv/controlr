"""``dojo.perception.base_refit`` (stream ORG2; default off): synthetic views, no images on disk, no GL."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from dojo.perception import base_refit as BR  # noqa: E402
from dojo.perception.cameras import head_camera  # noqa: E402


def _disc(r=0.04, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = np.sqrt(np.linspace(0.0, 1.0, n)) * r
    return np.c_[rr * np.cos(t * 7), rr * np.sin(t * 7), np.full(n, 0.002)]


def _views(centre, r=0.04, seen=True):
    cam = head_camera()
    fg = np.zeros((cam.h, cam.w), np.uint8)
    known = np.ones((cam.h, cam.w), np.uint8)
    vv, uu = np.mgrid[0:cam.h, 0:cam.w]
    X = cam.ray_plane(np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5], 0.767)
    d = np.hypot(X[:, 0] - centre[0], X[:, 1] - centre[1]).reshape(cam.h, cam.w)
    if seen:
        fg[np.nan_to_num(d, nan=9.0) < r] = 1
    return {"cam_head": SimpleNamespace(cam=cam, fg=fg, known=known)}


def _T(xy):
    T = np.eye(4)
    T[:3, 3] = [xy[0], xy[1], 0.765]
    return T


def test_refit_recovers_a_shifted_base():
    band = _disc()
    true = np.array([0.0, -0.15])
    r = BR.refit_xy(_views(true), _T(true + [0.012, -0.008]), band)
    assert r["ok"], r
    np.testing.assert_allclose(np.asarray(r["dxy"]), [-0.012, 0.008], atol=0.004)


def test_gate_keeps_the_pose_when_the_base_is_not_seen():
    band = _disc()
    r = BR.refit_xy(_views((0.0, -0.15), seen=False), _T((0.01, -0.15)), band)
    assert not r["ok"] and r["why"] in ("base not seen", "too few points")


def test_no_shift_when_already_right():
    band = _disc()
    r = BR.refit_xy(_views((0.0, -0.15)), _T((0.0, -0.15)), band)
    assert not r["ok"] and r["dxy"] == [0.0, 0.0]


def test_base_band_takes_the_lowest_band_only():
    V = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.1], [0.05, 0, 0.1], [0, 0.05, 0.1]], float)
    F = np.array([[0, 1, 2], [3, 4, 5]])
    P = BR.base_band(V, F)
    assert len(P) > 3 and P[:, 2].max() < BR.BAND
