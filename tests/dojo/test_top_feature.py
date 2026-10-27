"""dojo.perception.top_feature: the keyhole template from a mesh and the yaw read from a synthetic top-down view."""
import math

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from dojo.perception import top_feature as TF  # noqa: E402
from dojo.perception.cameras import TABLE_Z, Camera  # noqa: E402


def _in_keyhole(x, y):
    """insert_key's lock opening (asset frame, m): a 4.4 mm bulb at x = -2 mm and a 6 mm channel to x = +12 mm."""
    return (math.hypot(x + 0.002, y) < 0.0044) or (0.0 <= x <= 0.012 and abs(y) <= 0.003)


def _lock_mesh(hole=_in_keyhole, r=0.05, h=0.055, cell=0.001):
    """A disc top at z = h made of small quads (cells inside ``hole`` left out) over a bottom fan at z = 0."""
    V, F = [], []
    n = int(r / cell)
    for i in range(-n, n):
        for j in range(-n, n):
            x0, y0 = i * cell, j * cell
            cx, cy = x0 + cell / 2, y0 + cell / 2
            if math.hypot(cx, cy) > r or hole(cx, cy):
                continue
            k = len(V)
            V += [(x0, y0, h), (x0 + cell, y0, h), (x0 + cell, y0 + cell, h), (x0, y0 + cell, h)]
            F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    k = len(V)
    V.append((0.0, 0.0, 0.0))
    for a in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        V.append((r * math.cos(a), r * math.sin(a), 0.0))
    for m in range(64):
        F.append((k, k + 1 + (m + 1) % 64, k + 1 + m))
    return np.asarray(V, float), np.asarray(F, np.int64)


class _Shape:
    """The parts of ``inverse_layout.ShapeModel`` the read uses (rest pose = asset frame, lowest point on z = 0)."""
    key, tag, sym, zmin = "Geometry/lock/00000", "up", 36, 0.0
    R0 = np.eye(3)

    def pose(self, x, y, yaw, z_off=0.0):
        T = np.eye(4)
        c, s = math.cos(yaw), math.sin(yaw)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = (x, y, TABLE_Z + z_off)
        return T


def test_template_from_the_mesh():
    sm = _Shape()
    V, F = _lock_mesh()
    f = TF.top_feature_from_mesh(V, F, sm.pose(0, 0, 0))
    assert f is not None
    assert f.z_top == pytest.approx(0.055, abs=1e-6) and f.radius == pytest.approx(0.05, abs=0.002)
    assert abs(((f.axis_deg + 90) % 180) - 90) < 3          # the opening runs along x
    assert f.hole[:, 0].mean() > 0.001                       # its mass sits on the channel side (+x)
    assert f.elongation > TF.MIN_ELONGATION


def test_no_feature_without_an_elongated_opening():
    sm = _Shape()
    V, F = _lock_mesh(hole=lambda x, y: False)               # a plain disc
    assert TF.top_feature_from_mesh(V, F, sm.pose(0, 0, 0)) is None
    V, F = _lock_mesh(hole=lambda x, y: math.hypot(x, y) < 0.006)   # a round hole (a ring): no yaw in it
    assert TF.top_feature_from_mesh(V, F, sm.pose(0, 0, 0)) is None


def _top_down_view(sm, feat, x, y, yaw, height=0.18):
    """A copper disc, shaded across (dark on one side), with a black keyhole seen straight down from ``height`` above
    the top."""
    import cv2

    K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
    T_wc = np.eye(4)
    T_wc[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    z_top = TABLE_Z + feat.z_top
    T_wc[:3, 3] = (x + 0.004, y - 0.006, z_top + height)
    cam = Camera("cam_test", K, T_wc)
    img = np.full((cam.h, cam.w, 3), (120, 70, 40), np.uint8)              # table
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    uv, _ = cam.project(np.c_[x + feat.radius * np.cos(th), y + feat.radius * np.sin(th), np.full(120, z_top)])
    cv2.fillPoly(img, [np.round(uv - 0.5).astype(np.int32)], (190, 110, 80))  # copper top
    shade = np.clip(np.linspace(0.35, 1.0, cam.w)[None, :, None] * np.ones((cam.h, 1, 1)), 0, 1)
    img = (img * shade).astype(np.uint8)              # the gripper's shadow: a third as bright on one side
    H = feat.hole_world(sm.pose(x, y, yaw), sm.pose(0, 0, 0))
    uvh, _ = cam.project(np.c_[H, np.full(len(H), z_top)])
    for u, v in np.round(uvh - 0.5).astype(int):
        cv2.circle(img, (u, v), 1, (10, 10, 10), -1)
    return img, cam


@pytest.mark.parametrize("yaw_deg", [10.7, -63.0, 151.0])
def test_read_yaw_from_a_top_down_view(monkeypatch, yaw_deg):
    sm = _Shape()
    V, F = _lock_mesh()
    feat = TF.top_feature_from_mesh(V, F, sm.pose(0, 0, 0), sm.key, sm.tag)
    monkeypatch.setattr(TF, "_cached", lambda key, tag: feat)
    x, y = 0.157, -0.167
    img, cam = _top_down_view(sm, feat, x, y, math.radians(yaw_deg))
    r = TF.read_yaw(img, cam, sm, x + 0.003, y - 0.002)       # an outline fit 3.6 mm off (5.1 wrist hover: 0.3-7 mm)
    assert r["ok"] and r["sign_ok"], r
    err = (math.degrees(r["yaw"]) - yaw_deg + 180) % 360 - 180
    assert abs(err) < 2.0, r                                  # the full turn: axis and bulb side
    assert math.hypot(r["x"] - x, r["y"] - y) < 0.0015
