"""edge_yaw (default off): a cube's yaw from the image gradient along its rendered outline."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from types import SimpleNamespace  # noqa: E402

from dojo.perception import edge_yaw as ey  # noqa: E402


class Square:
    """A top-down square of side 40 px at the image centre, rotated by yaw (four-fold symmetric)."""
    sym = 4

    def pose(self, x, y, yaw, z_off=0.0):
        return yaw

    def sil(self, cam, yaw):
        m = np.zeros((100, 100), np.uint8)
        box = cv2.boxPoints(((50.0, 50.0), (40.0, 40.0), float(np.degrees(yaw))))
        cv2.fillConvexPoly(m, box.astype(np.int32), 1)
        return m


def _view(true_deg, texture=0.0, seed=0):
    img = np.full((100, 100, 3), 90, np.uint8)
    rng = np.random.default_rng(seed)
    if texture:
        img = np.clip(img + rng.normal(0, texture, img.shape), 0, 255).astype(np.uint8)
    m = Square().sil(None, np.radians(true_deg))
    img[m > 0] = (200, 60, 60)
    return SimpleNamespace(cam=None, img=img, known=np.ones((100, 100), np.uint8), robot=np.zeros((100, 100), np.uint8))


def test_the_outline_edges_give_the_cube_yaw():
    v = _view(30.0, texture=6.0)
    grads = {"cam_head": ey.gradient(v.img)}
    r = ey.edge_yaw(Square(), 0.0, 0.0, np.radians(0.0), 0.0, {"cam_head": v}, {}, grads)
    assert r["changed"] and abs(((np.degrees(r["yaw"]) - 30.0) + 45) % 90 - 45) < 3.0
    r = ey.edge_yaw(Square(), 0.0, 0.0, np.radians(29.0), 0.0, {"cam_head": v}, {}, grads)
    assert abs(np.degrees(r["yaw"]) - 29.0) < 2.5                                   # close already: small or no change


def test_a_flat_view_changes_nothing_and_other_shapes_are_skipped():
    v = _view(0.0)
    v.img[:] = 90                                                                   # no edges at all
    grads = {"cam_head": ey.gradient(v.img)}
    r = ey.edge_yaw(Square(), 0.0, 0.0, np.radians(10.0), 0.0, {"cam_head": v}, {}, grads)
    assert r is None or not r["changed"]
    sq = Square()
    sq.sym = 2
    assert ey.edge_yaw(sq, 0.0, 0.0, 0.0, 0.0, {"cam_head": v}, {}, grads) is None
