"""task_state.laid_pose: where a knocked-over object, laid face up where it stood, actually lies in the head view (the
new tile-face pixels since frame 0, searched around the laid pose); the service op names no task."""
import inspect

import numpy as np
import pytest

EXT = np.array([0.026, 0.035, 0.02])          # a lying tile's box (m)
Z = 0.765 + EXT[2] / 2


class _Box:
    """A shape model whose silhouette is the projected box (cv2 convex fill)."""

    def sil(self, cam, T):
        import cv2
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * EXT / 2
        P = c @ T[:3, :3].T + T[:3, 3]
        uv, _ = cam.project(P)
        m = np.zeros((cam.h, cam.w), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        return m


def _scene(true_xy, vacated_xy):
    from dojo.perception.cameras import head_camera, pose_to_T
    cam = head_camera()
    before = np.zeros((cam.h, cam.w, 3), np.uint8)
    before[:] = (110, 60, 35)                                     # the table
    after = before.copy()
    sb = _Box().sil(cam, pose_to_T([vacated_xy[0], vacated_xy[1], Z + 0.01], [1, 0, 0, 0])) > 0
    before[sb] = (40, 150, 60)                                    # a standing tile's green back, gone afterwards
    st = _Box().sil(cam, pose_to_T([true_xy[0], true_xy[1], Z], [0.0, 0.0, 0.0, 1.0])) > 0
    after[st] = (235, 232, 225)                                   # the new face-up tile: white
    return cam, before, after


def test_laid_pose_finds_the_tile_where_it_fell_not_where_it_stood():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception.task_state import laid_pose
    cam, before, after = _scene((0.0, 0.002), (0.0, 0.05))
    r = laid_pose(before, after, "Rigid/mahjong/00005", [0.0, 0.05, Z, 1.0, 0.0, 0.0, 0.0], cam=cam, shape=_Box())
    assert r["ok"] and r["iou"] > 0.8
    assert np.hypot(r["pos"][0] - 0.0, r["pos"][1] - 0.002) < 0.004 and abs(r["pos"][2] - Z) < 1e-9
    assert 44 < r["shift_mm"] < 52
    cam, before, after = _scene((0.03, 0.09), (0.0, 0.05))       # the other way: no fall direction assumed
    r = laid_pose(before, after, "k", [0.0, 0.05, Z, 1.0, 0.0, 0.0, 0.0], cam=cam, shape=_Box())
    assert r["ok"] and np.hypot(r["pos"][0] - 0.03, r["pos"][1] - 0.09) < 0.004


def test_laid_pose_stays_put_without_new_face_pixels_or_a_good_match():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception.task_state import laid_pose
    cam, before, after = _scene((0.0, 0.002), (0.0, 0.05))
    r = laid_pose(before, before.copy(), "k", [0.0, 0.05, Z, 1.0, 0.0, 0.0, 0.0], cam=cam, shape=_Box())
    assert not r["ok"] and r["why"] == "no new face pixels"
    cam2, b2, a2 = _scene((0.3, -0.2), (0.0, 0.05))               # white far outside the search window: not counted
    r = laid_pose(b2, a2, "k", [0.0, 0.05, Z, 1.0, 0.0, 0.0, 0.0], cam=cam2, shape=_Box())
    assert not r["ok"] and r["why"] == "no new face pixels"
    cam3, b3, a3 = _scene((0.0, 0.002), (0.0, 0.05))
    a3[:60, :] = (240, 240, 240)                                  # an arm passing high in the view: outside the window
    r = laid_pose(b3, a3, "k", [0.0, 0.05, Z, 1.0, 0.0, 0.0, 0.0], cam=cam3, shape=_Box())
    assert r["ok"] and np.hypot(r["pos"][0], r["pos"][1] - 0.002) < 0.004
    uv, _ = cam3.project(np.array([[0.0, 0.002, Z]]))
    u, v = np.round(uv[0]).astype(int)
    a4 = b3.copy()
    a4[v - 2:v + 2, u - 2:u + 2] = (235, 232, 225)                # a few new white pixels only: no tile-sized match
    r = laid_pose(b3, a4, "k", [0.0, 0.05, Z, 1.0, 0.0, 0.0, 0.0], cam=cam3, shape=_Box())
    assert not r["ok"] and "iou" in r["why"]


def test_the_locate_op_names_no_task():
    pytest.importorskip("cv2")
    from dojo.perception import task_state
    from dojo.runtime import perception_service as PS
    for src in (inspect.getsource(PS._Perception.locate_laid), inspect.getsource(task_state.laid_pose)):
        assert "make_kong" not in src and "mahjong" not in src and "vocabulary" not in src
