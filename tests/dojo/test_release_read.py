"""release_read (stream W): the 6-DoF pose of a just-released object on a synthetic two-camera scene (cv2 + scipy; the
robot, cameras and asset are stubbed, so no catalogue / URDF / GL is needed)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import held_data as HD
from dojo.perception import show_pose as SP


def _scene(monkeypatch, cams_used=("cam_left_wrist", "cam_head")):
    """A 50 x 30 x 70 mm box released at the prior, really swung 18 deg about world x and shifted 9 mm (a mug on its
    hook): seen by the releasing (left) wrist from 15 cm and by a far 'head' camera."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import held as H
    from dojo.perception import release_read as RR
    from dojo.perception.cameras import Camera, intrinsics

    hx, hy, hz = 0.025, 0.015, 0.035
    V = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    asset = SimpleNamespace(key="Rigid/box/00000", V=V, F=F, Vs=V, Fs=F, Vg=V, Fg=F,
                            info={"bbox_min": (-hx, -hy, -hz), "bbox_max": (hx, hy, hz)})
    for mod in (H, RR):
        monkeypatch.setattr(mod, "load_asset", lambda key: asset)
    monkeypatch.setattr(H, "gl_roi", lambda size: None)
    monkeypatch.setattr(RR, "gl", lambda: None)
    monkeypatch.setattr(RR, "robot_unknown", lambda cam, *a, **k: np.zeros((cam.h, cam.w), np.uint8))
    T_prior = np.eye(4)
    T_prior[:3, 3] = [0.0, -0.1, 0.95]
    T_gt = np.eye(4)
    T_gt[:3, :3] = HD.rotvec_R([math.radians(18.0), 0.0, 0.0])
    T_gt[:3, 3] = T_prior[:3, 3] + [0.004, 0.008, -0.003]
    c = T_prior[:3, 3]
    K = intrinsics(13.0, 20.955)
    all_cams = {"cam_left_wrist": Camera("cam_left_wrist", K, SP.look_at_camera(c + [-0.05, -0.08, 0.12], c)),
                "cam_head": Camera("cam_head", K, SP.look_at_camera(c + [0.0, -0.55, 0.35], c))}
    cams = {k: v for k, v in all_cams.items() if k in cams_used}
    monkeypatch.setattr(RR, "cameras_from_state", lambda state: cams)
    rng = np.random.default_rng(1)
    images = {}
    for name, cam in cams.items():
        img = (rng.normal(0, 10, (480, 640, 3)) + [100, 70, 50]).clip(0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        m = H.silhouette(cam, H.transform(V, T_gt), F)
        img[m > 0] = (220, 215, 205)
        images[name] = img
    state = {"left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6),
             "left_ee_joint_state": np.ones(1), "right_ee_joint_state": np.ones(1)}
    pose7 = list(T_prior[:3, 3]) + list(HD.R_to_quat_wxyz(T_prior[:3, :3]))
    return RR, {"images": images, "state": state}, pose7, T_gt


def _err(pose7, T_gt):
    T = HD.pose7_to_T(pose7)
    return 1000 * float(np.linalg.norm(T[:3, 3] - T_gt[:3, 3])), math.degrees(
        float(np.linalg.norm(HD.R_rotvec(T[:3, :3].T @ T_gt[:3, :3]))))


def test_release_read_recovers_a_swung_object(monkeypatch):
    RR, frame, pose7, T_gt = _scene(monkeypatch)
    r = RR.release_read(frame, "Rigid/box/00000", pose7, "left")
    assert r["ok"], r
    d0, a0 = _err(pose7, T_gt)
    d1, a1 = _err(r["pose7"], T_gt)
    assert d0 > 8 and a0 > 15
    assert d1 < 2.5 and a1 < 3.0, (d1, a1)
    assert set(r["views"]) == {"cam_left_wrist", "cam_head"} and min(r["ious"].values()) > 0.8


def test_release_read_keeps_the_prior_with_only_a_far_view(monkeypatch):
    # the head alone (~2.4 mm/px on the robot) does not pin 6 DoF: never adopted
    RR, frame, pose7, T_gt = _scene(monkeypatch, cams_used=("cam_head",))
    r = RR.release_read(frame, "Rigid/box/00000", pose7, "left")
    assert not r["ok"] and r["pose7"] == pose7


def test_release_read_without_views_keeps_the_prior(monkeypatch):
    RR, frame, pose7, T_gt = _scene(monkeypatch, cams_used=())
    r = RR.release_read(frame, "Rigid/box/00000", pose7, "left")
    assert not r["ok"] and r["why"] == "no_views" and r["pose7"] == pose7


def _search_stubs(monkeypatch, fit_at, wrist_ok=False, pre_occupied=False):
    """release_search with its head view, fits and 6-DoF read stubbed: a 200 x 200 px head view whose foreground holds
    one blob at pixel (150, 100); ``fit_at`` = {centre xy (rounded) or "pre": (T, iou)} of the head fits."""
    pytest.importorskip("cv2")
    from dojo.perception import release_read as RR

    fg = np.zeros((200, 200), np.uint8)
    fg[90:110, 140:160] = 1

    class Cam:
        w = h = 200

        def ray_plane(self, uv, z):                          # 1 px = 1 mm, origin at pixel (100, 100)
            uv = np.asarray(uv, float).reshape(-1, 2)
            return np.c_[(uv[:, 0] - 100) / 1000.0, (uv[:, 1] - 100) / 1000.0, np.full(len(uv), z)]

    v = SimpleNamespace(cam=Cam(), fg=fg, known=np.ones_like(fg), robot=np.zeros_like(fg))
    monkeypatch.setattr(RR, "_head_view", lambda frame, bgs, thr: v)
    V = np.array([[0, 0, 0], [0, 0, 0.04]], float)
    monkeypatch.setattr(RR, "load_asset", lambda key: SimpleNamespace(V=V, Vs=V, Fs=np.zeros((0, 3), int)))
    # the predicted head silhouette at the pre-grasp pose: the blob's pixels when the object is still there
    monkeypatch.setattr(RR, "silhouette", lambda cam, Vw, F, **k: fg.copy() if pre_occupied else np.zeros_like(fg))
    calls = []

    def head_fit(v_, key, T_init, om, yaws=1, iters=60, centre_xy=None):
        k = "pre" if centre_xy is None else tuple(round(c, 2) for c in centre_xy)
        calls.append(k)
        T, iou = fit_at.get(k, (None, 0.0))
        if T is None:
            return None
        s = np.zeros((200, 200), np.uint8)
        if k != "pre" or pre_occupied:                      # a fit's silhouette lies on the blob only where it is
            s[90:110, 140:160] = 1
        return T, iou, s

    monkeypatch.setattr(RR, "_head_fit", head_fit)
    monkeypatch.setattr(RR, "release_read", lambda *a, **k: {"ok": wrist_ok, "why": None if wrist_ok else "one_far_view",
                                                             "pose7": [9.0] * 7, "ious": {"cam_left_wrist": 0.9}})
    return RR, calls


def _T(x, y, z=0.78):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def test_release_search_takes_the_pre_grasp_pose_of_a_missed_grasp(monkeypatch):
    RR, calls = _search_stubs(monkeypatch, {"pre": (_T(0.2, 0.1), 0.8)}, pre_occupied=True)
    prior = [0.0, 0.0, 0.9, 1, 0, 0, 0]
    r = RR.release_search({}, "Rigid/x/00000", prior, "left", pre_pose7=[0.2, 0.1, 0.78, 1, 0, 0, 0], head_bgs={"x": 1})
    assert r["ok"] and r["via"] == "pre_grasp" and r["pose7"][:3] == pytest.approx([0.2, 0.1, 0.78])
    assert calls == ["pre"]                                   # found there: no head search
    # a thin object (a mallet): the head fit's IoU is noisy, its silhouette there is foreground: still taken
    RR, calls = _search_stubs(monkeypatch, {"pre": (_T(0.21, 0.1), 0.3)}, pre_occupied=True)
    r = RR.release_search({}, "Rigid/x/00000", prior, "left", pre_pose7=[0.2, 0.1, 0.78, 1, 0, 0, 0], head_bgs={"x": 1})
    assert r["ok"] and r["via"] == "pre_grasp" and r["pose7"][:3] == pytest.approx([0.21, 0.1, 0.78])   # the fit, 10 mm from it


def test_release_search_finds_a_fallen_object_near_the_release_and_ignores_far_blobs(monkeypatch):
    # the blob's centre is 50 mm from the release in x: found by the search (the pre-grasp check fails: iou 0.2)
    RR, calls = _search_stubs(monkeypatch, {"pre": (_T(0.3, 0.3), 0.2), (0.05, 0.0): (_T(0.05, 0.0), 0.75)},
                              wrist_ok=True)
    prior = [0.0, 0.0, 0.9, 1, 0, 0, 0]
    r = RR.release_search({}, "Rigid/x/00000", prior, "left", pre_pose7=[0.3, 0.3, 0.78, 1, 0, 0, 0], head_bgs={"x": 1})
    assert r["ok"] and r["via"] == "head_search" and r["pose7"] == [9.0] * 7   # a wrist saw it: the 6-DoF read's pose
    assert calls == ["pre", (0.05, 0.0)]
    # the same blob beyond the search radius: nothing
    RR, calls = _search_stubs(monkeypatch, {(0.05, 0.0): (_T(0.05, 0.0), 0.75)})
    r = RR.release_search({}, "Rigid/x/00000", prior, "left", head_bgs={"x": 1}, cfg={"radius_mm": 30.0})
    assert not r["ok"] and r["why"] == "not_found" and r["pose7"] == prior and calls == []
