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
