"""Held-object pose: show-pose geometry, DEV data math (pure numpy), and the estimator on a synthetic scene (cv2 +
scipy; the robot, cameras and asset are stubbed, so no catalogue / URDF is needed)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import held_data as HD
from dojo.perception import show_pose as SP


def test_look_at_camera_points_at_the_target_with_image_up_world_up():
    eye, tgt = np.array([0.2, -0.3, 1.0]), np.array([0.0, -0.1, 0.9])
    T = SP.look_at_camera(eye, tgt)
    z = T[:3, 2]
    assert np.allclose(z, (tgt - eye) / np.linalg.norm(tgt - eye))
    assert T[:3, 1] @ np.array([0, 0, 1.0]) < 0                 # image down points down in the world
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    T_tcp = SP.tcp_for_camera(T)
    assert np.allclose(SP.camera_for_tcp(T_tcp), T)


def test_observer_looks_sit_on_the_observer_side_at_the_chosen_distance():
    tgt = np.array([-0.15, -0.1, 0.95])
    looks = SP.observer_looks(tgt, "right", extent_m=0.06)
    assert looks and all(abs(np.linalg.norm(L["eye"] - tgt) - L["dist"]) < 1e-9 for L in looks)
    first = looks[0]
    assert first["eye"][0] > tgt[0]                              # the right arm looks from its own (+x) side
    cam = first["T_wc"]
    assert np.allclose(SP.camera_for_tcp(first["T_tcp"]), cam)


def test_show_poses_put_the_object_centre_at_the_show_point():
    T_tcp = np.eye(4)
    T_tcp[:3, 3] = [-0.2, -0.05, 0.9]
    T_tcp_obj = np.eye(4)
    T_tcp_obj[:3, 3] = [0.03, 0.0, 0.01]
    bmin, bmax = np.array([-0.01, -0.02, -0.005]), np.array([0.01, 0.02, 0.015])
    for s in SP.show_poses("left", T_tcp, T_tcp_obj, bmin, bmax):
        c = SP.object_centre_world(s["T_tcp"], T_tcp_obj, bmin, bmax)
        assert np.allclose(c, SP.show_point("left"), atol=1e-9)
    assert SP.show_point("left")[0] < SP.show_point("right")[0]


def test_quaternion_and_rotvec_roundtrips():
    R = HD.rotvec_R([0.3, -0.2, 0.9])
    assert np.allclose(HD.quat_wxyz_to_R(HD.R_to_quat_wxyz(R)), R)
    assert np.allclose(HD.R_rotvec(R), [0.3, -0.2, 0.9])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [0.1, 0.2, 0.3]
    assert np.allclose(HD.pose7_to_T(HD.T_to_pose7(T)), T)
    assert np.allclose(HD.inv_T(T) @ T, np.eye(4))


def test_pose_errors_lead_and_axis():
    T = np.eye(4)
    lead, axis = np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 1.0])
    e = HD.pose_errors(T, T, lead, axis)
    assert e["lead_norm_mm"] == 0 and e["axis_deg"] == 0
    T2 = np.eye(4)
    T2[:3, :3] = HD.rotvec_R([0.0, math.radians(2.0), 0.0])     # 2 deg about y through the origin
    e = HD.pose_errors(T2, T, lead, axis)
    assert e["axis_deg"] == pytest.approx(2.0, abs=1e-6)
    assert e["lead_norm_mm"] == pytest.approx(50 * math.radians(2.0), rel=1e-3)


def test_lead_axis_from_target_is_the_lowest_end_and_world_down():
    V = np.array([[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (0.0, 0.06)])
    T = np.eye(4)
    T[:3, :3] = HD.rotvec_R([math.pi, 0, 0])                     # upside down at the target: +z of the object is down
    lead, axis = HD.lead_axis_from_target(V, T)
    assert np.allclose(axis, [0, 0, 1.0], atol=1e-9)
    assert np.allclose(lead, [0, 0, 0.06])


# ---------------------------------------------------------------------------------------------- the estimator
def _synthetic(monkeypatch):
    """A 30 x 20 x 50 mm box held 2 cm beyond a TCP, two cameras (a close 'own' view along the approach, an
    'observer' from the side), a textured background; robot and cameras stubbed."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import held as H
    from dojo.perception.cameras import Camera, intrinsics

    hx, hy, hz = 0.025, 0.010, 0.015
    V = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    asset = SimpleNamespace(V=V, F=F, Vs=V, Fs=F, Vg=V, Fg=F,
                            info={"bbox_min": (-hx, -hy, -hz), "bbox_max": (hx, hy, hz), "extents": (2 * hx, 2 * hy, 2 * hz)})
    monkeypatch.setattr(H, "load_asset", lambda key: asset)
    monkeypatch.setattr(H, "gl_roi", lambda size: None)
    monkeypatch.setattr(H, "robot_unknown", lambda cam, *a, **k: np.zeros((cam.h, cam.w), np.uint8))
    monkeypatch.setattr(H, "finger_table", lambda: np.array([[0.0, 0.0, 0.0], [1.0, 0.088, 0.0]]))
    T_tcp = np.eye(4)
    T_tcp[:3, 3] = [0.0, -0.1, 1.0]
    T_tcp[:3, :3] = HD.rotvec_R([0.0, math.pi / 2, 0.0])            # approach (x) pointing down
    monkeypatch.setattr(H, "tcp_T", lambda arm, q, model=None: T_tcp)
    K = intrinsics(13.0, 20.955)
    T_gt = np.eye(4)
    T_gt[:3, 3] = [0.035, 0.0005, 0.001]
    T_gt[:3, :3] = HD.rotvec_R([0.0, math.radians(3.0), 0.0])
    c_obj = (T_tcp @ T_gt)[:3, 3]
    cams = {"cam_left_wrist": Camera("cam_left_wrist", K, SP.look_at_camera(c_obj + [0.0, -0.06, 0.09], c_obj)),
            "cam_right_wrist": Camera("cam_right_wrist", K, SP.look_at_camera(c_obj + [0.16, -0.05, 0.03], c_obj))}
    monkeypatch.setattr(H, "cameras_from_state", lambda state, model=None: cams)
    monkeypatch.setattr(H, "X5Model", SimpleNamespace(default=lambda: None))
    rng = np.random.default_rng(0)
    images = {}
    for name, cam in cams.items():
        img = (rng.normal(0, 12, (480, 640, 3)) + [90, 60, 40]).clip(0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        m = H.silhouette(cam, H.transform(V, T_tcp @ T_gt), F)
        img[m > 0] = (215, 210, 200)
        images[name] = img
    state = {"left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6),
             "left_ee_joint_state": np.zeros(1), "right_ee_joint_state": np.ones(1)}
    return H, {"images": images, "state": state}, T_gt


def test_held_pose_recovers_a_perturbed_prior_on_a_synthetic_scene(monkeypatch):
    H, frame, T_gt = _synthetic(monkeypatch)
    D = np.eye(4)
    D[:3, :3] = HD.rotvec_R([0.0, math.radians(-3.0), math.radians(1.0)])
    D[:3, 3] = [0.002, 0.0, -0.0015]
    T_prior = D @ T_gt
    lead, axis = np.array([0.025, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    r = H.held_pose(frame, "left", "Rigid/box/00000", T_prior, {"lead_obj": lead, "axis_obj": axis})
    assert r["ok"], r["why"]
    e0 = HD.pose_errors(T_prior, T_gt, lead, axis)
    e = HD.pose_errors(r["T_tcp_obj"], T_gt, lead, axis)
    assert e0["lead_norm_mm"] > 2.0
    assert e["lead_norm_mm"] < 0.5 and e["axis_deg"] < 1.0
    assert set(r["views"]) == {"cam_left_wrist", "cam_right_wrist"}
    assert len(r["sigma"]["lead_mm"]) == 3 and r["sigma"]["axis_deg"] >= 0


def test_held_pose_without_views_keeps_the_prior(monkeypatch):
    H, frame, T_gt = _synthetic(monkeypatch)
    r = H.held_pose(frame, "left", "Rigid/box/00000", T_gt, {"exclude_cams": ("cam_left_wrist", "cam_right_wrist")})
    assert not r["ok"] and r["why"] == "no_views"
    assert np.allclose(r["T_tcp_obj"], T_gt)


def test_show_pose_candidates_reach_their_targets_and_keep_the_arms_apart():
    from dojo.skills.kinematics import X5Kin

    kin = X5Kin()
    q_l = np.array([0.0, 1.2, 1.0, -0.6, 0.0, 0.0])
    T_tcp = kin.tcp("left", q_l)
    T_tcp_obj = np.eye(4)
    T_tcp_obj[:3, 3] = [0.03, 0.0, 0.0]
    for c in SP.show_poses("left", T_tcp, T_tcp_obj, [-0.02, -0.01, -0.01], [0.02, 0.01, 0.01]):
        r = kin.ik("left", c["T_tcp"], q_l)
        if r.ok:
            assert np.linalg.norm(kin.tcp("left", r.q)[:3, 3] - c["T_tcp"][:3, 3]) < 0.005
    looks = SP.observer_looks((T_tcp @ T_tcp_obj)[:3, 3], "right", 0.04)
    dirs = SP.observer_dirs(T_tcp)
    assert dirs[0][0] == "end_on" and all(abs(np.linalg.norm(u) - 1) < 1e-9 for _, u in dirs)
    assert all(L["arm"] == "right" for L in looks)
    assert SP.arm_clearance(kin, q_l, np.zeros(6)) > 0.1


def test_eval_summary_and_sigma_calibration_on_fake_rows():
    from dojo.perception import held_eval as HE

    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        s = rng.uniform(0.1, 0.5)
        e = rng.normal(0, 1.5 * s, 3)                       # errors 1.5x the reported sigma
        rows.append({"task": "t", "kind": "hover", "views_set": "all", "prior": "perturb", "seed": str(100000 + i),
                     "ok": True, "why": "", "t_s": 1.0, "sigma": {"lead_mm": [s] * 3, "axis_deg": 1.0},
                     "err_prior": {"lead_norm_mm": 2.0}, "err_fit": {"lead_mm": e.tolist(),
                                                                       "lead_norm_mm": float(np.linalg.norm(e)),
                                                                       "axis_deg": 0.5, "rot_deg": [0, 0.4, 0]},
                     "err_out": {"lead_mm": e.tolist(), "lead_norm_mm": float(np.linalg.norm(e)), "axis_deg": 0.5,
                                 "rot_deg": [0, 0.4, 0]}})
    S = HE.summarize(rows)
    assert S[0]["n"] == 80 and S[0]["ok"] == 80
    c = HE.calibrate(rows)
    assert 1.3 < c["all"]["k_lead"] < 2.6                 # the factor recovers the 1.5x under-statement
    for h in (0, 1):
        assert c[f"fit_on_{h}"]["held_out_cover_lead_axis"][0] > 0.85
