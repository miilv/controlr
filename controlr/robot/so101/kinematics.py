"""SO-101 follower kinematics (numpy only): FK, IK along a straight TCP line, wrist camera pose.

The chain is TheRobotStudio's ``Simulation/SO101/so101_new_calib.urdf`` (SO-ARM100 repo,
commit a758567, Apache-2.0) — the "new calibration" whose joint zero is the MIDDLE of each
joint's calibrated range. That is exactly what LeRobot's ``MotorNormMode.DEGREES`` reports
(``(raw - (range_min + range_max) / 2) * 360 / 4095``), so the server's joint angles go into
``fk`` unchanged (LeRobot's own placo kinematics uses the same URDF the same way).

Frames:
* base = URDF ``base_link``: origin on the mounting surface under the base, x forward (away
  from the cable side), y left, z up. The table surface is z = 0.
* TCP = URDF ``gripper_frame_link`` (LeRobot's end-effector frame): between the jaw tips;
  tool z points out of the gripper along the jaws.
* camera = OpenCV optical frame (x right, y down, z forward) of the wrist camera, rigidly
  attached to the TCP: ``T_base_cam = T_base_tcp @ T_tcp_cam`` (``T_tcp_cam`` from the
  hand-eye calibration, ``scripts/so101_calibrate.py``).

5 arm joints: position + tool axis + roll are 6 constraints, so not every pose is reachable.
The harness keeps the tool pointing straight down (``rotation=none|yaw``); then the pan joint
and the wrist roll together set any heading and every reachable point is reachable exactly.
"""

from __future__ import annotations

import math

import numpy as np

from controlr.robot.kinematics import matrix_to_rotvec, rotation_angle, rotvec_to_matrix

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MOTORS = ARM_JOINTS + ("gripper",)

# (joint, origin xyz in the parent link, origin rpy) — revolute about the child's z axis
_CHAIN = (
    ("shoulder_pan", (0.0388353, -8.97657e-09, 0.0624), (math.pi, 4.18253e-17, -math.pi)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-math.pi / 2, -math.pi / 2, 0.0)),
    ("elbow_flex", (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, math.pi / 2)),
    ("wrist_flex", (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -math.pi / 2)),
    ("wrist_roll", (5.55112e-17, -0.0611, 0.0181), (math.pi / 2, 0.0486795, math.pi)),
)
# gripper_link -> gripper_frame_link (fixed)
_TOOL = ((-0.0079, -0.000218121, -0.0981274), (0.0, math.pi, 0.0))
# URDF hard limits (rad); the server narrows them to the arm's calibrated range
URDF_LIMITS = np.array([[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69],
                        [-1.65806, 1.65806], [-2.74385, 2.84121]])

# Tool pointing straight down (tool z = -base z), jaw line along base y.
R_DOWN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _tf(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


_ORIGINS = tuple(_tf(xyz, rpy) for _, xyz, rpy in _CHAIN)
_T_TOOL = _tf(*_TOOL)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def fk(q) -> np.ndarray:
    """4x4 base-frame pose of the TCP for the 5 arm joint angles (rad)."""
    T = np.eye(4)
    for origin, a in zip(_ORIGINS, np.asarray(q, float)[:5]):
        T = T @ origin @ _rz(float(a))
    return T @ _T_TOOL


def joint_points(q) -> np.ndarray:
    """(6, 3) base-frame positions of the 5 joint axes' origins and the TCP (for clearance)."""
    T = np.eye(4)
    pts = []
    for origin, a in zip(_ORIGINS, np.asarray(q, float)[:5]):
        T = T @ origin @ _rz(float(a))
        pts.append(T[:3, 3].copy())
    pts.append((T @ _T_TOOL)[:3, 3])
    return np.array(pts)


def fk_pose(q) -> tuple[np.ndarray, np.ndarray]:
    T = fk(q)
    return T[:3, 3].copy(), matrix_to_rotvec(T[:3, :3])


def _pose_error(T: np.ndarray, p: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.concatenate([p - T[:3, 3], matrix_to_rotvec(R @ T[:3, :3].T)])


def ik(p, R, q_seed, limits=None, *, rot_weight: float = 0.05, iters: int = 100,
       tol_m: float = 2e-4, tol_rad: float = 2e-3) -> tuple[np.ndarray, float, float]:
    """Damped least squares from ``q_seed`` toward TCP position ``p`` (m) and rotation ``R``.

    Returns ``(q, position error m, angle error rad)`` of the best iterate; the caller decides
    whether the residual is acceptable (5 joints: a tilted target may be unreachable exactly).
    ``rot_weight`` (m per rad) trades orientation for position when both cannot hold — position
    wins. Joint limits are enforced by clamping every iterate."""
    lim = URDF_LIMITS if limits is None else np.asarray(limits, float)
    p, R = np.asarray(p, float), np.asarray(R, float)
    q = np.clip(np.asarray(q_seed, float)[:5].copy(), lim[:, 0], lim[:, 1])
    W = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
    best = (q.copy(), math.inf, math.inf)
    lam = 1e-3
    for _ in range(iters):
        T = fk(q)
        e = _pose_error(T, p, R)
        pe, ae = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        if pe + rot_weight * ae < best[1] + rot_weight * best[2]:
            best = (q.copy(), pe, ae)
        if pe < tol_m and ae < tol_rad:
            break
        J = np.zeros((6, 5))
        h = 1e-6
        for i in range(5):
            dq = q.copy()
            dq[i] += h
            J[:, i] = (_pose_error(T, p, R) - _pose_error(fk(dq), p, R)) / h
        Jw, ew = J * W[:, None], e * W
        step = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), ew)
        n = float(np.max(np.abs(step)))
        if n > 0.2:                      # at most ~11 deg per iteration: stay on this branch
            step *= 0.2 / n
        q = np.clip(q + step, lim[:, 0], lim[:, 1])
    return best


def heading_down(yaw: float) -> np.ndarray:
    """Tool pointing straight down, its x axis turned ``yaw`` about base z from ``R_DOWN``."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R_DOWN


def line_path(q0, p_target, R_target, limits=None, *, step_m: float = 0.005,
              step_rad: float = 0.05, tol_m: float = 0.002, tol_rad: float = 0.035,
              max_jump_rad: float = 0.35) -> tuple[list[np.ndarray], float, str]:
    """Joint waypoints along the straight TCP line (and rotation slerp) from FK(q0) to the target.

    Every waypoint is solved by IK seeded with the previous one. The path stops at the first
    waypoint that is not reachable within tolerance or needs a joint jump (a branch flip near a
    singularity). Returns ``(waypoints, fraction of the way done, reason)``; ``reason`` is ""
    when the whole line is reachable."""
    T0 = fk(q0)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    p_target, R_target = np.asarray(p_target, float), np.asarray(R_target, float)
    R_rel = R_target @ R0.T
    rv = matrix_to_rotvec(R_rel)
    n = max(1, int(math.ceil(max(np.linalg.norm(p_target - p0) / step_m, rotation_angle(R_rel) / step_rad))))
    q = np.asarray(q0, float)[:5].copy()
    path: list[np.ndarray] = []
    for k in range(1, n + 1):
        s = k / n
        p = p0 + s * (p_target - p0)
        R = rotvec_to_matrix(rv * s) @ R0
        qk, pe, ae = ik(p, R, q, limits)
        if pe > tol_m or ae > tol_rad:
            return path, (k - 1) / n, "out of reach" if pe > tol_m else "orientation out of reach"
        if float(np.max(np.abs(qk - q))) > max_jump_rad:
            return path, (k - 1) / n, "near a singularity"
        path.append(qk)
        q = qk
    return path, 1.0, ""


def camera_T_cam_base(q, T_tcp_cam) -> np.ndarray:
    """``CameraInfo.T_cam_base`` of the wrist camera at joint angles ``q``."""
    return np.linalg.inv(fk(q) @ np.asarray(T_tcp_cam, float))


def _urdf_T_tcp_cam() -> np.ndarray:
    """The stock wrist camera (``so101_new_calib_camera.urdf``: 32x32 UVC module on the wrist
    mount) as an OpenCV optical frame in the TCP frame. The URDF's camera link already has z
    along the view and y toward the jaws: the camera sits 74 mm beside the tool axis on the side
    of the jaw pivot axis, 105 mm behind the TCP, looking along the jaws tilted 25 deg toward
    them, so both jaws show side by side at the bottom of the image (as on the rig)."""
    G = np.eye(4)
    for origin in _ORIGINS:
        G = G @ origin
    cam = G @ _tf((-0.0150, 0.0240, -0.0315), (-math.pi / 2, -0.0008, -math.pi / 2)) \
        @ _tf((-0.04941, -0.03813, 0.0175), (0.0, -math.pi / 2, -2.0076))
    return np.linalg.inv(G @ _T_TOOL) @ cam


DEFAULT_T_TCP_CAM = _urdf_T_tcp_cam()


def tool_rotation(pitch: float, azimuth: float, roll: float = 0.0) -> np.ndarray:
    """Tool orientation reachable by the 5-joint arm: the tool axis (z) in the vertical plane at
    ``azimuth`` (rad, about base z), ``pitch`` below horizontal (pi/2 = straight down), turned by
    ``roll`` about its own axis. roll 0: jaw-pivot axis (tool y) horizontal."""
    cp, sp, ca, sa = math.cos(pitch), math.sin(pitch), math.cos(azimuth), math.sin(azimuth)
    z = np.array([cp * ca, cp * sa, -sp])
    y = np.array([-sa, ca, 0.0])
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    c, s = math.cos(roll), math.sin(roll)
    return R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


PAN_AXIS_XY = np.array([0.0388353, 0.0])


def reachable_rotation(p, R) -> np.ndarray:
    """``R`` turned about base z so its tool axis lies in the vertical plane through the pan axis
    and ``p`` — the closest orientation 5 joints can hold there (keeps the tool's pitch and its
    roll about the axis). A vertical tool axis is returned unchanged (every heading reachable)."""
    R = np.asarray(R, float)
    z = R[:, 2]
    h = math.hypot(z[0], z[1])
    if h < 1e-3:
        return R
    d = np.asarray(p, float)[:2] - PAN_AXIS_XY
    if math.hypot(d[0], d[1]) < 1e-6:
        return R
    dphi = math.atan2(d[1], d[0]) - math.atan2(z[1], z[0])
    c, s = math.cos(dphi), math.sin(dphi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
