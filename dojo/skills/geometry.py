"""Small rigid-body helpers (numpy only). Quaternions are ``[w, x, y, z]`` (RoboDojo's convention)."""

from __future__ import annotations

import math

import numpy as np


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R) -> np.ndarray:
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rpy_to_R(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def axis_angle_R(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rotvec(R) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(v)
    if s < 1e-9:  # angle ~ pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * ang
    return v / s * ang


def angle_between_R(Ra, Rb) -> float:
    """Rotation angle (rad) between two orientations."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return math.acos(min(1.0, max(-1.0, c)))


def T_from(pos=(0, 0, 0), R=None, quat=None) -> np.ndarray:
    T = np.eye(4)
    if quat is not None:
        T[:3, :3] = quat_to_R(quat)
    elif R is not None:
        T[:3, :3] = R
    T[:3, 3] = pos
    return T


def pose7_to_T(p) -> np.ndarray:
    p = np.asarray(p, float)
    return T_from(p[:3], quat=p[3:7])


def T_to_pose7(T) -> np.ndarray:
    return np.r_[T[:3, 3], R_to_quat(T[:3, :3])]


def inv_T(T) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def yaw_of_R(R) -> float:
    """Heading (rad) of the body x axis projected on the world xy plane."""
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def tilt_deg(R, axis=(0, 0, 1)) -> float:
    """Angle (deg) between the body ``axis`` and world +z."""
    v = np.asarray(R) @ np.asarray(axis, float)
    return math.degrees(math.acos(min(1.0, max(-1.0, v[2] / np.linalg.norm(v)))))


def slerp_R(Ra, Rb, t: float) -> np.ndarray:
    rv = rotvec(np.asarray(Ra).T @ np.asarray(Rb))
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.asarray(Ra, float).copy()
    return np.asarray(Ra) @ axis_angle_R(rv / ang, ang * t)
