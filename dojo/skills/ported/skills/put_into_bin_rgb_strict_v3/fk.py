# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/put_into_bin_rgb_strict_v3/fk.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""fk.py: forward kinematics of the ARX X5 arms from the PUBLIC URDF (Assets/Robots/x5/X5A.urdf joint origins and axes),
with the arm bases at (-0.30, -0.45, 0.765) left and (+0.30, -0.45, 0.765) right, yawed +90 deg (base x -> world +y).
The base pose follows from the documented bases and the home flange pose; fk_check() compares against logged
(q, flange) pairs. basket round r2 (Claude, 25 Oct 2026). Geometry only: no simulator state.

Also: link sample points (capsule model) for clearance checks, and a joint-line path sampler (the server's default
`joint_line` path interpolates joints linearly between the measured q and the IK goal q)."""
import math

import numpy as np

GRASP_OFF = 0.1501
BASE = {"left": np.array([-0.30, -0.45, 0.765]), "right": np.array([0.30, -0.45, 0.765])}
BASE_YAW = math.pi / 2
# (origin xyz, origin rpy, axis) for joint1..joint6 (X5A.urdf lines 49-199)
JOINTS = [((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0, 0, 1)),
          ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0, 1, 0)),
          ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0, 1, 0)),
          ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0, 1, 0)),
          ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0, 0, 1)),
          ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1, 0, 0))]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axang(ax, a):
    ax = np.asarray(ax, float)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def chain(arm, q):
    """World frames of joint1..joint6 (link1..link6) for joints q: list of (p, R)."""
    R = _rpy(0, 0, BASE_YAW)
    p = BASE[arm].astype(float).copy()
    out = []
    for (o, rpy, ax), qi in zip(JOINTS, q):
        p = p + R @ np.asarray(o, float)
        R = R @ _rpy(*rpy) @ _axang(ax, float(qi))
        out.append((p.copy(), R.copy()))
    return out


def flange(arm, q):
    p, R = chain(arm, q)[-1]
    return p, R


def tcp(arm, q):
    p, R = flange(arm, q)
    return p + GRASP_OFF * R[:, 0]


# capsule model: (from frame index, to frame index or 'tcp', radius); frames: 0..5 = joint1..joint6
LINKS = [(0, 1, 0.05), (1, 2, 0.045), (2, 3, 0.042), (3, 4, 0.04), (4, 5, 0.04)]
GRIPPER_BOX = {"a": (0.03, 0.135), "j": 0.055, "k": 0.036}      # along flange x from the flange, jaw half width, thickness


def link_points(arm, q, step=0.02, gripper=True, finger_half=0.055):
    """Sample points (N, 3) and radii (N,) on the arm links (capsules) and the gripper housing + fingers."""
    fr = chain(arm, q)
    pts, rad = [], []
    for a, b, r in LINKS:
        pa, pb = fr[a][0], fr[b][0]
        n = max(2, int(np.linalg.norm(pb - pa) / step) + 1)
        for t in np.linspace(0, 1, n):
            pts.append(pa + t * (pb - pa))
            rad.append(r)
    if gripper:
        p6, R6 = fr[5]
        for s in np.arange(GRIPPER_BOX["a"][0], GRASP_OFF + 0.001, 0.02):
            for jj in np.linspace(-finger_half, finger_half, 5):
                pts.append(p6 + s * R6[:, 0] + jj * R6[:, 1])
                rad.append(0.018)
    return np.asarray(pts), np.asarray(rad)


def joint_line(q0, q1, n=None, max_dq=0.08):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if n is None:
        n = max(2, int(np.abs(q1 - q0).max() / max_dq) + 2)
    return [q0 + t * (q1 - q0) for t in np.linspace(0, 1, n)]


def fk_check(pairs):
    """pairs: [{arm, q, fl}] -> max / median flange position error (m)."""
    e = [float(np.linalg.norm(flange(r["arm"], r["q"])[0] - np.asarray(r["fl"], float))) for r in pairs]
    return max(e), float(np.median(e))
