"""Analytic geometry for the pixel-level agent: the model points at things in the wrist image,
code computes every pose.

* ``Camera``: pinhole (horizontal FOV, image size) + the wrist mount ``T_tcp_cam`` (stock URDF
  mount) times an optional rotation correction ``R_fix`` (in the camera frame) fitted from
  observations of objects whose base-frame position is known (e.g. a later grasp's FK TCP).
* ``pixel_to_point``: pixel -> ray from the camera pose (FK of the measured joints) -> intersection
  with the horizontal plane at the object's height.
* ``fit_rotation``: Gauss-Newton on the angle between the observed ray and the known point.
* ``tool_pose_q``: IK for a TCP position with the tool pointing down as steeply as reachable
  (pitch tried from steep to shallow, azimuth through the pan axis, the camera kept upright).

No model output is ever turned into a joint angle directly: the model gives pixels, code gives
poses, and every pose still goes through ``geometry.plan_target``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from controlr.fkagent import geometry as G
from controlr.robot.kinematics import matrix_to_rotvec, rotvec_to_matrix
from controlr.robot.so101 import kinematics as K


# Rotation correction of the stock wrist mount (camera frame, rotation vector, deg), fitted with
# ``fit_rotation`` on 8 live photos of the rope ball (2026-10-15, episodes e1/e2/e4/e5 + 2 survey
# photos) whose position was known from the later grasp FK. Pixel -> table error: stock ~76 mm,
# fitted 18 mm median (leave-one-episode-out), 49 mm at the image edge. Refit if the camera is
# re-mounted (the jaws move in the image).
FITTED_R_FIX_DEG = [-9.992, -16.123, -2.75]


@dataclass
class Camera:
    width: int = 1280
    height: int = 720
    hfov_deg: float = 80.0
    R_fix: np.ndarray = field(default_factory=lambda: np.eye(3))   # camera-frame rotation correction
    T_tcp_cam: np.ndarray = field(default_factory=lambda: K.DEFAULT_T_TCP_CAM.copy())

    @property
    def K(self) -> np.ndarray:
        f = (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        return np.array([[f, 0.0, (self.width - 1) / 2], [0.0, f, (self.height - 1) / 2], [0.0, 0.0, 1.0]])

    def T_base_cam(self, q_deg) -> np.ndarray:
        T = G.fk_T(q_deg) @ self.T_tcp_cam
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ self.R_fix
        return T

    def ray(self, q_deg, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Origin (mm) and unit direction (base frame) of pixel (u, v) of the full-size image."""
        T = self.T_base_cam(q_deg)
        d = T[:3, :3] @ (np.linalg.inv(self.K) @ np.array([u, v, 1.0]))
        return T[:3, 3] * 1000.0, d / np.linalg.norm(d)

    def project(self, q_deg, p_mm) -> tuple[float, float, float]:
        T = np.linalg.inv(self.T_base_cam(q_deg))
        pc = T[:3, :3] @ (np.asarray(p_mm, float) / 1000.0) + T[:3, 3]
        uvw = self.K @ pc
        return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), float(pc[2])

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "hfov_deg": self.hfov_deg,
                "R_fix_rotvec_deg": np.degrees(matrix_to_rotvec(self.R_fix)).round(3).tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Camera":
        return cls(int(d.get("width", 1280)), int(d.get("height", 720)), float(d.get("hfov_deg", 80.0)),
                   rotvec_to_matrix(np.radians(d.get("R_fix_rotvec_deg", [0, 0, 0]))))

    @classmethod
    def fitted(cls) -> "Camera":
        return cls(R_fix=rotvec_to_matrix(np.radians(FITTED_R_FIX_DEG)))

    @classmethod
    def load(cls, path) -> "Camera":
        return cls.from_json(json.loads(Path(path).read_text()))


def pixel_to_point(cam: Camera, q_deg, u: float, v: float, z_mm: float) -> np.ndarray | None:
    """Base-frame point (mm) where the ray of pixel (u, v) meets the plane z = ``z_mm``; None if
    the ray does not go down to it."""
    o, d = cam.ray(q_deg, u, v)
    if d[2] >= -1e-6 or o[2] <= z_mm:
        return None
    s = (z_mm - o[2]) / d[2]
    return o + s * d


def ray_angle_error(cam: Camera, q_deg, u, v, p_mm) -> float:
    o, d = cam.ray(q_deg, u, v)
    w = np.asarray(p_mm, float) - o
    return math.acos(max(-1.0, min(1.0, float(d @ w / np.linalg.norm(w)))))


def fit_rotation(samples: list[tuple], base: Camera | None = None, iters: int = 30) -> Camera:
    """``samples``: (q_deg, u, v, p_mm). Fits ``R_fix`` (3 dof) minimising the ray-to-point angles."""
    cam = Camera() if base is None else Camera(base.width, base.height, base.hfov_deg, base.R_fix.copy(),
                                               base.T_tcp_cam.copy())
    r = np.zeros(3)
    R0 = cam.R_fix.copy()

    def resid(rv):
        c = Camera(cam.width, cam.height, cam.hfov_deg, R0 @ rotvec_to_matrix(rv), cam.T_tcp_cam)
        out = []
        for q, u, v, p in samples:
            o, d = c.ray(q, u, v)
            w = np.asarray(p, float) - o
            w /= np.linalg.norm(w)
            out.extend(np.cross(d, w))          # ~ angle, signed per axis
        return np.array(out)

    for _ in range(iters):
        e = resid(r)
        J = np.zeros((len(e), 3))
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = 1e-5
            J[:, i] = (resid(r + dr) - e) / 1e-5
        step = np.linalg.lstsq(J, -e, rcond=None)[0]
        r = r + step
        if np.linalg.norm(step) < 1e-7:
            break
    cam.R_fix = R0 @ rotvec_to_matrix(r)
    return cam


def tool_pose_q(p_mm, lim: G.Limits, seed_q, *, pitches=(85.0, 80.0, 75.0, 70.0, 65.0, 60.0),
                roll_deg: float = 90.0) -> tuple[np.ndarray | None, float | None]:
    """Joint angles (deg) putting the TCP at ``p_mm`` with the tool pointing down as steeply as
    reachable (5 joints: the tool axis in the vertical plane through the pan axis). Returns
    (q, pitch) or (None, None)."""
    p = np.asarray(p_mm, float) / 1000.0
    az = math.atan2(p[1] - K.PAN_AXIS_XY[1], p[0] - K.PAN_AXIS_XY[0])
    lim_rad = np.radians(np.stack([lim.joint_lo_deg, lim.joint_hi_deg], axis=1))
    seeds = [np.radians(np.asarray(seed_q, float)), np.radians([math.degrees(az) * -1, 0, 30, 60, -roll_deg]),
             np.radians([math.degrees(az) * -1, -20, 50, 50, -roll_deg])]
    for pitch in pitches:
        R = K.tool_rotation(math.radians(pitch), az, math.radians(roll_deg))
        for s in seeds:
            q, pe, ae = K.ik(p, R, s, lim_rad, iters=200)
            if pe < 0.002 and ae < 0.035:
                return np.degrees(q), pitch
    return None, None
