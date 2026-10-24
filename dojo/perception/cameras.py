"""Camera model from public constants: head camera from RoboDojo's camera config, wrist cameras from
forward kinematics of the public X5A URDF plus the mount from the robot config. No simulator query.

Conventions
- World = RoboDojo env frame (z up; table top at about z = 0.765; the head camera looks toward +y).
- ``Camera.T_wc`` is the 4x4 camera-to-world pose in the **OpenCV** convention (x right, y down,
  z forward). Isaac/USD cameras look down -z with y up; ``USD_TO_CV = diag(1, -1, -1)`` converts.
- Intrinsics: fx = fy = W * focal_length / horizontal_aperture, cx = W/2, cy = H/2 (RoboDojo's own
  ``CameraManager.get_camera_intrinsics``). Isaac derives the vertical aperture from the
  resolution, so pixels are square.

Sources (RoboDojo @ the commit in /root/controlr-robodojo/robodojo_commit.txt):
- ``env_cfg/camera/camera_config.yml``: cam_head type Gemini_345Lg, pos [0, -0.41, 1.308], ori [30, 0, 0]
  (intrinsic XYZ Euler, degrees; ``utils.rotations.euler_angles_to_quat``);
- ``env_cfg/camera/template.py``: GEMINI_345LG focal 10 mm / h-aperture 22.212 mm; D435 13 / 20.955;
- ``Assets/Robots/x5/robot_config.yml``: wrist camera on link ``camera``, pos [0, 0, 0.001], ori [0, -80, -90];
- ``env_cfg/robot/dual_x5.yml``: arm roots (-0.3, -0.45, 0.765) and (0.3, -0.45, 0.765), quat
  (0.707, 0, 0, 0.707) (w, x, y, z); the first arm is ``left``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

W, H = 640, 480
USD_TO_CV = np.diag([1.0, -1.0, -1.0])

HEAD = {"focal_length": 10.0, "horizontal_aperture": 22.212, "pos": (0.0, -0.41, 1.308), "ori_deg": (30.0, 0.0, 0.0)}
WRIST = {"focal_length": 13.0, "horizontal_aperture": 20.955, "link": "camera", "pos": (0.0, 0.0, 0.001),
         "ori_deg": (0.0, -80.0, -90.0)}
ARM_ROOTS = {
    "left": {"pos": (-0.3, -0.45, 0.765), "quat_wxyz": (0.707, 0.0, 0.0, 0.707)},
    "right": {"pos": (0.3, -0.45, 0.765), "quat_wxyz": (0.707, 0.0, 0.0, 0.707)},
}
TABLE_Z = 0.765  # table top: default_pos z 0.74 + half of the 0.05 thickness scale; refined from frames


def quat_wxyz_to_R(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def R_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = pos
    return T


def T_to_pose(T) -> np.ndarray:
    return np.r_[T[:3, 3], R_to_quat_wxyz(T[:3, :3])]


def euler_xyz_deg(e) -> np.ndarray:
    """RoboDojo's ``euler_angles_to_quat(..., degrees=True)`` default: intrinsic XYZ."""
    return Rotation.from_euler("XYZ", e, degrees=True).as_matrix()


def intrinsics(focal_length: float, horizontal_aperture: float, w: int = W, h: int = H) -> np.ndarray:
    fx = w * focal_length / horizontal_aperture
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


@dataclass
class Camera:
    name: str
    K: np.ndarray
    T_wc: np.ndarray  # camera-to-world, OpenCV axes
    w: int = W
    h: int = H
    meta: dict = field(default_factory=dict)

    @property
    def T_cw(self) -> np.ndarray:
        return np.linalg.inv(self.T_wc)

    @property
    def center(self) -> np.ndarray:
        return self.T_wc[:3, 3]

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixels (N,2) [u right, v down], depth (N,)."""
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        Xc = (X - self.T_wc[:3, 3]) @ self.T_wc[:3, :3]
        z = Xc[:, 2]
        uv = (Xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]) * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        return uv, z

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N,2) -> unit ray directions in world (N,3)."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        d = np.c_[(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(uv))]
        d = d @ self.T_wc[:3, :3].T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ray_plane(self, uv: np.ndarray, z: float) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height z. Returns (N,3); NaN if parallel/behind."""
        d = self.rays(uv)
        o = self.center
        t = (z - o[2]) / np.where(np.abs(d[:, 2]) < 1e-9, np.nan, d[:, 2])
        t = np.where(t > 0, t, np.nan)
        return o[None, :] + t[:, None] * d

    def T_wc_usd(self) -> np.ndarray:
        """Camera-to-world in USD/OpenGL axes (for pyrender)."""
        T = self.T_wc.copy()
        T[:3, :3] = T[:3, :3] @ USD_TO_CV
        return T


def head_camera() -> Camera:
    K = intrinsics(HEAD["focal_length"], HEAD["horizontal_aperture"])
    T = np.eye(4)
    T[:3, :3] = euler_xyz_deg(HEAD["ori_deg"]) @ USD_TO_CV
    T[:3, 3] = HEAD["pos"]
    return Camera("cam_head", K, T)


def wrist_mount_in_link() -> np.ndarray:
    """Camera (OpenCV axes) pose in the URDF ``camera`` link frame, from robot_config.yml."""
    T = np.eye(4)
    T[:3, :3] = euler_xyz_deg(WRIST["ori_deg"]) @ USD_TO_CV
    T[:3, 3] = WRIST["pos"]
    return T


def wrist_camera(arm: str, joints6, model=None, mount_correction: np.ndarray | None = None) -> Camera:
    """Wrist camera from FK on the public URDF. ``mount_correction`` (4x4, in the camera link frame) is an
    offline calibration term; identity by default (see ``calibrate_wrist``)."""
    from .robot_model import X5Model

    model = model or X5Model.default()
    links = model.fk_world(arm, joints6)
    T = links[WRIST["link"]] @ wrist_mount_in_link()
    if mount_correction is not None:
        T = links[WRIST["link"]] @ mount_correction @ wrist_mount_in_link()
    K = intrinsics(WRIST["focal_length"], WRIST["horizontal_aperture"])
    return Camera(f"cam_{arm}_wrist", K, T)


def cameras_from_state(state: dict, model=None) -> dict[str, Camera]:
    """All three cameras from an observation ``state`` dict (official fields only)."""
    cams = {"cam_head": head_camera()}
    for arm in ("left", "right"):
        q = state.get(f"{arm}_arm_joint_state")
        if q is not None:
            cams[f"cam_{arm}_wrist"] = wrist_camera(arm, q, model)
    return cams
