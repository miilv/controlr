"""Virtual cameras for the Isaac scene (numpy only; the server builds them, tests check them).

The rig has one calibrated, angled D435. On it base +y (away from the camera) and +z (up) both
point "up" the image, so a model reading a single frame confuses depth with height. Virtual
views make each axis readable on its own: ``top`` looks straight down (image right = +x, image up
= +y), ``side`` looks horizontally along base +x (image right = -y, image up = +z). Pinhole,
centred principal point (PHANTOM's ``legacy_centered_pinhole``), no distortion; the dict has the
keys PHANTOM's ``configure_camera_intrinsics`` / ``camera_projection`` read, plus
``world_from_cv`` (world == UR controller base frame in this scene).
"""

from __future__ import annotations

import numpy as np

VIRTUAL = ("top", "side")
# world_from_cv rotation columns: the cv camera's x (image right), y (image down), z (forward)
_R = {"top": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).T,
      "side": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]).T}
TOP_HEIGHT_M = 1.0        # above the table top
SIDE_DISTANCE_M = 0.8     # from the centre along -x
SIDE_HEIGHT_M = 0.12      # above the table top
F_PER_WIDTH = 0.9         # fx = fy = 0.9 * width: ~58 deg horizontal field of view


def virtual_camera(name: str, centre_xy, table_top_z: float, resolution=(640, 480)) -> dict:
    """Camera config for ``name`` (top | side) aimed at ``centre_xy`` on the table."""
    if name not in VIRTUAL:
        raise ValueError(f"unknown virtual camera {name!r} ({' | '.join(VIRTUAL)})")
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    if name == "top":
        pos = (cx, cy, table_top_z + TOP_HEIGHT_M)
    else:
        pos = (cx - SIDE_DISTANCE_M, cy, table_top_z + SIDE_HEIGHT_M)
    T = np.eye(4)
    T[:3, :3] = _R[name]
    T[:3, 3] = pos
    w, h = int(resolution[0]), int(resolution[1])
    f = F_PER_WIDTH * w
    return {"resolution": [w, h], "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
            "world_from_cv": T.tolist()}


def intrinsics(cam: dict) -> np.ndarray:
    return np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])


def view_centre(cfg: dict) -> tuple[float, float]:
    """Between the mat and the box, so both are in view (PHANTOM scene config)."""
    mat = np.asarray(cfg["mat"]["center"][:2], float)
    box = cfg.get("bin", {}).get("center")
    if box is None:
        return float(mat[0]), float(mat[1])
    return tuple(float(v) for v in (mat + np.asarray(box[:2], float)) / 2)
