"""Target-pose helpers shared by the expert programs."""

from __future__ import annotations

import math

import numpy as np

from dojo.skills.geometry import T_from, rot_z
from dojo.skills.kinematics import ARMS, TABLE_Z


def level_R(R: np.ndarray) -> np.ndarray:
    """Closest orientation with one body axis exactly vertical, same heading."""
    k = int(np.argmax(np.abs(R[2, :])))
    s = float(np.sign(R[2, k]))
    xi = (k + 1) % 3
    xw = R[:, xi].copy()
    xw[2] = 0.0
    xw /= np.linalg.norm(xw)
    cols = {k: np.array([0.0, 0.0, s]), xi: xw}
    yi = 3 - k - xi
    cols[yi] = np.cross(cols[(yi + 1) % 3], cols[(yi + 2) % 3])
    return np.stack([cols[0], cols[1], cols[2]], axis=1)


def heading_x(R: np.ndarray) -> float:
    """Heading of the body x axis in the world xy plane (for x-horizontal objects)."""
    return math.atan2(R[1, 0], R[0, 0])


def current_R(sk, label: str) -> np.ndarray:
    """The object's orientation now: in the hand if held (commanded TCP), else measured."""
    for a in ARMS:
        h = sk.held.get(a)
        if h is not None and h.label == label:
            return (sk.kin.tcp(a, sk.ex.q[a]) @ h.T_tcp_obj)[:3, :3]
    return sk.world.obj(label).R


def half_height(o, R) -> float:
    return float(np.abs(R[2, :]) @ (o.asset.extents / 2))


def rest_pose(sk, label: str, xy, z_bottom: float, yaw: float | None = None, sym: int = 2) -> np.ndarray:
    """Object pose resting level with its bbox bottom at ``z_bottom`` and bbox centre over ``xy``.
    ``yaw``: wanted heading of the body x axis (mod 2*pi/sym); None keeps the current heading."""
    o = sk.world.obj(label)
    R = level_R(current_R(sk, label))
    if yaw is not None:
        cur = heading_x(R)
        per = 2 * math.pi / max(1, sym)
        d = (yaw - cur + per / 2) % per - per / 2          # smallest rotation to an equivalent heading
        R = rot_z(d) @ R
    c = np.array([xy[0], xy[1], z_bottom + half_height(o, R)])
    return T_from(c - R @ o.asset.center, R)


def table_rest(sk, label, xy, yaw=None, sym=2) -> np.ndarray:
    return rest_pose(sk, label, xy, TABLE_Z, yaw, sym)


def on_top(sk, label: str, below: str, xy=None, yaw=None, sym: int = 2) -> np.ndarray:
    b = sk.world.obj(below)
    if xy is None:
        xy = b.center_w[:2]
    return rest_pose(sk, label, xy, b.top_z, yaw, sym)


def side_of(sk, label: str) -> str:
    return "left" if sk.world.obj(label).pos[0] < 0 else "right"


def other(arm: str) -> str:
    return "right" if arm == "left" else "left"
