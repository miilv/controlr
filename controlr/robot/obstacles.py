"""Known obstacles of the scene for the safety envelope's predictive link check.

WHY a hollow box and not a solid block: the gripper has to reach INTO the box to place
the packet, so only the walls and the floor are obstacles. The envelope checks the wrist /
gripper-housing centre line (``body_points``) against them, inflated by
``SafetyConfig.box_clearance_m`` (the Robotiq housing and the UR3 wrist are ~40-50 mm in
radius: in Isaac a 47 mm centre-line clearance already meant contact).

Numpy only: used by the envelope (any backend), the Isaac task code (CPU-testable) and
the Isaac server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Robot body centre line checked against obstacles, as indices into UR3Kinematics.frames(q):
# TCP (7) -> flange (6) -> wrist 3 (5) -> wrist 2 (4) -> wrist 1 (3). The first ``skip_m``
# from the TCP (the fingers) is left out: fingertips may touch (they are what grasps and
# places), the physical contact stop handles them.
BODY_CHAIN = (7, 6, 5, 4, 3)


@dataclass(frozen=True)
class BoxObstacle:
    """An open-top box (tote) in the base frame. ``center`` is the centre of the box's
    footprint at the underside of its floor, ``yaw`` its heading about base z; local
    coordinates are ``R(-yaw) @ (p - center)``. Walls are ``wall_xy`` thick (x walls, y
    walls), the floor ``floor`` thick, the outer size ``outer`` (x, y, height)."""
    name: str
    center: tuple[float, float, float]
    yaw: float
    outer: tuple[float, float, float]
    wall_xy: tuple[float, float]
    floor: float
    movable: bool = False          # a dynamic body (a push slides it) rather than a fixed one

    def slabs(self) -> np.ndarray:
        """(5, 2, 3) local AABBs (lo, hi): floor, -x wall, +x wall, -y wall, +y wall."""
        sx, sy, h = (float(v) for v in self.outer)
        wx, wy = (float(v) for v in self.wall_xy)
        f = float(self.floor)
        return np.array([
            [[-sx / 2, -sy / 2, 0.0], [sx / 2, sy / 2, f]],
            [[-sx / 2, -sy / 2, 0.0], [-sx / 2 + wx, sy / 2, h]],
            [[sx / 2 - wx, -sy / 2, 0.0], [sx / 2, sy / 2, h]],
            [[-sx / 2, -sy / 2, 0.0], [sx / 2, -sy / 2 + wy, h]],
            [[-sx / 2, sy / 2 - wy, 0.0], [sx / 2, sy / 2, h]],
        ])

    SLAB_NAMES = ("floor", "-x wall", "+x wall", "-y wall", "+y wall")

    def to_local(self, points) -> np.ndarray:
        p = np.asarray(points, float).reshape(-1, 3) - np.asarray(self.center, float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x = c * p[:, 0] + s * p[:, 1]
        y = -s * p[:, 0] + c * p[:, 1]
        return np.stack([x, y, p[:, 2]], axis=1)

    def distances(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Per point: distance (m) to the nearest slab (0 inside one) and that slab's index."""
        local = self.to_local(points)
        sl = self.slabs()
        d = np.maximum(sl[None, :, 0, :] - local[:, None, :], 0.0) + \
            np.maximum(local[:, None, :] - sl[None, :, 1, :], 0.0)
        dist = np.linalg.norm(d, axis=2)                     # (n, 5)
        k = np.argmin(dist, axis=1)
        return dist[np.arange(len(local)), k], k

    def clearance(self, points) -> tuple[float, str]:
        """Smallest distance of ``points`` to the box material and the part it is closest to."""
        if len(np.asarray(points).reshape(-1, 3)) == 0:
            return float("inf"), ""
        d, k = self.distances(points)
        i = int(np.argmin(d))
        return float(d[i]), self.part_name(int(k[i]))

    def part_name(self, k: int) -> str:
        """Human name of slab ``k`` as seen from the robot: the -y wall faces the camera and
        the robot side of the mat ("near wall")."""
        return {0: "floor", 1: "left (-x) wall", 2: "right (+x) wall", 3: "near (-y) wall",
                4: "far (+y) wall"}[k]

    def contains_xy(self, p, margin: float = 0.0) -> bool:
        """True if the base-frame point lies over the box footprint (shrunk by ``margin``)."""
        loc = self.to_local(p)[0]
        sx, sy, _ = self.outer
        return bool(abs(loc[0]) <= sx / 2 - margin and abs(loc[1]) <= sy / 2 - margin)

    def to_dict(self) -> dict:
        return {"name": self.name, "center": [float(v) for v in self.center], "yaw": float(self.yaw),
                "outer": [float(v) for v in self.outer], "wall_xy": [float(v) for v in self.wall_xy],
                "floor": float(self.floor), "movable": bool(self.movable)}

    @classmethod
    def from_dict(cls, d: dict) -> "BoxObstacle":
        return cls(str(d.get("name", "the blue box")), tuple(float(v) for v in d["center"]), float(d["yaw"]),
                   tuple(float(v) for v in d["outer"]), tuple(float(v) for v in d["wall_xy"]), float(d["floor"]),
                   bool(d.get("movable", False)))


def box_from_bin_info(b: dict, name: str = "the blue box") -> BoxObstacle:
    """The Isaac server's ``bin_info`` (interior ``lower``/``upper`` in the bin frame — absolute
    coordinates rotated about ``center`` by -yaw —, ``center`` at the floor underside, ``yaw``,
    ``wall``; optional ``wall_xy``) -> BoxObstacle."""
    lo, hi = np.asarray(b["lower"], float), np.asarray(b["upper"], float)
    c = np.asarray(b["center"], float)
    wall_xy = tuple(float(v) for v in b.get("wall_xy", (b.get("wall", 0.02),) * 2))
    outer = (float(hi[0] - lo[0]) + 2 * wall_xy[0], float(hi[1] - lo[1]) + 2 * wall_xy[1], float(hi[2] - c[2]))
    centre_xy = (lo[:2] + hi[:2]) / 2.0           # interior centre == outer centre (in the bin frame)
    # back to the base frame (bin-frame coordinates are rotated about c)
    cy, sy = np.cos(b.get("yaw", 0.0)), np.sin(b.get("yaw", 0.0))
    d = centre_xy - c[:2]
    cxy = c[:2] + np.array([cy * d[0] - sy * d[1], sy * d[0] + cy * d[1]])
    return BoxObstacle(name, (float(cxy[0]), float(cxy[1]), float(c[2])), float(b.get("yaw", 0.0)), outer,
                       wall_xy, float(lo[2] - c[2]), bool(b.get("dynamic", False)))


def body_points(kin, q, skip_m: float = 0.06, n_per_segment: int = 15) -> np.ndarray:
    """Points along the wrist / gripper-housing centre line (``BODY_CHAIN``) for joints ``q``,
    without the first ``skip_m`` from the TCP."""
    F = kin.frames(np.asarray(q, float))
    chain = [F[i][:3, 3] for i in BODY_CHAIN]
    pts = np.concatenate([np.linspace(a, b, n_per_segment) for a, b in zip(chain, chain[1:])])
    return pts[np.linalg.norm(pts - F[7][:3, 3], axis=1) > skip_m]


def body_clearance(kin, q, obstacles, skip_m: float = 0.06) -> tuple[float, str, str]:
    """(clearance m, obstacle name, part) of the robot body centre line to the nearest obstacle."""
    best = (float("inf"), "", "")
    if not obstacles:
        return best
    pts = body_points(kin, q, skip_m)
    for ob in obstacles:
        d, part = ob.clearance(pts)
        if d < best[0]:
            best = (d, ob.name, part)
    return best
