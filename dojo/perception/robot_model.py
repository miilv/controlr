"""Forward kinematics of the public ARX X5 URDF (Assets/Robots/x5/X5A.urdf) for both arms.

Minimal URDF reader (no dependency): joint origins (xyz + fixed-axis rpy), revolute/prismatic axes,
visual meshes. ``fk_world(arm, q6, gripper)`` returns every link's 4x4 world pose; arm roots come
from ``cameras.ARM_ROOTS``. Gripper: RoboDojo's normalised 0..1 maps to the finger joints via the
robot config's ``gripper_scale`` [-0.01, 0.044] (clipped to the URDF limits 0..0.044).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .cameras import ARM_ROOTS, pose_to_T

GRIPPER_SCALE = (-0.01, 0.044)
DEFAULT_URDF_DIRS = (
    os.environ.get("DOJO_X5_DIR", ""),
    "/root/dojo-data/perception/catalog/robot/x5",
    "/root/controlr-robodojo/hf_git/Assets/Robots/x5",
    os.path.expanduser("~/dojo-perception-data/catalog/robot/x5"),
)


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(x) for x in el.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()  # URDF: fixed-axis roll, pitch, yaw
    T[:3, 3] = xyz
    return T


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class X5Model:
    urdf_path: Path
    joints: list[Joint]
    visuals: dict[str, list[tuple[str, np.ndarray]]] = field(default_factory=dict)
    root: str = "base_link"
    arm_joints: tuple = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    gripper_joints: tuple = ("joint7", "joint8")

    @classmethod
    def load(cls, urdf_path: str | Path) -> "X5Model":
        urdf_path = Path(urdf_path)
        tree = ET.parse(urdf_path).getroot()
        joints = []
        for j in tree.findall("joint"):
            ax = j.find("axis")
            joints.append(
                Joint(
                    j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                    _origin(j.find("origin")),
                    np.array([float(x) for x in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
                )
            )
        visuals = {}
        for link in tree.findall("link"):
            vs = []
            for v in link.findall("visual"):
                m = v.find("geometry/mesh")
                if m is not None:
                    vs.append((m.get("filename"), _origin(v.find("origin"))))
            visuals[link.get("name")] = vs
        children = {j.child for j in joints}
        roots = [l for l in visuals if l not in children]
        return cls(urdf_path, joints, visuals, root=roots[0] if roots else "base_link")

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "X5Model":
        from .render import path_exists

        dirs = [d for d in os.environ.get("DOJO_X5_DIR", "").split(os.pathsep) if d] + list(DEFAULT_URDF_DIRS[1:])
        for d in dirs:
            if path_exists(Path(d, "X5A.urdf")):
                return cls.load(Path(d, "X5A.urdf"))
        raise FileNotFoundError(f"X5A.urdf not found in {dirs}; set DOJO_X5_DIR")

    def fk(self, q6, gripper: float | None = None) -> dict[str, np.ndarray]:
        """Link poses in the arm's base_link frame."""
        q = dict(zip(self.arm_joints, np.asarray(q6, dtype=float)))
        if gripper is not None:
            g = float(np.clip(GRIPPER_SCALE[0] + float(gripper) * (GRIPPER_SCALE[1] - GRIPPER_SCALE[0]), 0.0, 0.044))
            for gj in self.gripper_joints:
                q[gj] = g
        poses = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for j in pending:
                if j.parent not in poses:
                    rest.append(j)
                    continue
                M = np.eye(4)
                v = q.get(j.name, 0.0)
                if j.type in ("revolute", "continuous"):
                    M[:3, :3] = Rotation.from_rotvec(j.axis / np.linalg.norm(j.axis) * v).as_matrix()
                elif j.type == "prismatic":
                    M[:3, 3] = j.axis * v
                poses[j.child] = poses[j.parent] @ j.origin @ M
            if len(rest) == len(pending):
                break
            pending = rest
        return poses

    def fk_world(self, arm: str, q6, gripper: float | None = None) -> dict[str, np.ndarray]:
        r = ARM_ROOTS[arm]
        B = pose_to_T(r["pos"], r["quat_wxyz"])
        return {k: B @ v for k, v in self.fk(q6, gripper).items()}

    def mesh_path(self, filename: str) -> Path:
        return self.urdf_path.parent / filename
