"""Articulated assets: the checker's functional points in our asset frame, and link geometry at a joint position
(stream N; store_laptop's laptop stage).

* ``checker_point(o, tag, active)``: RoboDojo puts a functional frame that names a ``base_link`` on THAT link's pose
  (``layout_manager.get_functional_points``: ``link_pose(base_link) @ frame``), else on the instance (root prim) pose.
  Our ObjectState pose is the asset (default-prim) frame (``artroot``); the root link sits at ``artroot.root_in_asset``
  in it. So a base-link frame is ``root_in_asset @ frame`` in our asset frame (the laptop's "bottom": the body's front
  edge, (0, -0.076, 0.0025), not the frame's own (0, 0.001, -0.001), which is the body's middle).
* ``root_axis(o, axis)``: the checker's ``is_axis_up`` reads the instance (root prim) rotation; the root's rotation in
  the asset frame turns an axis of ours into the checker's.
* ``posed_asset(asset, joint, position)``: the asset with the joint's moving link turned about the joint axis from the
  authored position to ``position`` (links' points and boxes, the overall box); the mesh is dropped (its vertices carry
  no link), so grasps and collision see the posed box. Generic: any revolute joint.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from dojo.skills.geometry import T_from, axis_angle_R, quat_to_R


def _frames(o, tag: str, active: bool) -> list[tuple[np.ndarray, str | None]]:
    raw = ((o.asset.active_functional if active else o.asset.functional) or {}).get(tag) or {}
    out = []
    for f in raw.get("frame") or []:
        f = np.asarray(f, float)
        out.append((T_from(f[:3], quat=f[3:7] / np.linalg.norm(f[3:7])), raw.get("base_link")))
    return out


def link_in_asset(o, link: str | None) -> np.ndarray:
    """The pose of ``link`` in the asset frame: the root link (the articulation root prim, ``artroot``) where the asset
    has a root transform, else the asset frame itself."""
    from dojo.skills.artroot import root_in_asset
    T = root_in_asset(o.asset.key)
    return np.eye(4) if (T is None or link is None) else T


def checker_point(o, tag: str, active: bool = True) -> list[np.ndarray]:
    """Functional frames ``tag`` as the checker places them, in ``o``'s asset frame (4x4 each)."""
    return [link_in_asset(o, link) @ F for F, link in _frames(o, tag, active)]


def root_axis(o, axis) -> np.ndarray:
    """A root-frame axis (the checker's ``is_axis_up(label, axis)``) in ``o``'s asset frame."""
    from dojo.skills.artroot import root_in_asset
    T = root_in_asset(o.asset.key)
    a = np.asarray(axis, float)
    return a if T is None else T[:3, :3] @ a


def joint_turn(J: dict, position: float) -> np.ndarray:
    """4x4 (asset frame) taking the moving link from the joint's authored position to ``position`` (revolute; USD
    degrees when the range exceeds 2 pi)."""
    lo, hi = float(J.get("lower", 0.0)), float(J.get("upper", 0.0))
    deg = max(abs(lo), abs(hi)) > 2 * math.pi + 0.1
    d = float(position) - float(J.get("position") or lo)
    ang = math.radians(d) if deg else d
    axis = np.asarray(J["axis"], float)
    origin = np.asarray(J["origin"], float)
    R = axis_angle_R(axis / np.linalg.norm(axis), ang)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin - R @ origin
    return T


def posed_asset(asset, joint: str, position: float):
    """``asset`` with ``joint``'s moving link at ``position`` (see the module doc); the asset itself is not changed."""
    J = (asset.joints or {}).get(joint)
    if J is None or J.get("type") != "revolute" or not asset.links or J.get("body1") not in asset.links:
        return asset
    T = joint_turn(J, position)
    out = copy.copy(asset)
    links = {}
    allp = []
    for name, lk in asset.links.items():
        pts = np.asarray(lk.get("points"), float) if lk.get("points") is not None else None
        if pts is None or not len(pts):
            pts = np.array([[x, y, z] for x in (lk["bbox_min"][0], lk["bbox_max"][0])
                            for y in (lk["bbox_min"][1], lk["bbox_max"][1]) for z in (lk["bbox_min"][2], lk["bbox_max"][2])])
        if name == J["body1"]:
            pts = (T[:3, :3] @ pts.T).T + T[:3, 3]
        links[name] = {**lk, "points": pts, "bbox_min": pts.min(0), "bbox_max": pts.max(0)}
        allp.append(pts)
    P = np.vstack(allp)
    out.links = links
    out.bbox_min, out.bbox_max = P.min(0), P.max(0)
    out.mesh = None
    out.joints = {**asset.joints, joint: {**J, "position": float(position)}}
    return out


__all__ = ["checker_point", "root_axis", "link_in_asset", "joint_turn", "posed_asset"]
