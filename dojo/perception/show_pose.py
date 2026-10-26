"""Where to look at a held object: observer-arm looks and holding-arm "show poses" (generic, numpy only).

A held object's in-hand pose (``dojo.perception.held.held_pose``) is measured from the holding arm's own wrist camera,
the head camera and the OTHER arm's wrist camera. This module proposes where the cameras should be:

- ``observer_looks(target, observer_arm, ...)``: TCP poses for the other (free) arm whose wrist camera looks at a point
  (the held object at the pre-insert hover) from ``dist``, from its own side (the arm reaches out toward the target, so
  the camera sits between its base and the target), at a few azimuths / elevations / rolls.
- ``show_poses(arm, T_tcp, T_tcp_obj, bbox)``: TCP poses for the HOLDING arm that put the object's bbox centre at a
  canonical point in front of the head camera (on its optical axis, ``SHOW_HEAD_DIST`` away, shifted toward the holding
  arm's side), keeping the hover's orientation (``front``) or turned 90 deg about world z (``front_yaw90``).
- ``plan_costs(sk, ...)`` (dojo.skills present: G3 / the runtime): the skills' own IK + collision-checked moves to
  each candidate, their env-step cost and the arm-arm clearance at the end pose.

Nothing here keys on a task: candidates come from the object's bbox (in the TCP frame) and the camera geometry.
Conventions: world = RoboDojo env frame (z up); TCP frame as the skills' (``kinematics.TCP_OFFSET`` along flange +x;
x = approach, y = closing axis); cameras in OpenCV axes (x right, y down, z forward).
"""
from __future__ import annotations

import math

import numpy as np

# The wrist camera (OpenCV axes) in its own arm's TCP frame: link6 -> camera link -> mount (cameras.WRIST), measured by
# FK on the public URDF (identical for both arms and every joint state; tests/dojo/test_held_pose.py checks it)
T_TCP_CAM = np.array([[-0.000007, -0.500030, 0.866008, -0.060158],
                      [-1.000000, 0.000004, -0.000006, 0.000000],
                      [0.000000, -0.866008, -0.500030, 0.050940],
                      [0.0, 0.0, 0.0, 1.0]])
HEAD_EYE = np.array([0.0, -0.41, 1.308])          # cameras.HEAD pos
HEAD_AXIS = np.array([0.0, 0.5, -0.8660254])      # its optical axis (ori 30 deg about x; USD looks down -z)
ARM_BASES = {"left": np.array([-0.3, -0.45, 0.765]), "right": np.array([0.3, -0.45, 0.765])}
TABLE_Z = 0.765
SHOW_HEAD_DIST = 0.42      # m from the head camera along its axis: about 1.5 mm per head pixel
SHOW_SIDE = 0.06           # m toward the holding arm's side (room for the observer arm on the other side)
LOOK_DIST = (0.14, 0.24)   # observer camera distance range (m); 0.18 m = 0.45 mm per wrist pixel
MIN_ARM_CLEARANCE = 0.06   # m between the two arms' sampled link points at an end pose


def other(arm: str) -> str:
    return "right" if arm == "left" else "left"


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def look_at_camera(eye, target, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world (OpenCV axes) at ``eye`` looking at ``target``, image up = world up (then rolled)."""
    z = _unit(np.asarray(target, float) - np.asarray(eye, float))
    up = np.array([0.0, 0.0, 1.0])
    y = -(up - z * float(up @ z))
    if np.linalg.norm(y) < 1e-6:                  # looking straight down / up: image down = world +y
        y = np.array([0.0, 1.0, 0.0]) - z * z[1]
    y = _unit(y)
    x = np.cross(y, z)
    r = math.radians(roll_deg)
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def tcp_for_camera(T_wc: np.ndarray) -> np.ndarray:
    """The TCP pose that puts this arm's wrist camera at ``T_wc``."""
    return np.asarray(T_wc, float) @ np.linalg.inv(T_TCP_CAM)


def camera_for_tcp(T_tcp: np.ndarray) -> np.ndarray:
    return np.asarray(T_tcp, float) @ T_TCP_CAM


def look_distance(extent_m: float) -> float:
    """Observer distance: the object (largest bbox side) spans about a third of the wrist image width."""
    return float(np.clip(2.6 * float(extent_m), *LOOK_DIST))


def observer_looks(target, observer_arm: str, extent_m: float = 0.05, elevs=(30.0, 50.0), azims=(0.0, -35.0, 35.0),
                   rolls=(0.0, -40.0, 40.0), dist: float | None = None) -> list[dict]:
    """Candidate observer TCP poses (best first by construction: straight from the observer's side, lower first)."""
    p = np.asarray(target, float)
    b = ARM_BASES[observer_arm]
    phi0 = math.atan2(b[1] - p[1], b[0] - p[0])
    d = look_distance(extent_m) if dist is None else float(dist)
    out = []
    for el in elevs:
        for az in azims:
            for ro in rolls:
                phi, e = phi0 + math.radians(az), math.radians(el)
                u = np.array([math.cos(e) * math.cos(phi), math.cos(e) * math.sin(phi), math.sin(e)])
                eye = p + d * u
                T_wc = look_at_camera(eye, p, ro)
                out.append({"name": f"look_e{el:.0f}_a{az:+.0f}_r{ro:+.0f}", "arm": observer_arm,
                            "T_tcp": tcp_for_camera(T_wc), "T_wc": T_wc, "eye": eye, "look_at": p, "dist": d})
    return out


def bbox_corners(bmin, bmax) -> np.ndarray:
    lo, hi = np.asarray(bmin, float), np.asarray(bmax, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def object_centre_world(T_tcp, T_tcp_obj, bmin, bmax) -> np.ndarray:
    c = (np.asarray(bmin, float) + np.asarray(bmax, float)) / 2
    T = np.asarray(T_tcp, float) @ np.asarray(T_tcp_obj, float)
    return T[:3, :3] @ c + T[:3, 3]


def show_point(arm: str) -> np.ndarray:
    side = -SHOW_SIDE if arm == "left" else SHOW_SIDE
    return HEAD_EYE + SHOW_HEAD_DIST * HEAD_AXIS + np.array([side, 0.0, 0.0])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def show_poses(arm: str, T_tcp, T_tcp_obj, bmin, bmax) -> list[dict]:
    """Holding-arm TCP poses that put the object's bbox centre at the canonical show point."""
    T_tcp = np.asarray(T_tcp, float)
    c_obj = object_centre_world(T_tcp, T_tcp_obj, bmin, bmax)
    P = show_point(arm)
    out = []
    for name, yaw in (("front", 0.0), ("front_yaw90", math.pi / 2 if arm == "left" else -math.pi / 2)):
        R = _rot_z(yaw) @ T_tcp[:3, :3]
        off = T_tcp[:3, 3] - c_obj                 # TCP relative to the object centre, turned with it
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = P + _rot_z(yaw) @ off
        out.append({"name": name, "arm": arm, "T_tcp": T, "object_centre": P})
    return out


def arm_points(kin, arm: str, q) -> np.ndarray:
    """Points along an arm (joint frames + TCP + midpoints) for a coarse arm-arm clearance check."""
    F = kin.frames(arm, q)
    P = [f[:3, 3] for f in F] + [kin.tcp(arm, q)[:3, 3]]
    mids = [(P[i] + P[i + 1]) / 2 for i in range(2, len(P) - 1)]
    return np.array(P[2:] + mids)


def arm_clearance(kin, q_left, q_right) -> float:
    A, B = arm_points(kin, "left", q_left), arm_points(kin, "right", q_right)
    return float(np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)))


def plan_costs(sk, arm: str, obj_label: str, T_tcp_obj, bmin, bmax, max_looks: int = 3) -> dict:
    """Plan (no env steps) the observer looks at the hover and the show poses with the skills' own IK + collision
    model, from the current commanded joints. Returns {"looks": [...], "shows": [...]} with steps / ok / q_end."""
    from dojo.skills.motion import CARRY, FAST

    obs = other(arm)
    h = sk.held.get(arm)
    q_h, q_o = np.asarray(sk.ex.q[arm], float), np.asarray(sk.ex.q[obs], float)
    T_tcp = sk.kin.tcp(arm, q_h)
    ext = float(np.max(np.asarray(bmax, float) - np.asarray(bmin, float)))
    c_hover = object_centre_world(T_tcp, T_tcp_obj, bmin, bmax)

    def look_plans(target, q_hold, tag):
        rows = []
        for c in observer_looks(target, obs, ext):
            mv = sk.motion.to_pose(obs, q_o, c["T_tcp"], FAST, f"{tag}_look", gripper=float(sk.ex.g[obs]))
            row = {"name": c["name"], "ok": bool(mv.ok), "steps": int(mv.steps) if mv.ok else None,
                   "note": "" if mv.ok else str(mv.note)[:120], "eye": c["eye"].tolist(), "dist": c["dist"]}
            if mv.ok:
                qe = np.asarray(mv.q_end, float)
                clr = arm_clearance(sk.kin, *((q_hold, qe) if arm == "left" else (qe, q_hold)))
                row.update({"q_end": qe.tolist(), "clearance_m": round(clr, 3)})
                row["ok"] = clr >= MIN_ARM_CLEARANCE
            rows.append(row)
            if sum(r["ok"] for r in rows) >= max_looks:
                break
        return rows

    out = {"observer": obs, "hover_object_centre": c_hover.tolist(), "looks": look_plans(c_hover, q_h, "hover"),
           "shows": []}
    for s in show_poses(arm, T_tcp, T_tcp_obj, bmin, bmax):
        mv = sk.motion.to_pose(arm, q_h, s["T_tcp"], CARRY, s["name"], gripper=0.0, held=None if h is None else h.box,
                               ignore=(obj_label,), ease=3)
        row = {"name": s["name"], "ok": bool(mv.ok), "steps_out": int(mv.steps) if mv.ok else None,
               "note": "" if mv.ok else str(mv.note)[:120], "T_tcp": s["T_tcp"].tolist()}
        if mv.ok:
            qe = np.asarray(mv.q_end, float)
            back = sk.motion.joint(arm, qe, q_h, CARRY, "show_back", gripper=0.0, check=False, ease=3)
            row.update({"q_end": qe.tolist(), "steps_back": int(back.steps),
                        "looks": look_plans(s["object_centre"], qe, s["name"])})
        out["shows"].append(row)
    return out
