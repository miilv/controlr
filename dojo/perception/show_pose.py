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


def _ik_steps(kin, arm: str, q0, T_tcp, step_rad: float, ease: int):
    """IK (the skills' solver, nearest to ``q0``) and the env steps of a straight joint move there (no collision
    check: the caller plans the real move)."""
    from dojo.skills.motion import segment_targets

    r = kin.ik(arm, np.asarray(T_tcp, float), np.asarray(q0, float))
    if not r.ok:
        return None, None
    return r.q, len(segment_targets(np.stack([np.asarray(q0, float), r.q]), step_rad, ease))


def propose(arm: str, bbox_min, bbox_max, T_tcp_obj, state: dict, kin=None) -> dict:
    """The service's ``show_pose``: candidates for (a) an observer look at the object where it is now (the holding arm
    stays: ``hover_look``) and (b) the canonical show poses (``front``, ``front_yaw90``) with the observer looking.
    Every entry carries IK hints and a straight-move step estimate (holding arm at CARRY speed, observer at FAST); the
    skills plan the actual moves with their own collision model. Ordered: hover looks first (cheapest), then shows."""
    from dojo.skills.kinematics import X5Kin
    from dojo.skills.motion import CARRY, FAST, segment_targets

    kin = kin or X5Kin()
    obs = other(arm)
    q_h = np.asarray(state[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]
    q_o = np.asarray(state[f"{obs}_arm_joint_state"], float).reshape(-1)[:6]
    T_tcp = kin.tcp(arm, q_h)
    ext = float(np.max(np.asarray(bbox_max, float) - np.asarray(bbox_min, float)))

    def look(target, q_hold):
        for c in observer_looks(target, obs, ext):
            q, n = _ik_steps(kin, obs, q_o, c["T_tcp"], FAST, 2)
            if q is None:
                continue
            clr = arm_clearance(kin, *((q_hold, q) if arm == "left" else (q, q_hold)))
            if clr < MIN_ARM_CLEARANCE:
                continue
            return {"arm": obs, "name": c["name"], "tcp": c["T_tcp"].tolist(), "q": q.tolist(), "eye": c["eye"].tolist(),
                    "look_at": c["look_at"].tolist(), "dist": c["dist"], "est_steps": int(n), "clearance_m": round(clr, 3)}
        return None

    out = []
    c_now = object_centre_world(T_tcp, T_tcp_obj, bbox_min, bbox_max)
    lk = look(c_now, q_h)
    if lk is not None:
        out.append({"name": "hover_look", "arm": arm, "tcp": T_tcp.tolist(), "q": q_h.tolist(), "est_steps_out": 0,
                    "est_steps_back": 0, "observer": lk, "est_steps": lk["est_steps"]})
    for s in show_poses(arm, T_tcp, T_tcp_obj, bbox_min, bbox_max):
        q, n = _ik_steps(kin, arm, q_h, s["T_tcp"], CARRY, 3)
        if q is None:
            continue
        lk = look(s["object_centre"], q)
        back = len(segment_targets(np.stack([q, q_h]), CARRY, 3))
        out.append({"name": s["name"], "arm": arm, "tcp": s["T_tcp"].tolist(), "q": q.tolist(), "est_steps_out": int(n),
                    "est_steps_back": int(back), "observer": lk,
                    "est_steps": int(max(n, lk["est_steps"] if lk else 0) + back)})
    return {"poses": out, "object_centre": c_now.tolist()}


# ------------------------------------------------------------------------------------------------ predicted accuracy
# (the perception venv: CPU silhouettes of the object and of both arms from FK)
FREE_DOF = ((1, "ry", math.radians(1.0)), (3, "x", 0.001), (5, "z", 0.001))   # the in-pad DoF a grasp leaves free
SIGMA_PX = 0.35            # effective boundary noise of a silhouette fit (px), fitted to the dev errors (held_eval)


def view_information(cam, state: dict, arm: str, key: str, T_tcp, T_tcp_obj, opening: float | None = None,
                     occ: np.ndarray | None = None):
    """Fisher information (3x3, per px^2 of boundary noise) of one camera about the in-pad DoF (turn about the closing
    axis, slide along approach / across): the signed silhouette change per unit of each DoF, counted only where the
    object is not hidden by either arm. Returns (I, visible_frac, n_px) or None when the object is not in view."""
    from .held import apply_delta, robot_unknown
    from .render import load_asset, silhouette, transform

    a = load_asset(key)
    T_tcp, T_tcp_obj = np.asarray(T_tcp, float), np.asarray(T_tcp_obj, float)
    c = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    c_tcp = T_tcp_obj[:3, :3] @ c + T_tcp_obj[:3, 3]
    Vw = transform(a.Vs, T_tcp @ T_tcp_obj)
    uv, z = cam.project(Vw)
    if (z <= 0.02).any():
        return None
    S = silhouette(cam, Vw, a.Fs).astype(np.float32)
    n = float(S.sum())
    if n < 30:
        return None
    if occ is None:
        from .held import est_opening
        o = est_opening(a.V, T_tcp_obj) if opening is None else opening
        occ = robot_unknown(cam, state, arm, max(0.0, o - 0.05), min(1.0, o + 0.05), None, dilate_px=1)
    vis = 1.0 - occ.astype(np.float32)
    dS = []
    for idx, _, step in FREE_DOF:
        d = np.zeros(6)
        d[idx] = step
        Sk = silhouette(cam, transform(a.Vs, T_tcp @ apply_delta(T_tcp_obj, c_tcp, d)), a.Fs).astype(np.float32)
        unit = 1.0 if idx < 3 else 1000.0                                       # per degree / per mm
        dS.append((Sk - S) * vis / (step * (math.degrees(1) if idx < 3 else unit)))
    G = np.stack([g.ravel() for g in dS])
    return G @ G.T, float((S * vis).sum() / n), int(n)


def predicted_sigma(I: np.ndarray, sigma_px: float = SIGMA_PX) -> dict:
    """Predicted 1-sigma of (ry deg, x mm, z mm) from summed view information (boundary noise ``sigma_px``)."""
    w, V = np.linalg.eigh(I)
    w = np.maximum(w, 1e-6 * max(float(w.max()), 1e-9))
    C = V @ np.diag(1.0 / w) @ V.T * sigma_px ** 2
    s = np.sqrt(np.maximum(np.diag(C), 0))
    return {"ry_deg": float(s[0]), "x_mm": float(s[1]), "z_mm": float(s[2]), "cov": C}


def lead_sigma(C: np.ndarray, T_tcp_obj, lead_obj) -> float:
    """Leading-point 1-sigma (mm, norm) from the (ry deg, x mm, z mm) covariance: the turn moves the point by its
    lever about the closing axis through the bbox centre (approximated by the TCP-frame lever)."""
    p = np.asarray(T_tcp_obj, float)[:3, :3] @ np.asarray(lead_obj, float) + np.asarray(T_tcp_obj, float)[:3, 3]
    lever = math.hypot(p[0], p[2]) * 1000 * math.radians(1.0)                   # mm per degree about y
    J = np.array([[lever, 1.0, 0.0], [lever, 0.0, 1.0]]) / math.sqrt(2)
    return float(math.sqrt(max(np.trace(J @ C @ J.T), 0.0)))


def observer_dirs(T_tcp) -> list[tuple[str, np.ndarray]]:
    """Viewing directions (object -> eye, world) defined in the TCP frame, so the holding gripper stays out of the line
    of sight: beside the pads (+-y) and across them (+-z), tilted toward the protruding side (+x) by 20 / 45 deg, and
    end-on (+x)."""
    R = np.asarray(T_tcp, float)[:3, :3]
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    out = [("end_on", x)]
    for name, s in (("side+y", y), ("side-y", -y), ("across+z", z), ("across-z", -z)):
        for t in (20.0, 45.0):
            r = math.radians(t)
            out.append((f"{name}@{t:.0f}", math.cos(r) * s + math.sin(r) * x))
    return out


def rank_observer_looks(state: dict, arm: str, key: str, T_tcp_obj, kin=None, base_I: np.ndarray | None = None,
                        lead_obj=None, max_out: int = 4) -> list[dict]:
    """Observer look candidates for the object ``arm`` holds, ranked by the predicted in-pad accuracy WITH the views
    the caller already has (``base_I``): reachable (the skills' IK), clear of the holding arm, the object visible."""
    from dojo.skills.kinematics import X5Kin
    from dojo.skills.motion import FAST, segment_targets

    from .cameras import Camera, intrinsics, WRIST
    from .render import load_asset

    kin = kin or X5Kin()
    obs = other(arm)
    q_h = np.asarray(state[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]
    q_o = np.asarray(state[f"{obs}_arm_joint_state"], float).reshape(-1)[:6]
    T_tcp = kin.tcp(arm, q_h)
    a = load_asset(key)
    bmin, bmax = np.asarray(a.info["bbox_min"]), np.asarray(a.info["bbox_max"])
    ext = float(np.max(bmax - bmin))
    target = object_centre_world(T_tcp, T_tcp_obj, bmin, bmax)
    K = intrinsics(WRIST["focal_length"], WRIST["horizontal_aperture"])
    base_I = np.zeros((3, 3)) if base_I is None else base_I
    out = []
    for name, u in observer_dirs(T_tcp):
        if u[2] < -0.35:                         # an eye under the object (the table / the receptacle is there)
            continue
        d = look_distance(ext)
        eye = target + d * u
        if eye[2] < TABLE_Z + 0.05:
            continue
        for roll in (0.0, -45.0, 45.0, 90.0, -90.0):
            T_wc = look_at_camera(eye, target, roll)
            T_t = tcp_for_camera(T_wc)
            r = kin.ik(obs, T_t, q_o)
            if not r.ok:
                continue
            clr = arm_clearance(kin, *((q_h, r.q) if arm == "left" else (r.q, q_h)))
            if clr < MIN_ARM_CLEARANCE:
                continue
            st = {**state, f"{obs}_arm_joint_state": r.q}
            cam = Camera(f"cam_{obs}_wrist", K, T_wc)
            vi = view_information(cam, st, arm, key, T_tcp, T_tcp_obj)
            if vi is None or vi[1] < 0.3:
                continue
            ps = predicted_sigma(base_I + vi[0])
            row = {"name": f"{name}_r{roll:+.0f}", "arm": obs, "tcp": T_t.tolist(), "q": r.q.tolist(),
                   "eye": eye.tolist(), "look_at": target.tolist(), "dist": d, "visible_frac": round(vi[1], 3),
                   "est_steps": len(segment_targets(np.stack([q_o, r.q]), FAST, 2)), "clearance_m": round(clr, 3),
                   "pred_sigma": {k: round(v, 3) for k, v in ps.items() if k != "cov"}}
            if lead_obj is not None:
                row["pred_sigma"]["lead_mm"] = round(lead_sigma(ps["cov"], T_tcp_obj, lead_obj), 3)
            out.append(row)
            break                                 # the first reachable roll per direction
    key_ = (lambda r: r["pred_sigma"].get("lead_mm", r["pred_sigma"]["ry_deg"]))
    return sorted(out, key=key_)[:max_out]


def base_information(state: dict, arm: str, key: str, T_tcp, T_tcp_obj, cams=("own", "head")) -> tuple[np.ndarray, dict]:
    """Summed information of the holding arm's own wrist camera and the head camera at the current joints."""
    from .cameras import head_camera, wrist_camera

    I = np.zeros((3, 3))
    per = {}
    q = np.asarray(state[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]
    for name, cam in (("own", wrist_camera(arm, q)), ("head", head_camera())):
        if name not in cams:
            continue
        vi = view_information(cam, state, arm, key, T_tcp, T_tcp_obj)
        if vi is not None and vi[1] >= 0.2:
            I = I + vi[0]
            per[name] = round(vi[1], 3)
    return I, per


def rank_show_poses(state: dict, arm: str, key: str, T_tcp_obj, kin=None, lead_obj=None, yaws_deg=range(0, 360, 45),
                    n_observer: int = 2) -> list[dict]:
    """Show poses of the holding arm: the object's bbox centre at the canonical show point, the hover's TCP turned
    about world z by each of ``yaws_deg``; ranked by the predicted accuracy of own + head (+ the best observer look
    for the top ``n_observer``). Each with IK hints and step estimates (out at CARRY, back the same; the observer
    moves in parallel)."""
    from dojo.skills.kinematics import X5Kin
    from dojo.skills.motion import CARRY, segment_targets

    from .render import load_asset

    kin = kin or X5Kin()
    q_h = np.asarray(state[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]
    T_tcp = kin.tcp(arm, q_h)
    a = load_asset(key)
    bmin, bmax = np.asarray(a.info["bbox_min"]), np.asarray(a.info["bbox_max"])
    c_now = object_centre_world(T_tcp, T_tcp_obj, bmin, bmax)
    P = show_point(arm)
    rows = []
    for yd in yaws_deg:
        Rz = _rot_z(math.radians(yd))
        T = np.eye(4)
        T[:3, :3] = Rz @ T_tcp[:3, :3]
        T[:3, 3] = P + Rz @ (T_tcp[:3, 3] - c_now)
        r = kin.ik(arm, T, q_h)
        if not r.ok:
            continue
        st = {**state, f"{arm}_arm_joint_state": r.q}
        I, per = base_information(st, arm, key, T, T_tcp_obj)
        ps = predicted_sigma(I)
        n_out = len(segment_targets(np.stack([q_h, r.q]), CARRY, 3))
        row = {"name": f"show_yaw{yd:+d}", "arm": arm, "tcp": T.tolist(), "q": r.q.tolist(), "est_steps_out": n_out,
               "est_steps_back": n_out, "views": per, "pred_sigma": {k: round(v, 3) for k, v in ps.items() if k != "cov"},
               "_I": I}
        if lead_obj is not None:
            row["pred_sigma"]["lead_mm"] = round(lead_sigma(ps["cov"], T_tcp_obj, lead_obj), 3)
        rows.append(row)
    key_ = (lambda r: r["pred_sigma"].get("lead_mm", r["pred_sigma"]["ry_deg"]))
    rows.sort(key=key_)
    for row in rows[:n_observer]:
        st = {**state, f"{arm}_arm_joint_state": np.asarray(row["q"])}
        looks = rank_observer_looks(st, arm, key, T_tcp_obj, kin=kin, base_I=row["_I"], lead_obj=lead_obj, max_out=1)
        row["observer"] = looks[0] if looks else None
        if looks:
            row["pred_sigma"] = looks[0]["pred_sigma"]
    for row in rows:
        row.pop("_I", None)
        row.setdefault("observer", None)
        row["est_steps"] = int(max(row["est_steps_out"], (row["observer"] or {}).get("est_steps", 0)) + row["est_steps_back"])
    return sorted(rows, key=key_)
