"""Task registry for the Isaac backend: scene choice, seeded randomisation,
default instruction and success evaluation.

WHY pure numpy and no Isaac / PHANTOM imports: this module is imported by the
Isaac server (to place objects and score) *and* by CPU unit tests, so the
randomisation ranges and success rules are testable without a GPU. The server
hands ``evaluate`` a plain "privileged snapshot" dict (object pose/velocity,
pad/robot contact forces, the box's CURRENT interior bounds and pose — the box is a
dynamic body and can be pushed —, TCP); everything here is arithmetic on that dict.

Targets are text only (Ilia: "the text instruction must be enough"): no visual markers.
A reach target is a point stated relative to a visible object ("60 mm above the centre
of the top of the wafer packet"), a push target a distance and direction from the
packet's start pose; both are checked against the objects' current poses.

Tasks share ONE stage (PHANTOM's measured waffle rig): switching between them
moves the packet instead of rebuilding the scene, so the ~1 min Isaac startup is
paid once per server.

Ranges follow PHANTOM where it documents them: packet centre N(0, 10 mm)
clipped to +-20 mm and yaw N(0, 5 deg) clipped to +-10 deg
(``tools/sim/expert_campaign.py`` defaults); success for the pick-place mirrors
``phantom/sim/policy_metrics.py`` full_task at a single settled instant
(oriented box inside the bin interior with 2 mm tolerance, unloaded pads, no
robot load on the packet, speed <= 0.03 m/s and <= 0.5 rad/s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np

# PHANTOM scene config (relative to the PHANTOM repo). The adaptive W2L gripper
# at 1 ms physics is the configuration PHANTOM's scripted-expert data campaign
# runs (zoo inputs scene_dt1ms.json differs only in solver iterations).
WAFFLE_SCENE = "configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json"

# Real recorded start state (PHANTOM configs/sim/initial_states/
# waffles_aug22_1787395928_000.json): the arm hovers above the mat with the
# tool tilted the way every real demonstration holds it. A top-down tool can
# not reach the packet with a UR3 on this rig (verified by IK), so the start
# orientation is the demonstrated one.
START_Q = (0.24457107484340668, -1.5101473967181605, 0.9251918792724609,
           0.8998333215713501, 1.1859229803085327, -3.203757349644796)


# ---------------------------------------------------------------------------
# small geometry helpers (numpy only)
# ---------------------------------------------------------------------------

def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, float) / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def box_corners(pos, quat_wxyz, size) -> np.ndarray:
    """(8,3) world corners of an oriented box."""
    R = quat_wxyz_to_matrix(quat_wxyz)
    local = np.asarray(list(product([-1, 1], repeat=3)), float) * np.asarray(size, float) / 2
    return local @ R.T + np.asarray(pos, float)


def to_bin_frame(points, center, yaw) -> np.ndarray:
    """Same convention as PHANTOM ``BinGeometry.to_interior_frame``."""
    p = np.asarray(points, float).copy()
    if not yaw:
        return p
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    p[..., :2] = (p[..., :2] - center[:2]) @ rot + center[:2]
    return p


def tilt_angle(quat_wxyz) -> float:
    """Angle between the object's local z and world z (rad)."""
    return float(np.arccos(np.clip(quat_wxyz_to_matrix(quat_wxyz)[2, 2], -1.0, 1.0)))


def bin_info_at(b0: dict, pose0, pos, quat_wxyz) -> dict:
    """The box geometry ``b0`` (server ``bin_info`` at the authored pose ``pose0`` =
    (position, quaternion wxyz) of the ``/World/Bin`` body) moved rigidly to the body's
    CURRENT pose ``(pos, quat_wxyz)``: centre and yaw follow the body; the interior bounds,
    kept in the bin frame (absolute coordinates rotated about the centre by -yaw), shift
    with the centre. ``tilt_deg`` reports a tipped box (the interior test assumes upright)."""
    p0, q0 = np.asarray(pose0[0], float), np.asarray(pose0[1], float)
    R0, R = quat_wxyz_to_matrix(q0), quat_wxyz_to_matrix(quat_wxyz)
    dR = R @ R0.T
    c0 = np.asarray(b0["center"], float)
    c = dR @ (c0 - p0) + np.asarray(pos, float)
    dyaw = float(np.arctan2(dR[1, 0], dR[0, 0]))
    out = dict(b0)
    out["center"] = c
    out["yaw"] = float(b0.get("yaw", 0.0)) + dyaw
    out["lower"] = np.asarray(b0["lower"], float) - c0 + c
    out["upper"] = np.asarray(b0["upper"], float) - c0 + c
    out["tilt_deg"] = float(np.degrees(np.arccos(np.clip(dR[2, 2], -1.0, 1.0))))
    out["shift_m"] = float(np.linalg.norm(c - c0))
    # exact frame for the containment test of a tilted box (a dynamic box resting partly on
    # the 3 mm mat tips by ~0.7 deg): points -> authored box frame -> authored bounds
    out["frame0"] = {"R": dR.tolist(), "center": c.tolist(), "center0": c0.tolist(),
                     "yaw0": float(b0.get("yaw", 0.0)), "lower0": np.asarray(b0["lower"], float).tolist(),
                     "upper0": np.asarray(b0["upper"], float).tolist()}
    return out


def to_box_interior(points, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points in the box's interior frame, lower, upper): with a moved / tilted box
    (``bin_info_at``'s ``frame0``) the points are carried back into the authored box pose
    first, so containment is exact for any rigid displacement; otherwise the yaw-only
    convention of PHANTOM ``BinGeometry.to_interior_frame``."""
    f = b.get("frame0")
    pts = np.asarray(points, float)
    if f:
        R = np.asarray(f["R"], float)
        back = (pts - np.asarray(f["center"], float)) @ R + np.asarray(f["center0"], float)
        return (to_bin_frame(back, np.asarray(f["center0"], float), float(f["yaw0"])),
                np.asarray(f["lower0"], float), np.asarray(f["upper0"], float))
    return (to_bin_frame(pts, np.asarray(b["center"], float), float(b["yaw"])),
            np.asarray(b["lower"], float), np.asarray(b["upper"], float))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    name: str
    scene: str                          # PHANTOM scene config (relative path)
    instruction: str                    # default plain-language instruction
    sample: Callable[[np.random.Generator, dict, dict], dict]
    evaluate: Callable[[dict, dict], dict]
    defaults: dict = field(default_factory=dict)


def _param(params: dict, defaults: dict, key: str):
    return params.get(key, defaults[key])


# -- waffle pick-place ------------------------------------------------------

WAFFLE_DEFAULTS = {
    "xy_sigma_m": 0.010, "xy_max_m": 0.020, "yaw_sigma_deg": 5.0, "yaw_max_deg": 10.0,
    # packet yaw offset from the nominal pose: "normal" = N(0, yaw_sigma) clipped to +-yaw_max
    # (PHANTOM), "uniform" = U(-yaw_max, +yaw_max). Graspable with the demo tool tilt and
    # rotation=yaw: offsets -40..+50 deg (UR3 reach; -45 is out of reach, see ROTATION_REPORT).
    "yaw_dist": "normal",
    # tool yaw offset about the vertical through the start TCP, drawn after the packet:
    # a number v -> U(-v, v) deg, a pair (lo, hi) -> U(lo, hi) deg; 0 = the recorded start
    # (no draw, so seeds keep their packet poses). Start-pose reach: -27..+50 deg.
    "start_yaw_deg": 0.0,
    "yaw_offset_deg": None,             # explicit packet yaw offset (deg); overrides the draw
    "nominal": False,                   # True -> PHANTOM's measured nominal pose
    "bin_tolerance_m": 0.002, "contact_force_n": 0.1,
    "settle_speed_m_s": 0.03, "settle_angular_speed_rad_s": 0.5, "lift_height_m": 0.03,
    "start_q": START_Q, "start_gripper": "open",
    # the blue box: a dynamic rigid body (a push slides it) since 2026-10-02; False = the
    # immovable box of every run before (kinematic, like PHANTOM's static colliders)
    "box_dynamic": True, "box_mass_kg": 0.4,
}


def _sample_object(rng: np.random.Generator, params: dict, scene: dict, defaults: dict) -> dict:
    obj = scene["object"]
    center = np.asarray(obj["center"], float).copy()
    yaw = float(obj["yaw"])
    if not _param(params, defaults, "nominal"):
        xy_max = float(_param(params, defaults, "xy_max_m"))
        yaw_max = np.radians(float(_param(params, defaults, "yaw_max_deg")))
        center[:2] += np.clip(rng.normal(0.0, float(_param(params, defaults, "xy_sigma_m")), 2),
                              -xy_max, xy_max)
        dist = str(params.get("yaw_dist", defaults.get("yaw_dist", "normal")))
        if dist == "uniform":
            yaw += float(rng.uniform(-yaw_max, yaw_max))
        elif dist == "normal":
            yaw += float(np.clip(rng.normal(0.0, np.radians(float(_param(params, defaults, "yaw_sigma_deg")))),
                                 -yaw_max, yaw_max))
        else:
            raise ValueError(f"yaw_dist {dist!r}: normal | uniform")
    fixed = params.get("yaw_offset_deg", defaults.get("yaw_offset_deg"))
    if fixed is not None:                       # after the draws: the RNG stream is unchanged
        yaw = float(obj["yaw"]) + float(np.radians(float(fixed)))
    return {"object_pos": center, "object_yaw": yaw, "object_yaw_offset": yaw - float(obj["yaw"])}


def rotate_start_q(start_q, yaw: float, step_rad: float = np.radians(2.0)) -> np.ndarray:
    """Joints with the start TCP kept and the tool turned by ``yaw`` (rad) about the vertical
    through the TCP (roll/pitch kept). Followed in small steps from ``start_q`` with IK seeded
    by the previous step, so the result is on the same branch as the recorded start (the arm
    could drive there with ``rotation=yaw`` moves). Raises if a step has no solution."""
    from controlr.robot.kinematics import IKOptions, UR3Kinematics, matrix_to_rotvec
    from controlr.robot.spec import ur3_cb3_spec

    q = np.asarray(start_q, float).copy()
    if abs(yaw) < 1e-9:
        return q
    kin = UR3Kinematics()
    margin = 0.0873
    lim = np.array([[j.lower + margin, j.upper - margin] for j in ur3_cb3_spec().joints])
    T0 = kin.fk_matrix(q)
    n = max(1, int(np.ceil(abs(yaw) / step_rad)))
    for k in range(1, n + 1):
        qn = kin.ik(T0[:3, 3], matrix_to_rotvec(_rz(yaw * k / n) @ T0[:3, :3]), q, lim,
                    options=IKOptions(restarts=0))
        if qn is None or float(np.max(np.abs(qn - q))) > 0.3:
            raise ValueError(f"start_yaw {np.degrees(yaw):.1f} deg: the start TCP cannot be kept at "
                             f"{np.degrees(yaw * k / n):.1f} deg (reach limit); narrow start_yaw_deg")
        q = qn
    return q


def _start(params: dict, defaults: dict, rng: np.random.Generator | None = None) -> dict:
    q0 = np.asarray(_param(params, defaults, "start_q"), float)
    spec = params.get("start_yaw_deg", defaults.get("start_yaw_deg", 0.0))
    lo, hi = (-float(spec), float(spec)) if np.isscalar(spec) else (float(spec[0]), float(spec[1]))
    yaw = 0.0
    if rng is not None and (lo != 0.0 or hi != 0.0):
        yaw = float(np.radians(rng.uniform(min(lo, hi), max(lo, hi))))
    return {"start_q": rotate_start_q(q0, yaw), "start_yaw_offset": yaw,
            "start_gripper": _param(params, defaults, "start_gripper")}


def sample_waffle(rng, params, scene):
    return {**_sample_object(rng, params, scene, WAFFLE_DEFAULTS), **_start(params, WAFFLE_DEFAULTS, rng)}


def scene_record(episode: dict, scene: dict) -> dict:
    """The sampled task scene of one episode in loggable units (mm / deg, base frame):
    packet pose (sampled and settled) and its yaw offset from the nominal pose, box pose,
    start joints / TCP / tool yaw (+ the randomised start yaw offset), the text reach / push
    target. ``episode`` is the server's episode dict, ``scene`` its ``scene_info`` (with the
    box at its pose after the reset settle)."""
    from controlr.robot.kinematics import UR3Kinematics, matrix_to_rpy

    def mm(v):
        return None if v is None else [round(float(x) * 1000.0, 2) for x in np.asarray(v, float).reshape(-1)]

    def deg(a):
        return None if a is None else round(float(np.degrees(a)), 3)

    obj = scene.get("object", {})
    nominal_yaw = float(obj.get("yaw", 0.0))
    quat = episode.get("object_quat_wxyz")
    settled_yaw = None
    tilt = None
    if quat is not None:
        R = quat_wxyz_to_matrix(quat)
        settled_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        tilt = float(np.degrees(tilt_angle(quat)))
    sampled_yaw = episode.get("object_yaw_sampled")
    kin = UR3Kinematics()
    q0 = episode.get("start_q")
    tcp_yaw0 = None
    tcp0 = None
    if q0 is not None:
        T = kin.fk_matrix(np.asarray(q0, float))
        tcp0, tcp_yaw0 = T[:3, 3], float(matrix_to_rpy(T[:3, :3])[2])
    b = scene.get("bin") or {}
    rt = episode.get("reach_target") or None
    pt = episode.get("push_target") or None
    return {
        "task": episode.get("task"), "seed": episode.get("seed"),
        "packet": {"pos_mm": mm(episode.get("object_pos")), "pos_sampled_mm": mm(episode.get("object_pos_sampled")),
                   "yaw_deg": deg(settled_yaw), "yaw_sampled_deg": deg(sampled_yaw),
                   "nominal_yaw_deg": deg(nominal_yaw),
                   "yaw_offset_deg": deg(None if settled_yaw is None else
                                         (settled_yaw - nominal_yaw + np.pi) % (2 * np.pi) - np.pi),
                   "tilt_deg": None if tilt is None else round(tilt, 3),
                   "size_mm": mm(obj.get("size")),
                   "settle_drift_mm": None if episode.get("settle_drift_m") is None
                   else round(float(episode["settle_drift_m"]) * 1000, 2)},
        "box": {"center_mm": mm(b.get("center")), "yaw_deg": deg(b.get("yaw")),
                "interior_lower_mm": mm(b.get("lower")), "interior_upper_mm": mm(b.get("upper")),
                "dynamic": episode.get("box_dynamic", (episode.get("params") or {}).get("box_dynamic"))},
        "start": {"q_deg": None if q0 is None else [round(float(np.degrees(v)), 3) for v in q0],
                  "tcp_mm": mm(tcp0), "tcp_yaw_deg": deg(tcp_yaw0),
                  "yaw_offset_deg": deg(episode.get("start_yaw_offset", 0.0)),
                  "gripper": episode.get("start_gripper")},
        "reach_target": None if not rt else {"text": rt.get("text"), "ref": rt.get("ref"),
                                             "offset_mm": mm(rt.get("offset")), "point_mm": mm(rt.get("point0"))},
        "push_target": None if not pt else {"text": pt.get("text"), "distance_mm": round(float(pt["distance"]) * 1000, 1),
                                            "point_mm": mm(pt.get("point"))},
    }


def evaluate_waffle(snap: dict, episode: dict) -> dict:
    p = {**WAFFLE_DEFAULTS, **episode.get("params", {})}
    tol = float(p["bin_tolerance_m"])
    corners = box_corners(snap["object_pos"], snap["object_quat_wxyz"], snap["object_size"])
    local, lower, upper = to_box_interior(corners, snap["bin"])        # the box's CURRENT pose
    over_bin = bool(np.all((local[:, :2] >= lower[:2] - tol) & (local[:, :2] <= upper[:2] + tol)))
    inside = bool(over_bin and local[:, 2].min() >= lower[2] - tol and local[:, 2].max() <= upper[2] + tol)
    pads = np.asarray(snap["pad_object_force_n"], float)
    thr = float(p["contact_force_n"])
    gripped = bool(np.all(pads > thr))
    unloaded = bool(np.all(pads <= thr)) and float(snap["robot_object_force_n"]) <= thr
    speed = float(np.linalg.norm(snap["object_lin_vel"]))
    ang = float(np.linalg.norm(snap["object_ang_vel"]))
    settled = bool(speed <= p["settle_speed_m_s"] and ang <= p["settle_angular_speed_rad_s"])
    lift = float(snap["object_pos"][2] - np.asarray(episode["object_pos"], float)[2])
    success = bool(inside and unloaded and settled)
    progress = (1.0 if success else 0.9 if inside else 0.7 if over_bin and gripped
                else 0.5 if gripped and lift >= p["lift_height_m"] else 0.25 if gripped else 0.0)
    if success:
        msg = "goal reached: the packet rests inside the box and the gripper has released it"
    elif inside and not unloaded:
        msg = "the packet is inside the box but still touched/held by the robot"
    elif inside:
        msg = "the packet is inside the box but still moving"
    elif gripped:
        msg = f"the packet is held, {lift * 1000:.0f} mm above its start height, not inside the box yet"
    else:
        msg = "the packet is not in the box"
    return {"success": success, "progress": progress, "message": msg, "metrics": {
        "inside_bin": inside, "over_bin": over_bin, "gripped": gripped, "unloaded": unloaded,
        "settled": settled, "lift_m": lift, "object_speed_m_s": speed,
        "object_tilt_deg": float(np.degrees(tilt_angle(snap["object_quat_wxyz"]))),
        "pad_object_force_n": pads.tolist()}}


# -- reach --------------------------------------------------------------------

# Text targets relative to visible objects (Ilia: "the text instruction must be enough"):
# (reference point, metres above it, how the instruction names it). The reference point
# is computed from the object's CURRENT pose when the goal is checked.
REACH_TARGETS = (
    ("packet_top", 0.06, "60 mm above the centre of the top face of the wafer packet"),
    ("packet_top", 0.10, "100 mm above the centre of the top face of the wafer packet"),
    ("box_opening", 0.05, "50 mm above the rim of the blue box, over the centre of its opening"),
    ("box_near_rim", 0.05, "50 mm above the middle of the blue box's near rim (the rim closest to the camera)"),
    ("box_near_right", 0.05, "50 mm above the near-right corner of the blue box's rim (closest to the camera, "
                             "on the right in the image)"),
    ("box_near_left", 0.05, "50 mm above the near-left corner of the blue box's rim (closest to the camera, "
                            "on the left in the image)"),
)

REACH_DEFAULTS = {
    **WAFFLE_DEFAULTS,
    "targets": None,                     # subset of REACH_TARGETS indices (None = all)
    "target_index": None,                # explicit choice (index into REACH_TARGETS)
    "tolerance_m": 0.015,
    # the gripper housing / wrist must stay this far from the blue box at the target
    "body_clearance_m": 0.045,
}


def project(scene: dict, p) -> np.ndarray | None:
    """Pixel (u, v) of base-frame point ``p`` in the scene camera, if known."""
    cam = scene.get("camera")
    if not cam:
        return None
    T, K = np.asarray(cam["T_cam_base"], float), np.asarray(cam["K"], float)
    pc = T[:3, :3] @ np.asarray(p, float) + T[:3, 3]
    uv = K @ pc
    return uv[:2] / uv[2]


def _box(b: dict | None):
    from controlr.robot.obstacles import box_from_bin_info
    return None if not b else box_from_bin_info(b)


def reach_reference(ref: str, obj_pos, obj_quat_wxyz, obj_size, bin_info: dict | None) -> np.ndarray:
    """Base-frame reference point of a reach target from the objects' poses: ``packet_top``
    = centre of the packet's top face; ``box_*`` = points of the box's top rim (outer
    corners: near = -y, toward the camera; right = +x, image right)."""
    if ref == "packet_top":
        R = quat_wxyz_to_matrix(obj_quat_wxyz)
        return np.asarray(obj_pos, float) + R @ np.array([0.0, 0.0, float(obj_size[2]) / 2])
    box = _box(bin_info)
    if box is None:
        raise ValueError("reach target relative to the box, but the scene has no box")
    sx, sy, h = box.outer
    local = {"box_opening": (0.0, 0.0), "box_near_rim": (0.0, -sy / 2), "box_near_right": (sx / 2, -sy / 2),
             "box_near_left": (-sx / 2, -sy / 2)}[ref]
    c, sn = np.cos(box.yaw), np.sin(box.yaw)
    xy = np.asarray(box.center[:2], float) + np.array([c * local[0] - sn * local[1], sn * local[0] + c * local[1]])
    return np.array([xy[0], xy[1], box.center[2] + h])


def reach_target_point(target: dict, obj_pos, obj_quat_wxyz, obj_size, bin_info: dict | None) -> np.ndarray:
    return reach_reference(target["ref"], obj_pos, obj_quat_wxyz, obj_size, bin_info) + np.asarray(
        target["offset"], float)


def reach_feasibility(marker, scene: dict, start_q, *, body_clearance: float = 0.045,
                      table_clearance: float = 0.005, open_width: float = 0.091) -> str:
    """Why a reach target cannot be touched with the START orientation ("" if it can).

    With action.rotation=none the tool keeps its start orientation, so (a) IK must
    reach the target with that orientation inside the soft joint limits, (b) the
    open fingertips must stay above the table at the target, and (c) the tool body
    (TCP -> flange -> wrist, inflated by ``body_clearance``) must stay clear of the blue
    box's walls and floor: at seed 0 the old sampler put the housing inside the box's near
    wall, so no controller could succeed (review control-safety #1)."""
    from controlr.robot.kinematics import UR3Kinematics, matrix_to_rotvec   # numpy only
    from controlr.robot.obstacles import body_clearance as _body_clearance
    from controlr.robot.safety import finger_drop
    from controlr.robot.spec import ur3_cb3_spec

    spec = ur3_cb3_spec()
    kin = UR3Kinematics()
    q0 = np.asarray(start_q, float)
    R0 = kin.fk_matrix(q0)[:3, :3]
    margin = 0.0873
    lim = np.array([[j.lower + margin, j.upper - margin] for j in spec.joints])
    m = np.asarray(marker, float)
    q = kin.ik(m, matrix_to_rotvec(R0), q0, lim)
    if q is None:
        return "not reachable with the start orientation"
    if m[2] - finger_drop(spec, R0, open_width) < float(scene["table_top_z"]) + table_clearance:
        return "open fingertips would be below the table"
    box = _box(scene.get("bin"))
    if box is not None and _body_clearance(kin, q, [box], skip_m=0.0)[0] < body_clearance:
        return "the gripper body would hit the blue box"
    return ""


def sample_reach(rng, params, scene):
    d = REACH_DEFAULTS
    out = {**_sample_object(rng, params, scene, d), **_start(params, d, rng)}
    body = float(_param(params, d, "body_clearance_m"))
    obj = scene["object"]
    quat = yaw_quat_wxyz(out["object_yaw"])
    cand = list(range(len(REACH_TARGETS))) if params.get("targets") is None else [int(i) for i in params["targets"]]
    if params.get("target_index") is not None:
        cand = [int(params["target_index"])]
    feasible = []
    for i in cand:
        ref, up, text = REACH_TARGETS[i]
        target = {"ref": ref, "offset": [0.0, 0.0, up], "text": text, "index": i}
        pt = reach_target_point(target, out["object_pos"], quat, obj["size"], scene.get("bin"))
        if not reach_feasibility(pt, scene, out["start_q"], body_clearance=body):
            feasible.append((target, pt))
    if not feasible:
        raise ValueError(f"reach: none of the targets {cand} is reachable from this start pose "
                         f"(check start_q / targets)")
    # drawn AFTER the packet / start draws: the RNG stream of those is unchanged
    target, pt = feasible[int(rng.integers(len(feasible)))]
    target["point0"] = pt
    out["reach_target"] = target
    out["instruction"] = (f"Move the gripper so that the point between its fingertips is {target['text']}. "
                          f"Do not touch anything.")
    return out


def evaluate_reach(snap: dict, episode: dict) -> dict:
    p = {**REACH_DEFAULTS, **episode.get("params", {})}
    tgt = episode["reach_target"]
    target = reach_target_point(tgt, snap["object_pos"], snap["object_quat_wxyz"], snap["object_size"],
                                snap.get("bin"))
    d = float(np.linalg.norm(np.asarray(snap["tcp_pos"], float) - target))
    d0 = float(np.linalg.norm(np.asarray(episode["tcp_pos0"], float) - np.asarray(tgt["point0"], float))) or 1.0
    success = bool(d <= float(p["tolerance_m"]))
    return {"success": success, "progress": float(np.clip(1.0 - d / d0, 0.0, 1.0)),
            "message": ("goal reached: the fingertips are at the target point" if success
                        else f"the fingertips are {d * 1000:.0f} mm from the target point"),
            "metrics": {"distance_m": d, "tolerance_m": float(p["tolerance_m"]), "target": target.tolist()}}


# -- push ---------------------------------------------------------------------

PUSH_DEFAULTS = {**WAFFLE_DEFAULTS, "push_distance_m": (0.06, 0.10), "tolerance_m": 0.025}


def sample_push(rng, params, scene):
    d = PUSH_DEFAULTS
    out = {**_sample_object(rng, params, scene, d), **_start(params, d, rng)}
    lo, hi = _param(params, d, "push_distance_m")
    # Push along the packet's long axis: pushing its broad face tips the 90 mm
    # tall, 35 mm thick packet over. The target lies on the -x side (left in
    # the image), which keeps it on the mat and out from under the gripper.
    axis = np.array([np.cos(out["object_yaw"]), np.sin(out["object_yaw"])])
    if axis[0] > 0:
        axis = -axis
    dist = float(rng.uniform(lo, hi))
    target = out["object_pos"][:2] + axis * dist
    mm = int(round(dist * 1000 / 10.0) * 10)
    text = (f"about {mm} mm along its long side, toward the left of the image (-x), so that its centre ends "
            f"{mm} mm from where it stands now")
    out["push_target"] = {"point": np.array([target[0], target[1], float(scene["table_top_z"])]),
                          "distance": dist, "text": text}
    out["instruction"] = (f"Push the wafer packet {text}. Do not pick it up; keep it upright if you can.")
    return out


def evaluate_push(snap: dict, episode: dict) -> dict:
    """Pushed, not carried: besides the state at the check instant, the server's running
    record of the episode (``episode["max_lift_m"]``, ``episode["ever_held"]``) must show
    that the packet was never lifted > 20 mm or gripped (picking it up, carrying it and
    releasing it on the target used to count as a push)."""
    p = {**PUSH_DEFAULTS, **episode.get("params", {})}
    target = np.asarray(episode["push_target"]["point"], float)[:2]
    d = float(np.linalg.norm(np.asarray(snap["object_pos"], float)[:2] - target))
    d0 = float(np.linalg.norm(np.asarray(episode["object_pos"], float)[:2] - target)) or 1.0
    held = bool(np.all(np.asarray(snap["pad_object_force_n"], float) > p["contact_force_n"])
                or episode.get("ever_held", False))
    lift_now = float(snap["object_pos"][2] - np.asarray(episode["object_pos"], float)[2])
    lifted = max(lift_now, float(episode.get("max_lift_m", 0.0))) > 0.02
    success = bool(d <= float(p["tolerance_m"]) and not held and not lifted)
    return {"success": success, "progress": float(np.clip(1.0 - d / d0, 0.0, 1.0)),
            "message": ("goal reached: the packet stands at the target" if success
                        else f"the packet centre is {d * 1000:.0f} mm from the target"
                        + (" (it must be pushed, not carried)" if held or lifted else "")),
            "metrics": {"distance_m": d, "held": held, "lifted": lifted,
                        "object_tilt_deg": float(np.degrees(tilt_angle(snap["object_quat_wxyz"])))}}


TASKS: dict[str, TaskSpec] = {
    "waffle_pick_place": TaskSpec(
        "waffle_pick_place", WAFFLE_SCENE,
        "Pick up the wafer packet (the small upright wrapped bar standing on the dark mat) "
        "and put it into the blue box. Open the gripper so the packet rests inside the box, "
        "then move the gripper up and away.",
        sample_waffle, evaluate_waffle, WAFFLE_DEFAULTS),
    # reach / push: the instruction is drawn per seed (sample -> "instruction"); these are
    # the generic forms
    "reach": TaskSpec(
        "reach", WAFFLE_SCENE,
        "Move the gripper so that the point between its fingertips is at the point the task names "
        "(relative to the wafer packet or the blue box). Do not touch anything.",
        sample_reach, evaluate_reach, REACH_DEFAULTS),
    "push": TaskSpec(
        "push", WAFFLE_SCENE,
        "Push the wafer packet along the mat by the distance and in the direction the task names. "
        "Do not pick it up; keep it upright if you can.",
        sample_push, evaluate_push, PUSH_DEFAULTS),
}
ALIASES = {"pick_place": "waffle_pick_place", "waffles": "waffle_pick_place"}


def get_task(name: str) -> TaskSpec:
    key = ALIASES.get(name, name)
    if key not in TASKS:
        raise KeyError(f"unknown isaac task {name!r}; available: {sorted(TASKS)}")
    return TASKS[key]


# ---------------------------------------------------------------------------
# scripted reference solution (tests / demos; never shown to the model)
# ---------------------------------------------------------------------------

# Rotation-vector targets (base frame) are PHANTOM's scripted-expert phase
# means from 20 real teleop demos (phantom/sim/scripted_expert.py,
# ExpertParams.rot_grasp / rot_carry), measured at the nominal packet yaw.
EXPERT_NOMINAL_YAW = 0.2595497954083121
EXPERT_ROT_GRASP = (-1.70, -1.92, 1.26)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotvec_to_matrix(rv) -> np.ndarray:
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def scripted_pick_place_plan(object_pos, object_yaw: float, bin_center, *, grasp_above_center_m: float = 0.035,
                             approach_m: float = 0.10, lift_z_m: float = 0.16, carry_z_m: float = 0.29,
                             place_tcp_z_m: float = 0.11, retreat_z_m: float = 0.26,
                             open_width_m: float = 0.085) -> list[dict]:
    """Phases ``{"name", "pos"(3,), "R"(3,3), "gripper": m | None, "linear": bool}``
    for a privileged IK pick of the packet into the bin.

    The tool is tilted the way the real demonstrations hold it (a top-down
    UR3 tool cannot reach the packet on this rig); the fingers close along
    tool x, which ``EXPERT_ROT_GRASP`` aligns with the packet's thin axis.
    Approach runs along the tool axis so the open fingers never sweep through
    the packet. With this orientation the arm cannot rise straight above the
    packet to wall-clearing height (outside the reachable set on the same
    IK branch), so it lifts to ``lift_z_m`` first and climbs while moving
    toward the box, whose centre is closer to the robot base."""
    p = np.asarray(object_pos, float)
    R = _rz(float(object_yaw) - EXPERT_NOMINAL_YAW) @ _rotvec_to_matrix(EXPERT_ROT_GRASP)
    z_tool = R[:, 2]
    grasp = p + np.array([0.0, 0.0, grasp_above_center_m])
    pre = grasp - approach_m * z_tool
    lift = np.array([grasp[0], grasp[1], lift_z_m])
    b = np.asarray(bin_center, float)
    front = np.array([b[0], b[1] - 0.165, carry_z_m])       # just outside the box's near wall
    over = np.array([b[0], b[1], carry_z_m])
    place = np.array([b[0], b[1], place_tcp_z_m])
    return [
        {"name": "pregrasp", "pos": pre, "R": R, "gripper": open_width_m, "linear": False},
        {"name": "approach", "pos": grasp, "R": R, "gripper": None, "linear": True},
        {"name": "close", "pos": grasp, "R": R, "gripper": 0.0, "linear": False},
        {"name": "lift", "pos": lift, "R": R, "gripper": None, "linear": True},
        {"name": "climb", "pos": front, "R": R, "gripper": None, "linear": True},
        {"name": "carry", "pos": over, "R": R, "gripper": None, "linear": True},
        {"name": "lower", "pos": place, "R": R, "gripper": None, "linear": True},
        {"name": "release", "pos": place, "R": R, "gripper": open_width_m, "linear": False},
        {"name": "retreat", "pos": np.array([b[0], b[1], retreat_z_m]), "R": R, "gripper": None, "linear": True},
    ]


def body_box_clearance(q, scene: dict, skip_m: float = 0.06) -> float:
    """Distance (m) from the tool/wrist centre line (TCP-``skip_m`` along the tool -> flange ->
    wrist 3 -> wrist 2 -> wrist 1, ``controlr.robot.obstacles.body_points``) to the blue box's
    walls and floor (``scene["bin"]``, e.g. the CURRENT box from the server state); 0 inside a
    wall. The Robotiq housing / wrist links are ~40-50 mm in radius: in Isaac 47 mm already
    meant contact. The same check guards moves in ``SafetyEnvelope`` (safety.box_collision)."""
    from controlr.robot.kinematics import UR3Kinematics
    from controlr.robot.obstacles import body_clearance
    box = _box(scene.get("bin"))
    if box is None:
        return float("inf")
    return body_clearance(UR3Kinematics(), q, [box], skip_m)[0]


def jaw_yaw_for(object_yaw: float, near: float) -> float:
    """Tool heading (STATE yaw: extrinsic RPY yaw = heading of the jaw line, tool x) that
    closes the jaws across the packet's thin axis (local y): object_yaw + 90 deg, taken
    modulo 180 deg (the jaws are symmetric) nearest to ``near``."""
    target = float(object_yaw) + np.pi / 2
    return near + ((target - near + np.pi / 2) % np.pi - np.pi / 2)


def run_scripted_yaw_pick_place(object_pos, object_yaw: float, bin_center, act: Callable, T0,
                                *, grasp_above_center_m: float = 0.035, approach_m: float = 0.08,
                                lift0_m: float = 0.03, lift_z_m: float = 0.16, carry_z_m: float = 0.29,
                                place_tcp_z_m: float = 0.11, retreat_z_m: float = 0.26,
                                carry_yaw: float | None = None, open_width_m: float = 0.085,
                                grasp_shift_m: float = 0.05, turn_above_m: float = 0.02,
                                turn_back_m: float = 0.0,
                                max_step_m: float = 0.09, fine_step_m: float = 0.02,
                                max_yaw_rad: float = np.radians(30.0), max_moves: int = 120) -> list[dict]:
    """Closed-loop scripted expert for ``action.rotation=yaw`` that uses ONLY the model's action
    space (``MOVE ee_delta dx dy dz dyaw`` + GRIP), never shown to the model.

    ``act(values, gripper, phase)`` executes one ee_delta action (SI values ``(dx, dy, dz, 0, 0, dyaw)``,
    gripper width or None) and returns the measured TCP pose (4x4) afterwards; ``T0`` is the
    measured pose now. Phases: hover ``turn_above_m`` above (and ``turn_back_m`` toward -y of)
    the pregrasp point of the TARGET heading, keeping the current heading; turn the jaws across
    the packet (``jaw_yaw_for``) there — low enough for the turn to be in reach (at the high start
    pose a negative turn straightens the elbow after ~27 deg) and with the housing clear of the
    box's near wall (turning at the old-heading pregrasp swung it into the wall); pregrasp; approach
    along the tool axis to the packet centre — or, when the jaws turn more than 10 deg past the
    demo heading, a point ``grasp_shift_m`` from it along the long axis toward -y (with the jaws
    turned +25..+40 deg the housing otherwise sits over the box's near wall); close; lift ``lift0_m``; turn back to
    ``carry_yaw`` (default: the demo heading, ``EXPERT_ROT_GRASP``) while low — a lift with the
    jaws turned -40 deg runs out of reach at ~135 mm; then lift, climb, carry, lower, release,
    retreat as ``scripted_pick_place_plan``. Returns one record per action:
    ``{"phase", "values", "gripper", "T_before", "T_after"}``."""
    from controlr.robot.kinematics import matrix_to_rpy

    p_obj = np.asarray(object_pos, float)
    axis = np.array([np.cos(object_yaw), np.sin(object_yaw), 0.0])
    if axis[1] < 0:
        axis = -axis                                  # long axis pointing toward +y (the box)
    b = np.asarray(bin_center, float)
    if carry_yaw is None:
        carry_yaw = float(matrix_to_rpy(_rotvec_to_matrix(EXPERT_ROT_GRASP))[2])
    T = np.asarray(T0, float)
    jaw = jaw_yaw_for(object_yaw, float(matrix_to_rpy(T[:3, :3])[2]))
    # jaws turned positive (past the demo heading) swing the housing toward the box: grasp off-centre
    positive = (jaw - carry_yaw + np.pi) % (2 * np.pi) - np.pi > np.radians(10.0)
    grasp = p_obj - (grasp_shift_m if positive else 0.0) * axis + np.array([0.0, 0.0, grasp_above_center_m])
    log: list[dict] = []

    def heading(Tm) -> float:
        return float(matrix_to_rpy(Tm[:3, :3])[2])

    def do(phase, d=(0.0, 0.0, 0.0), dyaw=0.0, grip=None):
        nonlocal T
        if len(log) >= max_moves:
            raise RuntimeError(f"scripted yaw expert: more than {max_moves} moves (stuck in {phase})")
        vals = (float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, float(dyaw))
        T_new = np.asarray(act(vals, grip, phase), float)
        log.append({"phase": phase, "values": vals, "gripper": grip, "T_before": T, "T_after": T_new})
        T = T_new

    def goto(phase, target, step, tol=0.001):
        for _ in range(40):
            d = np.asarray(target, float) - T[:3, 3]
            n = float(np.linalg.norm(d))
            if n < tol:
                return
            do(phase, d * min(1.0, step / n))
        raise RuntimeError(f"scripted yaw expert: {phase} did not converge ({n * 1000:.1f} mm left)")

    def turn_to(phase, yaw, tol=np.radians(0.3)):
        for _ in range(10):
            dy = (yaw - heading(T) + np.pi) % (2 * np.pi) - np.pi
            if abs(dy) < tol:
                return
            do(phase, dyaw=float(np.clip(dy, -max_yaw_rad, max_yaw_rad)))
        raise RuntimeError(f"scripted yaw expert: {phase} did not converge")

    z_target = (_rz(jaw - heading(T)) @ T[:3, :3])[:, 2]        # tool axis after the turn
    goto("hover", grasp - approach_m * z_target + np.array([0.0, -turn_back_m, turn_above_m]), max_step_m)
    turn_to("align", jaw)
    goto("pregrasp", grasp - approach_m * T[:3, 2], max_step_m)
    goto("approach", grasp, fine_step_m)
    do("close", grip=0.0)
    goto("lift0", grasp + np.array([0.0, 0.0, lift0_m]), fine_step_m)
    turn_to("unrotate", carry_yaw)
    goto("lift", np.array([grasp[0], grasp[1], lift_z_m]), max_step_m)
    goto("climb", np.array([b[0], b[1] - 0.165, carry_z_m]), max_step_m)
    goto("carry", np.array([b[0], b[1], carry_z_m]), max_step_m)
    goto("lower", np.array([b[0], b[1], place_tcp_z_m]), max_step_m)
    do("release", grip=open_width_m)
    goto("retreat", np.array([b[0], b[1], retreat_z_m]), max_step_m)
    return log
