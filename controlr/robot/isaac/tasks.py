"""Task registry for the Isaac backend: scene choice, seeded randomisation,
default instruction and success evaluation.

WHY pure numpy and no Isaac / PHANTOM imports: this module is imported by the
Isaac server (to place objects and score) *and* by CPU unit tests, so the
randomisation ranges and success rules are testable without a GPU. The server
hands ``evaluate`` a plain "privileged snapshot" dict (object pose/velocity,
pad/robot contact forces, bin interior bounds, TCP); everything here is
arithmetic on that dict.

Tasks share ONE stage (PHANTOM's measured waffle rig): switching between them
moves the packet and toggles visual-only markers instead of rebuilding the
scene, so the ~1 min Isaac startup is paid once per server.

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
    "nominal": False,                   # True -> PHANTOM's measured nominal pose
    "bin_tolerance_m": 0.002, "contact_force_n": 0.1,
    "settle_speed_m_s": 0.03, "settle_angular_speed_rad_s": 0.5, "lift_height_m": 0.03,
    "start_q": START_Q, "start_gripper": "open",
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
        yaw += float(np.clip(rng.normal(0.0, np.radians(float(_param(params, defaults, "yaw_sigma_deg")))),
                             -yaw_max, yaw_max))
    return {"object_pos": center, "object_yaw": yaw}


def _start(params: dict, defaults: dict) -> dict:
    return {"start_q": np.asarray(_param(params, defaults, "start_q"), float),
            "start_gripper": _param(params, defaults, "start_gripper")}


def sample_waffle(rng, params, scene):
    return {**_sample_object(rng, params, scene, WAFFLE_DEFAULTS), **_start(params, WAFFLE_DEFAULTS),
            "marker": None, "zone": None}


def evaluate_waffle(snap: dict, episode: dict) -> dict:
    p = {**WAFFLE_DEFAULTS, **episode.get("params", {})}
    tol = float(p["bin_tolerance_m"])
    corners = box_corners(snap["object_pos"], snap["object_quat_wxyz"], snap["object_size"])
    b = snap["bin"]
    lower, upper = np.asarray(b["lower"], float), np.asarray(b["upper"], float)
    local = to_bin_frame(corners, np.asarray(b["center"], float), float(b["yaw"]))
    over_bin = bool(np.all((local[:, :2] >= lower[:2] - tol) & (local[:, :2] <= upper[:2] + tol)))
    inside = bool(over_bin and corners[:, 2].min() >= lower[2] - tol and corners[:, 2].max() <= upper[2] + tol)
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

REACH_DEFAULTS = {
    **WAFFLE_DEFAULTS,
    # marker centre box in the base frame (over the mat); height measured from the
    # table top. Every sample is also checked for feasibility (reach_feasibility).
    "marker_x": (-0.50, -0.26), "marker_y": (-0.33, -0.12), "marker_height": (0.05, 0.16),
    "min_object_clearance_m": 0.08, "tolerance_m": 0.015,
    # the gripper housing / wrist must stay this far from the blue box at the marker
    "body_clearance_m": 0.045,
    # D435 pixel box hidden behind the gripper at START_Q (u0, v0, u1, v1):
    # markers projecting into it are re-drawn so the model can see them
    "occluded_px": (285, 225, 525, 455),
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


def _occluded(scene: dict, p, box) -> bool:
    uv = project(scene, p)
    return uv is not None and box[0] <= uv[0] <= box[2] and box[1] <= uv[1] <= box[3]


def _bin_solid(scene: dict, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Outer box of the bin (walls included) in the bin frame, inflated by ``margin``:
    (lower, upper, centre, yaw), or None without bin geometry."""
    b = scene.get("bin")
    if not b:
        return None
    wall = float(b.get("wall", 0.02))
    lo = np.asarray(b["lower"], float).copy()
    hi = np.asarray(b["upper"], float).copy()
    lo[:2] -= wall + margin
    hi[:2] += wall + margin
    lo[2] = float(b["center"][2]) - margin
    hi[2] += margin
    return lo, hi, np.asarray(b["center"], float), float(b["yaw"])


def reach_feasibility(marker, scene: dict, start_q, *, body_clearance: float = 0.045,
                      table_clearance: float = 0.005, open_width: float = 0.091) -> str:
    """Why a reach target cannot be touched with the START orientation ("" if it can).

    With action.rotation=none the tool keeps its start orientation, so (a) IK must
    reach the marker with that orientation inside the soft joint limits, (b) the
    open fingertips must stay above the table at the marker, and (c) the tool body
    (TCP -> flange -> wrist, inflated by ``body_clearance``) must not pass through
    the blue box: at seed 0 the old sampler put the housing inside the box's near
    wall, so no controller could succeed (review control-safety #1)."""
    from controlr.robot.kinematics import UR3Kinematics, matrix_to_rotvec   # numpy only
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
    solid = _bin_solid(scene, body_clearance)
    if solid is not None:
        lo, hi, c, yaw = solid
        F = kin.frames(q)
        chain = [F[7][:3, 3], F[6][:3, 3], F[5][:3, 3], F[4][:3, 3]]     # TCP, flange, wrist 3, wrist 2
        pts = np.concatenate([np.linspace(a, b, 12) for a, b in zip(chain, chain[1:])])
        local = to_bin_frame(pts, c, yaw)
        if np.any(np.all((local >= lo) & (local <= hi), axis=1)):
            return "the gripper body would be inside the blue box"
    return ""


def sample_reach(rng, params, scene):
    d = REACH_DEFAULTS
    out = {**_sample_object(rng, params, scene, d), **_start(params, d), "zone": None}
    lo_x, hi_x = _param(params, d, "marker_x")
    lo_y, hi_y = _param(params, d, "marker_y")
    lo_h, hi_h = _param(params, d, "marker_height")
    clear = float(_param(params, d, "min_object_clearance_m"))
    body = float(_param(params, d, "body_clearance_m"))
    box = _param(params, d, "occluded_px")
    top = float(scene["table_top_z"])
    if "marker" in params:                       # explicit marker: still refuse an impossible one
        m = np.asarray(params["marker"], float)
        why = reach_feasibility(m, scene, out["start_q"], body_clearance=body)
        if why:
            raise ValueError(f"reach marker {m.tolist()} is infeasible: {why}")
        out["marker"] = m
        return out
    for _ in range(500):
        m = np.array([rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y), top + rng.uniform(lo_h, hi_h)])
        if np.linalg.norm(m[:2] - out["object_pos"][:2]) < clear:
            continue
        if _occluded(scene, m, box) or _occluded(scene, (m[0], m[1], top), box):
            continue
        if reach_feasibility(m, scene, out["start_q"], body_clearance=body):
            continue
        out["marker"] = m
        return out
    raise RuntimeError("sample_reach: no visible, reachable, collision-free marker in 500 draws; "
                       "check marker_x/marker_y/marker_height and start_q")


def evaluate_reach(snap: dict, episode: dict) -> dict:
    p = {**REACH_DEFAULTS, **episode.get("params", {})}
    target = np.asarray(episode["marker"], float)
    d = float(np.linalg.norm(np.asarray(snap["tcp_pos"], float) - target))
    d0 = float(np.linalg.norm(np.asarray(episode["tcp_pos0"], float) - target)) or 1.0
    success = bool(d <= float(p["tolerance_m"]))
    return {"success": success, "progress": float(np.clip(1.0 - d / d0, 0.0, 1.0)),
            "message": ("goal reached: the fingertips are at the red ball" if success
                        else f"the fingertips are {d * 1000:.0f} mm from the red ball"),
            "metrics": {"distance_m": d, "tolerance_m": float(p["tolerance_m"])}}


# -- push ---------------------------------------------------------------------

PUSH_DEFAULTS = {**WAFFLE_DEFAULTS, "push_distance_m": (0.06, 0.10), "tolerance_m": 0.025}


def sample_push(rng, params, scene):
    d = PUSH_DEFAULTS
    out = {**_sample_object(rng, params, scene, d), **_start(params, d), "marker": None}
    lo, hi = _param(params, d, "push_distance_m")
    # Push along the packet's long axis: pushing its broad face tips the 90 mm
    # tall, 35 mm thick packet over. The target lies on the -x side (left in
    # the image), which keeps it on the mat and out from under the gripper.
    axis = np.array([np.cos(out["object_yaw"]), np.sin(out["object_yaw"])])
    if axis[0] > 0:
        axis = -axis
    target = out["object_pos"][:2] + axis * rng.uniform(lo, hi)
    out["zone"] = np.array([target[0], target[1], float(scene["table_top_z"])])
    return out


def evaluate_push(snap: dict, episode: dict) -> dict:
    """Pushed, not carried: besides the state at the check instant, the server's running
    record of the episode (``episode["max_lift_m"]``, ``episode["ever_held"]``) must show
    that the packet was never lifted > 20 mm or gripped (picking it up, carrying it and
    releasing it on the square used to count as a push)."""
    p = {**PUSH_DEFAULTS, **episode.get("params", {})}
    target = np.asarray(episode["zone"], float)[:2]
    d = float(np.linalg.norm(np.asarray(snap["object_pos"], float)[:2] - target))
    d0 = float(np.linalg.norm(np.asarray(episode["object_pos"], float)[:2] - target)) or 1.0
    held = bool(np.all(np.asarray(snap["pad_object_force_n"], float) > p["contact_force_n"])
                or episode.get("ever_held", False))
    lift_now = float(snap["object_pos"][2] - np.asarray(episode["object_pos"], float)[2])
    lifted = max(lift_now, float(episode.get("max_lift_m", 0.0))) > 0.02
    success = bool(d <= float(p["tolerance_m"]) and not held and not lifted)
    return {"success": success, "progress": float(np.clip(1.0 - d / d0, 0.0, 1.0)),
            "message": ("goal reached: the packet is on the green square" if success
                        else f"the packet centre is {d * 1000:.0f} mm from the green square"
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
    "reach": TaskSpec(
        "reach", WAFFLE_SCENE,
        "Move the gripper so that the point between its fingertips touches the red ball. "
        "Do not touch anything else.",
        sample_reach, evaluate_reach, REACH_DEFAULTS),
    "push": TaskSpec(
        "push", WAFFLE_SCENE,
        "Push the wafer packet along the mat until it stands on the green square. "
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
