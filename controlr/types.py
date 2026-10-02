"""Shared data types — the contract between the loop, the LLM layer, the
protocol, observation renderers and robot backends.

Units: everything inside the harness is SI (metres, radians, seconds).
Conversion to the LLM-facing units (mm / deg by default) happens only in
``controlr.protocol`` (parsing actions) and ``controlr.protocol.feedback`` /
``controlr.observation`` (text the model reads).

Frames: all poses are in the ROBOT BASE frame (for the UR3 this is the UR
controller "base" frame: z up, the frame the teach pendant reports TCP poses
in). Orientation is a rotation vector (axis * angle, rad), like UR's
``getActualTCPPose``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# robot description and state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointSpec:
    name: str
    lower: float           # rad, hard limit
    upper: float           # rad, hard limit
    max_velocity: float    # rad/s used for execution


@dataclass(frozen=True)
class RobotSpec:
    """Static description of the robot; rendered into the system prompt."""
    name: str                          # e.g. "UR3 CB3 + Robotiq 2F-85"
    joints: tuple[JointSpec, ...]
    base_frame_doc: str                # one paragraph: where the base frame is and how axes point
    tcp_doc: str                       # where the TCP is (e.g. "between the fingertips, 180 mm from flange")
    gripper_max_mm: float              # fully open width
    workspace_lo: tuple[float, float, float]   # m, base frame, safe box for the TCP
    workspace_hi: tuple[float, float, float]
    home_q: tuple[float, ...]          # rad


@dataclass(frozen=True)
class RobotState:
    t: float                           # robot/sim time, s
    q: np.ndarray                      # (n_joints,) rad
    tcp_pos: np.ndarray                # (3,) m, base frame
    tcp_rotvec: np.ndarray             # (3,) rad, base frame
    gripper_mm: float                  # current opening width
    gripper_closed: bool               # last commanded state is "closed"
    holding: bool | None = None        # backend's grasp detection; None = unknown


@dataclass(frozen=True)
class CameraInfo:
    """Pinhole intrinsics + extrinsics so renderers can project base-frame
    points (TCP marker, axes, grid) into the image."""
    name: str
    width: int
    height: int
    K: np.ndarray                      # (3,3)
    T_cam_base: np.ndarray             # (4,4) maps base-frame points into the OpenCV camera frame
                                       # (x right, y down, z forward)


@dataclass(frozen=True)
class Observation:
    t: float
    images: dict[str, np.ndarray]      # camera name -> HxWx3 uint8 RGB, native resolution
    cameras: dict[str, CameraInfo]
    state: RobotState


# ---------------------------------------------------------------------------
# actions (already converted to SI by the protocol layer)
# ---------------------------------------------------------------------------

class ActionMode(str, Enum):
    EE_DELTA = "ee_delta"          # translate (and optionally rotate) the TCP relative to its current pose, base frame
    EE_ABS = "ee_abs"              # absolute TCP target, base frame
    JOINT_DELTA = "joint_delta"    # add to current joint angles
    JOINT_ABS = "joint_abs"        # absolute joint angles


@dataclass(frozen=True)
class Action:
    """One motion step. ``values`` length depends on mode:
    ee_*: 3 (xyz, m) or 6 (xyz m + rotvec-style rx ry rz in rad, see protocol);
    joint_*: n_joints (rad). ``values`` may be None for gripper-only / hold.
    ``rotation`` for ee modes is given as extrinsic roll/pitch/yaw (rad) about
    base-frame x/y/z when 6 values are present (protocol converts deg->rad)."""
    mode: ActionMode | None
    values: tuple[float, ...] | None
    gripper: float | None = None   # target opening, m (0 = closed); None = unchanged
    raw: str = ""                  # the source line, for logs/feedback


class Status(str, Enum):
    OK = "OK"          # continue
    DONE = "DONE"      # model claims the task is complete
    FAIL = "FAIL"      # model gives up
    STUCK = "STUCK"    # model reports it cannot make progress (episode continues)
    LIMIT = "LIMIT"    # model reports it is at / near a robot limit (episode continues)


@dataclass
class ParsedReply:
    actions: list[Action]              # 0..max_chunk, executed in order
    status: Status | None
    note: str = ""
    errors: list[str] = field(default_factory=list)   # parse problems, fed back to the model
    complete: bool = False             # True once a STATUS line was seen


# ---------------------------------------------------------------------------
# execution / safety reports (robot -> loop -> feedback)
# ---------------------------------------------------------------------------

class EventLevel(str, Enum):
    INFO = "info"
    WARN = "warn"          # e.g. approaching a limit, clamp applied
    STOP = "stop"          # safety stop: episode must end or robot holds


@dataclass(frozen=True)
class SafetyEvent:
    level: EventLevel
    kind: str              # "clamp" | "joint_limit_near" | "workspace" | "collision" | "ik_fail" | "step_limit" | ...
    message: str           # human/LLM-readable, already in LLM units where relevant


@dataclass
class ExecReport:
    requested: list[Action]                 # after parsing
    executed: list[Action]                  # after safety filtering (what was actually commanded)
    events: list[SafetyEvent]
    state_before: RobotState
    state_after: RobotState
    duration_s: float                       # robot/sim time spent moving + settling
    stopped: bool = False                   # a STOP-level event ended execution early


@dataclass(frozen=True)
class GoalReport:
    success: bool
    progress: float | None                  # 0..1 if the task can measure it
    message: str                            # text that may be shown to the model (depends on config)
    metrics: dict = field(default_factory=dict)
