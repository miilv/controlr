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
    # --- optional (added by the robot backends; defaults keep older constructors valid) ---
    table_z: float | None = None       # m, base frame height of the table surface (None: no table)
    tcp_offset: tuple[float, ...] | None = None   # flange(tool0)->TCP pose xyz+rotvec; None: kinematics default
    # finger pad extent around the TCP (m): (half length along tool z, half width along tool y,
    # thickness outward along the jaw axis). Lets the safety envelope keep the lowest FINGERTIP
    # (not only the TCP) above the table. None: only the TCP point is checked.
    finger_pad: tuple[float, float, float] | None = None


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
    ee_*: 3 (xyz, m), 4 (xyz m + yaw rad; ee_abs only: heading about base z, the tool keeps
    its tilt) or 6 (xyz m + extrinsic roll/pitch/yaw in rad about base x/y/z, see
    controlr.protocol.grammar); joint_*: n_joints (rad). ``values`` may be None for
    gripper-only / hold."""
    mode: ActionMode | None
    values: tuple[float, ...] | None
    gripper: float | None = None   # target opening, m (0 = closed); None = unchanged
    raw: str = ""                  # the source line, for logs/feedback
    # Resolved absolute joint target (rad), attached by SafetyEnvelope.filter so
    # backends need not redo IK. None for unfiltered actions.
    q_target: tuple[float, ...] | None = None
    # Intermediate joint waypoints (rad, excluding q_target) along the straight TCP line of
    # an ee move, attached by SafetyEnvelope.filter. Backends that interpolate should pass
    # through them (joint-linear interpolation to q_target alone bends the TCP path).
    q_path: tuple[tuple[float, ...], ...] | None = None


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
    kind: str              # "clamp" | "joint_limit_near" | "workspace" | "collision" | "ik_fail" | "step_limit"
                           # | "box" | "tactile" (fingertip sensing; shown only with observation.tactile) | ...
    message: str           # human/LLM-readable, already in LLM units where relevant
    # one-line notice without measurement numbers (feedback.level=minimal); "" -> message
    brief: str = ""


@dataclass
class ExecReport:
    requested: list[Action]                 # after parsing
    executed: list[Action]                  # after safety filtering (what was actually commanded)
    events: list[SafetyEvent]
    state_before: RobotState
    state_after: RobotState
    duration_s: float                       # robot/sim time spent moving + settling
    stopped: bool = False                   # a STOP-level event ended execution early
    # backend diagnostics for the run log only (never shown to the model): e.g. Isaac's
    # per-execute profile (physics / contact-read / wall seconds), contact peaks, pad forces
    backend: dict | None = None


@dataclass(frozen=True)
class GoalReport:
    success: bool
    progress: float | None                  # 0..1 if the task can measure it
    message: str                            # text that may be shown to the model (depends on config)
    metrics: dict = field(default_factory=dict)
