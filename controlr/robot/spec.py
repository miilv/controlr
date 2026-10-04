"""The ONE RobotSpec constructor for the UR3 CB3 + Robotiq 2F-85 rig.

Shared by the mock and the Isaac backend so the system prompt, the safety
envelope and the feedback describe the same robot whichever backend runs.

Numbers and their provenance (PHANTOM, /home/agent/skoltech/research):
* DH / TCP: ``phantom/sim/kinematics.py``, ``configs/hardware.yaml``
  (arm.tcp_offset_m = +0.18 m along tool z, fingertip centre).
* table surface z = -0.0095 m: the mat top of PHANTOM's current measured
  waffle scene (``waffles_w2l_adaptive_parallel_dt1ms_20260909.json``), the
  value the Isaac backend reports. (The older ``waffles.json`` fit said 0.053 m,
  which put real demo TCP heights of 41.5 mm below the table.)
* workspace box: union of the demo TCP envelopes in ``configs/start_poses.yaml``
  (x -0.518..-0.230, y -0.377..0.166, z up to 0.476) padded by ~3 cm; the lower
  z bound is the table, the clearance itself is SafetyConfig's job.
* joint speed 1.0 rad/s: hardware.yaml arm.limits.joint_speed_rad_s.
* joint limits +-360 deg except the elbow, +-180 deg: PHANTOM's URDF
  (assets/sim/ur3/ur3_cb3.urdf) limits the elbow to +-pi — the Isaac drive stalls
  there, and the real elbow self-collides near +-170 deg anyway.
* finger pads: PHANTOM gripper.pad_size (W2L gel pads ~48 mm long along the tool
  axis, ~39 mm wide), centred on the TCP; ~6 mm finger thickness outward of the gap.
"""

from __future__ import annotations

import math

import numpy as np

from controlr.robot.kinematics import DEFAULT_TCP_OFFSET, JOINT_NAMES
from controlr.types import JointSpec, RobotSpec

TABLE_Z = -0.0095
FINGER_PAD = (0.024, 0.019, 0.006)     # half length (tool z), half width (tool y), outward thickness
# Tool pointing straight down (tool z = -base z), TCP at about (-0.30, -0.15, 0.15):
# the natural pose for top-down tasks with rotation=none (see tests/test_kinematics.py).
HOME_Q_TOPDOWN = (0.12248, -1.49581, 1.41546, -1.49026, -1.57119, -4.58983)
# PHANTOM Carton demo mean start pose (tilted gripper; TCP ~(-0.383, -0.277, 0.312)).
HOME_Q_PHANTOM = (0.1804, -1.5300, 1.2218, 0.5795, 1.2155, -3.1032)

_BASE_DOC = (
    "Robot base frame (UR controller 'base'): origin at the centre of the robot's "
    "mounting flange, z points up, the table surface is at z = {table_mm:.0f} mm. "
    "The work area (mat, objects) lies at negative x (x about -150 to -550 mm); "
    "y runs across the table (about -400 to +200 mm)."
)
_TCP_DOC = (
    "The TCP is the centre point between the two Robotiq 2F-85 fingertips, "
    "{off_mm:.0f} mm from the tool flange along the tool axis. Positions you command "
    "and read are of this point."
)


def ur3_cb3_spec(*, home_q: tuple[float, ...] = HOME_Q_TOPDOWN, table_z: float = TABLE_Z,
                 tcp_offset: tuple[float, ...] = DEFAULT_TCP_OFFSET,
                 gripper_max_mm: float = 85.0, joint_speed: float = 1.0,
                 workspace_lo: tuple[float, float, float] | None = None,
                 workspace_hi: tuple[float, float, float] = (-0.15, 0.22, 0.48),
                 finger_pad: tuple[float, float, float] | None = FINGER_PAD,
                 name: str = "UR3 CB3 + Robotiq 2F-85") -> RobotSpec:
    """RobotSpec for the Skoltech UR3 CB3 rig. Backends override only what
    their scene differs in (e.g. the Isaac backend's effective pad aperture
    or a measured TCP offset) — keep the defaults in sync with PHANTOM."""
    lims = {n: (np.pi if n == "elbow" else 2 * np.pi) for n in JOINT_NAMES}
    joints = tuple(JointSpec(n, -lims[n], lims[n], joint_speed) for n in JOINT_NAMES)
    lo = workspace_lo if workspace_lo is not None else (-0.55, -0.42, table_z)
    return RobotSpec(
        name=name,
        joints=joints,
        base_frame_doc=_BASE_DOC.format(table_mm=table_z * 1000),
        tcp_doc=_TCP_DOC.format(off_mm=float(np.linalg.norm(tcp_offset[:3])) * 1000),
        gripper_max_mm=gripper_max_mm,
        workspace_lo=tuple(float(v) for v in lo),
        workspace_hi=tuple(float(v) for v in workspace_hi),
        home_q=tuple(float(v) for v in home_q),
        table_z=float(table_z),
        tcp_offset=tuple(float(v) for v in tcp_offset),
        finger_pad=tuple(float(v) for v in finger_pad) if finger_pad is not None else None,
    )


# ---------------------------------------------------------------------------
# RoboDojo: dual ARX X5 (Isaac Sim 5.1, RoboDojo's own scene)
# ---------------------------------------------------------------------------

# RoboDojo world frame (env_cfg/robot/dual_x5.yml, env_cfg/scene/default.yml): z up, origin
# on the floor below the table centre-front; arm bases at x = -/+0.30, y = -0.45 on the table
# top (z 0.765), both facing +y. Grasp point = flange + 0.1501 m along the flange's +x (the
# centre of the gripping face; XPolicyLab GPT_6_Astra_Direct_EEF pose.py, from X5A.urdf).
ARX_X5_TABLE_Z = 0.765
ARX_X5_GRASP_OFFSET_M = 0.1501
ARX_X5_BASES = {"L": (-0.30, -0.45, 0.765), "R": (0.30, -0.45, 0.765)}


def arx_x5_dual_spec(*, gripper_max_mm: float = 80.0,
                     workspace_lo: tuple[float, float, float] = (-0.75, -0.65, 0.765),
                     workspace_hi: tuple[float, float, float] = (0.75, 0.45, 1.40)) -> RobotSpec:
    """Both RoboDojo ARX X5 arms as one multi-arm ``RobotSpec`` (shared world frame; the
    backend's cuRobo plans the joints, so the joint table is informational only)."""
    joints = tuple(JointSpec(f"joint{i}", -math.pi, math.pi, 3.0) for i in range(1, 7))
    return RobotSpec(
        name="two ARX X5 6-DoF arms with parallel grippers (RoboDojo, simulated)",
        joints=joints,
        base_frame_doc=(
            "One WORLD frame for both arms, in metres internally: z points up; the table top is at "
            f"z = {ARX_X5_TABLE_Z * 1000:.0f} mm. The two arms stand on the near edge of the table, "
            "600 mm apart, both reaching forward along +y: arm L at x = -300 mm, arm R at x = +300 mm, "
            "y = -450 mm. +x points from L to R (to the right seen from behind the arms), +y away "
            "from the arms across the table."),
        tcp_doc=(
            "Each arm's TCP is the grasp point: the centre of its gripping face, between the finger "
            f"pads, {ARX_X5_GRASP_OFFSET_M * 1000:.0f} mm out from the wrist flange along the tool "
            "axis. Tool z points out of the gripper (from the wrist toward the fingertips)."),
        gripper_max_mm=gripper_max_mm,
        workspace_lo=tuple(workspace_lo),
        workspace_hi=tuple(workspace_hi),
        home_q=(0.0,) * 12,
        table_z=ARX_X5_TABLE_Z,
        kinematics="backend",
        arms=("L", "R"),
    )
