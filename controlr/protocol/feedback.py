"""Feedback text for the next user turn.

Every turn after the first starts with a compact, line-oriented receipt of
what happened, in the model's units::

    TURN 7
    EXEC: MOVE ee_delta 20 0 -10 -> achieved dx=19.6 dy=0.2 dz=-9.8 mm (0.41 s)
    CLAMP: z target 3 mm -> 20 mm (table clearance)
    WARN: wrist_2 at 171 deg, soft limit 175 deg
    EVENT: contact finger-object
    GOAL: not reached
    STATE: tcp x=312 y=-45 z=88 mm yaw=12 deg | grip 42 mm open | holding: no

Why this shape (lessons from prior harnesses, see research/sources/):
  * report the post-motion OUTCOME (achieved deltas, clamps, contacts), never
    "executing over N steps" — Robocurve's inspect-robots transcripts show the
    model needs the residual to self-correct; RoboProbe's "left 7 mm away"
    arrival feedback is what enabled self-repair;
  * a requested target is not evidence of arrival (GPT-as-Policy contract), so
    EXEC always shows achieved motion next to the request;
  * fixed decimals and a fixed line order: this text becomes part of the cached
    prefix on the next turn, and the run logs are diffed across experiments.

The canonical STATE formatter lives here (``format_state``); the observation
renderers and the prompt builder reuse it rather than formatting state twice.
"""

from __future__ import annotations

import math
import re

import numpy as np

from controlr.config import ActionConfig, Config
from controlr.protocol.grammar import (
    EE_MODES,
    JOINT_MODES,
    ang_decimals,
    ang_factor,
    fmt_num,
    grammar_reminder,
    pos_decimals,
    pos_factor,
)
from controlr.types import (
    Action,
    EventLevel,
    ExecReport,
    GoalReport,
    Observation,
    ParsedReply,
    RobotSpec,
    RobotState,
    SafetyEvent,
    Status,
)


# ---------------------------------------------------------------------------
# rotation helpers: ONE implementation (controlr.robot.kinematics, numpy only);
# thin wrappers keep this module's (roll, pitch, yaw) call style for callers.
# ---------------------------------------------------------------------------

from controlr.robot.kinematics import matrix_to_rpy as _matrix_to_rpy  # noqa: E402
from controlr.robot.kinematics import rotvec_to_matrix  # noqa: E402,F401  (re-exported for renderers)
from controlr.robot.kinematics import rpy_to_matrix as _rpy_to_matrix  # noqa: E402


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic x-y-z (= Rz(yaw) @ Ry(pitch) @ Rx(roll)), the grammar convention."""
    return _rpy_to_matrix((roll, pitch, yaw))


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of ``rpy_to_matrix`` (rad). At pitch = ±90 deg roll is set to 0."""
    r, p, y = _matrix_to_rpy(R)
    return float(r), float(p), float(y)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _roll_display(roll: float) -> float:
    """Roll in [0, 2pi): a downward tool (roll ~ 180 deg) then reads a steady
    ~180 instead of flickering between +179 and -179."""
    return roll % (2 * math.pi)


# ---------------------------------------------------------------------------
# formatting primitives
# ---------------------------------------------------------------------------

def _p(v_m: float, a: ActionConfig, fine: bool = False) -> str:
    return fmt_num(v_m / pos_factor(a), pos_decimals(a, fine))


def _ang(v_rad: float, a: ActionConfig, fine: bool = False) -> str:
    return fmt_num(v_rad / ang_factor(a), ang_decimals(a, fine))


def _grip_text(width_m: float | None, a: ActionConfig) -> str:
    if width_m is None:
        return ""
    if a.gripper == "binary":
        return "close" if width_m <= 1e-6 else "open"
    return _p(width_m, a)


def format_action(action: Action, a: ActionConfig) -> str:
    """Canonical one-line rendering of an SI Action in LLM units (same grammar
    the model writes). Used for EXEC lines so they never depend on how sloppily
    the model formatted its own line."""
    parts: list[str] = []
    if action.mode is not None and action.values is not None:
        mode = action.mode.value
        v = list(action.values)
        if mode in JOINT_MODES:
            nums = [_ang(x, a, fine=True) for x in v]
        else:
            nums = [_p(x, a) for x in v[:3]]
            if len(v) == 4:                         # ee_abs + yaw: absolute heading
                nums.append(_ang(v[3], a))
            elif len(v) >= 6:
                if a.rotation == "yaw":
                    nums.append(_ang(v[5], a))
                elif a.rotation == "full":
                    nums += [_ang(x, a) for x in v[3:6]]
        parts.append(f"MOVE {mode} {' '.join(nums)}")
        if action.gripper is not None:
            parts.append(f"GRIP {_grip_text(action.gripper, a)}")
    elif action.gripper is not None:
        parts.append(f"GRIP {_grip_text(action.gripper, a)}")
    else:
        parts.append("HOLD")
    return " ".join(parts)


def format_state(state: RobotState, cfg: Config) -> str:
    """The canonical STATE line (deterministic, LLM units).

    Orientation is shown only as far as the model can command it
    (rotation=none -> omitted, yaw -> yaw, full -> roll pitch yaw); joint
    angles only in joint modes. Keeping the line short matters: it is repeated
    every turn."""
    a = cfg.action
    pu = a.pos_unit
    x, y, z = (float(v) for v in np.asarray(state.tcp_pos).reshape(3))
    s = f"STATE: tcp x={_p(x, a)} y={_p(y, a)} z={_p(z, a)} {pu}"
    if a.mode in EE_MODES and a.rotation != "none":
        roll, pitch, yaw = matrix_to_rpy(rotvec_to_matrix(state.tcp_rotvec))
        if a.rotation == "yaw":
            s += f" yaw={_ang(yaw, a)} {a.ang_unit}"
        else:
            s += (f" roll={_ang(_roll_display(roll), a)} pitch={_ang(pitch, a)}"
                  f" yaw={_ang(yaw, a)} {a.ang_unit}")
    if a.mode in JOINT_MODES:
        q = " ".join(_ang(float(v), a, fine=True) for v in np.asarray(state.q).reshape(-1))
        s += f" | q=[{q}] {a.ang_unit}"
    grip_state = "closed" if state.gripper_closed else "open"
    s += f" | grip {_p(state.gripper_mm * 1e-3, a)} {pu} {grip_state}"
    holding = "unknown" if state.holding is None else ("yes" if state.holding else "no")
    s += f" | holding: {holding}"
    return s


def _achieved(report: ExecReport, a: ActionConfig) -> str:
    """Measured change over the whole chunk. Gripper-only chunks report only
    the gripper (unless the TCP moved anyway, e.g. pushed by contact)."""
    b, f = report.state_before, report.state_after
    moved = float(np.linalg.norm(np.asarray(f.tcp_pos, float) - np.asarray(b.tcp_pos, float)))
    motion = any(x.mode is not None for x in report.executed) or moved >= 5e-4
    if not motion:
        g = f"grip {_p(b.gripper_mm * 1e-3, a)}->{_p(f.gripper_mm * 1e-3, a)} {a.pos_unit}"
        return g
    if a.mode in JOINT_MODES:
        dq = np.asarray(f.q, dtype=float) - np.asarray(b.q, dtype=float)
        txt = "dq=[" + " ".join(_ang(float(v), a, fine=True) for v in dq) + f"] {a.ang_unit}"
    else:
        d = np.asarray(f.tcp_pos, dtype=float) - np.asarray(b.tcp_pos, dtype=float)
        txt = (f"dx={_p(d[0], a, True)} dy={_p(d[1], a, True)} dz={_p(d[2], a, True)} {a.pos_unit}")
        if a.rotation != "none":
            rb = matrix_to_rpy(rotvec_to_matrix(b.tcp_rotvec))
            rf = matrix_to_rpy(rotvec_to_matrix(f.tcp_rotvec))
            dr = [_wrap_pi(rf[i] - rb[i]) for i in range(3)]
            if a.rotation == "yaw":
                txt += f" dyaw={_ang(dr[2], a, True)} {a.ang_unit}"
            else:
                txt += (f" droll={_ang(dr[0], a, True)} dpitch={_ang(dr[1], a, True)}"
                        f" dyaw={_ang(dr[2], a, True)} {a.ang_unit}")
    if abs(f.gripper_mm - b.gripper_mm) >= 0.5:
        txt += f", grip {_p(b.gripper_mm * 1e-3, a)}->{_p(f.gripper_mm * 1e-3, a)} {a.pos_unit}"
    return txt


_CLAMP_KINDS = {"clamp", "step_limit", "workspace", "table_clearance", "table", "reject", "rejected", "reach"}

_LEN_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(\s*)mm\b")
_ANG_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(\s*)deg\b")


def to_llm_units(text: str, a: ActionConfig) -> str:
    """Rewrite ``<n> mm`` / ``<n> deg`` in backend and safety messages into the configured
    LLM units. The envelope and the backends write canonical mm/deg (they need not know
    the experiment's units); this is the single place that converts, so ``pos_unit=cm``
    never yields mixed-unit feedback ("Never mix in other units", says the manual)."""
    if a.pos_unit != "mm":
        f = 1e-3 / pos_factor(a)
        text = _LEN_RE.sub(lambda m: f"{fmt_num(float(m.group(1)) * f, pos_decimals(a))}{m.group(2)}{a.pos_unit}",
                           text)
    if a.ang_unit != "deg":
        f = math.pi / 180.0 / ang_factor(a)
        text = _ANG_RE.sub(lambda m: f"{fmt_num(float(m.group(1)) * f, ang_decimals(a, True))}{m.group(2)}"
                                     f"{a.ang_unit}", text)
    return text


def _event_tag(ev: SafetyEvent) -> str:
    if ev.level == EventLevel.STOP:
        return "STOP"
    k = ev.kind.lower()
    if k in _CLAMP_KINDS or k.startswith("clamp") or k.startswith("reject"):
        return "CLAMP"
    if ev.level == EventLevel.WARN:
        return "WARN"
    return "EVENT"


def format_events(events: list[SafetyEvent], a: ActionConfig | None = None) -> list[str]:
    """Group events as CLAMP, WARN, EVENT, STOP (that order; original order
    within a group; exact duplicates collapsed with a count). With ``a`` the
    messages are converted to the LLM units (``to_llm_units``)."""
    order = ["CLAMP", "WARN", "EVENT", "STOP"]
    groups: dict[str, list[str]] = {t: [] for t in order}
    counts: dict[tuple[str, str], int] = {}
    for ev in events:
        tag = _event_tag(ev)
        msg = to_llm_units(ev.message, a) if a is not None else ev.message
        key = (tag, msg)
        if key in counts:
            counts[key] += 1
            continue
        counts[key] = 1
        groups[tag].append(msg)
    out = []
    for t in order:
        for msg in groups[t]:
            n = counts[(t, msg)]
            out.append(f"{t}: {msg}" + (f" (x{n})" if n > 1 else ""))
    return out


def format_goal(goal: GoalReport, a: ActionConfig | None = None) -> str:
    s = "GOAL: " + ("reached" if goal.success else "not reached")
    if goal.progress is not None and not goal.success:
        s += f" (progress {int(round(100 * goal.progress))}%)"
    if goal.message:
        s += f" - {to_llm_units(goal.message, a) if a is not None else goal.message}"
    return s


def _show_goal(parsed: ParsedReply | None, goal: GoalReport | None, cfg: Config) -> bool:
    if goal is None:
        return False
    mode = cfg.episode.goal_feedback
    if mode == "always":
        return True
    if mode == "on_done":
        return parsed is not None and parsed.status == Status.DONE
    return False


def format_feedback(turn: int, parsed: ParsedReply | None, report: ExecReport | None,
                    goal: GoalReport | None, obs: Observation, cfg: Config,
                    spec: RobotSpec | None = None) -> str:
    """Feedback text for the user turn that follows the model's reply ``turn``.

    Turn 0 (no reply yet): call with ``parsed=None, report=None`` to get just
    ``TURN 0`` + the STATE line. ``spec`` (optional, contract addition) only
    sharpens the grammar reminder with joint names; without it a generic
    reminder is used."""
    a = cfg.action
    lines = [f"TURN {turn}"]
    if report is not None:
        req = report.requested or (parsed.actions if parsed else [])
        if req:
            what = "; ".join(format_action(x, a) for x in req)
        else:
            what = "HOLD"
        if not report.executed and req:
            lines.append(f"EXEC: {what} -> nothing executed ({report.duration_s:.2f} s)")
        else:
            lines.append(f"EXEC: {what} -> achieved {_achieved(report, a)} ({report.duration_s:.2f} s)")
        lines += format_events(report.events, a)
        if report.stopped and not any(e.level == EventLevel.STOP for e in report.events):
            lines.append("STOP: execution stopped early")
    elif parsed is not None:
        if parsed.actions:
            lines.append("EXEC: not executed")
        else:
            lines.append("EXEC: HOLD")
    if parsed is not None and parsed.errors:
        for err in parsed.errors:
            lines.append(f"PARSE ERROR: {err}")
        lines.append(grammar_reminder(a, spec))
    if _show_goal(parsed, goal, cfg):
        lines.append(format_goal(goal, a))
    if cfg.observation.state_text:
        lines.append(format_state(obs.state, cfg))
    return "\n".join(lines)

