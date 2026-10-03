"""Safety envelope: the backend-independent filter between the parser and
``Robot.execute``.

WHY here and not in the backends: the model must get the SAME feedback for
the same mistake whatever runs underneath (mock, Isaac, the real UR3), and
the real robot must never be the first place a bad target is noticed.

What it does, per action, in order (a chunk chains: each action starts from
the previous action's resolved target, not from the measured state):

1. gripper width clamped to [0, gripper_max];
2. ee modes: resolve the absolute TCP target, limit the per-line translation
   / rotation (``max_step_m`` / ``max_step_rad``), clamp to the workspace box
   (shrunk by ``workspace_margin_m``) and to the table clearance — applied to
   the LOWEST point of the finger pads (``RobotSpec.finger_pad``), not just the
   TCP: a tilted, open gripper reaches 15-45 mm below its TCP. Then the TCP is
   moved along the commanded straight line in steps of ``path_step_m`` with IK
   per step (soft joint limits, seeded by the previous step, no branch
   switching): the backend gets the waypoints (``Action.q_path``) so the robot
   follows the line instead of a joint-space arc. A step that has no IK
   solution, or needs a joint jump far out of proportion to the TCP step
   (a wrist singularity), ends the move there: the move is SHORTENED to the
   feasible fraction (>= 10 %, else skipped as ``ik_fail``) and reported;
3. rotation=none: the tool orientation target is the reference orientation
   captured by ``reset(state0)`` (not the measured one), so a tilt caused by a
   contact is undone by the next move instead of being locked in (WARN when the
   tool is off by > 3 deg). rotation=yaw (``SafetyEnvelope(..., rotation="yaw")``):
   only roll/pitch are held at the reference (the tilt); the heading is the
   current yaw + the commanded dyaw (ee_delta) or the commanded yaw (ee_abs), so a
   yaw turn is never undone and a contact tilt still is;
4. joint modes: per-line joint step limit, TCP step limit (``max_step_m``, via
   FK), soft joint limits, then the TCP / fingertips are checked against the
   workspace/table at samples ALONG the joint path, and the move is shortened
   to the largest safe fraction if any sample leaves the envelope;
5. known obstacles (``set_obstacles``: the backend's current scene, e.g. the blue box
   from its CURRENT pose): the wrist / gripper-housing centre line (``obstacles.body_points``)
   along the move is checked against the box walls and floor inflated by
   ``box_clearance_m``. ``box_collision=block`` shortens (or skips) a move that would get
   closer than that, reported like a clamp; ``warn`` executes it and adds a WARN; ``off``
   does not check. A move that does not get closer is always allowed (backing out);
6. near-limit warnings for joints that end within ``near_limit_rad`` of a
   soft limit.

``cfg.clamp`` False turns every clamp into a rejection (the action is dropped).
Executed actions keep the requested mode with the clamped values (values are
untouched when nothing was clamped, so feedback can compare requested vs
executed exactly) and carry ``q_target`` (+ ``q_path``) so backends need not
redo IK. Messages are written for the model in canonical mm / deg;
``controlr.protocol.feedback`` converts them to the configured LLM units. Every clamp also
carries a ``brief`` (what was cut, a few words, no measurements) for ``feedback.level=short``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from controlr.config import SafetyConfig
from controlr.robot.kinematics import (
    DEFAULT_TCP_OFFSET, IKOptions, UR3Kinematics, matrix_to_rotvec, matrix_to_rpy,
    rotation_angle, rotvec_to_matrix, rpy_to_matrix,
)
from controlr.robot.obstacles import body_clearance
from controlr.types import Action, ActionMode, EventLevel, RobotSpec, RobotState, SafetyEvent

_EE = (ActionMode.EE_DELTA, ActionMode.EE_ABS)
_AXES = "xyz"
_TILT_WARN_RAD = math.radians(3.0)


def _mm(v: float) -> str:
    return f"{v * 1000:.0f} mm"


def _deg(v: float) -> str:
    return f"{np.degrees(v):.1f} deg"


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a: float) -> float:
    """Angle wrapped to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def finger_drop(spec: RobotSpec, R: np.ndarray, width_m: float) -> float:
    """How far (m) the lowest finger-pad corner sits below the TCP for tool
    orientation ``R`` and finger opening ``width_m`` (0 without pad geometry).

    The pads are boxes centred on the TCP: half length ``hl`` along tool z, half
    width ``hw`` along tool y, and their outer faces at ``width/2 + thickness``
    along the jaw axis (tool x). The lowest corner is the sum of the absolute
    vertical components."""
    if spec.finger_pad is None:
        return 0.0
    hl, hw, th = spec.finger_pad
    R = np.asarray(R, float)
    return float(abs(R[2, 0]) * (max(width_m, 0.0) / 2 + th) + abs(R[2, 2]) * hl + abs(R[2, 1]) * hw)


class SafetyEnvelope:
    # Branch guard for one interpolation step of an ee move: max |dq| may be at most
    # DQ_PER_M * (TCP step) + DQ_PER_RAD * (tool rotation step) + DQ_ABS. A bigger jump
    # means IK swung the wrist through a singularity (seen live: 78 deg of wrist_3 for a
    # 50 mm move) or flipped branch — the move is shortened before that point.
    DQ_PER_M = 20.0          # 0.02 rad per mm
    DQ_PER_RAD = 2.0
    DQ_ABS = 0.05
    MIN_FRACTION = 0.10      # shorter remainders are not worth executing: skip + ik_fail
    JOINT_PATH_SAMPLES = 10

    def __init__(self, spec: RobotSpec, cfg: SafetyConfig, kin: UR3Kinematics | None = None,
                 rotation: str | None = None) -> None:
        self.spec = spec
        self.cfg = cfg
        # ActionConfig.rotation of the experiment. "yaw": ee_delta values (dx dy dz 0 0 dyaw)
        # hold the reference roll/pitch; None/"full": 6 values are a free extrinsic rotation.
        self.rotation = rotation
        self.kin = kin or UR3Kinematics(tcp_offset=spec.tcp_offset or DEFAULT_TCP_OFFSET)
        hard = np.array([[j.lower, j.upper] for j in spec.joints], dtype=float)
        self.hard = hard
        self.soft = hard + np.array([cfg.joint_margin_rad, -cfg.joint_margin_rad])
        m = cfg.workspace_margin_m
        self.ws_lo = np.asarray(spec.workspace_lo, float) + m
        self.ws_hi = np.asarray(spec.workspace_hi, float) - m
        # height the lowest finger point (or the TCP, without pad geometry) must keep
        self.table_floor = (spec.table_z + cfg.table_clearance_m) if spec.table_z is not None else None
        self.z_floor = self.ws_lo[2] if self.table_floor is None else max(self.ws_lo[2], self.table_floor)
        self.path_options = IKOptions(restarts=0)
        self.names = [j.name for j in spec.joints]
        self.R_ref: np.ndarray | None = None
        self.elbow_sign = 0.0
        self.obstacles: list = []

    BOX_TOL_M = 0.0005           # a move "gets closer" to an obstacle only beyond this

    def set_obstacles(self, obstacles) -> None:
        """Known obstacles (``controlr.robot.obstacles.BoxObstacle``) at their CURRENT pose;
        the loop refreshes them from the backend before every filter."""
        self.obstacles = list(obstacles or [])

    # ------------------------------------------------------------------ api
    def reset(self, state0: RobotState) -> None:
        """Capture the reference tool orientation (rotation=none keeps the tool at
        this orientation for the whole episode). Call after every robot reset with
        the nominal reset state (``Robot.reference_state()`` or the measured one)."""
        self.R_ref = self.kin.fk_matrix(np.asarray(state0.q, float))[:3, :3].copy()
        q0 = np.asarray(state0.q, float)
        # elbow branch of the episode (UR: elbow up/down = sign of the elbow joint)
        self.elbow_sign = float(np.sign(q0[2])) if len(q0) == 6 and abs(q0[2]) > 1e-3 else 0.0

    def tcp_floor(self, R: np.ndarray, width_m: float) -> float:
        """Lowest allowed TCP z for orientation ``R`` and opening ``width_m``."""
        if self.table_floor is None:
            return float(self.ws_lo[2])
        return float(max(self.ws_lo[2], self.table_floor + finger_drop(self.spec, R, width_m)))

    def hold_tilt(self, R_now: np.ndarray, yaw: float) -> np.ndarray:
        """Reference roll/pitch (the tool's tilt) with heading ``yaw``: the orientation a
        rotation=yaw move targets. Pre-multiplying by Rz leaves extrinsic roll/pitch
        unchanged, so this is Rz(yaw - yaw_ref) @ R_ref."""
        R_base = self.R_ref if self.R_ref is not None else R_now
        r, p, _ = matrix_to_rpy(R_base)
        return rpy_to_matrix([r, p, yaw])

    def filter(self, actions: list[Action], state: RobotState) -> tuple[list[Action], list[SafetyEvent]]:
        events: list[SafetyEvent] = []
        out: list[Action] = []
        q = np.asarray(state.q, dtype=float).copy()
        width = float(state.gripper_mm) / 1000.0
        if self.R_ref is not None and any(a.mode in _EE and a.values is not None and len(a.values) == 3
                                          for a in actions):
            tilt = rotation_angle(self.R_ref @ self.kin.fk_matrix(q)[:3, :3].T)
            if tilt > _TILT_WARN_RAD:
                self._ev(events, "tilt", f"the tool is tilted {_deg(tilt)} away from its fixed orientation "
                                         f"(after a contact?); this move turns it back",
                         brief="tool tilted by a contact; this move turns it back")
        elif self.R_ref is not None and self.rotation == "yaw" and any(
                a.mode in _EE and a.values is not None and len(a.values) in (4, 6) for a in actions):
            R_now = self.kin.fk_matrix(q)[:3, :3]
            tilt = rotation_angle(self.hold_tilt(R_now, matrix_to_rpy(R_now)[2]) @ R_now.T)
            if tilt > _TILT_WARN_RAD:
                self._ev(events, "tilt", f"the tool is tilted {_deg(tilt)} away from its fixed tilt "
                                         f"(after a contact?); this move turns it back (yaw is kept)",
                         brief="tool tilted by a contact; this move turns it back")
        for a in actions:
            res = self._one(a, q, width, events)
            if res is None:
                continue
            act, q = res
            if act.gripper is not None:
                width = act.gripper
            out.append(act)
        events.extend(self._near_limit_events(q) if out else [])
        return out, events

    # ------------------------------------------------------------ internals
    def _ev(self, events: list[SafetyEvent], kind: str, msg: str, level: EventLevel = EventLevel.WARN,
            brief: str = "") -> None:
        events.append(SafetyEvent(level, kind, msg, brief))

    def _violation(self, events, kind: str, what: str, clamped_to: str, brief: str = "") -> bool:
        """Record a limit violation; returns True if the action may continue
        (clamp mode) and False if it must be dropped (reject mode). ``brief``: the
        clamp in a few words (the reject brief is derived from it)."""
        if self.cfg.clamp:
            self._ev(events, kind, f"{what} -> {clamped_to}", brief=brief)
            return True
        rej = (brief.replace("shortened", "rejected", 1) if "shortened" in brief
               else f"move rejected: {brief}" if brief else "move rejected")
        self._ev(events, kind, f"{what} -> action rejected", brief=rej)
        return False

    def _gripper(self, a: Action, events) -> float | None:
        if a.gripper is None:
            return None
        gmax = self.spec.gripper_max_mm / 1000.0
        g = float(np.clip(a.gripper, 0.0, gmax))
        if abs(g - a.gripper) > 1e-9:
            self._ev(events, "clamp", f"gripper width {_mm(a.gripper)} -> {_mm(g)} (range 0..{_mm(gmax)})",
                     brief="gripper width limited to its range")
        return g

    def _one(self, a: Action, q: np.ndarray, width: float, events) -> tuple[Action, np.ndarray] | None:
        g = self._gripper(a, events)
        if a.mode is None or a.values is None:
            return replace(a, gripper=g, q_target=tuple(float(v) for v in q), q_path=None), q
        # the fingers are as wide as now during the motion, and as commanded after it
        w_eff = max(width, g) if g is not None else width
        if a.mode in _EE:
            return self._ee(a, g, q, w_eff, events)
        return self._joint(a, g, q, w_eff, events)

    # ---------------------------------------------------------------- ee modes
    def _ee(self, a: Action, g, q: np.ndarray, w: float, events):
        v = np.asarray(a.values, dtype=float)
        n = len(v)
        if n not in (3, 4, 6):
            self._ev(events, "invalid", f"{a.mode.value} needs 3, 4 or 6 values, got {n} -> action skipped",
                     brief="move skipped: wrong number of values")
            return None
        T = self.kin.fk_matrix(q)
        p0, R0 = T[:3, 3], T[:3, :3]
        R_base = self.R_ref if self.R_ref is not None else R0     # orientation the tool should keep
        delta = a.mode is ActionMode.EE_DELTA
        yaw_mode = self.rotation == "yaw" and n in (4, 6)
        # -- requested absolute target
        if yaw_mode:
            # rotation=yaw: heading = current yaw + dyaw (delta) or the commanded heading (abs);
            # roll/pitch = the reference tilt (a contact tilt is undone, a yaw turn is kept)
            p = p0 + v[:3] if delta else v[:3].copy()
            yaw0 = matrix_to_rpy(R0)[2]
            dyaw = float(v[-1]) if delta else _wrap(float(v[-1]) - yaw0)
            if abs(dyaw) > self.cfg.max_step_rad + 1e-9:
                if not self._violation(events, "step_limit",
                                       f"yaw change {_deg(dyaw)} exceeds the per-line limit "
                                       f"{_deg(self.cfg.max_step_rad)}",
                                       f"scaled to {_deg(math.copysign(self.cfg.max_step_rad, dyaw))}",
                                       brief=f"turn shortened: {_deg(self.cfg.max_step_rad)} per-line limit"):
                    return None
                dyaw = math.copysign(self.cfg.max_step_rad, dyaw)
                changed_yaw = True
            else:
                changed_yaw = False
            R = self.hold_tilt(R0, yaw0 + dyaw)
        elif delta:
            p = p0 + v[:3]
            R = {3: lambda: R_base, 4: lambda: _rz(v[3]) @ R0, 6: lambda: rpy_to_matrix(v[3:]) @ R0}[n]()
        else:
            p = v[:3].copy()
            if n == 3:
                R = R_base
            elif n == 4:   # heading only: keep the reference tilt (roll/pitch), set yaw
                r0, p0_, _ = matrix_to_rpy(R_base)
                R = rpy_to_matrix([r0, p0_, v[3]])
            else:
                R = rpy_to_matrix(v[3:])
        changed = yaw_mode and changed_yaw
        # -- per-line step limits
        d = p - p0
        dn = float(np.linalg.norm(d))
        if dn > self.cfg.max_step_m + 1e-9:
            if not self._violation(events, "step_limit",
                                   f"translation {_mm(dn)} exceeds the per-line limit {_mm(self.cfg.max_step_m)}",
                                   f"scaled to {_mm(self.cfg.max_step_m)}",
                                   brief=f"move shortened: {_mm(self.cfg.max_step_m)} per-line limit"):
                return None
            p = p0 + d * (self.cfg.max_step_m / dn)
            changed = True
        R_rel = R @ R0.T
        ang = rotation_angle(R_rel)
        if ang > self.cfg.max_step_rad + 1e-9:
            if n == 3 or yaw_mode:   # re-aligning a tilted tool: silently partial (the tilt WARN says why)
                pass
            elif not self._violation(events, "step_limit",
                                     f"rotation {_deg(ang)} exceeds the per-line limit {_deg(self.cfg.max_step_rad)}",
                                     f"scaled to {_deg(self.cfg.max_step_rad)}",
                                     brief=f"rotation shortened: {_deg(self.cfg.max_step_rad)} per-line limit"):
                return None
            else:
                changed = True
            rv = matrix_to_rotvec(R_rel)
            R = rotvec_to_matrix(rv * (self.cfg.max_step_rad / ang)) @ R0
        # -- workspace box and table clearance (lowest fingertip, not just the TCP)
        floor = self.tcp_floor(R, w)
        lo = self.ws_lo.copy()
        lo[2] = max(lo[2], floor)
        for i in range(3):
            c = float(np.clip(p[i], lo[i], self.ws_hi[i]))
            if abs(c - p[i]) > 1e-9:
                if i == 2 and p[i] < lo[2] and self.table_floor is not None and lo[2] > self.ws_lo[2] + 1e-12:
                    drop = finger_drop(self.spec, R, w)
                    why = (f"table clearance: table surface z={_mm(self.spec.table_z)}, the lowest fingertip "
                           f"must stay {_mm(self.cfg.table_clearance_m)} above it"
                           + (f" and is {_mm(drop)} below the TCP" if drop > 0 else ""))
                    kind = "table"
                    brief = "move shortened: table clearance"
                else:
                    why = f"workspace {_AXES[i]} range {_mm(lo[i])}..{_mm(self.ws_hi[i])}"
                    kind = "workspace"
                    brief = f"move shortened: workspace edge ({_AXES[i]})"
                if not self._violation(events, kind, f"{_AXES[i]} target {_mm(p[i])} outside ({why})",
                                       f"{_mm(c)}", brief=brief):
                    return None
                p[i] = c
                changed = True
        # -- straight-line path with IK per step
        path, frac, why = self._line_path(q, p0, R0, p, R)
        if frac < 1.0 - 1e-9:
            T_end = self.kin.fk_matrix(path[-1]) if path else self.kin.fk_matrix(q)
            moved = float(np.linalg.norm(T_end[:3, 3] - p0))
            target = f"TCP target x={_mm(p[0])} y={_mm(p[1])} z={_mm(p[2])}"
            turn = rotation_angle(R @ R0.T)
            done = f"{_mm(moved)}"
            if turn > math.radians(0.5):      # a turn is part of the move: say so (a pure turn moves 0 mm)
                yaw_t = matrix_to_rpy(R)[2]
                target += (f" with yaw {_deg(yaw_t)}" if self.rotation == "yaw" or n == 4 else " with this rotation")
                done += f", turned {_deg(rotation_angle(T_end[:3, :3] @ R0.T))} of {_deg(turn)}"
            short = self._why_brief(why)
            if not path or frac < self.MIN_FRACTION:
                self._ev(events, "ik_fail",
                         f"{target} is not reachable {why} -> action skipped",
                         brief=f"move skipped: {short}")
                return None
            if not self.cfg.clamp:
                self._ev(events, "ik_fail", f"{target} is not reachable {why} -> action rejected",
                         brief=f"move rejected: {short}")
                return None
            self._ev(events, "reach", f"{target} is not reachable {why} -> moved {frac * 100:.0f} % of the way "
                                      f"({done}); the reachable edge is in that direction",
                     brief=f"move shortened: {short}")
            p, R = T_end[:3, 3].copy(), T_end[:3, :3].copy()
            changed = True
        # -- known obstacles (the box) along the reachable part of the line
        if path and self.obstacles and self.cfg.box_collision != "off":
            k = self._box_check(q, path, events, frac)
            if k is None:
                return None
            if k < len(path):
                path = path[:k]
                T_end = self.kin.fk_matrix(path[-1])
                p, R = T_end[:3, 3].copy(), T_end[:3, :3].copy()
                changed = True
        qt = path[-1] if path else q.copy()
        # -- executed values in the requested mode
        if changed:
            if delta and yaw_mode:
                dy = _wrap(matrix_to_rpy(R)[2] - matrix_to_rpy(R0)[2])
                vals = list(p - p0) + ([dy] if n == 4 else [0.0, 0.0, dy])
            elif delta:
                R_ex = R @ R0.T
                rot = {3: [], 4: [matrix_to_rpy(R_ex)[2]], 6: list(matrix_to_rpy(R_ex))}[n]
                vals = list(p - p0) + rot
            else:
                rot = {3: [], 4: [matrix_to_rpy(R)[2]], 6: list(matrix_to_rpy(R))}[n]
                vals = list(p) + rot
            values = tuple(float(x) for x in vals)
        else:
            values = a.values
        q_path = tuple(tuple(float(x) for x in qq) for qq in path[:-1]) or None
        return replace(a, values=values, gripper=g, q_target=tuple(float(x) for x in qt), q_path=q_path), qt

    def _why_brief(self, why: str) -> str:
        if "edge of its reach" in why:
            return "edge of reach"
        if "singularity" in why:
            return "wrist singularity"
        if "joint swing" in why:
            return "it needs a large joint swing"
        return "not reachable with this orientation"

    def _box_check(self, q0: np.ndarray, qs: list[np.ndarray], events, frac_of_full: float = 1.0) -> int | None:
        """Predictive wrist / housing-vs-obstacle check along the joint waypoints ``qs`` of
        one move (``box_collision`` block | warn). Returns how many waypoints may be executed
        (``len(qs)`` = all), or None when the move is skipped / rejected. A waypoint is a
        violation when its clearance is below ``box_clearance_m`` AND smaller than anything
        seen before on this move (from the start pose on): a move that keeps or gains distance
        is always allowed, so the arm can back away from a box it is already close to."""
        margin = float(self.cfg.box_clearance_m)
        c0, _, _ = body_clearance(self.kin, q0, self.obstacles)
        best = c0
        first_bad = None
        worst = (c0, "", "")
        for k, qk in enumerate(qs):
            ck, name, part = body_clearance(self.kin, qk, self.obstacles)
            if ck < worst[0]:
                worst = (ck, name, part)
            if first_bad is None and ck < margin and ck < best - self.BOX_TOL_M:
                first_bad = (k, ck, name, part)
            best = min(best, ck)
        if first_bad is None:
            return len(qs)
        k, ck, name, part = first_bad
        d, name, part = worst if self.cfg.box_collision == "warn" else (ck, name, part)
        body = "the wrist / gripper housing"
        if self.cfg.box_collision == "warn":
            self._ev(events, "box_warn", f"this move brings {body} within {_mm(d)} of {name} ({part}; keep "
                                         f"{_mm(margin)}); a contact stops the arm",
                     brief=f"move brings the wrist / gripper housing close to {name} ({part})")
            return len(qs)
        what = f"{body} would come within {_mm(ck)} of {name} ({part}; it must stay {_mm(margin)} clear)"
        frac = frac_of_full * k / max(len(qs), 1)
        if k == 0 or frac < self.MIN_FRACTION:
            self._ev(events, "box", f"{what} -> action skipped",
                     brief=f"move skipped: the wrist / gripper housing would hit {name} ({part})")
            return None
        if not self.cfg.clamp:
            self._ev(events, "box", f"{what} -> action rejected",
                     brief=f"move rejected: the wrist / gripper housing would hit {name} ({part})")
            return None
        self._ev(events, "box", f"{what} -> moved {frac * 100:.0f} % of the way",
                 brief=f"move shortened: the wrist / gripper housing would hit {name} ({part})")
        return k

    def _line_path(self, q: np.ndarray, p0: np.ndarray, R0: np.ndarray, p: np.ndarray, R: np.ndarray
                   ) -> tuple[list[np.ndarray], float, str]:
        """Joint waypoints along the straight TCP line p0 -> p (orientation slerped
        R0 -> R). Returns (waypoints incl. the end, feasible fraction, reason)."""
        dist = float(np.linalg.norm(p - p0))
        rv = matrix_to_rotvec(R @ R0.T)
        ang = float(np.linalg.norm(rv))
        step = max(self.cfg.path_step_m, 1e-4)
        n = max(1, math.ceil(max(dist / step, ang / math.radians(2.0)) - 1e-9))
        guard = self.DQ_PER_M * dist / n + self.DQ_PER_RAD * ang / n + self.DQ_ABS
        path: list[np.ndarray] = []
        q_prev = q
        for k in range(1, n + 1):
            s = k / n
            pk = p0 + s * (p - p0)
            Rk = rotvec_to_matrix(rv * s) @ R0
            qk = self.kin.ik(pk, matrix_to_rotvec(Rk), q_prev, self.soft, orientation="full",
                             options=self.path_options)
            if qk is None:
                return path, (k - 1) / n, ("with this gripper orientation inside the joint limits"
                                           + self._why_stuck(q_prev))
            jump = float(np.max(np.abs(qk - q_prev)))
            if jump > guard and not self._leaves_stretch(q_prev, qk, jump):
                return path, (k - 1) / n, "without a large joint swing" + (
                    self._why_stuck(q_prev, qk) or " (near a wrist singularity or a joint limit)")
            if self._too_straight(qk) and not self._too_straight(q_prev):
                # never END a move on the stretched-arm singularity: from elbow ~0 every
                # direction needs a large first-step joint change (live run 184337: trapped)
                return path, (k - 1) / n, "without a large joint swing" + self._why_stuck(qk)
            path.append(qk)
            q_prev = qk
        return path, 1.0, ""

    ELBOW_STRAIGHT_RAD = math.radians(20.0)
    WRIST_SINGULAR_RAD = math.radians(10.0)

    ELBOW_MIN_RAD = math.radians(8.0)        # moves stop before the elbow gets this straight
    LEAVE_STRETCH_RAD = 0.8                  # allowed first-step joint change when bending out of it

    def _too_straight(self, q: np.ndarray) -> bool:
        return len(self.names) == 6 and abs(float(q[2])) < self.ELBOW_MIN_RAD

    def _leaves_stretch(self, q_prev: np.ndarray, qk: np.ndarray, jump: float) -> bool:
        """At a (nearly) straight elbow the first step of any inward move needs a large
        elbow change (the Jacobian is singular): allow it when the elbow BENDS (|elbow| grows)
        toward the episode's branch sign, so the arm can always back out of the reach edge."""
        if len(self.names) != 6 or abs(float(q_prev[2])) >= self.ELBOW_STRAIGHT_RAD:
            return False
        bends = abs(float(qk[2])) > abs(float(q_prev[2]))
        branch_ok = self.elbow_sign == 0.0 or np.sign(qk[2]) == self.elbow_sign
        return bends and branch_ok and jump <= self.LEAVE_STRETCH_RAD

    def _why_stuck(self, *qs: np.ndarray) -> str:
        """Name the kinematic reason an ee path ends (UR geometry): a straight elbow is
        the edge of the arm's reach; wrist_2 near 0 / 180 deg is the wrist singularity."""
        if len(self.names) != 6:
            return ""
        if any(abs(float(q[2])) < self.ELBOW_STRAIGHT_RAD for q in qs):
            return (" (the arm is stretched to the edge of its reach: elbow almost straight; come "
                    "closer to the robot base or lower)")
        if any(min(abs(math.remainder(float(q[4]), math.pi)), math.pi) < self.WRIST_SINGULAR_RAD for q in qs):
            return " (wrist singularity: wrist_2 near 0/180 deg)"
        return ""

    # ------------------------------------------------------------- joint modes
    def _inside(self, qx: np.ndarray, w: float) -> bool:
        T = self.kin.fk_matrix(qx)
        p = T[:3, 3]
        lo = self.ws_lo.copy()
        lo[2] = max(lo[2], self.tcp_floor(T[:3, :3], w))
        return bool(np.all(p >= lo - 1e-9) and np.all(p <= self.ws_hi + 1e-9))

    def _path_inside(self, q: np.ndarray, qt: np.ndarray, w: float) -> bool:
        """Envelope check at samples along the joint-linear path (the TCP of a
        joint move is not monotone: it can leave the box and come back)."""
        return all(self._inside(q + s * (qt - q), w)
                   for s in np.linspace(0.0, 1.0, self.JOINT_PATH_SAMPLES + 1)[1:])

    def _largest_fraction(self, ok) -> float:
        lo_s, hi_s = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo_s + hi_s)
            lo_s, hi_s = (mid, hi_s) if ok(mid) else (lo_s, mid)
        return lo_s

    def _joint(self, a: Action, g, q: np.ndarray, w: float, events):
        v = np.asarray(a.values, dtype=float)
        nj = len(self.names)
        if len(v) != nj:
            self._ev(events, "invalid", f"{a.mode.value} needs {nj} values, got {len(v)} -> action skipped",
                     brief="move skipped: wrong number of values")
            return None
        delta = a.mode is ActionMode.JOINT_DELTA
        qt = q + v if delta else v.copy()
        changed = False
        dq = qt - q
        m = float(np.max(np.abs(dq)))
        if m > self.cfg.max_step_rad + 1e-9:
            j = int(np.argmax(np.abs(dq)))
            if not self._violation(events, "step_limit",
                                   f"{self.names[j]} change {_deg(dq[j])} exceeds the per-line limit "
                                   f"{_deg(self.cfg.max_step_rad)}",
                                   f"whole move scaled by {self.cfg.max_step_rad / m:.2f}",
                                   brief=f"joint move shortened: {_deg(self.cfg.max_step_rad)} per-line limit"):
                return None
            qt = q + dq * (self.cfg.max_step_rad / m)
            changed = True
        for j in range(nj):
            c = float(np.clip(qt[j], self.soft[j, 0], self.soft[j, 1]))
            if abs(c - qt[j]) > 1e-9:
                if not self._violation(events, "joint_limit",
                                       f"{self.names[j]} target {_deg(qt[j])} beyond the soft limit "
                                       f"{_deg(self.soft[j, 0] if qt[j] < c else self.soft[j, 1])}",
                                       _deg(c), brief=f"move shortened: {self.names[j]} joint limit"):
                    return None
                qt[j] = c
                changed = True
        # -- TCP travel per line (max_step_m applies in every mode)
        p0 = self.kin.fk_matrix(q)[:3, 3]

        def travel(s: float) -> float:
            return float(np.linalg.norm(self.kin.fk_matrix(q + s * (qt - q))[:3, 3] - p0))

        if travel(1.0) > self.cfg.max_step_m + 1e-9:
            s = self._largest_fraction(lambda x: travel(x) <= self.cfg.max_step_m)
            if not self._violation(events, "step_limit",
                                   f"joint move would carry the TCP {_mm(travel(1.0))}, more than the per-line "
                                   f"limit {_mm(self.cfg.max_step_m)}", f"move shortened to {s * 100:.0f} %",
                                   brief=f"joint move shortened: {_mm(self.cfg.max_step_m)} per-line limit"):
                return None
            qt = q + s * (qt - q)
            changed = True
        if not self._path_inside(q, qt, w):
            if not self._inside(q, w):
                # Already outside (e.g. an odd reset pose): allow the move unless it
                # goes further down below the table clearance, and say so.
                T0, Tt = self.kin.fk_matrix(q), self.kin.fk_matrix(qt)
                z0 = T0[2, 3] - finger_drop(self.spec, T0[:3, :3], w)
                zt = Tt[2, 3] - finger_drop(self.spec, Tt[:3, :3], w)
                fl = self.table_floor if self.table_floor is not None else self.z_floor
                if zt < fl and zt < z0 - 1e-6:
                    self._ev(events, "table", f"joint move would lower the fingertips to z={_mm(zt)}, below the "
                                              f"table clearance z >= {_mm(fl)} -> action skipped",
                             brief="joint move skipped: table clearance")
                    return None
                self._ev(events, "workspace", "TCP is currently outside the workspace/table envelope; "
                                              "move toward the work area",
                         brief="TCP outside the workspace; move toward the work area")
            else:
                p = self.kin.fk_matrix(qt)[:3, 3]
                s = self._largest_fraction(lambda x: self._path_inside(q, q + x * (qt - q), w))
                what = (f"joint move would put the TCP at x={_mm(p[0])} y={_mm(p[1])} z={_mm(p[2])} or pass "
                        f"outside the workspace / table clearance on the way")
                if s < 0.01:
                    self._ev(events, "workspace", f"{what} -> action skipped",
                             brief="joint move skipped: workspace / table clearance")
                    return None
                if not self._violation(events, "workspace", what, f"move shortened to {s * 100:.0f} %",
                                       brief="joint move shortened: workspace / table clearance"):
                    return None
                qt = q + s * (qt - q)
                changed = True
        if self.obstacles and self.cfg.box_collision != "off":
            fr = np.linspace(0.0, 1.0, self.JOINT_PATH_SAMPLES + 1)[1:]
            k = self._box_check(q, [q + f * (qt - q) for f in fr], events)
            if k is None:
                return None
            if k < len(fr):
                qt = q + fr[k - 1] * (qt - q)
                changed = True
        if changed:
            values = tuple(float(x) for x in (qt - q if delta else qt))
        else:
            values = a.values
        return replace(a, values=values, gripper=g, q_target=tuple(float(x) for x in qt), q_path=None), qt

    # --------------------------------------------------------------- warnings
    def _near_limit_events(self, q: np.ndarray) -> list[SafetyEvent]:
        out = []
        for j, name in enumerate(self.names):
            lo, hi = self.soft[j]
            if q[j] - lo < self.cfg.near_limit_rad:
                out.append(SafetyEvent(EventLevel.WARN, "joint_limit_near",
                                       f"{name} at {_deg(q[j])}, soft limit {_deg(lo)}", f"{name} near its joint limit"))
            elif hi - q[j] < self.cfg.near_limit_rad:
                out.append(SafetyEvent(EventLevel.WARN, "joint_limit_near",
                                       f"{name} at {_deg(q[j])}, soft limit {_deg(hi)}", f"{name} near its joint limit"))
        return out
