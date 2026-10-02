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
   tool is off by > 3 deg);
4. joint modes: per-line joint step limit, TCP step limit (``max_step_m``, via
   FK), soft joint limits, then the TCP / fingertips are checked against the
   workspace/table at samples ALONG the joint path, and the move is shortened
   to the largest safe fraction if any sample leaves the envelope;
5. near-limit warnings for joints that end within ``near_limit_rad`` of a
   soft limit.

``cfg.clamp`` False turns every clamp into a rejection (the action is dropped).
Executed actions keep the requested mode with the clamped values (values are
untouched when nothing was clamped, so feedback can compare requested vs
executed exactly) and carry ``q_target`` (+ ``q_path``) so backends need not
redo IK. Messages are written for the model in canonical mm / deg;
``controlr.protocol.feedback`` converts them to the configured LLM units.
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

    def __init__(self, spec: RobotSpec, cfg: SafetyConfig, kin: UR3Kinematics | None = None) -> None:
        self.spec = spec
        self.cfg = cfg
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

    # ------------------------------------------------------------------ api
    def reset(self, state0: RobotState) -> None:
        """Capture the reference tool orientation (rotation=none keeps the tool at
        this orientation for the whole episode). Call after every robot reset with
        the nominal reset state (``Robot.reference_state()`` or the measured one)."""
        self.R_ref = self.kin.fk_matrix(np.asarray(state0.q, float))[:3, :3].copy()

    def tcp_floor(self, R: np.ndarray, width_m: float) -> float:
        """Lowest allowed TCP z for orientation ``R`` and opening ``width_m``."""
        if self.table_floor is None:
            return float(self.ws_lo[2])
        return float(max(self.ws_lo[2], self.table_floor + finger_drop(self.spec, R, width_m)))

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
                                         f"(after a contact?); this move turns it back")
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
    def _ev(self, events: list[SafetyEvent], kind: str, msg: str, level: EventLevel = EventLevel.WARN) -> None:
        events.append(SafetyEvent(level, kind, msg))

    def _violation(self, events, kind: str, what: str, clamped_to: str) -> bool:
        """Record a limit violation; returns True if the action may continue
        (clamp mode) and False if it must be dropped (reject mode)."""
        if self.cfg.clamp:
            self._ev(events, kind, f"{what} -> {clamped_to}")
            return True
        self._ev(events, kind, f"{what} -> action rejected")
        return False

    def _gripper(self, a: Action, events) -> float | None:
        if a.gripper is None:
            return None
        gmax = self.spec.gripper_max_mm / 1000.0
        g = float(np.clip(a.gripper, 0.0, gmax))
        if abs(g - a.gripper) > 1e-9:
            self._ev(events, "clamp", f"gripper width {_mm(a.gripper)} -> {_mm(g)} (range 0..{_mm(gmax)})")
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
            self._ev(events, "invalid", f"{a.mode.value} needs 3, 4 or 6 values, got {n} -> action skipped")
            return None
        T = self.kin.fk_matrix(q)
        p0, R0 = T[:3, 3], T[:3, :3]
        R_base = self.R_ref if self.R_ref is not None else R0     # orientation the tool should keep
        delta = a.mode is ActionMode.EE_DELTA
        # -- requested absolute target
        if delta:
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
        changed = False
        # -- per-line step limits
        d = p - p0
        dn = float(np.linalg.norm(d))
        if dn > self.cfg.max_step_m + 1e-9:
            if not self._violation(events, "step_limit",
                                   f"translation {_mm(dn)} exceeds the per-line limit {_mm(self.cfg.max_step_m)}",
                                   f"scaled to {_mm(self.cfg.max_step_m)}"):
                return None
            p = p0 + d * (self.cfg.max_step_m / dn)
            changed = True
        R_rel = R @ R0.T
        ang = rotation_angle(R_rel)
        if ang > self.cfg.max_step_rad + 1e-9:
            if n == 3:     # re-aligning a tilted tool: silently partial (the tilt WARN says why)
                pass
            elif not self._violation(events, "step_limit",
                                     f"rotation {_deg(ang)} exceeds the per-line limit {_deg(self.cfg.max_step_rad)}",
                                     f"scaled to {_deg(self.cfg.max_step_rad)}"):
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
                else:
                    why = f"workspace {_AXES[i]} range {_mm(lo[i])}..{_mm(self.ws_hi[i])}"
                    kind = "workspace"
                if not self._violation(events, kind, f"{_AXES[i]} target {_mm(p[i])} outside ({why})",
                                       f"{_mm(c)}"):
                    return None
                p[i] = c
                changed = True
        # -- straight-line path with IK per step
        path, frac, why = self._line_path(q, p0, R0, p, R)
        if frac < 1.0 - 1e-9:
            p_end = self.kin.fk_matrix(path[-1])[:3, 3] if path else p0
            moved = float(np.linalg.norm(p_end - p0))
            target = f"TCP target x={_mm(p[0])} y={_mm(p[1])} z={_mm(p[2])}"
            if not path or frac < self.MIN_FRACTION:
                self._ev(events, "ik_fail",
                         f"{target} is not reachable {why} -> action skipped")
                return None
            if not self.cfg.clamp:
                self._ev(events, "ik_fail", f"{target} is not reachable {why} -> action rejected")
                return None
            self._ev(events, "reach", f"{target} is not reachable {why} -> moved {frac * 100:.0f} % of the way "
                                      f"({_mm(moved)}); the reachable edge is in that direction")
            T_end = self.kin.fk_matrix(path[-1])
            p, R = T_end[:3, 3].copy(), T_end[:3, :3].copy()
            changed = True
        qt = path[-1] if path else q.copy()
        # -- executed values in the requested mode
        if changed:
            if delta:
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
                return path, (k - 1) / n, "with this gripper orientation inside the joint limits"
            if float(np.max(np.abs(qk - q_prev))) > guard:
                return path, (k - 1) / n, ("without a large joint swing (near a wrist singularity or a "
                                           "joint limit)")
            path.append(qk)
            q_prev = qk
        return path, 1.0, ""

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
            self._ev(events, "invalid", f"{a.mode.value} needs {nj} values, got {len(v)} -> action skipped")
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
                                   f"whole move scaled by {self.cfg.max_step_rad / m:.2f}"):
                return None
            qt = q + dq * (self.cfg.max_step_rad / m)
            changed = True
        for j in range(nj):
            c = float(np.clip(qt[j], self.soft[j, 0], self.soft[j, 1]))
            if abs(c - qt[j]) > 1e-9:
                if not self._violation(events, "joint_limit",
                                       f"{self.names[j]} target {_deg(qt[j])} beyond the soft limit "
                                       f"{_deg(self.soft[j, 0] if qt[j] < c else self.soft[j, 1])}",
                                       _deg(c)):
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
                                   f"limit {_mm(self.cfg.max_step_m)}", f"move shortened to {s * 100:.0f} %"):
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
                                              f"table clearance z >= {_mm(fl)} -> action skipped")
                    return None
                self._ev(events, "workspace", "TCP is currently outside the workspace/table envelope; "
                                              "move toward the work area")
            else:
                p = self.kin.fk_matrix(qt)[:3, 3]
                s = self._largest_fraction(lambda x: self._path_inside(q, q + x * (qt - q), w))
                what = (f"joint move would put the TCP at x={_mm(p[0])} y={_mm(p[1])} z={_mm(p[2])} or pass "
                        f"outside the workspace / table clearance on the way")
                if s < 0.01:
                    self._ev(events, "workspace", f"{what} -> action skipped")
                    return None
                if not self._violation(events, "workspace", what, f"move shortened to {s * 100:.0f} %"):
                    return None
                qt = q + s * (qt - q)
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
                                       f"{name} at {_deg(q[j])}, soft limit {_deg(lo)}"))
            elif hi - q[j] < self.cfg.near_limit_rad:
                out.append(SafetyEvent(EventLevel.WARN, "joint_limit_near",
                                       f"{name} at {_deg(q[j])}, soft limit {_deg(hi)}"))
        return out
