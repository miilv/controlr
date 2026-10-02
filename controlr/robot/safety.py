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
   (shrunk by ``workspace_margin_m``) and to the table clearance, then IK
   within the soft joint limits (hard limits minus ``joint_margin_rad``);
   no IK solution -> ``ik_fail`` event and the action is dropped;
3. joint modes: per-line step limit, soft joint limits, then the TCP of the
   joint target is checked against the workspace/table and the move is
   shortened (bisection along the joint path) if it would leave them;
4. near-limit warnings for joints that end within ``near_limit_rad`` of a
   soft limit (one per joint per filter call, last state wins).

``cfg.clamp`` False turns every clamp into a rejection (the action is dropped).
Executed actions keep the requested mode with the clamped values (values are
untouched when nothing was clamped, so feedback can compare requested vs
executed exactly) and carry ``q_target`` so backends need not redo IK.
Messages are written for the model: mm / deg, what was asked, what happens.
"""

from __future__ import annotations

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


def _mm(v: float) -> str:
    return f"{v * 1000:.0f} mm"


def _deg(v: float) -> str:
    return f"{np.degrees(v):.1f} deg"


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SafetyEnvelope:
    # Largest joint change an ee-mode IK solution may need for one MOVE line.
    # A bigger jump means IK flipped to another arm branch: the robot would
    # swing through the workspace to reach a nearby TCP pose — refuse it.
    BRANCH_GUARD_RAD = np.pi / 2

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
        self.z_floor = self.ws_lo[2]
        if spec.table_z is not None:
            self.z_floor = max(self.z_floor, spec.table_z + cfg.table_clearance_m)
        self.ik_options = IKOptions(max_joint_delta=self.BRANCH_GUARD_RAD)
        self.names = [j.name for j in spec.joints]

    # ------------------------------------------------------------------ api
    def filter(self, actions: list[Action], state: RobotState) -> tuple[list[Action], list[SafetyEvent]]:
        events: list[SafetyEvent] = []
        out: list[Action] = []
        q = np.asarray(state.q, dtype=float).copy()
        for a in actions:
            res = self._one(a, q, events)
            if res is None:
                continue
            act, q = res
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

    def _one(self, a: Action, q: np.ndarray, events) -> tuple[Action, np.ndarray] | None:
        g = self._gripper(a, events)
        if a.mode is None or a.values is None:
            return replace(a, gripper=g, q_target=tuple(float(v) for v in q)), q
        if a.mode in _EE:
            return self._ee(a, g, q, events)
        return self._joint(a, g, q, events)

    # ---------------------------------------------------------------- ee modes
    def _ee(self, a: Action, g, q: np.ndarray, events):
        v = np.asarray(a.values, dtype=float)
        n = len(v)
        if n not in (3, 4, 6):
            self._ev(events, "invalid", f"{a.mode.value} needs 3, 4 or 6 values, got {n} -> action skipped")
            return None
        T = self.kin.fk_matrix(q)
        p0, R0 = T[:3, 3], T[:3, :3]
        delta = a.mode is ActionMode.EE_DELTA
        # -- requested absolute target
        if delta:
            p = p0 + v[:3]
            R = {3: lambda: R0, 4: lambda: _rz(v[3]) @ R0, 6: lambda: rpy_to_matrix(v[3:]) @ R0}[n]()
        else:
            p = v[:3].copy()
            if n == 3:
                R = R0
            elif n == 4:
                r0, p0_, _ = matrix_to_rpy(R0)
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
            if not self._violation(events, "step_limit",
                                   f"rotation {_deg(ang)} exceeds the per-line limit {_deg(self.cfg.max_step_rad)}",
                                   f"scaled to {_deg(self.cfg.max_step_rad)}"):
                return None
            rv = matrix_to_rotvec(R_rel)
            R = rotvec_to_matrix(rv * (self.cfg.max_step_rad / ang)) @ R0
            changed = True
        # -- workspace box and table clearance
        lo = self.ws_lo.copy()
        lo[2] = self.z_floor
        for i in range(3):
            c = float(np.clip(p[i], lo[i], self.ws_hi[i]))
            if abs(c - p[i]) > 1e-9:
                if i == 2 and p[i] < lo[2] and self.spec.table_z is not None and lo[2] > self.ws_lo[2]:
                    why = (f"table clearance: table surface z={_mm(self.spec.table_z)}"
                           f" + {_mm(self.cfg.table_clearance_m)}")
                    kind = "table"
                else:
                    why = f"workspace {_AXES[i]} range {_mm(lo[i])}..{_mm(self.ws_hi[i])}"
                    kind = "workspace"
                if not self._violation(events, kind, f"{_AXES[i]} target {_mm(p[i])} outside ({why})",
                                       f"{_mm(c)}"):
                    return None
                p[i] = c
                changed = True
        # -- IK within soft joint limits, near the current branch
        qt = self.kin.ik(p, matrix_to_rotvec(R), q, self.soft, orientation="full", options=self.ik_options)
        if qt is None:
            self._ev(events, "ik_fail",
                     f"TCP target x={_mm(p[0])} y={_mm(p[1])} z={_mm(p[2])} is not reachable with this "
                     f"gripper orientation inside the joint limits -> action skipped")
            return None
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
        return replace(a, values=values, gripper=g, q_target=tuple(float(x) for x in qt)), qt

    # ------------------------------------------------------------- joint modes
    def _inside(self, qx: np.ndarray) -> bool:
        p = self.kin.fk_matrix(qx)[:3, 3]
        lo = self.ws_lo.copy()
        lo[2] = self.z_floor
        return bool(np.all(p >= lo - 1e-9) and np.all(p <= self.ws_hi + 1e-9))

    def _joint(self, a: Action, g, q: np.ndarray, events):
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
        if not self._inside(qt):
            if not self._inside(q):
                # Already outside (e.g. an odd reset pose): allow the move unless it
                # goes further down below the table clearance, and say so.
                z0, zt = self.kin.fk_matrix(q)[2, 3], self.kin.fk_matrix(qt)[2, 3]
                if zt < self.z_floor and zt < z0 - 1e-6:
                    self._ev(events, "table", f"joint move would lower the TCP to z={_mm(zt)}, below the "
                                              f"table clearance z >= {_mm(self.z_floor)} -> action skipped")
                    return None
                self._ev(events, "workspace", "TCP is currently outside the workspace/table envelope; "
                                              "move toward the work area")
            else:
                p = self.kin.fk_matrix(qt)[:3, 3]
                lo_s, hi_s = 0.0, 1.0
                for _ in range(20):                      # largest safe fraction of the joint move
                    mid = 0.5 * (lo_s + hi_s)
                    lo_s, hi_s = (mid, hi_s) if self._inside(q + mid * (qt - q)) else (lo_s, mid)
                what = (f"joint move would put the TCP at x={_mm(p[0])} y={_mm(p[1])} z={_mm(p[2])}, "
                        f"outside the workspace/table clearance (z >= {_mm(self.z_floor)})")
                if lo_s < 0.01:
                    self._ev(events, "workspace", f"{what} -> action skipped")
                    return None
                if not self._violation(events, "workspace", what, f"move shortened to {lo_s * 100:.0f} %"):
                    return None
                qt = q + lo_s * (qt - q)
                changed = True
        if changed:
            values = tuple(float(x) for x in (qt - q if delta else qt))
        else:
            values = a.values
        return replace(a, values=values, gripper=g, q_target=tuple(float(x) for x in qt)), qt

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
