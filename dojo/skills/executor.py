"""Executor: planned joint segments -> one official action per env step, both arms at once.

Every env step commands both arms (``{arm}_arm_joint_state``, 6) and both grippers
(``{arm}_ee_joint_state``, 1). An arm without a plan holds its last COMMANDED joints, and a
gripper re-sends its last command (PyRUA's KEEP: re-sending the measured opening relaxes the
squeeze). The action list is recorded so a run replays bit-for-bit (simlab ``Trajectory``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from dojo.skills.kinematics import ARMS
from dojo.skills.motion import MoveResult, Segment, merge_steps, segment_targets


@dataclass
class ArmPlan:
    """One arm's queue of ops, compiled to one (joint target, gripper command) per env step.

    ops: ("move", [Segment, ...]) | ("grip", g, n_steps) | ("hold", n_steps) | ("mark", name)
    """
    arm: str
    q0: np.ndarray
    g0: float
    ops: list = field(default_factory=list)

    def move(self, mv: MoveResult | list[Segment], name: str = "") -> "ArmPlan":
        segs = mv.segments if isinstance(mv, MoveResult) else mv
        self.ops.append(("move", list(segs), name))
        return self

    def grip(self, g: float, n: int = 3, name: str = "") -> "ArmPlan":
        self.ops.append(("grip", float(g), int(n), name))
        return self

    def hold(self, n: int, name: str = "") -> "ArmPlan":
        if n > 0:
            self.ops.append(("hold", int(n), name))
        return self

    def mark(self, name: str) -> "ArmPlan":
        self.ops.append(("mark", name))
        return self

    @property
    def q_end(self) -> np.ndarray:
        q = np.asarray(self.q0, float)
        for op in self.ops:
            if op[0] == "move" and op[1]:
                q = np.asarray(op[1][-1].q[-1], float)
        return q

    @property
    def g_end(self) -> float:
        g = self.g0
        for op in self.ops:
            if op[0] == "grip":
                g = op[1]
        return g

    def compile(self) -> tuple[list[tuple[np.ndarray, float]], list[tuple[int, str]]]:
        """Per-step (q, g) and the step index at which each named op starts."""
        q, g = np.asarray(self.q0, float), float(self.g0)
        out, marks = [], []
        for op in self.ops:
            kind = op[0]
            if kind == "move":
                marks.append((len(out), op[2]))
                for s in op[1]:
                    path = np.asarray(s.q, float)
                    if np.abs(path[0] - q).max() > 1e-6:
                        path = np.vstack([q, path])
                    for t in segment_targets(path, s.step_rad, getattr(s, "ease", 0)):
                        out.append((t, g))
                    q = path[-1]
            elif kind == "grip":
                marks.append((len(out), op[3]))
                g = op[1]
                out.extend([(q, g)] * op[2])
            elif kind == "hold":
                marks.append((len(out), op[2]))
                out.extend([(q, g)] * op[1])
            elif kind == "mark":
                marks.append((len(out), op[1]))
        return out, marks

    @property
    def steps(self) -> int:
        return len(self.compile()[0])


class EpisodeEnded(Exception):
    """The env ended the episode (success, failure or step limit)."""


class BudgetExceeded(Exception):
    """Not enough env steps left for the requested action plus the reserve for ``finish``."""


@dataclass
class ExecLog:
    actions: list[dict] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)


class Executor:
    def __init__(self, env, reserve: int = 0, record: bool = True):
        self.env = env
        st = env.proprio()
        self.q = {a: np.asarray(st[f"{a}_arm_joint_state"], float).reshape(-1)[:6].copy() for a in ARMS}
        self.g = {a: float(np.asarray(st[f"{a}_ee_joint_state"]).reshape(-1)[0]) for a in ARMS}
        self.home_q = copy.deepcopy(self.q)
        self.reserve = reserve
        self.record = record
        self.log = ExecLog()
        self.ended = False
        self.on_step = None

    # ------------------------------------------------------------------ basics
    @property
    def steps_used(self) -> int:
        return int(self.env.steps_used)

    @property
    def steps_left(self) -> int:
        return int(self.env.step_lim) - self.steps_used

    def action(self) -> dict:
        d = {}
        for a in ARMS:
            d[f"{a}_arm_joint_state"] = np.asarray(self.q[a], np.float32)
            d[f"{a}_ee_joint_state"] = np.asarray([self.g[a]], np.float32)
        return d

    def _step(self) -> None:
        if self.ended:
            raise EpisodeEnded("episode already ended")
        act = self.action()
        if self.record:
            self.log.actions.append({k: v.copy() for k, v in act.items()})
        self.ended = bool(self.env.step(act))
        if self.on_step is not None:
            self.on_step(self)
        if self.ended:
            raise EpisodeEnded(f"ended at step {self.steps_used}")

    def measured(self) -> dict[str, np.ndarray]:
        st = self.env.proprio()
        return {a: np.asarray(st[f"{a}_arm_joint_state"], float).reshape(-1)[:6] for a in ARMS}

    def _budget(self, n: int, use_reserve: bool) -> None:
        avail = self.steps_left - (0 if use_reserve else self.reserve)
        if n > avail:
            raise BudgetExceeded(f"need {n} steps, {avail} available (reserve {self.reserve})")

    # ------------------------------------------------------------------ motion
    def execute(self, plans: dict[str, MoveResult | list[Segment]], grip: dict[str, float] | None = None,
                name: str = "", use_reserve: bool = False, settle: int = 0) -> int:
        """Run each arm's segments concurrently (both arms move in the same env steps).
        ``grip`` sets gripper commands at the start (fingers move during the motion)."""
        per_arm = {}
        for a, p in plans.items():
            segs = p.segments if isinstance(p, MoveResult) else p
            tg = []
            q = self.q[a]
            for s in segs:
                path = np.asarray(s.q, float)
                if np.abs(path[0] - q).max() > 1e-6:      # start from where the arm is commanded
                    path = np.vstack([q, path])
                tg.extend(segment_targets(path, s.step_rad, getattr(s, "ease", 0)))
                q = path[-1]
            per_arm[a] = tg
        steps = merge_steps(per_arm, self.q)
        self._budget(len(steps) + settle, use_reserve)
        if grip:
            for a, v in grip.items():
                self.g[a] = float(np.clip(v, 0.0, 1.0))
        for st in steps:
            for a, q in st.items():
                self.q[a] = np.asarray(q, float)
            self._step()
        if settle:
            self.hold(settle, use_reserve=use_reserve)
        if self.record:
            self.log.segments.append({"name": name, "kind": "arm", "arms": sorted(plans), "steps": len(steps) + settle})
        return len(steps) + settle

    def plan(self, arm: str) -> ArmPlan:
        """A new plan starting from this arm's current command."""
        return ArmPlan(arm, self.q[arm].copy(), self.g[arm])

    def run(self, plans: dict[str, ArmPlan] | list[ArmPlan], name: str = "", use_reserve: bool = False,
            align: str = "start") -> dict:
        """Execute arm plans concurrently. ``align='end'`` delays the shorter plan so both end together
        (useful when one arm must not leave before the other is done)."""
        if isinstance(plans, list):
            plans = {p.arm: p for p in plans}
        comp = {a: p.compile()[0] for a, p in plans.items()}
        n = max((len(v) for v in comp.values()), default=0)
        if align == "end":
            for a, v in comp.items():
                if len(v) < n:
                    pad = [(self.q[a].copy(), self.g[a])] * (n - len(v))
                    comp[a] = pad + v
        self._budget(n, use_reserve)
        for i in range(n):
            for a, v in comp.items():
                if i < len(v):
                    self.q[a] = np.asarray(v[i][0], float)
                    self.g[a] = float(v[i][1])
            self._step()
        if self.record:
            self.log.segments.append({"name": name, "kind": "plan", "arms": {a: len(v) for a, v in comp.items()},
                                      "steps": n})
        return {"steps": n, "per_arm": {a: len(v) for a, v in comp.items()}}

    def run_lists(self, lists: dict[str, list], name: str = "", use_reserve: bool = False) -> int:
        """Execute already-compiled per-step (q, g) lists concurrently."""
        n = max((len(v) for v in lists.values()), default=0)
        self._budget(n, use_reserve)
        for i in range(n):
            for a, v in lists.items():
                if i < len(v):
                    self.q[a] = np.asarray(v[i][0], float)
                    self.g[a] = float(v[i][1])
            self._step()
        if self.record and n:
            self.log.segments.append({"name": name, "kind": "lists", "arms": {a: len(v) for a, v in lists.items()},
                                      "steps": n})
        return n

    def actions_for(self, lists: dict[str, list]) -> list[dict]:
        """The exact action dicts ``run_lists`` would send for these per-step (q, g) lists."""
        q = {a: self.q[a].copy() for a in ARMS}
        g = dict(self.g)
        n = max((len(v) for v in lists.values()), default=0)
        out = []
        for i in range(n):
            for a, v in lists.items():
                if i < len(v):
                    q[a] = np.asarray(v[i][0], float)
                    g[a] = float(v[i][1])
            act = {}
            for a in ARMS:
                act[f"{a}_arm_joint_state"] = np.asarray(q[a], np.float32)
                act[f"{a}_ee_joint_state"] = np.asarray([g[a]], np.float32)
            out.append(act)
        return out

    def run_actions(self, actions: list[dict], name: str = "", use_reserve: bool = False) -> int:
        """Send exact action dicts (e.g. a verified segment); the commanded state follows them."""
        self._budget(len(actions), use_reserve)
        for act in actions:
            for a in ARMS:
                self.q[a] = np.asarray(act[f"{a}_arm_joint_state"], float)
                self.g[a] = float(np.asarray(act[f"{a}_ee_joint_state"]).reshape(-1)[0])
            self._step()
        if self.record and actions:
            self.log.segments.append({"name": name, "kind": "actions", "steps": len(actions)})
        return len(actions)

    def grip(self, values: dict[str, float], steps: int = 3, name: str = "grip", use_reserve: bool = False) -> int:
        self._budget(steps, use_reserve)
        for a, v in values.items():
            self.g[a] = float(np.clip(v, 0.0, 1.0))
        for _ in range(steps):
            self._step()
        if self.record:
            self.log.segments.append({"name": name, "kind": "grip", "values": dict(values), "steps": steps})
        return steps

    def hold(self, n: int, name: str = "hold", use_reserve: bool = False) -> int:
        self._budget(n, use_reserve)
        for _ in range(n):
            self._step()
        return n

    def settle(self, tol: float = 0.01, max_steps: int = 6, use_reserve: bool = False) -> int:
        """Hold until every measured joint is within ``tol`` of its command (or ``max_steps``)."""
        n = 0
        while n < max_steps:
            m = self.measured()
            if all(np.abs(m[a] - self.q[a]).max() < tol for a in ARMS):
                break
            self.hold(1, use_reserve=use_reserve)
            n += 1
        return n

    def lag(self) -> dict[str, float]:
        m = self.measured()
        return {a: float(np.abs(m[a] - self.q[a]).max()) for a in ARMS}

    def trajectory_json(self) -> dict:
        return {"actions": [{k: np.asarray(v, np.float32).astype(np.float64).tolist() for k, v in a.items()}
                            for a in self.log.actions], "segments": self.log.segments}
