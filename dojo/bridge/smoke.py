"""``joint_steps`` smoke for the first GPU run behind the strict server (docs/BRIDGE-RUNBOOK.md §4, practice seed 0
layout 9 only): a policy for the bridge driver that moves each arm's base and wrist-roll joints out and back at
``dq`` rad/step (0.18, the skills' free-space rate), toggles each gripper with a step command, holds between phases
and ends. Every phase is its own ``next()`` reply, hence its own ``joint_steps`` chunk, so ``bridge.json`` →
``joint_steps_chunks`` carries the server's tracking per phase (``tracking_rad``, ``tracking_max_rad``,
``max_step_dq``, ``grip_change_while_moving``) before a full episode runs. About 70 env steps, no perception, no
twin, no LLM:

    scripts/dojo/bridge_episode.sh --id br-prac-smoke-s0L9-1 --task stack_blocks --layout 9 --seed 0 \\
        --policy dojo.bridge.smoke:JointStepsSmoke --perception server --verify off
"""

from __future__ import annotations

import numpy as np

ARMS = ("left", "right")


class JointStepsSmoke:
    def __init__(self, cfg=None, dq: float = 0.18, n_steps: int = 2, joints=(0, 5), hold: int = 3, grip_steps: int = 6):
        self.cfg = cfg                      # the driver reads cfg.verify (no twin for the smoke: run with --verify off)
        self.dq, self.n_steps, self.joints = float(dq), int(n_steps), tuple(joints)
        self.hold, self.grip_steps = int(hold), int(grip_steps)
        self.report: dict = {}

    def reset(self, info: dict) -> None:
        self.phases, self.i = None, 0
        self.report = {"error": None, "info": dict(info), "phases": [], "smoke": {"dq": self.dq, "n_steps": self.n_steps,
                                                                                 "joints": list(self.joints)}}

    def _act(self, q: dict, g: dict) -> dict:
        out = {}
        for a in ARMS:
            out[f"{a}_arm_joint_state"] = [float(v) for v in q[a]]
            out[f"{a}_ee_joint_state"] = [float(g[a])]
        return out

    def _plan(self, state: dict) -> list[tuple[str, list[dict]]]:
        q = {a: np.asarray(state[f"{a}_arm_joint_state"], float).copy() for a in ARMS}
        g = {a: float(np.asarray(state[f"{a}_ee_joint_state"], float).reshape(-1)[0]) for a in ARMS}
        phases = [("hold", [self._act(q, g)] * self.hold)]
        for a in ARMS:
            for j in self.joints:
                for sign, tag in ((1.0, "+"), (-1.0, "-")):
                    acts = []
                    for _ in range(self.n_steps):
                        q[a][j] += sign * self.dq
                        acts.append(self._act(q, g))
                    acts += [self._act(q, g)] * self.hold
                    phases.append((f"{a} j{j} {tag}{self.dq * self.n_steps:.2f} rad", acts))
        for a in ARMS:
            g0 = g[a]
            for tag, val in (("toggle", 0.0 if g0 > 0.5 else 1.0), ("back", g0)):
                g[a] = val
                phases.append((f"{a} grip {tag} {val:.1f}", [self._act(q, g)] * self.grip_steps))
        return phases

    def next(self, obs: dict) -> dict:
        if self.phases is None:
            self.phases = self._plan(obs["state"])
        if self.i >= len(self.phases):
            return {"done": True}
        name, acts = self.phases[self.i]
        self.i += 1
        self.report["phases"].append({"name": name, "n": len(acts), "step0": int(obs["steps_used"])})
        return {"actions": acts}

    def episode_report(self) -> dict:
        return self.report


__all__ = ["JointStepsSmoke"]
