"""What a policy reads as each arm's gripper: the strict contract's command, or RoboDojo's observation (``DOJO_GRIP_SOURCE``).

RoboDojo's ``{arm}_ee_joint_state`` is the APPLIED gripper control, not the command. RoboDojo's
``control_manager.process_gripper_val`` moves the MEASURED finger value at most 20 % of the range toward the target per
step. So while the jaws are blocked by an object, the observation reads about the measured opening minus 0.2: 0.30-0.35
on a stack block whose command is 0.0. The strict server on the leaderboard path (robot-harness, tier B: no finger
measurement) sends the pure command instead, its ``grip_cmd`` (the shim's ``hold_g``, rounded to 1e-4).

``DOJO_GRIP_SOURCE`` (``DOJO_GRIP_OBS`` is its deprecated alias, read only when ``DOJO_GRIP_SOURCE`` is unset):
- ``command`` (default since 2026-10-27, journal 2026-10-27-dojo-official §1.5): the strict contract's value. That is
  the last commanded gripper, rounded as the strict server reports it. Before the first command it is the step-0
  observation, as the shim initialises ``hold_g``.
- ``robodojo``: the observation exactly as RoboDojo gives it (the default until 2026-10-27). The strict server v2 is
  to send this value too, under its own key (``ee_joint_state_official``), never in place of the command.

The names avoid robot-harness's ``eef.<arm>.grip_obs``, which its strict v1 server already uses for the command.

Used where a dev harness hands an observation to the policy: ``ProxyEpisode`` / ``RenderedEpisode``
(``dojo.runtime.eval`` = integ_eval, ``proxy_eval``) and the skills' ``RunnerEnv`` (G3). Physics never changes: only
what the policy reads does. Journal 2026-10-27-dojo-official §1.4-1.5.
"""

from __future__ import annotations

import os

import numpy as np

MODES = ("robodojo", "command")
ARMS = ("left", "right")
ENV = "DOJO_GRIP_SOURCE"
ALIAS = "DOJO_GRIP_OBS"          # deprecated name (PR #104); remove after 2026-10-28
DEFAULT = "command"


def grip_source_mode(value: str | None = None) -> str:
    if value is None:
        value = os.environ.get(ENV) or os.environ.get(ALIAS) or DEFAULT
    m = value.strip().lower() or DEFAULT
    if m not in MODES:
        raise ValueError(f"{ENV}={m!r}: one of {MODES}")
    return m


def _key(arm: str) -> str:
    return f"{arm}_ee_joint_state"


class GripSource:
    """Per episode: ``reset()``, then ``stepped(action)`` for every env step and ``apply(state)`` for every observation."""

    def __init__(self, mode: str | None = None):
        self.mode = grip_source_mode(mode)
        self.cmd: dict[str, float] | None = None

    def reset(self) -> None:
        self.cmd = None

    def stepped(self, action: dict) -> None:
        if self.cmd is None:
            self.cmd = {}
        for a in ARMS:
            if _key(a) in action:
                self.cmd[a] = float(np.clip(np.asarray(action[_key(a)], float).reshape(-1)[0], 0.0, 1.0))

    def apply(self, state: dict) -> dict:
        if self.cmd is None:                    # the step-0 observation: the shim starts its hold_g from it
            self.cmd = {a: float(np.asarray(state[_key(a)], float).reshape(-1)[0]) for a in ARMS if _key(a) in state}
        for a in ARMS:                          # an arm no action has commanded yet keeps the first value it was seen at
            if a not in self.cmd and _key(a) in state:
                self.cmd[a] = float(np.asarray(state[_key(a)], float).reshape(-1)[0])
        if self.mode == "robodojo":
            return state
        out = dict(state)
        for a in ARMS:
            if _key(a) in out and a in self.cmd:
                out[_key(a)] = np.array([round(self.cmd[a], 4)])
        return out


__all__ = ["GripSource", "grip_source_mode", "MODES", "ENV", "ALIAS", "DEFAULT"]
