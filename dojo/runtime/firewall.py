"""Observation firewall: what may reach the runtime policy.

The policy sees only the official observation contract (research/robodojo-teardown/robodojo-task-anatomy.md
§5): three RGB frames 480x640, the six proprio fields (+ optionally RoboDojo's own gripper value per arm under
``{arm}_ee_joint_state_official``, when the gripper field carries the command: strict v2), the instruction, and the step counters the reference
adapter also reads (``steps_used`` / ``step_lim``). The proxy render service adds a ``renderer`` label
(a string naming the renderer, no state). Everything else is refused:

* ``check_obs`` raises ``FirewallError`` on any other key at any level (fail closed). In mode ``gt``
  (dev-only upper bound) exactly one extra key is allowed, ``DEV_GT_KEY``, and it is split off so the
  rest of the policy handles the same clean observation in both modes.
* ``sanitize_info`` keeps only whitelisted episode-info keys. simlab's proxy_eval passes
  ``layout_file`` (our dev record = ground truth) to every policy; it is dropped here and never read.

Episode info keeps only what the official observation also carries (``instruction``, ``step_lim``). The
launcher's task name, the layout id, the reset's stability flag and timings, and any value derived from them
(the old ``dev_calib_fold`` = crc32(layout id) % 2) never reach the policy: the drivers log them on the scorer
side. The perception calibration (backgrounds, colour map) is ONE frozen set per perception service
(``DOJO_CALIB_SET``), the same for every episode (legality audit 2026-10-28, P1 finding 1).
"""

from __future__ import annotations

import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = {"left_arm_joint_state": 6, "right_arm_joint_state": 6, "left_ee_pose": 7, "right_ee_pose": 7,
              "left_ee_joint_state": 1, "right_ee_joint_state": 1}
# Optional per arm: RoboDojo's official ``{arm}_ee_joint_state`` (the APPLIED gripper control) when the path carries it
# under its own key next to the command: robot-harness's strict server v2 (``eef.<arm>.ee_joint_state_official``,
# D-C11) and the dev harnesses' ``DOJO_GRIP_SOURCE=v2`` (``dojo.simlab.grip_source``). Part of RoboDojo's own
# observation, never a simulator reading beyond it. Absent on strict v1 / ``command`` / ``robodojo``.
OPTIONAL_STATE_KEYS = {"left_ee_joint_state_official": 1, "right_ee_joint_state_official": 1}
OBS_KEYS = {"images", "state", "instruction", "steps_used", "step_lim"}
PROXY_META_KEYS = {"renderer"}
DEV_GT_KEY = "_dev_gt"
INFO_KEYS = {"instruction", "step_lim"}
MODES = ("gt", "perception")


class FirewallError(RuntimeError):
    """Something other than the official observation tried to reach the policy."""


def check_obs(obs: dict, mode: str) -> tuple[dict, list | None]:
    """Validate one observation; returns (clean obs, dev GT rows or None).

    Mode ``perception``: any key outside the contract raises. Mode ``gt``: ``DEV_GT_KEY`` is also
    accepted (and required), returned separately."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}")
    extra = set(obs) - OBS_KEYS - PROXY_META_KEYS
    gt = None
    if mode == "gt":
        if DEV_GT_KEY not in obs:
            raise FirewallError("mode gt needs the driver's dev GT rows in the observation")
        gt = obs[DEV_GT_KEY]
        extra.discard(DEV_GT_KEY)
    if extra:
        raise FirewallError(f"non-contract observation keys: {sorted(extra)}")
    missing = OBS_KEYS - set(obs)
    if missing:
        raise FirewallError(f"observation lacks {sorted(missing)}")
    imgs = obs["images"]
    if set(imgs) - set(CAMS):
        raise FirewallError(f"non-contract cameras: {sorted(set(imgs) - set(CAMS))}")
    for k, v in imgs.items():
        a = np.asarray(v)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise FirewallError(f"{k}: expected HxWx3 uint8 RGB, got {a.dtype} {a.shape}")
    st = obs["state"]
    if not set(STATE_KEYS) <= set(st) or set(st) - set(STATE_KEYS) - set(OPTIONAL_STATE_KEYS):
        raise FirewallError(f"state keys {sorted(st)} != contract {sorted(STATE_KEYS)} (+ optional "
                            f"{sorted(OPTIONAL_STATE_KEYS)})")
    clean_state = {}
    for k, n in {**STATE_KEYS, **{k: v for k, v in OPTIONAL_STATE_KEYS.items() if k in st}}.items():
        a = np.asarray(st[k], dtype=float).reshape(-1)
        if a.size != n:
            raise FirewallError(f"state {k}: {a.size} values, expected {n}")
        clean_state[k] = a
    if not isinstance(obs["instruction"], str):
        raise FirewallError("instruction must be a string")
    clean = {"images": {k: np.asarray(v) for k, v in imgs.items()}, "state": clean_state,
             "instruction": obs["instruction"], "steps_used": int(obs["steps_used"]), "step_lim": int(obs["step_lim"])}
    return clean, gt


def sanitize_info(info: dict) -> tuple[dict, list[str]]:
    """Whitelisted episode info + the names of the dropped keys (logged, never read)."""
    keep = {k: v for k, v in info.items() if k in INFO_KEYS}
    return keep, sorted(set(info) - INFO_KEYS)


__all__ = ["CAMS", "STATE_KEYS", "OPTIONAL_STATE_KEYS", "OBS_KEYS", "DEV_GT_KEY", "INFO_KEYS", "FirewallError", "check_obs",
           "sanitize_info", "MODES"]
