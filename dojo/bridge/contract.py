"""Strict mode: what of a robot-server reply may reach our runtime.

The strict server (robot-harness ``harness/robodojo_server_strict.py --strict``) already strips simulator readings
from its shim replies. This module is OUR side of the same rule, fail-closed, so a misconfigured server (legacy
tier B, tier A, a future verb that leaks) cannot feed the runtime anything outside the official observation contract:

* **Whitelist.** ``contract_obs`` builds the runtime's observation (``dojo.runtime.firewall`` contract) from exactly:
  the three RGB frames (the PNGs the server wrote for this obs), per arm the MEASURED joints ``eef.<arm>.q``, the
  MEASURED flange pose ``eef.<arm>.flange`` and the COMMANDED gripper ``eef.<arm>.grip_cmd``, the instruction, and
  the step counters (plus, from a strict v2 server, RoboDojo's official gripper value ``eef.<arm>.ee_joint_state_official``
  under the runtime key ``<arm>_ee_joint_state_official``; robot-harness D-C11) (``step``; ``step_lim`` from the reset reply's ``step_budget``). Every other reply field (TCP,
  ``grip_obs``, ``home_dpos_max_m``, ``at_home``, ``held_look``, image paths, residuals, hints) stays in the backend.
  Camera calibration is never taken from a reply: our perception computes K / T from the public camera config and
  the measured joints (``dojo.perception.cameras.cameras_from_state``), as the server's ``strict_calib.py`` does.
* **Banned keys.** ``scan`` finds keys that only a non-strict or leaking server sends (the strict server's own
  BANNED list plus dev / grader keys) anywhere in a reply; ``check_reply`` raises ``ContractViolation`` (a
  ``dojo.runtime.firewall.FirewallError``) on any hit. Nothing is stripped and carried on: a leak stops the episode
  before the runtime sees that reply.
* **Server tier.** ``check_tier`` refuses a server whose ``info`` is not strict tier B.

``ended`` (RoboDojo's natural termination: success latch or step limit) is read by the driver only, to stop calling
the policy, exactly as the proxy / official harness stops calling ``next``; it never enters an observation.
"""

from __future__ import annotations

import numpy as np

from dojo.runtime.firewall import CAMS, FirewallError

# the strict server's BANNED list (robodojo_server_strict.py) + what a dev / privileged / grader reply carries
# (a superset of robot-harness J1's IDENTITY_KEYS, robodojo_server_j1.py)
BANNED = ("objects", "calib", "finger_q", "gap_m", "grasp_detected", "fingers_stopped", "blocked", "grasps",
          "objects_dev", "objects_dev_pos", "final_objects_dev", "success", "success_dev", "reward", "score",
          "layout_id", "layout", "seed", "task", "task_name", "env_seeds", "marked_failed")
ARMS = ("left", "right")


class ContractViolation(FirewallError):
    """A server reply carried something outside the strict observation contract."""


def scan(x, path: str = "", banned=BANNED) -> list[str]:
    hits = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}" if path else str(k)
            if k in banned:
                hits.append(p)
            else:
                hits += scan(v, p, banned)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            hits += scan(v, f"{path}[{i}]", banned)
    return hits


def check_reply(verb: str, reply: dict, banned=BANNED) -> dict:
    hits = scan(reply, "", banned)
    if hits:
        raise ContractViolation(f"{verb} reply carries non-contract fields {hits[:10]} (strict mode: refused)")
    return reply


def check_tier(info: dict) -> str:
    """The server must be strict tier B (no finger measurement, computed calibration, no privileged poses)."""
    tier = str(info.get("tier") or "")
    st = info.get("strict") or {}
    if not tier.startswith("B-strict") or st.get("finger_measurement") is not False:
        raise ContractViolation(f"server tier {tier!r} is not strict tier B (strict={st}); start it with "
                                "robot-start-strict / robot-start-j1, never --privileged")
    return tier


def state_from_eef(eef: dict) -> dict:
    """The runtime's six proprio fields from a reply's ``eef`` (measured joints + flange, commanded gripper), plus
    ``<arm>_ee_joint_state_official`` when a strict v2 server sends RoboDojo's own value (its first element; a v1 server
    or a ``null`` adds nothing). The runtime reads it only behind ``RuntimeConfig.grip_check``."""
    st = {}
    for arm in ARMS:
        e = eef.get(arm)
        if e is None:
            raise ContractViolation(f"reply has no eef.{arm}")
        q = np.asarray(e.get("q"), float).reshape(-1)
        fl = e.get("flange") or {}
        pose = np.asarray(list(fl.get("pos") or []) + list(fl.get("quat") or []), float).reshape(-1)
        g = e.get("grip_cmd")
        if q.size != 6 or pose.size != 7 or g is None:
            raise ContractViolation(f"eef.{arm}: need q[6], flange pos+quat[7], grip_cmd (got {q.size}, {pose.size}, {g})")
        st[f"{arm}_arm_joint_state"] = q
        st[f"{arm}_ee_pose"] = pose
        st[f"{arm}_ee_joint_state"] = np.array([float(g)])
        off = e.get("ee_joint_state_official")
        if off is not None:
            v = np.asarray(off, float).reshape(-1)
            if v.size < 1 or not np.all(np.isfinite(v)):
                raise ContractViolation(f"eef.{arm}.ee_joint_state_official: {off!r}")
            st[f"{arm}_ee_joint_state_official"] = v[:1].copy()
    return st


def contract_obs(reply: dict, images: dict, step_lim: int, instruction: str | None = None) -> dict:
    """The runtime observation from an ``obs`` reply and the frames decoded from its PNGs."""
    for c in images:
        if c not in CAMS:
            raise ContractViolation(f"non-contract camera {c!r}")
    instr = reply.get("instruction") if reply.get("instruction") is not None else instruction
    if not isinstance(instr, str):
        raise ContractViolation("obs reply has no instruction")
    return {"images": {c: np.asarray(images[c], np.uint8) for c in CAMS if c in images},
            "state": state_from_eef(reply.get("eef") or {}), "instruction": instr,
            "steps_used": int(reply.get("step", 0)), "step_lim": int(step_lim)}


__all__ = ["BANNED", "ContractViolation", "scan", "check_reply", "check_tier", "state_from_eef", "contract_obs"]
