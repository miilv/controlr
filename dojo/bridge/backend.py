"""RobotServerBackend: our runtime and skills acting through a robot-harness robot server (strict tier B).

Two faces over one server connection (``client.RobotServerClient``):

* **Observation** (``observe()``): the runtime's contract observation (``dojo.runtime.firewall``) built by
  ``contract.contract_obs`` from the ``obs`` verb: the three RGB frames (PNGs the server writes into our
  ``frames_dir``, decoded here), measured joints and flange poses, commanded grippers, the instruction, steps.
* **Action** (the skills ``Env`` protocol: ``step(action)``, ``proprio()``, ``steps_used``, ``step_lim``; plus
  ``execute(actions)`` for a whole list): our per-step joint action dicts (``{arm}_arm_joint_state`` [6] and
  ``{arm}_ee_joint_state`` [1], both arms every step, from ``dojo.skills.executor``) are buffered and flushed
  through the ``joint_steps`` verb, one RoboDojo ``take_action`` per entry, verbatim (no re-timing, no IK, no
  gripper ramp on the server), so what the twin verified is what runs. ``joint_steps`` is the one verb this needs
  beyond the strict server's frozen set: requested in robot-harness issue #8 (contract in ``docs/BRIDGE-RUNBOOK.md``
  §9); a server without it is refused at ``connect()`` unless ``allow_no_joint_steps`` (observation-only use).
  ``reserve=0`` on every call: the runtime keeps its own fixed reserve for ``finish()`` (``RuntimeConfig.reserve``,
  the same for every episode, never chosen by task).

Plus thin wrappers for the server's own verbs (``gripper``, ``hold``, ``return_home``, ``perceive`` and
``held_look`` on the J1 / strict servers), each flushing the buffer first and passing ``contract.check_reply``. The
client is available as ``robot`` for robot-harness skills (``skill_robodojo.run(robot, **args)``; flushes too).

Strict mode (default): the server's ``info`` must be strict tier B, every reply is scanned for banned keys
(``contract.BANNED``: fail closed), and only whitelisted fields leave this class.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import contract
from .client import RobotServerClient, ServerError

ACTION_KEYS = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")
JOINT_STEPS = "joint_steps"


def read_png(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), np.uint8).copy()


def actions_md5(actions: list[dict]) -> str:
    """robot-harness's ``actions_md5`` of a ``joint_steps`` list (robodojo_server_j2.actions_md5): what the server
    logs in episode.jsonl, so an executed chunk can be matched against what our twin verified."""
    return hashlib.md5(json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def action_json(a: dict) -> dict:
    """One action dict as plain lists of float32-exact numbers (bit-for-bit through JSON: float32 -> double repr
    -> float32 on the server)."""
    if set(a) != set(ACTION_KEYS):
        raise ValueError(f"action keys {sorted(a)} != {sorted(ACTION_KEYS)}")
    out = {}
    for k in ACTION_KEYS:
        v = np.asarray(a[k], np.float32).reshape(-1)
        if v.size != (6 if k.endswith("arm_joint_state") else 1) or not np.all(np.isfinite(v)):
            raise ValueError(f"action {k}: {v.size} values / non-finite")
        out[k] = [float(x) for x in v]
    return out


class RobotServerBackend:
    def __init__(self, client: RobotServerClient, strict: bool = True, frames_dir: str | Path | None = None,
                 keep_every: int = 20, chunk: int = 500, allow_no_joint_steps: bool = False, log=None):
        self.c = client
        self.strict = strict
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.keep_every = int(keep_every)
        self.chunk = int(chunk)
        self.allow_no_joint_steps = allow_no_joint_steps
        self.log = log or (lambda msg: None)
        self.pending: list[dict] = []
        self.executed: list[dict] = []
        self.info_reply: dict = {}
        self.commands: tuple = ()
        self.instruction: str | None = None
        self._step_lim = 0
        self._steps = 0
        self.ended = False
        self._obs: dict | None = None
        self.fresh = False
        self.n_obs = 0
        self.stats = {"obs": 0, "obs_s": 0.0, "png_s": 0.0, "joint_steps_calls": 0, "joint_steps": 0, "exec_s": 0.0,
                      "perceive": 0, "held_look": 0, "frames_kept": 0, "grip_change_while_moving": {},
                      "tracking_max_rad": 0.0}
        self.chunks: list[dict] = []        # per joint_steps call: first step, size, executed, actions_md5
        self.c.on_request = self._before_foreign_verb
        self.c.on_reply = self._facts
        self._internal = False

    # ------------------------------------------------------------------ plumbing
    def _call(self, verb: str, **kw) -> dict:
        self._internal = True
        try:
            rep = self.c.ok(verb, **kw)
        finally:
            self._internal = False
        if self.strict:
            contract.check_reply(verb, rep)
        self._facts(rep)
        return rep

    def _facts(self, rep: dict) -> None:
        if rep.get("step") is not None:
            self._steps = int(rep["step"])
        if rep.get("ended"):
            self.ended = True

    def _before_foreign_verb(self, verb: str) -> None:
        """A robot-harness skill (or anyone holding ``robot``) is about to call the server: our buffered steps
        must run first, and the cached observation is stale after it."""
        if self._internal:
            return
        self.flush()
        self.fresh = False

    @property
    def robot(self) -> RobotServerClient:
        return self.c

    # ------------------------------------------------------------------ session
    def connect(self) -> dict:
        info = self.c.ok("info")              # server metadata, never passed to the runtime (only the tier is checked)
        if self.strict:
            contract.check_tier(info)
        self.info_reply = info
        self.commands = tuple(info.get("commands") or ())
        if JOINT_STEPS not in self.commands and not self.allow_no_joint_steps:
            raise ServerError("info", {"error": f"server has no `{JOINT_STEPS}` verb (commands {list(self.commands)})",
                                       "hint": "our skills emit per-step joint actions: robot-harness issue #8, docs/BRIDGE-RUNBOOK.md §9"})
        return info

    def wait_episode(self, timeout_s: float = 900.0, poll_s: float = 3.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            info = self.c.ok("info")
            if info.get("episode_pending") or info.get("episode_active"):
                return
            time.sleep(poll_s)
        raise TimeoutError(f"no RoboDojo episode within {timeout_s:.0f} s")

    def reset(self, timeout_s: float = 600.0) -> dict:
        """Take the next RoboDojo episode. Returns the episode info the runtime gets (instruction, step_lim only)."""
        rep = self._call("reset", timeout=float(timeout_s))
        self.pending, self.executed, self.ended, self.chunks = [], [], False, []
        self.instruction = rep.get("instruction")
        self._step_lim = int(rep.get("step_budget") or 0)
        self._steps = int(rep.get("step", 0))
        self.n_obs = 0
        self._obs = self._obs_from(rep.get("obs") or {})
        self.fresh = True
        return {"instruction": self.instruction, "step_lim": self._step_lim}

    def close(self) -> dict:
        """Ends the episode (an unfinished one counts as failed) and stops the server; RoboDojo writes its result."""
        self.flush()
        return self.c.call("close")

    # ------------------------------------------------------------------ observation
    def _save_kw(self) -> dict:
        if self.frames_dir is None:
            return {}
        self.frames_dir.mkdir(parents=True, exist_ok=True)      # the server names them s<step>_<cam>.png
        return {"save_dir": str(self.frames_dir)}

    def _obs_from(self, rep: dict) -> dict:
        t0 = time.perf_counter()
        paths = rep.get("images") or {}
        imgs = {c: read_png(p) for c, p in paths.items() if c in contract.CAMS}
        self.stats["png_s"] += time.perf_counter() - t0
        keep = self.n_obs == 0 or (self.keep_every > 0 and self.n_obs % self.keep_every == 0)
        if self.frames_dir is not None and not keep:
            for p in paths.values():                   # our own frames dir: the server wrote them for us
                try:
                    if Path(p).resolve().is_relative_to(self.frames_dir.resolve()):
                        os.unlink(p)
                except OSError:
                    pass
        elif self.frames_dir is not None:
            self.stats["frames_kept"] += 1
        self.n_obs += 1
        return contract.contract_obs(rep, imgs, self._step_lim, self.instruction)

    def observe(self) -> dict:
        """The contract observation after every buffered action ran."""
        self.flush()
        if self.fresh and self._obs is not None:
            return self._obs
        t0 = time.perf_counter()
        rep = self._call("obs", **self._save_kw())
        self._obs = self._obs_from(rep)
        self.fresh = True
        self.stats["obs"] += 1
        self.stats["obs_s"] += time.perf_counter() - t0
        return self._obs

    # ------------------------------------------------------------------ skills Env protocol
    @property
    def step_lim(self) -> int:
        return self._step_lim

    @property
    def steps_used(self) -> int:
        return self._steps + len(self.pending)

    @property
    def steps_left(self) -> int:
        return self._step_lim - self.steps_used

    def step(self, action: dict) -> bool:
        """Buffer one env step; True when it uses the last step of the budget."""
        self.pending.append({k: np.asarray(v, np.float32).copy() for k, v in action.items()})
        self.fresh = False
        return self.steps_used >= self._step_lim

    def proprio(self) -> dict:
        self.flush()
        if self.fresh and self._obs is not None:
            return self._obs["state"]
        rep = self._call("obs", cameras=[])
        return contract.state_from_eef(rep.get("eef") or {})

    def flush(self) -> dict:
        """Run the buffered actions (``joint_steps`` in chunks); stops at the episode's end."""
        acts, self.pending = self.pending, []
        return self._run(acts)

    def execute(self, actions: list[dict]) -> dict:
        """Run a list of action dicts now (the driver: one ``next()`` reply). -> {"steps", "ended"}."""
        self.flush()
        return self._run(actions)

    def _run(self, acts: list[dict]) -> dict:
        if not acts:
            return {"steps": 0, "ended": self.ended}
        if self.ended:
            self.log(f"[bridge] episode ended: {len(acts)} actions not sent")
            return {"steps": 0, "ended": True, "dropped": len(acts)}
        t0, n0 = time.perf_counter(), self._steps
        self.fresh = False
        left = self._step_lim - self._steps
        if len(acts) > left:                    # like the proxy / official harness: steps past the limit never run
            self.log(f"[bridge] {len(acts) - left} actions past the step limit dropped")
            acts = acts[:max(0, left)]
        sent = 0
        for i in range(0, len(acts), self.chunk):
            part = [action_json(a) for a in acts[i:i + self.chunk]]
            md5 = actions_md5(part)
            rep = self._call(JOINT_STEPS, actions=part, reserve=0)
            self.stats["joint_steps_calls"] += 1
            if rep.get("refused"):                    # the whole chunk was refused at 0 env steps
                raise ServerError(JOINT_STEPS, {"error": f"refused: {rep.get('refused')} at entry {rep.get('index')}",
                                                "hint": rep.get("hint"), "refused": rep.get("refused")})
            if rep.get("actions_md5") not in (None, md5):
                raise ServerError(JOINT_STEPS, {"error": f"actions_md5 {rep.get('actions_md5')} != sent {md5}",
                                                "hint": "the server did not receive what we sent"})
            ran = int(rep.get("executed", rep.get("steps", 0)))
            self.executed.extend(acts[i:i + ran])
            sent += ran
            self.chunks.append({"step0": self._steps - int(rep.get("steps", 0)), "n": len(part), "ran": ran, "md5": md5})
            for arm, k in (rep.get("grip_change_while_moving") or {}).items():
                self.stats["grip_change_while_moving"][arm] = self.stats["grip_change_while_moving"].get(arm, 0) + int(k)
            for arm, e in (rep.get("tracking_max_rad") or {}).items():
                self.stats["tracking_max_rad"] = max(self.stats["tracking_max_rad"], float(e))
            if self.ended or ran < len(part):
                break
        self.stats["joint_steps"] += sent
        self.stats["exec_s"] += time.perf_counter() - t0
        return {"steps": self._steps - n0, "ended": self.ended}

    # bookkeeping the runtime's verifier uses (same names as dojo.runtime.bridge.Bridge)
    @property
    def n_executed(self) -> int:
        return len(self.executed) + len(self.pending)

    def actions_since(self, n0: int) -> list[dict]:
        return (self.executed + self.pending)[n0:]

    # ------------------------------------------------------------------ the server's own verbs
    def hold(self, n: int = 1) -> dict:
        self.flush()
        self.fresh = False
        return {"steps": self._call("hold", steps=int(n)).get("steps", 0)}

    def gripper(self, arm: str, g: float, wait_steps: int | None = None) -> dict:
        self.flush()
        self.fresh = False
        kw = {"arm": arm, "value": float(g)}
        if wait_steps is not None:
            kw["wait_steps"] = int(wait_steps)
        rep = self._call("gripper", **kw)
        return {"arm": arm, "grip_cmd": rep.get("grip_cmd"), "steps": rep.get("steps", 0)}

    def return_home(self, arms=("left", "right"), open_grippers: bool = True) -> dict:
        self.flush()
        self.fresh = False
        rep = self._call("return_home", arms=list(arms), open=bool(open_grippers))
        return {"all_home": rep.get("all_home"), "steps": rep.get("steps", 0)}

    def perceive(self, want=None, prior_n: int | None = None) -> dict:
        """J1 server: our B1 adapter on this step's three frames, inside the server (inputs and outputs logged
        there). Returns the validated output: perceived objects, regions, needs_look, last_seen, call number."""
        self.flush()
        kw = {}
        if want:
            kw["want"] = list(want)
        if prior_n is not None:
            kw["prior_n"] = int(prior_n)
        rep = self._call("perceive", **kw)
        self.stats["perceive"] += 1
        return {k: rep.get(k) for k in ("perceived", "regions", "needs_look", "last_seen", "n")}

    def held_look(self, arm: str, stage: str = "") -> dict:
        """Strict held signal: B1 (the adapter's held_look on the wrist image) or the operator. 0 env steps."""
        self.flush()
        rep = self._call("held_look", arm=arm, stage=str(stage)[:80])
        self.stats["held_look"] += 1
        return {"arm": arm, "held": bool(rep.get("held")), "confidence": rep.get("confidence"), "source": rep.get("source")}


__all__ = ["RobotServerBackend", "ACTION_KEYS", "JOINT_STEPS", "action_json", "actions_md5", "read_png"]
