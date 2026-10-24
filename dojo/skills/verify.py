"""Plan + verify: candidate segments are rolled in a twin before one is executed.

``TwinBackend`` is the seam: ``rollout(candidates, k, held_labels, src) -> list[Outcome]``. The skills
(candidate generation, judges, selection: ``Skills.verify_choose``) do not care where the twin lives:

* ``InProcessTwin`` (this module, DEV-ONLY, G3): env 0 of the same RoboDojo process runs the
  episode, envs 1..K are twin copies seeded with env 0's TRUE internal state;
* production: a separate twin process loaded from OUR estimate (perception's layout record +
  proprio) via simlab's ``load_from_estimate`` (integration's ``TwinClient``); it never sees the
  eval env.

Judges only read an ``Outcome`` (object poses and robot joints simulated by the twin), which both
backends provide.

InProcessTwin details: env 0's full state

Before a segment runs on env 0, env 0's full state (robot joints, velocities and targets, every
object's pose and velocity, the controller's last command, RoboDojo's reward latches and step
counter) is broadcast into the twin envs, optionally with object pose noise, and the candidate
segments are rolled there in one batch (``take_action_batch`` on the twin envs only). Env 0 is
not stepped while the twins run; its robot holds its last command (the eval env is paused between
actions as well). The first candidate whose every sample passes the skill's judge is executed on
env 0 exactly as verified. This copies privileged simulator state and is a dev tool only.

The state copy follows simlab's ``Twin.snapshot/restore`` (dojo/simlab/twin.py), per env index;
restored rollouts are close to, not bitwise equal to, a straight run (PhysX contact caches are not
part of the state; simlab measured 0.4-0.6 mm over 264 contact steps).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dojo.skills.kinematics import ARMS

REWARD_KEYS = ("check_list", "final_check_list", "query_list", "trigger_check_list", "trigger_query_list",
               "score_list", "score_achieved", "score_completed_count", "score_meta", "score_trigger_meta",
               "final_score_list", "final_score_achieved", "final_score_completed_count", "final_score_meta")
PARSER_KEYS = ("pre_state", "robot_origin_endpose", "joint_ratio_transition_state")


@dataclass
class Outcome:
    candidate: int
    sample: int
    objects: dict                      # label -> (pos, quat)
    q: dict                            # arm -> measured joints
    success: bool
    failed: bool                       # a query (e.g. extra press, lifted T) failed the episode
    score: float
    ended: bool
    extra: dict = field(default_factory=dict)


class TwinBackend(Protocol):
    """Where candidate segments are simulated. ``candidates``: one list of official action dicts per
    candidate; ``k`` samples each (sample 0 nominal, the rest with pose noise); ``held_labels``: objects
    in the grippers (not perturbed); ``src``: which state to start from (backend-specific)."""

    def rollout(self, candidates: list[list[dict]], k: int = 2, held_labels: tuple = (), src: int = 0) -> list[Outcome]:
        ...


class InProcessTwin:
    """DEV-ONLY TwinBackend: twin envs 1..K in the episode's own process, seeded with env 0's TRUE state."""

    def __init__(self, runner, twin_envs: list[int], perturb_mm: float = 1.5, perturb_deg: float = 1.0,
                 seed: int = 0, log=None):
        self.r = runner
        self.env = runner.env
        self.twins = list(twin_envs)
        self.perturb_mm = perturb_mm
        self.perturb_deg = perturb_deg
        self.rng = np.random.default_rng(seed)
        self.log = log or (lambda *a, **k: None)
        self.stats = {"calls": 0, "rollouts": 0, "steps": 0, "wall_s": 0.0}

    # ------------------------------------------------------------------ state copy
    def _objects(self, env_idx: int):
        lm = self.env.scene_manager.layout_manager
        for otype in ("Rigid", "Dynamic", "Articulation"):
            for rec in lm.object_records_by_type[otype].layout_records_by_env[env_idx]:
                obj = lm.get_scene_object(env_idx, rec["inst_name"])
                if obj is not None:
                    yield otype, rec["inst_name"], rec.get("label"), obj

    def snapshot(self, src: int = 0) -> dict:
        from dojo.simlab.runner import _np
        env, rm = self.env, self.env.robot_manager
        robots = [{"q": k.data.joint_pos[src].clone(), "qd": k.data.joint_vel[src].clone(),
                   "target": k.data.joint_pos_target[src].clone()} for k in rm.robot_key]
        objs = {}
        for otype, name, label, obj in self._objects(src):
            t, q = obj.get_local_pose()
            rec = {"type": otype, "label": label, "pos": _np(t).copy(), "quat": _np(q).copy()}
            try:
                rec["v"] = _np(obj.get_linear_velocity()).copy()
                rec["w"] = _np(obj.get_angular_velocity()).copy()
            except Exception:  # noqa: BLE001
                pass
            if otype == "Articulation":
                try:
                    rec["joints"] = _np(obj.get_current_joint_positions()).copy()
                except Exception:  # noqa: BLE001
                    pass
            objs[name] = rec
        rwm = env.reward_manager
        return {"robots": robots, "objects": objs,
                "prev_control": copy.deepcopy(rm.control_manager.prev_control[src]),
                "counters": (int(env.take_action_cnt[src]), bool(env.success[src]), bool(env.end_flag[src])),
                "reward": {k: copy.deepcopy(getattr(rwm, k)[src]) for k in REWARD_KEYS},
                "parser": {k: copy.deepcopy(getattr(rwm.func_parser, k)[src]) for k in PARSER_KEYS}}

    def broadcast(self, snap: dict, envs: list[int], perturb: list[bool] | None = None,
                  held_labels: tuple = ()) -> None:
        import torch
        env, rm = self.env, self.env.robot_manager
        for key, r in zip(rm.robot_key, snap["robots"]):
            eid = torch.tensor(envs, dtype=torch.long, device=key.data.joint_pos.device)
            n = len(envs)
            key.write_joint_state_to_sim(r["q"].unsqueeze(0).repeat(n, 1), r["qd"].unsqueeze(0).repeat(n, 1), env_ids=eid)
            key.set_joint_position_target(r["target"].unsqueeze(0).repeat(n, 1), env_ids=eid)
        for j, e in enumerate(envs):
            live = {name: obj for _, name, _, obj in self._objects(e)}
            noisy = bool(perturb and perturb[j])
            for name, rec in snap["objects"].items():
                obj = live.get(name)
                if obj is None:
                    continue
                pos, quat = rec["pos"].copy(), rec["quat"].copy()
                if noisy and rec["type"] == "Rigid" and rec.get("label") and rec["label"] not in held_labels:
                    d = self.rng.normal(0, self.perturb_mm / 1000.0, 2)
                    pos[:2] += d
                    a = np.radians(self.rng.normal(0, self.perturb_deg)) / 2
                    dq = np.array([np.cos(a), 0, 0, np.sin(a)])
                    w0, x0, y0, z0 = dq
                    w1, x1, y1, z1 = quat
                    quat = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
                obj.set_local_pose(translation=pos, orientation=quat)
                if "v" in rec:
                    obj.set_linear_velocity(torch.tensor(rec["v"], dtype=torch.float32))
                    obj.set_angular_velocity(torch.tensor(rec["w"], dtype=torch.float32))
                if "joints" in rec:
                    try:
                        obj.set_current_joint_positions(rec["joints"])
                    except Exception:  # noqa: BLE001
                        pass
            rm.control_manager.prev_control[e] = copy.deepcopy(snap["prev_control"])
            env.take_action_cnt[e], env.success[e], env.end_flag[e] = snap["counters"]
            rwm = env.reward_manager
            for k, v in snap["reward"].items():
                getattr(rwm, k)[e] = copy.deepcopy(v)
            for k, v in snap["parser"].items():
                getattr(rwm.func_parser, k)[e] = copy.deepcopy(v)
        env.sim.scene.write_data_to_sim()

    # ------------------------------------------------------------------ rollouts
    def rollout(self, candidates: list[list[dict]], k: int = 2, held_labels: tuple = (), src: int = 0) -> list[Outcome]:
        """Each candidate is a list of official action dicts (one per env step). ``k`` samples per
        candidate; sample 0 is nominal, the others carry object pose noise."""
        t0 = time.perf_counter()
        snap = self.snapshot(src)
        jobs = [(ci, s) for ci in range(len(candidates)) for s in range(k)]
        out: list[Outcome] = []
        for b0 in range(0, len(jobs), len(self.twins)):
            batch = jobs[b0:b0 + len(self.twins)]
            envs = self.twins[:len(batch)]
            self.broadcast(snap, envs, [s > 0 for _, s in batch], held_labels)
            n = max(len(candidates[ci]) for ci, _ in batch)
            for t in range(n):
                acts = [candidates[ci][min(t, len(candidates[ci]) - 1)] for ci, _ in batch]
                acts = [{kk: np.asarray(v, np.float32) for kk, v in a.items()} for a in acts]
                self.env.take_action_batch(acts, env_idx_list=envs)
            self.stats["steps"] += n
            for (ci, s), e in zip(batch, envs):
                objs = {o["label"]: (np.asarray(o["pos"], float), np.asarray(o["quat"], float))
                        for o in self.r.objects_gt(e) if o.get("label") and o.get("pos") is not None}
                st = self.r.obs_state(e)
                q = {a: np.asarray(st[f"{a}_arm_joint_state"], float) for a in ARMS}
                rew = self.r.reward(e)
                out.append(Outcome(ci, s, objs, q, success=bool(rew["success"]),
                                   failed=bool(not self.env.success[e] and not rew["ended"]) or
                                   (rew["ended"] and not rew["success"] and rew["steps_used"] < rew["step_lim"]),
                                   score=float(self.r.score(e)), ended=bool(rew["ended"])))
        dt = time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["rollouts"] += len(jobs)
        self.stats["wall_s"] += dt
        return out


TwinVerifier = InProcessTwin       # old name
