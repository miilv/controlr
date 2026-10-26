"""Twin API: RoboDojo itself as the twin, loaded from a layout record (ours or an inverse layout).

    tw = Twin("stack_blocks", num_envs=4)
    tw.load(layouts)                    # one record per env (or one for all)
    s = tw.snapshot()                   # objects, robot joints + targets, reward latches, counters
    tw.step(actions)                    # one RoboDojo env step per env (official action dicts)
    imgs = tw.render(["cam_head"])      # official RGB frames
    tw.reward(); tw.score()             # the task's own checker
    tw.restore(s)                       # back to the snapshot (no reload)

``restore`` writes poses/velocities/joint states straight into PhysX; contact warm-start caches
are not part of the snapshot, so a restored rollout is close to, not bitwise equal to, the
original (``bench`` measures the gap). ``load`` is the exact path (full RoboDojo reset).

Bench:  scripts/simlab/rdpy.sh -m dojo.simlab.twin --task stack_blocks --num-envs 4 [--physics-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.runner import RoboDojoRunner, _np, jsonable

ARMS = ("left", "right")


HELD_WINDOW = 10          # env steps (0.4 s) used to decide which object moves with which TCP


def held_objects(hist, grip_cmd: dict, blocked: dict, near_m: float = 0.06, rigid_mm: float = 5.0) -> dict:
    """Per arm, the label of the object held at the end of a rollout, else None. Held = the gripper is
    commanded closed (< 0.5) or blocked, the object is within ``near_m`` of the TCP, and over the last
    window it moved rigidly with the TCP (displacement equal within ``rigid_mm``). When the TCP did not
    move in the window, the closest object within ``near_m`` counts."""
    out = {}
    if not hist:
        return {arm: None for arm in grip_cmd}
    (tcp0, obj0), (tcp1, obj1) = hist[0], hist[-1]
    for arm in grip_cmd:
        if not (grip_cmd[arm] < 0.5 or blocked.get(arm)):
            out[arm] = None
            continue
        best, best_d = None, near_m
        d_tcp = tcp1[arm] - tcp0[arm]
        for lab, p1 in obj1.items():
            dist = float(np.linalg.norm(p1 - tcp1[arm]))
            if dist > best_d:
                continue
            d_obj = p1 - obj0.get(lab, p1)
            if np.linalg.norm(d_tcp) > 0.005 and np.linalg.norm(d_obj - d_tcp) * 1000 > rigid_mm:
                continue
            best, best_d = lab, dist
        out[arm] = best
    return out



def _safe_copy(v):
    """deepcopy of env / checker state; a member that cannot be copied (a checker keeps a USD prim / stage handle for
    some tasks, e.g. stack_bowls: "Pickling of pxr.Usd.Stage instances is not enabled") is kept by reference, a
    handle the snapshot only reads back."""
    try:
        return copy.deepcopy(v)
    except Exception:  # noqa: BLE001
        if isinstance(v, dict):
            return {k: _safe_copy(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_safe_copy(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_safe_copy(x) for x in v)
        return v

class IncompleteWorld(ValueError):
    """An estimated world lacks objects the task's reward refers to (perception missed them)."""


def missing_labels(layout: dict, required: set[str]) -> list[str]:
    from dojo.simlab.layout import iter_objects
    present = {inst.get("label") for _, _, inst in iter_objects(layout)}
    return sorted(required - present)


class Twin:
    def __init__(self, task: str, num_envs: int = 1, device_id: int = 0, workdir: str | None = None,
                 physics_only: bool = False, need_planner: bool = True):
        self.run = RoboDojoRunner(task, num_envs=num_envs, device_id=device_id,
                                  workdir=workdir or str(paths.DATA / "simlab" / "work" / f"twin_{task}_{num_envs}"),
                                  physics_only=physics_only, need_planner=need_planner)
        self.physics_only = physics_only
        self.task = task
        self.env = self.run.env
        self.num_envs = self.run.num_envs
        from dojo.simlab.gen_layouts import required_labels
        self.required = required_labels(task)

    # ------------------------------------------------------------------ episode
    def load(self, layouts, fresh: bool = True) -> dict:
        return self.run.load(layouts, fresh=fresh)

    def step(self, actions) -> list[bool]:
        return self.run.step(actions)

    def render(self, cams: list[str] | None = None, env_idx_list: list[int] | None = None) -> list[dict]:
        """RGB frames (all three cameras are captured by RoboDojo's tiled capture; ``cams`` filters)."""
        out = []
        for o in self.run.obs_batch(env_idx_list):
            out.append({c: im for c, im in o["images"].items() if cams is None or c in cams})
        return out

    def reward(self, env_idx: int = 0) -> dict:
        return self.run.reward(env_idx)

    def score(self, env_idx: int = 0) -> float:
        return self.run.score(env_idx)

    def obs(self, env_idx: int = 0) -> dict:
        return self.run.obs(env_idx)

    def gt(self, env_idx: int = 0) -> dict:
        return self.run.gt(env_idx)

    # ------------------------------------------------------------------ state
    def _objects(self, env_idx: int):
        lm = self.env.scene_manager.layout_manager
        for otype in ("Rigid", "Dynamic", "Articulation"):
            for rec in lm.object_records_by_type[otype].layout_records_by_env[env_idx]:
                obj = lm.get_scene_object(env_idx, rec["inst_name"])
                if obj is not None:
                    yield otype, rec["inst_name"], obj

    def snapshot(self, env_idx_list: list[int] | None = None) -> dict:
        env = self.env
        ids = list(range(self.num_envs)) if env_idx_list is None else list(env_idx_list)
        snap: dict = {"envs": {}}
        rm = env.robot_manager
        snap["robot"] = []
        for key in rm.robot_key:
            snap["robot"].append({"q": key.data.joint_pos.clone(), "qd": key.data.joint_vel.clone(),
                                  "target": key.data.joint_pos_target.clone()})
        for i in ids:
            objs = {}
            for otype, name, obj in self._objects(i):
                t, q = obj.get_local_pose()
                rec = {"type": otype, "pos": _np(t).copy(), "quat": _np(q).copy()}
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
            fp = rwm.func_parser
            snap["envs"][i] = {
                "objects": objs, "keys": self._name_keys(i),
                "prev_control": _safe_copy(rm.control_manager.prev_control[i]),
                "counters": {"take_action_cnt": int(env.take_action_cnt[i]), "success": bool(env.success[i]),
                             "end_flag": bool(env.end_flag[i])},
                "reward": {k: _safe_copy(getattr(rwm, k)[i]) for k in (
                    "check_list", "final_check_list", "query_list", "trigger_check_list", "trigger_query_list",
                    "score_list", "score_achieved", "score_completed_count", "score_meta", "score_trigger_meta",
                    "final_score_list", "final_score_achieved", "final_score_completed_count", "final_score_meta")},
                "func_parser": {k: _safe_copy(getattr(fp, k)[i]) for k in (
                    "pre_state", "robot_origin_endpose", "joint_ratio_transition_state")},
            }
        snap["gated_score"] = _safe_copy(env.reward_manager._gated_score_lst)
        return snap

    def _name_keys(self, env_idx: int) -> dict:
        """inst_name -> (type, category, index, label, clutter): RoboDojo's instance names carry a global
        counter and differ between envs, the keys do not."""
        lm = self.env.scene_manager.layout_manager
        out = {}
        for otype in ("Rigid", "Dynamic", "Articulation", "Geometry", "Garment", "Fluid"):
            for rec in lm.object_records_by_type[otype].layout_records_by_env[env_idx]:
                out[rec["inst_name"]] = self._object_key(rec, otype, lm.decode_inst_name(rec["inst_name"])[0])
        return out

    def broadcast_snapshot(self, env_idx: int) -> dict:
        """Snapshot of ONE env, restorable into every env (``restore`` maps objects and reward state by
        object key). Used for chaining verification through a grasp."""
        s = self.snapshot([env_idx])
        s["broadcast_src"] = env_idx
        return s

    def _restore_broadcast(self, snap: dict) -> None:
        import torch
        env, rm = self.env, self.env.robot_manager
        src = int(snap["broadcast_src"])
        e = snap["envs"][src]
        n = self.num_envs
        eid = None
        for key, r in zip(rm.robot_key, snap["robot"]):
            eid = torch.arange(n, dtype=torch.long, device=key.data.joint_pos.device)
            key.write_joint_state_to_sim(r["q"][src].repeat(n, 1), r["qd"][src].repeat(n, 1), env_ids=eid)
            key.set_joint_position_target(r["target"][src].repeat(n, 1), env_ids=eid)
        for i in range(n):
            dst = {}
            for name, k in self._name_keys(i).items():
                dst.setdefault(k, []).append(name)
            rename = {}
            for name_src, k in e["keys"].items():
                if dst.get(k):
                    rename[name_src] = dst[k].pop(0)
            live = {name: obj for _, name, obj in self._objects(i)}
            for name_src, rec in e["objects"].items():
                obj = live.get(rename.get(name_src))
                if obj is None:
                    continue
                obj.set_local_pose(translation=rec["pos"], orientation=rec["quat"])
                if "v" in rec:
                    obj.set_linear_velocity(torch.tensor(rec["v"], dtype=torch.float32))
                    obj.set_angular_velocity(torch.tensor(rec["w"], dtype=torch.float32))
                if "joints" in rec:
                    try:
                        obj.set_current_joint_positions(rec["joints"])
                    except Exception:  # noqa: BLE001
                        pass
            rm.control_manager.prev_control[i] = _safe_copy(e["prev_control"])
            c = e["counters"]
            env.take_action_cnt[i], env.success[i], env.end_flag[i] = c["take_action_cnt"], c["success"], c["end_flag"]
            rwm = env.reward_manager
            for k2, v in e["reward"].items():
                getattr(rwm, k2)[i] = _safe_copy(v)
            for k2, v in e["func_parser"].items():           # keyed by inst_name where it is per object
                v = _safe_copy(v)
                if isinstance(v, dict):
                    v = {rename.get(kk, kk): vv for kk, vv in v.items()}
                getattr(rwm.func_parser, k2)[i] = v
        gs = snap.get("gated_score")
        env.reward_manager._gated_score_lst = None if gs is None else [_safe_copy(gs[src]) for _ in range(n)]
        env.sim.scene.write_data_to_sim()

    def restore(self, snap: dict, settle_steps: int = 0) -> None:
        """Back to ``snap``: env by env, or (``broadcast_snapshot``) one env's state into every env."""
        if snap.get("broadcast_src") is not None:
            self._restore_broadcast(snap)
            for _ in range(settle_steps):
                self.env.sim_step(render=False)
            return
        import torch
        env = self.env
        rm = env.robot_manager
        ids = sorted(snap["envs"])
        for key, r in zip(rm.robot_key, snap["robot"]):
            eid = torch.tensor(ids, dtype=torch.long, device=key.data.joint_pos.device)
            key.write_joint_state_to_sim(r["q"][eid], r["qd"][eid], env_ids=eid)
            key.set_joint_position_target(r["target"][eid], env_ids=eid)
        for i in ids:
            e = snap["envs"][i]
            live = {name: obj for _, name, obj in self._objects(i)}
            for name, rec in e["objects"].items():
                obj = live.get(name)
                if obj is None:
                    continue
                obj.set_local_pose(translation=rec["pos"], orientation=rec["quat"])
                if "v" in rec:
                    obj.set_linear_velocity(torch.tensor(rec["v"], dtype=torch.float32))
                    obj.set_angular_velocity(torch.tensor(rec["w"], dtype=torch.float32))
                if "joints" in rec:
                    try:
                        obj.set_current_joint_positions(rec["joints"])
                    except Exception:  # noqa: BLE001
                        pass
            rm.control_manager.prev_control[i] = _safe_copy(e["prev_control"])
            c = e["counters"]
            env.take_action_cnt[i], env.success[i], env.end_flag[i] = c["take_action_cnt"], c["success"], c["end_flag"]
            rwm = env.reward_manager
            for k, v in e["reward"].items():
                getattr(rwm, k)[i] = _safe_copy(v)
            for k, v in e["func_parser"].items():
                getattr(rwm.func_parser, k)[i] = _safe_copy(v)
        env.reward_manager._gated_score_lst = _safe_copy(snap["gated_score"])
        env.sim.scene.write_data_to_sim()
        for _ in range(settle_steps):
            env.sim_step(render=False)

    # ------------------------------------------------------------------ world states from estimates
    def _object_key(self, rec: dict, otype: str, cat: str) -> tuple:
        return (otype, cat, int(rec.get("category_idx", -1)), rec.get("label"), rec.get("type") == "cluttered")

    def _layout_signature(self, layout: dict) -> list:
        from dojo.simlab.layout import iter_objects
        return sorted(self._object_key(inst, t, cat) for t, cat, inst in iter_objects(layout))

    def set_robot(self, env_idx: int, robot: dict) -> None:
        """Put both arms at ``{left_q, right_q, left_grip, right_grip}`` (joint state, drive targets and
        RoboDojo's control-manager targets), velocities zero."""
        import torch
        rm = self.env.robot_manager
        for robot_obj, key in zip(rm.robot_list, rm.robot_key):
            if robot_obj.type != "target":
                continue
            arm = robot_obj.arm_name.split("_")[0]
            q = key.data.joint_pos[env_idx].clone()
            if robot.get(f"{arm}_q") is not None:
                q[list(robot_obj.arm_joint_indices)] = torch.tensor(robot[f"{arm}_q"], dtype=q.dtype)
            g = robot.get(f"{arm}_grip")
            if g is not None and robot_obj.ee_type == "gripper":
                sc, mv = robot_obj.gripper_scale, robot_obj.gripper_move
                val = g * (sc[1] - sc[0]) + sc[0] if mv["sign"] == 1 else (1 - g) * (sc[1] - sc[0]) + sc[0]
                gi = list(robot_obj.gripper_joint_indices)
                q[gi[0]] = float(val)
                if len(gi) > 1:
                    q[gi[1]] = float(val * mv["mimic"][1] + mv["mimic"][2])
            eid = torch.tensor([env_idx], dtype=torch.long, device=q.device)
            key.write_joint_state_to_sim(q.unsqueeze(0), torch.zeros_like(q).unsqueeze(0), env_ids=eid)
            key.set_joint_position_target(q.unsqueeze(0), env_ids=eid)
            prev = rm.control_manager.prev_control[env_idx]
            ak = rm.process_name(robot_obj.arm_name)
            if ak in prev:
                prev[ak]["position"] = q[list(robot_obj.arm_joint_indices)].tolist()
            gk = rm.process_name(robot_obj.gripper_name)
            if gk in prev and g is not None:
                prev[gk]["position"] = q[list(robot_obj.gripper_joint_indices)].tolist()
        self.env.sim.scene.write_data_to_sim()

    def set_object_poses(self, env_idx: int, layout: dict) -> int:
        """Write every object's record pose (``default_pos/ori``) onto the live object of the same
        (type, category, index, label) in env ``env_idx``; velocities zero. Returns objects written."""
        import torch

        from dojo.simlab.layout import iter_objects
        lm = self.env.scene_manager.layout_manager
        live = {}
        for otype in ("Rigid", "Dynamic", "Articulation"):
            for rec in lm.object_records_by_type[otype].layout_records_by_env[env_idx]:
                cat = lm.decode_inst_name(rec["inst_name"])[0]
                live.setdefault(self._object_key(rec, otype, cat), []).append(rec["inst_name"])
        n = 0
        for t, cat, inst in iter_objects(layout):
            if t not in ("Rigid", "Dynamic", "Articulation"):
                continue
            names = live.get(self._object_key(inst, t, cat))
            if not names:
                continue
            obj = lm.get_scene_object(env_idx, names.pop(0))
            obj.set_local_pose(translation=np.asarray(inst["default_pos"], float),
                               orientation=np.asarray(inst["default_ori"], float))
            try:
                obj.set_linear_velocity(torch.zeros(3))
                obj.set_angular_velocity(torch.zeros(3))
            except Exception:  # noqa: BLE001 (articulations)
                pass
            n += 1
        self.env.sim.scene.write_data_to_sim()
        return n

    def _rearm_reward(self) -> None:
        """RoboDojo's post-reset bookkeeping (latches, initial poses for is_lift/is_moved, counters)."""
        env = self.env
        for i in range(self.num_envs):
            env.success[i], env.end_flag[i], env.take_action_cnt[i] = True, False, 0
        env.reward_manager.reset()
        env.robot_manager.set_origin_endpose()
        env.reward_manager.init_state()
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()

    def load_from_estimate(self, layout: dict, robot: dict | None = None, settle_steps: int = 50) -> dict:
        """World state from an estimate (inverse layout + proprio) in every env. If the object set equals
        the loaded one, the open sim is reused: restore the post-reset snapshot, write the estimated poses,
        place the robot, settle ``settle_steps`` physics steps (4 ms each, no env steps counted) and re-arm
        RoboDojo's reward state. Otherwise a full RoboDojo reset (``load``)."""
        t0 = time.perf_counter()
        missing = missing_labels(layout, self.required)
        if missing:            # RoboDojo's reward would look them up and crash mid-rollout (None poses)
            raise IncompleteWorld(f"{self.task}: the world lacks {missing}, which the task's checks need; "
                                  "add them (e.g. last known pose) or skip verification")
        sig = self._layout_signature(layout)
        fast = getattr(self, "_sig", None) == sig and getattr(self, "_home", None) is not None
        if not fast:
            info = self.load([layout] * self.num_envs)
            self._sig, self._home = sig, self.snapshot()
        else:
            self.restore(self._home)
            info = {"stable": True}
        for i in range(self.num_envs):
            if fast:
                self.set_object_poses(i, layout)
            if robot:
                self.set_robot(i, robot)
        if fast or robot:
            for _ in range(settle_steps):
                self.env.sim_step(render=False)
            self._rearm_reward()
        return {"fast": fast, "stable": bool(info.get("stable", True)), "s": time.perf_counter() - t0}

    # ------------------------------------------------------------------ batched verification
    def _task_labels(self, env_idx: int = 0) -> dict:
        return {o["label"]: o for o in self.run.objects_gt(env_idx) if o["label"] and not o["clutter"]
                and o["type"] in ("Rigid", "Dynamic", "Articulation")}

    def _perturb_env(self, env_idx: int, perturb: dict, rng: np.random.Generator) -> None:
        import torch

        from dojo.simlab.layout import quat_mul, quat_z
        labels = perturb.get("labels")
        lm = self.env.scene_manager.layout_manager
        for lab, o in self._task_labels(env_idx).items():
            if labels and lab not in labels:
                continue
            obj = lm.get_scene_object(env_idx, o["inst_name"])
            pos, q = obj.get_local_pose()
            pos, q = _np(pos).copy(), _np(q).copy()
            pos[:2] += rng.normal(0.0, perturb.get("pos_mm", 0.0) / 1000.0, 2)
            q = quat_mul(quat_z(rng.normal(0.0, perturb.get("yaw_deg", 0.0))), q)
            obj.set_local_pose(translation=pos, orientation=q / np.linalg.norm(q))
            try:
                obj.set_linear_velocity(torch.zeros(3))
                obj.set_angular_velocity(torch.zeros(3))
            except Exception:  # noqa: BLE001
                pass
        self.env.sim.scene.write_data_to_sim()

    def verify(self, world: dict, candidates: list, perturb: dict | None = None, k: int = 1,
               record_traj: bool = False, seed: int = 0, keep=None) -> list[dict]:
        """Roll every candidate segment (x ``k`` perturbed samples; sample 0 nominal) from ``world`` in
        batches of ``num_envs`` and score each with the task's own checker. See the module docstring of
        the segment format; ``world`` = {"layout", "robot"?} or {"snapshot"} (a ``broadcast_snapshot``
        starts every env from that one state). ``keep``: (candidate, sample), a list of such pairs, or
        "nominal" (every (candidate, 0)): those rollouts' end states are kept as broadcast snapshots in
        ``self.kept[(c, s)]`` (chain the next segment from the chosen one, e.g. a place after a verified
        grasp); their outcomes get ``"kept": True``."""
        self.kept = {}
        if keep == "nominal":
            keep_set = {(ci, 0) for ci in range(len(candidates))}
        elif keep is None:
            keep_set = set()
        elif len(keep) == 2 and all(isinstance(v, (int, np.integer)) for v in keep):
            keep_set = {tuple(int(v) for v in keep)}
        else:
            keep_set = {tuple(int(v) for v in kv) for kv in keep}
        from dojo.simlab.motion import Trajectory
        if "snapshot" in world:
            base = world["snapshot"]
        else:
            self.load_from_estimate(world["layout"], world.get("robot"))
            base = self.snapshot()
        acts = []
        for c in candidates:
            a = c.get("actions") if isinstance(c, dict) else c
            acts.append(Trajectory.from_json({"actions": a}).actions if a and isinstance(a[0]["left_arm_joint_state"], list)
                        else a)
        jobs = [(ci, si) for ci in range(len(candidates)) for si in range(max(1, k))]
        out: list[dict] = []
        rng = np.random.default_rng(seed)
        for b0 in range(0, len(jobs), self.num_envs):
            batch = jobs[b0:b0 + self.num_envs]
            self.restore(base)
            if perturb:
                for e, (_, si) in enumerate(batch):
                    if si > 0:
                        self._perturb_env(e, perturb, rng)
            from collections import deque

            from dojo.simlab.motion import tcp_of
            start = [self._task_labels(e) for e in range(self.num_envs)]
            hist = [deque(maxlen=HELD_WINDOW) for _ in batch]   # (tcp per arm, object positions) per step
            holder = [{} for _ in batch]             # label -> arm that lifted it with closed jaws
            slipped = [set() for _ in batch]
            trajs = [[] for _ in batch]
            n_max = max(len(acts[ci]) for ci, _ in batch)
            for t in range(n_max):
                step_acts = []
                for e in range(self.num_envs):
                    if e < len(batch):
                        a = acts[batch[e][0]]
                        step_acts.append(a[min(t, len(a) - 1)])
                    else:
                        step_acts.append(acts[batch[0][0]][min(t, len(acts[batch[0][0]]) - 1)])
                self.step(step_acts)
                for e in range(len(batch)):
                    cur = self._task_labels(e)
                    st = self.run.obs_state(e)
                    hist[e].append(({arm: tcp_of(st[f"{arm}_ee_pose"]) for arm in ARMS},
                                    {lab: np.asarray(o["pos"], float) for lab, o in cur.items()}))
                    tcp, _ = hist[e][-1]
                    for arm in ARMS:
                        arm_closed = float(st[f"{arm}_ee_joint_state"][0]) < 0.5
                        for lab, o in cur.items():
                            z0 = start[e][lab]["pos"][2] if lab in start[e] else o["pos"][2]
                            near = float(np.linalg.norm(np.asarray(o["pos"]) - tcp[arm])) < 0.06
                            if arm_closed and near and o["pos"][2] - z0 > 0.01:
                                holder[e][lab] = arm
                            elif holder[e].get(lab) == arm:
                                if not arm_closed:
                                    holder[e].pop(lab)          # released on purpose
                                elif float(np.linalg.norm(np.asarray(o["pos"]) - tcp[arm])) > 0.04:
                                    slipped[e].add(lab)         # left the closed jaws
                                    holder[e].pop(lab)
                    if record_traj:
                        trajs[e].append({lab: o["pos"] + o["quat"] for lab, o in cur.items()})
            rob = None
            for e, (ci, si) in enumerate(batch):
                cur = self._task_labels(e)
                rob = self.run.robot_gt(e)
                z0 = {lab: o["pos"][2] for lab, o in start[e].items()}
                lifted = [lab for lab, o in cur.items() if o["pos"][2] - z0.get(lab, o["pos"][2]) > 0.01]
                dropped = sorted(slipped[e])
                rew = self.reward(e)
                res = {"candidate": ci, "sample": si, "steps": len(acts[ci]), "ended": rew["ended"],
                       "success": rew["success"], "score": self.score(e), "reward": rew,
                       "final_objects": {lab: o["pos"] + o["quat"] for lab, o in cur.items()},
                       "moved_mm": {lab: float(np.linalg.norm(np.subtract(o["pos"], start[e][lab]["pos"]))) * 1000
                                    for lab, o in cur.items() if lab in start[e]},
                       "lifted": lifted, "dropped": dropped,      # dropped = left the closed jaws of the arm that lifted it
                       "fell": [lab for lab, o in cur.items() if o["pos"][2] < 0.7],
                       "gripper_blocked": {arm: bool(r.get("gripper_open_measured", 1.0) >
                                                     r.get("gripper_open_cmd", 1.0) + 0.05)
                                           for arm, r in rob.items() if "gripper_open_cmd" in r}}
                st = self.run.obs_state(e)
                res["final_robot"] = {f"{arm}_q": np.asarray(st[f"{arm}_arm_joint_state"], float).tolist() for arm in ARMS}
                res["final_robot"].update({f"{arm}_grip_cmd": float(st[f"{arm}_ee_joint_state"][0]) for arm in ARMS})
                res["held"] = held_objects(hist[e], {arm: float(st[f"{arm}_ee_joint_state"][0]) for arm in ARMS},
                                           res["gripper_blocked"])
                if isinstance(candidates[ci], dict):
                    res.update({k2: v for k2, v in candidates[ci].items() if k2 not in ("actions",)})
                if (ci, si) in keep_set:
                    self.kept[(ci, si)] = self.broadcast_snapshot(e)
                    res["kept"] = True
                if record_traj:
                    res["traj"] = trajs[e]
                out.append(res)
        return out

    def close(self) -> None:
        self.run.close()


# --------------------------------------------------------------------------- benchmark
def _hold_actions(tw: Twin) -> list[dict]:
    return [tw.run.hold_action(i) for i in range(tw.num_envs)]


def _wiggle(tw: Twin, k: int) -> list[dict]:
    acts = []
    for i in range(tw.num_envs):
        a = tw.run.hold_action(i)
        for arm in ARMS:
            q = a[f"{arm}_arm_joint_state"].copy()
            q[0] += 0.03 * np.sin(k / 5.0)
            q[1] += 0.03 * np.cos(k / 7.0)
            a[f"{arm}_arm_joint_state"] = q
        acts.append(a)
    return acts


def _cpu_s() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF)          # all threads of this process
    return r.ru_utime + r.ru_stime


def _max_rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _obj_poses(tw: Twin, env_idx: int = 0) -> dict:
    return {o["inst_name"]: o["pos"] + o["quat"] for o in tw.run.objects_gt(env_idx) if o["pos"] is not None}


def contact_restore_fidelity(tw: Twin, g2_dir: Path, split: int = 120, seg: int = 50) -> dict:
    """Snapshot in the middle of the recorded G2 ``stack`` run (block in the gripper), roll ``seg``
    steps, restore, roll the same steps again: does restore reproduce a contact-rich segment?"""
    from dojo.simlab.motion import Trajectory
    lay = json.loads((g2_dir / "stack_layout.json").read_text())
    traj = Trajectory.from_json(json.loads((g2_dir / "stack_traj.json").read_text()))
    tw.load(lay)
    for act in traj.actions[:split]:
        tw.step(act)
    t0 = time.perf_counter()
    s = tw.snapshot()
    t_snap = time.perf_counter() - t0
    a = []
    for act in traj.actions[split:split + seg]:
        tw.step(act)
        a.append(_obj_poses(tw))
    t1 = time.perf_counter()
    tw.restore(s)
    t_rest = time.perf_counter() - t1
    b = []
    t2 = time.perf_counter()
    for act in traj.actions[split:split + seg]:
        tw.step(act)
        b.append(_obj_poses(tw))
    t_seg = time.perf_counter() - t2
    dmax, first = 0.0, None
    for k, (fa, fb) in enumerate(zip(a, b)):
        for n, pa in fa.items():
            d = float(np.linalg.norm(np.subtract(pa[:3], fb[n][:3]))) * 1000
            if d > 0 and first is None:
                first = k
            dmax = max(dmax, d)
    return {"split_step": split, "segment_steps": seg, "snapshot_s": t_snap, "restore_s": t_rest,
            "segment_rollout_s": t_seg, "max_obj_divergence_mm": dmax, "first_divergent_step": first,
            "bitwise": a == b}


def bench(task: str, num_envs: int, n_steps: int, out: Path, physics_only: bool = False,
          g2_dir: Path | None = None) -> dict:
    from dojo.simlab.gen_layouts import generate
    t0 = time.perf_counter()
    tw = Twin(task, num_envs=num_envs, physics_only=physics_only, need_planner=False)
    boot = time.perf_counter() - t0
    lays = [generate(task, paths.SEED_BASE + 500 + i) for i in range(tw.num_envs)]
    lays2 = [generate(task, paths.SEED_BASE + 600 + i) for i in range(tw.num_envs)]
    res = {"task": task, "num_envs": tw.num_envs, "boot_s": boot, "physics_only": physics_only}
    info = tw.load(lays)
    res["load_first"] = {"close_s": info["close_s"], "reset_s": info["reset_s"], "stable": info["per_env_ok"]}
    info = tw.load(lays2)                               # a NEW layout (what an inverse-layout update costs)
    res["load_new_layout"] = {"close_s": info["close_s"], "reset_s": info["reset_s"], "stable": info["per_env_ok"]}
    info = tw.load(lays2, fresh=False)                  # same object set, no sim close (CPU physics respawns)
    res["load_no_close"] = {"reset_s": info["reset_s"], "stable": info["per_env_ok"]}
    tw.step(_hold_actions(tw))
    c0, t1 = _cpu_s(), time.perf_counter()
    for k in range(n_steps):
        tw.step(_wiggle(tw, k))
    dt, dc = time.perf_counter() - t1, _cpu_s() - c0
    res["steps"] = {"n": n_steps, "wall_s": dt, "env_steps_per_s": n_steps * tw.num_envs / dt,
                    "per_env_steps_per_s": n_steps / dt, "sim_realtime_factor_per_env": n_steps * 0.04 / dt,
                    "cpu_cores_used": dc / dt}
    times = []
    for _ in range(0 if physics_only else 10):
        t2 = time.perf_counter()
        tw.render()
        times.append(time.perf_counter() - t2)
    med = float(np.median(times)) if times else float("nan")
    res["render"] = {"obs_all_cams_s_median": med, "n_cams": 0 if physics_only else 3 * tw.num_envs,
                     "per_camera_s": med / (3 * tw.num_envs) if times else None}
    # verify-before-execute unit: snapshot once, then (restore + 40-step segment) per candidate
    s = tw.snapshot()
    seg = []
    for rep_ in range(3):
        t3 = time.perf_counter()
        tw.restore(s)
        t4 = time.perf_counter()
        for k in range(40):
            tw.step(_wiggle(tw, k))
        seg.append({"restore_s": t4 - t3, "rollout40_s": time.perf_counter() - t4})
    t5 = time.perf_counter()
    tw.snapshot()
    res["segment"] = {"snapshot_s": time.perf_counter() - t5, "runs": seg,
                      "restore_plus_40_steps_s": float(np.median([r["restore_s"] + r["rollout40_s"] for r in seg])),
                      "candidates_per_s": tw.num_envs / float(np.median([r["restore_s"] + r["rollout40_s"]
                                                                         for r in seg]))}
    if g2_dir is not None and tw.num_envs == 1 and task == "stack_blocks":
        res["contact_restore"] = contact_restore_fidelity(tw, Path(g2_dir))
    res["max_rss_gb"] = _max_rss_gb()
    out.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("SIMLAB_BENCH_TAG", "")
    (out / f"bench_{task}_n{tw.num_envs}{tag}.json").write_text(json.dumps(jsonable(res), indent=1))
    cr = res.get("contact_restore", {})
    print(f"[twin] {task} n={tw.num_envs}{tag}: boot {boot:.1f}s, new layout {res['load_new_layout']['reset_s'] + res['load_new_layout']['close_s']:.1f}s"
          f" (no-close {res['load_no_close']['reset_s']:.1f}s), {res['steps']['env_steps_per_s']:.1f} env-steps/s total, "
          f"{res['steps']['per_env_steps_per_s']:.1f}/env, {res['steps']['cpu_cores_used']:.1f} cores, "
          f"restore+40 steps {res['segment']['restore_plus_40_steps_s']:.2f}s, snapshot {res['segment']['snapshot_s']*1000:.0f} ms, "
          f"rss {res['max_rss_gb']:.1f} GB"
          + (f", contact restore max {cr['max_obj_divergence_mm']:.4f} mm bitwise={cr['bitwise']}" if cr else ""),
          flush=True)
    tw.close()
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_blocks")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default=str(paths.DATA / "simlab" / "twin_bench"))
    ap.add_argument("--physics-only", action="store_true")
    ap.add_argument("--g2-dir", default=None, help="G2 record dir (stack_*.json) for the contact-restore check")
    a = ap.parse_args(argv)
    bench(a.task, a.num_envs, a.steps, Path(a.out), physics_only=a.physics_only,
          g2_dir=Path(a.g2_dir) if a.g2_dir else None)


if __name__ == "__main__":
    main()
