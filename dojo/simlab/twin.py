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

Bench (compute2):  scripts/simlab/rdpy.sh -m dojo.simlab.twin --task stack_blocks --num-envs 1 4 8
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.runner import RoboDojoRunner, _np, jsonable

ARMS = ("left", "right")


class Twin:
    def __init__(self, task: str, num_envs: int = 1, device_id: int = 0, workdir: str | None = None):
        self.run = RoboDojoRunner(task, num_envs=num_envs, device_id=device_id,
                                  workdir=workdir or str(paths.DATA / "simlab" / "work" / f"twin_{task}_{num_envs}"))
        self.env = self.run.env
        self.num_envs = self.run.num_envs

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
                "objects": objs,
                "prev_control": copy.deepcopy(rm.control_manager.prev_control[i]),
                "counters": {"take_action_cnt": int(env.take_action_cnt[i]), "success": bool(env.success[i]),
                             "end_flag": bool(env.end_flag[i])},
                "reward": {k: copy.deepcopy(getattr(rwm, k)[i]) for k in (
                    "check_list", "final_check_list", "query_list", "trigger_check_list", "trigger_query_list",
                    "score_list", "score_achieved", "score_completed_count", "score_meta", "score_trigger_meta",
                    "final_score_list", "final_score_achieved", "final_score_completed_count", "final_score_meta")},
                "func_parser": {k: copy.deepcopy(getattr(fp, k)[i]) for k in (
                    "pre_state", "robot_origin_endpose", "joint_ratio_transition_state")},
            }
        snap["gated_score"] = copy.deepcopy(env.reward_manager._gated_score_lst)
        return snap

    def restore(self, snap: dict, settle_steps: int = 0) -> None:
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
            rm.control_manager.prev_control[i] = copy.deepcopy(e["prev_control"])
            c = e["counters"]
            env.take_action_cnt[i], env.success[i], env.end_flag[i] = c["take_action_cnt"], c["success"], c["end_flag"]
            rwm = env.reward_manager
            for k, v in e["reward"].items():
                getattr(rwm, k)[i] = copy.deepcopy(v)
            for k, v in e["func_parser"].items():
                getattr(rwm.func_parser, k)[i] = copy.deepcopy(v)
        env.reward_manager._gated_score_lst = copy.deepcopy(snap["gated_score"])
        env.sim.scene.write_data_to_sim()
        for _ in range(settle_steps):
            env.sim_step(render=False)

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


def bench(task: str, num_envs: int, n_steps: int, out: Path) -> dict:
    from dojo.simlab.gen_layouts import generate
    t0 = time.perf_counter()
    tw = Twin(task, num_envs=num_envs)
    boot = time.perf_counter() - t0
    lays = [generate(task, paths.SEED_BASE + 500 + i) for i in range(tw.num_envs)]
    res = {"task": task, "num_envs": tw.num_envs, "boot_s": boot, "loads": []}
    for rep in range(2):
        info = tw.load(lays)
        res["loads"].append({"close_s": info["close_s"], "reset_s": info["reset_s"], "stable": info["per_env_ok"]})
    # steps/s with a small joint wiggle (contacts-free), all envs stepped together
    tw.step(_hold_actions(tw))
    t1 = time.perf_counter()
    for k in range(n_steps):
        tw.step(_wiggle(tw, k))
    dt = time.perf_counter() - t1
    res["steps"] = {"n": n_steps, "wall_s": dt, "env_steps_per_s": n_steps * tw.num_envs / dt,
                    "batch_steps_per_s": n_steps / dt, "sim_realtime_factor": n_steps * 0.04 / dt}
    # render: all three cameras of all envs, as get_obs does it
    times = []
    for _ in range(10):
        t2 = time.perf_counter()
        tw.render()
        times.append(time.perf_counter() - t2)
    res["render"] = {"obs_all_cams_s_median": float(np.median(times)), "n_cams": 3 * tw.num_envs,
                     "per_camera_s": float(np.median(times)) / (3 * tw.num_envs)}
    # snapshot / restore cost and fidelity: run 40 steps, restore, run them again, compare poses
    tw.step(_hold_actions(tw))
    t3 = time.perf_counter()
    s = tw.snapshot()
    t_snap = time.perf_counter() - t3
    traj_a = []
    for k in range(40):
        tw.step(_wiggle(tw, k))
        traj_a.append(tw.run.objects_gt(0))
    t4 = time.perf_counter()
    tw.restore(s)
    t_rest = time.perf_counter() - t4
    traj_b = []
    for k in range(40):
        tw.step(_wiggle(tw, k))
        traj_b.append(tw.run.objects_gt(0))
    dmax = 0.0
    for fa, fb in zip(traj_a, traj_b):
        for oa, ob in zip(fa, fb):
            if oa["pos"] is not None and ob["pos"] is not None:
                dmax = max(dmax, float(np.linalg.norm(np.subtract(oa["pos"], ob["pos"]))) * 1000)
    qa = tw.run.robot_gt(0)
    res["snapshot"] = {"snapshot_s": t_snap, "restore_s": t_rest, "replay_after_restore_max_obj_mm": dmax,
                       "robot_q_after": {a: qa[a]["joint_pos"][:6] for a in ARMS}}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bench_{task}_n{tw.num_envs}.json").write_text(json.dumps(jsonable(res), indent=1))
    print(f"[twin] {task} n={tw.num_envs}: boot {boot:.1f}s, load {res['loads'][-1]['reset_s']:.1f}s "
          f"(+close {res['loads'][-1]['close_s']:.1f}s), {res['steps']['env_steps_per_s']:.1f} env-steps/s "
          f"(x{res['steps']['sim_realtime_factor']:.2f} real time), obs {res['render']['obs_all_cams_s_median']*1000:.0f} ms "
          f"for {res['render']['n_cams']} cams, snapshot {t_snap*1000:.0f} ms, restore {t_rest*1000:.0f} ms, "
          f"restore replay gap {dmax:.3f} mm", flush=True)
    tw.close()
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_blocks")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default=str(paths.DATA / "simlab" / "twin_bench"))
    a = ap.parse_args(argv)
    bench(a.task, a.num_envs, a.steps, Path(a.out))


if __name__ == "__main__":
    main()
