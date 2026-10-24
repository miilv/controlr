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


class Twin:
    def __init__(self, task: str, num_envs: int = 1, device_id: int = 0, workdir: str | None = None,
                 physics_only: bool = False, need_planner: bool = True):
        self.run = RoboDojoRunner(task, num_envs=num_envs, device_id=device_id,
                                  workdir=workdir or str(paths.DATA / "simlab" / "work" / f"twin_{task}_{num_envs}"),
                                  physics_only=physics_only, need_planner=need_planner)
        self.physics_only = physics_only
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
