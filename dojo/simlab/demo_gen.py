"""Support-arm demonstrations for imitate_sorting_sequence dev layouts (RoboDojo's own pkl format).

RoboDojo plays the support arm (a Franka at (0, 0.6, table), planner-less) from
``Assets/Traj/<bench>/imitate_sorting_sequence/<eval_seed>/<seed>.pkl``. Those files are paired with the eval
layouts and are off limits (``paths.eval_asset_violation``). For our dev layouts this plans an equivalent demo
with cuRobo:
- the support arm picks aim0..aim4 top-down, in the seed-fixed order of ``runner.demo_order``;
- it drops each into basket1;
- ``target_label`` lists the matching t-objects in that order, which is what the policy must reproduce in
  basket0.

Format (what ``ImitateSortingSequenceCommon.load_support_arm_traj`` reads)::

    {"joint_path": [segment, ...], "target_label": [5], "aim_label": [5], "target_place_tag": [5]}
    segment = [step, ...] at the physics rate (RoboDojo pops one per 4 ms control sub-step)
    step = {"support_arm0_joint_state": {"position": [7], "velocity": [7]},
            "support_ee0_joint_state": {"position": [2], "velocity": [2]}}

Validation: RoboDojo itself replays the demo (fresh reset, policy arms holding, ``runner``'s routed loader).
The demo is kept only if:
- RoboDojo's own post-demo check passes (every aim in basket1, env not flagged unstable);
- no t-object moved.

Output: ``DATA/simlab/traj/imitate_sorting_sequence/<seed>.pkl`` plus ``demo_index.json`` (status per layout).

    scripts/simlab/rdpy.sh -m dojo.simlab.demo_gen --task imitate_sorting_sequence [--ids ...] [--probe]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths

TASK = "imitate_sorting_sequence"
ARM, EE = "support_arm0", "support_ee0"
HAND_TCP = 0.102            # panda_hand -> grasp point along the hand z (robot_config gripper_bias)
OPEN, CLOSE = 0.04, -0.01   # finger joint targets (gripper_scale); closing stops on the object
MAX_WIDTH = 0.074           # widest graspable footprint across the fingers (2 x 0.04 minus margin)
HOME_STEP_RAD = 0.004       # per 4 ms sub-step for joint-space moves (1 rad/s)


def demo_dir() -> Path:
    return paths.DATA / "simlab" / "traj" / TASK


def world_vertices(meta: dict, pos, quat) -> np.ndarray:
    from dojo.simlab.motion import quat_to_mat
    v = np.asarray(meta["geometry"]["oriented_bbox"]["vertices"], float)
    q = np.asarray(quat, float)
    return v @ quat_to_mat(q / np.linalg.norm(q)).T + np.asarray(pos, float)


def top_down_grasp(verts: np.ndarray, max_width: float = MAX_WIDTH) -> dict | None:
    """Narrowest horizontal direction of the object's world bbox: fingers close along it. Returns the footprint
    centre, the closing yaw, the width across the fingers, top and bottom z (None if too wide)."""
    best = None
    for deg in range(0, 180, 5):
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        e = np.array([-d[1], d[0]])
        pd, pe = verts[:, :2] @ d, verts[:, :2] @ e
        w = float(pd.max() - pd.min())
        if best is None or w < best["width"]:
            c = d * (pd.max() + pd.min()) / 2 + e * (pe.max() + pe.min()) / 2
            best = {"width": w, "yaw": math.radians(deg), "xy": c, "length": float(pe.max() - pe.min())}
    best["top"], best["bottom"] = float(verts[:, 2].max()), float(verts[:, 2].min())
    return best if best["width"] <= max_width else None


def hand_quat(yaw: float) -> np.ndarray:
    """panda_hand pointing down (z = -world z), fingers closing along world (cos yaw, sin yaw)."""
    from dojo.simlab.look_render import mat_to_quat
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return mat_to_quat(np.stack([np.cross(y, z), y, z], axis=1))


def hand_pose(tcp_xyz, yaw: float) -> np.ndarray:
    return np.concatenate([np.asarray(tcp_xyz, float) + [0.0, 0.0, HAND_TCP], hand_quat(yaw)])


def step_rec(q, dq, fingers) -> dict:
    return {f"{ARM}_joint_state": {"position": [float(v) for v in q], "velocity": [float(v) for v in dq]},
            f"{EE}_joint_state": {"position": [float(fingers)] * 2, "velocity": [0.0, 0.0]}}


class DemoPlanner:
    """Plans one demo with the support arm's cuRobo planner. The scene stands still: every target comes from
    the reset state, and nothing is stepped."""

    def __init__(self, run):
        self.run = run
        rm = run.env.robot_manager
        self.robot = rm.get_robot_by_arm_name(ARM)
        self.planner = rm.planner.get(self.robot.robot_name)
        if self.planner is None:
            raise RuntimeError("support arm has no planner: RoboDojoRunner(..., support_planner=True)")
        self.q0 = np.asarray(rm.get_joint(self.robot, env_idx_list=[0])[0], float).reshape(-1)
        self.segments: list[list[dict]] = []
        self.q = self.q0.copy()
        self.fingers = OPEN

    def plan_to(self, tcp_xyz, yaw: float, name: str) -> bool:
        for y in (yaw, yaw + math.pi, yaw + 0.2, yaw - 0.2, yaw + math.pi + 0.2, yaw + math.pi - 0.2):
            res = self.planner.plan_path(self.q.copy(), hand_pose(tcp_xyz, y),
                                         real_robot_pose=copy.deepcopy(self.robot.entity_origin_pose))
            if str(res.get("status")) != "Success" or res.get("position") is None:
                continue
            p, v = res["position"], res.get("velocity")
            p = np.asarray(p.detach().cpu().numpy() if hasattr(p, "detach") else p, float).reshape(-1, len(self.q))
            v = (np.zeros_like(p) if v is None else
                 np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, float).reshape(p.shape))
            if not len(p):
                continue
            self.segments.append([step_rec(p[i], v[i], self.fingers) for i in range(len(p))])
            self.q = p[-1].copy()
            return True
        return False

    def grip(self, target: float, steps: int = 60, hold: int = 40) -> None:
        f0 = self.fingers
        seg = [step_rec(self.q, np.zeros_like(self.q), f0 + (target - f0) * (i + 1) / steps) for i in range(steps)]
        seg += [step_rec(self.q, np.zeros_like(self.q), target)] * hold
        self.segments.append(seg)
        self.fingers = target

    def home(self) -> None:
        n = max(1, math.ceil(float(np.abs(self.q0 - self.q).max()) / HOME_STEP_RAD))
        path = [self.q + (self.q0 - self.q) * (i + 1) / n for i in range(n)]
        self.segments.append([step_rec(q, np.zeros_like(q), self.fingers) for q in path])
        self.q = self.q0.copy()

    def n_steps(self) -> int:
        return sum(len(s) for s in self.segments)


def plan_demo(run, seed: int) -> tuple[dict | None, dict]:
    """Plan the demo on the env's current (reset) state. Returns (pkl data or None, info)."""
    from dojo.simlab.gen_layouts import metadata
    from dojo.simlab.runner import demo_order
    order = demo_order(seed)
    rows = {o["label"]: o for o in run.objects_gt() if o.get("label") and not o.get("clutter")}
    origin = np.asarray(run.gt().get("env_origin") or [0.0, 0.0, 0.0], float)[:3]

    def local(o):
        return np.asarray(o["pos"], float) - origin

    basket = rows["basket1"]
    bv = world_vertices(metadata(basket["type"], basket["category"], int(basket["category_idx"])),
                        local(basket), basket["quat"])
    b_xy, b_top = (bv[:, :2].max(axis=0) + bv[:, :2].min(axis=0)) / 2, float(bv[:, 2].max())
    dp = DemoPlanner(run)
    info: dict = {"order": order, "objects": {}}
    for i in order:
        o = rows[f"aim{i}"]
        g = top_down_grasp(world_vertices(metadata(o["type"], o["category"], int(o["category_idx"])), local(o),
                                          o["quat"]))
        if g is None:
            return None, {**info, "status": f"aim{i} ({o['category']}) wider than the gripper"}
        h = g["top"] - g["bottom"]
        grasp_z = max(g["top"] - min(0.025, 0.5 * h), g["bottom"] + 0.01)
        info["objects"][f"aim{i}"] = {"category": o["category"], "width_cm": round(100 * g["width"], 1),
                                      "height_cm": round(100 * h, 1)}
        carry_z = max(b_top, g["top"]) + h + 0.06
        steps = [((*g["xy"], g["top"] + 0.08), g["yaw"], f"pre aim{i}"),
                 ((*g["xy"], grasp_z), g["yaw"], f"grasp aim{i}")]
        for xyz, yaw, name in steps:
            if not dp.plan_to(xyz, yaw, name):
                return None, {**info, "status": f"no plan: {name}"}
        dp.grip(CLOSE)
        lift = (*g["xy"], carry_z)
        if not dp.plan_to(lift, g["yaw"], f"lift aim{i}"):
            return None, {**info, "status": f"no plan: lift aim{i}"}
        if not dp.plan_to((*b_xy, carry_z), g["yaw"], f"over basket1 with aim{i}"):
            return None, {**info, "status": f"no plan: over basket1 with aim{i}"}
        dp.grip(OPEN, steps=40, hold=60)
    dp.home()
    data = {"joint_path": dp.segments, "target_label": [f"t{i}" for i in order],
            "aim_label": [f"aim{i}" for i in order], "target_place_tag": [None] * 5,
            "_simlab": {"generator": "simlab/demo_gen v1", "seed": seed, "order": order}}
    return data, {**info, "status": "planned", "substeps": dp.n_steps()}


def validate(run, layout: dict, n_substeps: int, margin_env_steps: int = 60) -> dict:
    """Replay through RoboDojo (the routed loader reads the pkl just written): policy arms hold, the
    support arm plays. Ok if RoboDojo did not flag the env unstable, every aim ended in basket1, no t moved."""
    info = run.load(layout)
    if not info["stable"]:
        return {"ok": False, "why": "reset unstable"}
    env = run.env
    t0 = {o["label"]: np.asarray(o["pos"], float) for o in run.objects_gt()
          if o.get("label", "").startswith("t") and not o.get("clutter")}
    hold = run.hold_action(0)
    n = math.ceil(n_substeps / 10) + margin_env_steps
    ended_at = None
    for k in range(n):
        if run.step(hold)[0]:
            ended_at = k + 1
            break
    rm = env.reward_manager
    in_b = [float(rm.call_func_parser(rm.is_A_in_B(label_A=f"aim{i}", label_B="basket1"), 0)) for i in range(5)]
    t1 = {o["label"]: np.asarray(o["pos"], float) for o in run.objects_gt() if o.get("label") in t0}
    moved = {lab: round(float(np.linalg.norm(t1[lab] - p)) * 1000, 1) for lab, p in t0.items() if lab in t1}
    unstable = 0 in getattr(env, "unstable_envs", set())
    ok = (not unstable) and all(v >= 0.5 for v in in_b) and max(moved.values(), default=0.0) < 5.0
    return {"ok": bool(ok), "unstable": bool(unstable), "aim_in_basket1": in_b, "t_moved_mm": moved,
            "env_steps": n if ended_at is None else ended_at, "ended_early": ended_at is not None,
            "support_checked": bool(getattr(env, "support_checked", [False])[0])}


def main(argv=None) -> None:
    from dojo.simlab.presettle import layout_files
    from dojo.simlab.runner import RoboDojoRunner, jsonable
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="print the support arm / planner facts and stop")
    ap.add_argument("--raw", action="store_true", help="use <id>.json instead of the pre-settled <id>.settled.json")
    a = ap.parse_args(argv)
    if a.task != TASK:
        raise SystemExit(f"demos are implemented for {TASK} only")
    files = [p for p in layout_files(TASK) if not a.ids or int(p.stem) in a.ids]
    if not a.raw:
        files = [f.with_name(f"{f.stem}.settled.json") for f in files if f.with_name(f"{f.stem}.settled.json").exists()]
    out = demo_dir()
    paths.assert_not_eval(out)
    out.mkdir(parents=True, exist_ok=True)
    run = RoboDojoRunner(TASK, physics_only=True, support_planner=True,
                         workdir=str(paths.DATA / "simlab" / "work" / f"demo_gen_{TASK}"))
    index_p = out / "demo_index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {}
    for f in files:
        t0 = time.perf_counter()
        seed = int(f.name.split(".")[0])
        lay = json.loads(f.read_text())
        pk = out / f"{seed}.pkl"
        pk.unlink(missing_ok=True)                    # the plan runs on the stub-loaded reset
        if not run.load(lay)["stable"]:
            index[str(seed)] = {"status": "reset unstable"}
            continue
        if a.probe:
            dp = DemoPlanner(run)
            rm = run.env.robot_manager
            print("[demo] probe", json.dumps(jsonable({
                "robot": dp.robot.robot_name, "arm_name": dp.robot.arm_name, "q0": dp.q0,
                "origin": dp.robot.entity_origin_pose, "planner_dt": getattr(dp.planner, "dt", None),
                "fingers": rm.get_end_effector_real_val(dp.robot, env_idx_list=[0])[0],
                "collect_interval": run.env.obs_manager.collect_interval,
                "objects": {o["label"]: o["pos"] for o in run.objects_gt() if o.get("label") and not o.get("clutter")}})),
                flush=True)
            break
        data, info = plan_demo(run, seed)
        rec = {"from": f.name, **info}
        if data is not None:
            pk.write_bytes(pickle.dumps(data, protocol=4))
            v = validate(run, lay, info["substeps"])
            rec["validation"] = v
            rec["status"] = "ok" if v["ok"] else "dropped: validation"
            if not v["ok"]:
                pk.unlink(missing_ok=True)
        rec["s"] = round(time.perf_counter() - t0, 1)
        index[str(seed)] = rec
        index_p.write_text(json.dumps(jsonable(index), indent=1))
        print(f"[demo] {seed}: {rec['status']} {json.dumps(jsonable(rec.get('validation', {})))[:300]} ({rec['s']}s)",
              flush=True)
    run.close()


if __name__ == "__main__":
    main()
