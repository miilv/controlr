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
MAX_WIDTH = 0.077           # widest graspable width across the fingers (2 x 0.04 open, 1.5 mm per side)
GRASP_DEPTH = 0.03          # grasp point below the object's top (at most 60 % of its height)
MIN_PAD_Z = 0.016           # pad centre above the object's bottom: the planner's table sits 2.5 cm low
                            # (RoboDojo passes table_height 0.74, the top is 0.765), so lower tips stall
# rad per 4 ms sub-step (RoboDojo's own demos: median 0.002, peak 0.008). Free moves fast; near and with the
# object slow (fast descents left the hand lagging above the object when the fingers closed).
DQ_FREE, DQ_CARRY, DQ_NEAR = 0.008, 0.004, 0.002
SETTLE = 50                 # sub-steps holding still before the fingers close
HOME_STEP_RAD = 0.004       # per 4 ms sub-step for joint-space moves (1 rad/s)
MAX_DEMO_ENV_STEPS = 900    # leaves the policy >= 700 of the 1600 env steps (RoboDojo-paced demos: ~575)
ORDER_TRIES = 4             # the seed's order, then other seeded orders (a transport can knock a later object)
CARRY_CLEARANCE = 0.08
STRAIGHT = [1, 1, 1, 1, 1, 0]  # cuRobo pose-cost hold: orientation + x/y fixed, free along the hand z (vertical)


def demo_dir() -> Path:
    return paths.DATA / "simlab" / "traj" / TASK


def world_vertices(meta: dict, pos, quat) -> np.ndarray:
    from dojo.simlab.motion import quat_to_mat
    v = np.asarray(meta["geometry"]["oriented_bbox"]["vertices"], float)
    q = np.asarray(quat, float)
    return v @ quat_to_mat(q / np.linalg.norm(q)).T + np.asarray(pos, float)


def mesh_points(prim_path: str, origin) -> np.ndarray | None:
    """World points of every mesh under ``prim_path`` on the live stage, minus the env origin (env frame)."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except ImportError:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None and prim_path else None
    if prim is None or not prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not pts:
                continue
            m = np.asarray(cache.GetLocalToWorldTransform(p), float)      # row-vector convention
            v = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ m
            out.append(v[:, :3])
    return np.concatenate(out) - np.asarray(origin, float)[:3] if out else None


def top_down_grasp(verts: np.ndarray, max_width: float = MAX_WIDTH, band: float | None = None) -> dict | None:
    """Narrowest horizontal direction of the object (bbox corners or mesh points; with ``band``, only the points
    within ``band`` of the top, where the fingers hold it): fingers close along it. Returns the footprint
    centre, the closing yaw, the width across the fingers, top and bottom z (None if too wide)."""
    top_all, bottom_all = float(verts[:, 2].max()), float(verts[:, 2].min())
    if band is not None:
        verts = verts[verts[:, 2] >= top_all - band]
    best = None
    for deg in range(0, 180, 5):
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        e = np.array([-d[1], d[0]])
        pd, pe = verts[:, :2] @ d, verts[:, :2] @ e
        w = float(pd.max() - pd.min())
        if best is None or w < best["width"]:
            c = d * (pd.max() + pd.min()) / 2 + e * (pe.max() + pe.min()) / 2
            best = {"width": w, "yaw": math.radians(deg), "xy": c, "length": float(pe.max() - pe.min())}
    best["top"], best["bottom"] = top_all, bottom_all
    return best if best["width"] <= max_width else {**best, "too_wide": True}


def hand_quat(yaw: float) -> np.ndarray:
    """panda_hand pointing down (z = -world z), fingers closing along world (cos yaw, sin yaw)."""
    from dojo.simlab.look_render import mat_to_quat
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return mat_to_quat(np.stack([np.cross(y, z), y, z], axis=1))


def hand_pose(tcp_xyz, yaw: float) -> np.ndarray:
    return np.concatenate([np.asarray(tcp_xyz, float) + [0.0, 0.0, HAND_TCP], hand_quat(yaw)])


def retime(path: np.ndarray, dq_max: float, min_n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Resample a joint path to RoboDojo's own demo pace: at most ``dq_max`` rad per 4 ms sub-step on any
    joint (its make_kong / tic-tac-toe demos peak at 0.008), at least ``min_n`` sub-steps; smoothstep
    timing (zero velocity at both ends). Returns (positions, velocities in rad/s)."""
    path = np.asarray(path, float)
    travel = float((path.max(axis=0) - path.min(axis=0)).max()) if len(path) > 1 else 0.0   # cuRobo jitter-free
    n = max(min_n, math.ceil(1.5 * travel / dq_max))            # smoothstep peaks at 1.5x the mean speed
    u = np.linspace(0.0, 1.0, n + 1)[1:]
    u = u * u * (3.0 - 2.0 * u)
    src = np.linspace(0.0, 1.0, len(path)) if len(path) > 1 else np.zeros(1)
    p = np.stack([np.interp(u, src, path[:, j]) for j in range(path.shape[1])], axis=1)
    v = np.diff(np.vstack([path[:1], p]), axis=0) / 0.004
    return p, v


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
        self.log: list[dict] = []
        self.last_yaw: float | None = None
        self.q = self.q0.copy()
        self.fingers = OPEN

    def plan_to(self, tcp_xyz, yaw: float, name: str, straight: bool = False, dq: float = DQ_FREE) -> bool:
        """Plan the hand to ``tcp_xyz`` (top-down, fingers along ``yaw`` or yaw + pi). ``straight``: a vertical
        line from the current pose (cuRobo pose-cost hold on orientation and x/y), else free space."""
        pair = [yaw, yaw + math.pi]
        if straight and self.last_yaw is not None:      # keep the hand as it is: no turn on a vertical line
            pair.sort(key=lambda y: abs(math.remainder(y - self.last_yaw, 2 * math.pi)))
        tries = [(y, STRAIGHT if straight else None) for y in pair]
        if straight:
            tries += [(y, None) for y in pair]
        else:
            tries += [(y, None) for y in (yaw + 0.2, yaw - 0.2, yaw + math.pi + 0.2, yaw + math.pi - 0.2)]
        for y, hold in tries:
            res = self.planner.plan_path(self.q.copy(), hand_pose(tcp_xyz, y),
                                         real_robot_pose=copy.deepcopy(self.robot.entity_origin_pose),
                                         constraint_pose=hold)
            if str(res.get("status")) != "Success" or res.get("position") is None:
                continue
            p, v = res["position"], res.get("velocity")
            p = np.asarray(p.detach().cpu().numpy() if hasattr(p, "detach") else p, float).reshape(-1, len(self.q))
            v = (np.zeros_like(p) if v is None else
                 np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, float).reshape(p.shape))
            if not len(p):
                continue
            p, v = retime(p, dq)
            self.segments.append([step_rec(p[i], v[i], self.fingers) for i in range(len(p))])
            self.q = p[-1].copy()
            self.last_yaw = y
            self.log.append({"name": name, "steps": len(p), "straight": hold is not None,
                             "yaw_deg": round(math.degrees(y), 1)})
            return True
        return False

    def grip(self, target: float, steps: int = 100, hold: int = 50, settle: int = 0) -> None:
        self.log.append({"name": "close" if target < self.fingers else "open", "steps": settle + steps + hold})
        f0 = self.fingers
        seg = [step_rec(self.q, np.zeros_like(self.q), f0)] * settle
        seg += [step_rec(self.q, np.zeros_like(self.q), f0 + (target - f0) * (i + 1) / steps) for i in range(steps)]
        seg += [step_rec(self.q, np.zeros_like(self.q), target)] * hold
        self.segments.append(seg)
        self.fingers = target

    def home(self) -> None:
        n = max(1, math.ceil(float(np.abs(self.q0 - self.q).max()) / HOME_STEP_RAD))
        self.log.append({"name": "home", "steps": n})
        path = [self.q + (self.q0 - self.q) * (i + 1) / n for i in range(n)]
        self.segments.append([step_rec(q, np.zeros_like(q), self.fingers) for q in path])
        self.q = self.q0.copy()

    def n_steps(self) -> int:
        return sum(len(s) for s in self.segments)


def plan_demo(run, seed: int, order: list[int] | None = None) -> tuple[dict | None, dict]:
    """Plan the demo on the env's current (reset) state, in ``order`` (default: ``demo_order(seed)``).
    Returns (pkl data or None, info)."""
    from dojo.simlab.gen_layouts import metadata
    from dojo.simlab.runner import demo_order
    order = list(order) if order is not None else demo_order(seed)
    rows = {o["label"]: o for o in run.objects_gt() if o.get("label") and not o.get("clutter")}
    tallest = max(float(world_vertices(metadata(o["type"], o["category"], int(o["category_idx"])),
                                       np.asarray(o["pos"], float), o["quat"])[:, 2].max())
                  for lab, o in rows.items() if lab.startswith("aim"))
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
        pts = mesh_points(o.get("prim_path"), origin)
        if pts is not None and len(pts) >= 8:          # the real shape: the top band the fingers hold
            src = "mesh"
            h_all = float(pts[:, 2].max() - pts[:, 2].min())
            g = top_down_grasp(pts, band=min(GRASP_DEPTH + 0.01, 0.6 * h_all))
        else:
            src = "bbox"
            g = top_down_grasp(world_vertices(metadata(o["type"], o["category"], int(o["category_idx"])), local(o),
                                              o["quat"]))
        h = g["top"] - g["bottom"]
        info["objects"][f"aim{i}"] = {"category": o["category"], "idx": int(o["category_idx"]), "shape": src,
                                      "width_cm": round(100 * g["width"], 1), "height_cm": round(100 * h, 1)}
        if g.get("too_wide"):
            return None, {**info, "status": f"aim{i} ({o['category']}) {100 * g['width']:.1f} cm wider than the gripper"}
        grasp_z = max(g["top"] - min(GRASP_DEPTH, 0.6 * h), g["bottom"] + MIN_PAD_Z)
        carry_z = max(b_top, tallest - origin[2]) + h + CARRY_CLEARANCE   # the carried object clears everything
        steps = [((*g["xy"], g["top"] + 0.08), g["yaw"], f"pre aim{i}"),
                 ((*g["xy"], grasp_z), g["yaw"], f"grasp aim{i}")]
        for xyz, yaw, name in steps:
            if not dp.plan_to(xyz, yaw, name, straight=name.startswith("grasp"),
                              dq=DQ_NEAR if name.startswith("grasp") else DQ_FREE):
                return None, {**info, "status": f"no plan: {name}"}
        dp.grip(CLOSE, settle=SETTLE)
        lift = (*g["xy"], carry_z)
        if not dp.plan_to(lift, g["yaw"], f"lift aim{i}", straight=True, dq=DQ_NEAR):
            return None, {**info, "status": f"no plan: lift aim{i}"}
        if not dp.plan_to((*b_xy, carry_z), g["yaw"], f"over basket1 with aim{i}", dq=DQ_CARRY):
            return None, {**info, "status": f"no plan: over basket1 with aim{i}"}
        below_tcp = grasp_z - g["bottom"]                # object bottom under the TCP while carried
        release_z = b_top + below_tcp + 0.015            # bottom 1.5 cm above the rim: placed, not dropped
        if release_z < carry_z - 0.01 and not dp.plan_to((*b_xy, release_z), g["yaw"], f"lower aim{i}",
                                                         straight=True, dq=DQ_NEAR):
            return None, {**info, "status": f"no plan: lower aim{i}"}
        dp.grip(OPEN, steps=40, hold=30)
        if not dp.plan_to((*b_xy, carry_z), g["yaw"], f"clear basket1 after aim{i}", straight=True):
            return None, {**info, "status": f"no plan: clear basket1 after aim{i}"}
    dp.home()
    data = {"joint_path": dp.segments, "target_label": [f"t{i}" for i in order],
            "aim_label": [f"aim{i}" for i in order], "target_place_tag": [None] * 5,
            "_simlab": {"generator": "simlab/demo_gen v1", "seed": seed, "order": order}}
    if dp.n_steps() / 10 > MAX_DEMO_ENV_STEPS:
        return None, {**info, "status": f"too long: {dp.n_steps() / 10:.0f} env steps", "segments": dp.log}
    return data, {**info, "status": "planned", "substeps": dp.n_steps(), "segments": dp.log}


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
    aims = [f"aim{i}" for i in range(5)]
    z0 = {o["label"]: float(o["pos"][2]) for o in run.objects_gt() if o.get("label") in aims}
    zmax = dict(z0)
    rm_ = env.robot_manager
    sup = rm_.get_robot_by_arm_name(ARM)
    trace = {"step": [], "fingers": [], **{lab: [] for lab in aims}}    # every 5 env steps (diagnostics)
    for k in range(n):
        if run.step(hold)[0]:
            ended_at = k + 1
            break
        if k % 5 == 0:
            trace["step"].append(k + 1)
            trace["fingers"].append(round(float(np.asarray(rm_.get_end_effector_real_val(sup, env_idx_list=[0])[0],
                                                           float).reshape(-1)[0]), 4))
            for o in run.objects_gt():
                if o.get("label") in zmax and o["pos"] is not None:
                    zmax[o["label"]] = max(zmax[o["label"]], float(o["pos"][2]))
                    trace[o["label"]].append(round(float(o["pos"][2]), 3))
    rm = env.reward_manager
    in_b = [float(rm.call_func_parser(rm.is_A_in_B(label_A=f"aim{i}", label_B="basket1"), 0)) for i in range(5)]
    fin = {o["label"]: o["pos"] for o in run.objects_gt() if o.get("label") in aims + ["basket1"]}
    b1 = np.asarray(fin.get("basket1"), float)
    final = {lab: {"lifted_cm": round(100 * (zmax[lab] - z0[lab]), 1),
                   "from_basket1_cm": [round(100 * float(v), 1) for v in np.asarray(fin[lab], float) - b1]}
             for lab in aims if lab in fin and lab in z0}
    t1 = {o["label"]: np.asarray(o["pos"], float) for o in run.objects_gt() if o.get("label") in t0}
    moved = {lab: round(float(np.linalg.norm(t1[lab] - p)) * 1000, 1) for lab, p in t0.items() if lab in t1}
    unstable = 0 in getattr(env, "unstable_envs", set())
    ok = (not unstable) and all(v >= 0.5 for v in in_b) and max(moved.values(), default=0.0) < 5.0
    return {"ok": bool(ok), "unstable": bool(unstable), "aim_in_basket1": in_b, "t_moved_mm": moved, "aims": final,
            **({} if ok else {"trace": trace}),
            "env_steps": n if ended_at is None else ended_at, "ended_early": ended_at is not None,
            "support_checked": bool(getattr(env, "support_checked", [False])[0])}


def main(argv=None) -> None:
    from dojo.simlab.presettle import layout_files
    from dojo.simlab.runner import RoboDojoRunner, demo_order, jsonable
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="print the support arm / planner facts and stop")
    ap.add_argument("--raw", action="store_true", help="use <id>.json instead of the pre-settled <id>.settled.json")
    ap.add_argument("--tries", type=int, default=ORDER_TRIES, help="orders to try per layout")
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
        if a.probe:                                   # the facts this module relies on, once, then plan as usual
            dp = DemoPlanner(run)
            rm = run.env.robot_manager
            print("[demo] probe", json.dumps(jsonable({
                "robot": dp.robot.robot_name, "arm_name": dp.robot.arm_name, "q0": dp.q0,
                "origin": list(dp.robot.entity_origin_pose), "planner_dt": getattr(dp.planner, "dt", None),
                "fingers": rm.get_end_effector_real_val(dp.robot, env_idx_list=[0])[0],
                "collect_interval": run.env.obs_manager.collect_interval,
                "objects": {o["label"]: o["pos"] for o in run.objects_gt() if o.get("label") and not o.get("clutter")}}),
                default=str), flush=True)
            a.probe = False
        attempts = []
        for k in range(a.tries):
            if k:
                pk.unlink(missing_ok=True)
                if not run.load(lay)["stable"]:          # plan on a clean reset again
                    break
            order = demo_order(seed) if k == 0 else demo_order(seed * 31 + k)
            data, info = plan_demo(run, seed, order)
            rec = {"from": f.name, "attempt": k, **info}
            if data is not None:
                pk.write_bytes(pickle.dumps(data, protocol=4))
                v = validate(run, lay, info["substeps"])
                rec["validation"] = v
                rec["status"] = "ok" if v["ok"] else "dropped: validation"
                if not v["ok"]:
                    pk.unlink(missing_ok=True)
            attempts.append({"order": order, "status": rec["status"],
                             "in_basket1": (rec.get("validation") or {}).get("aim_in_basket1")})
            if rec["status"] == "ok":
                break
        rec["attempts"] = attempts
        rec["s"] = round(time.perf_counter() - t0, 1)
        index[str(seed)] = rec
        index_p.write_text(json.dumps(jsonable(index), indent=1))
        print(f"[demo] {seed}: {rec['status']} {json.dumps(jsonable(rec.get('validation', {})))[:300]} ({rec['s']}s)",
              flush=True)
    run.close()


if __name__ == "__main__":
    main()
