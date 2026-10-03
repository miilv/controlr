"""Isaac speed + contact checks without an LLM (compute3, controlr's venv).

Launches (or connects to) an Isaac server with the given physics settings and runs scripted
scenarios under each per-execute variant, printing a table and appending JSON lines:

  expert     the closed-loop scripted yaw expert (tasks.run_scripted_yaw_pick_place) through
             the rotation=yaw envelope: typical moves (hover, turn, approach, close, lift,
             climb, carry, lower, release, retreat), one observe() per action like the loop
  free_air   grasp, then lift + yaw turn + climb with the packet held: must NOT stop
  held_jam   grasp, lift, then carry the packet into the box's near wall: must stop with
             "the held packet pushed against the box wall"
  box_push   the gripper driven sideways into the box's near wall (no envelope): a STOP at the
             box limit; with box_dynamic the box slides a little; never unstable
  mat_push   the TCP driven 30 mm into the mat (no envelope): STOP overshoot vs the 80 N limit

  uv run python scripts/isaac_profile.py --port 7821 --out runs/profile.jsonl \\
      --physics '{"dt": 0.001}' --variants baseline,matrix,direct --scenarios expert,box_push

No LLM calls. Never touches other processes; the server it launched is shut down at exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controlr.config import SafetyConfig  # noqa: E402
from controlr.robot.isaac.tasks import run_scripted_yaw_pick_place  # noqa: E402
from controlr.robot.kinematics import UR3Kinematics, pose_to_matrix  # noqa: E402
from controlr.robot.safety import SafetyEnvelope  # noqa: E402
from controlr.types import Action, ActionMode, EventLevel  # noqa: E402

KIN = UR3Kinematics()
SIM_START_Q = [0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406]          # configs/sim_waffle.yaml

# per-execute variants (robot.params keys); merged over the defaults below
VARIANTS = {
    "baseline": {"reader": "phantom", "direct": False, "substeps": 1, "contact_every": 1},
    "matrix": {"reader": "matrix", "direct": False, "substeps": 1, "contact_every": 1},
    "direct": {"reader": "matrix", "direct": True, "substeps": 1, "contact_every": 1},
    "sub1": {"reader": "matrix", "direct": True, "substeps": 1, "contact_every": 1},
    "sub2": {"reader": "matrix", "direct": True, "substeps": 2, "contact_every": 1},
    "sub5": {"reader": "matrix", "direct": True, "substeps": 5, "contact_every": 1},
    "sub5p": {"reader": "matrix", "direct": True, "substeps": 5, "contact_every": 1, "predict_stop": True},
    "sub10p": {"reader": "matrix", "direct": True, "substeps": 10, "contact_every": 1, "predict_stop": True},
    "ce3": {"reader": "matrix", "direct": True, "substeps": 1, "contact_every": 3, "predict_stop": True},
}
BASE = {"reader": "phantom", "direct": False, "substeps": 1, "contact_every": 1, "predict_stop": False,
        "tcp_speed_m_s": None, "settle": {}}


def yaw_env(robot):
    env = SafetyEnvelope(robot.spec, robot.safety_cfg, rotation="yaw")
    env.reset(robot.reference_state())
    return env


class Recorder:
    def __init__(self, robot):
        self.robot = robot
        self.rows: list[dict] = []
        self.render_s = 0.0
        self.client_s = 0.0

    def execute(self, actions):
        t = time.perf_counter()
        rep = self.robot.execute(actions)
        self.client_s += time.perf_counter() - t
        b = dict(rep.backend or {})
        b["stops"] = [(e.kind, e.message, e.brief) for e in rep.events if e.level is EventLevel.STOP]
        self.rows.append(b)
        return rep

    def observe(self):
        o = self.robot.call("observe")
        self.render_s += float(o.get("render_s", 0.0))
        return o

    def summary(self) -> dict:
        prof_keys = ("targets_s", "simulate_s", "contacts_s", "bookkeeping_s", "settle_checks_s")
        out = {"n_exec": len(self.rows), "sim_s": sum(r.get("sim_s", 0.0) for r in self.rows),
               "exec_wall_s": sum(r.get("wall_s", 0.0) for r in self.rows), "render_s": self.render_s,
               "client_s": self.client_s,
               "steps": sum(r.get("physics_steps", 0) for r in self.rows),
               "settle_s": sum(r.get("settle_s", 0.0) for r in self.rows)}
        for k in prof_keys:
            out[k] = sum((r.get("profile") or {}).get(k, 0.0) for r in self.rows)
        out["other_s"] = out["exec_wall_s"] - sum(out[k] for k in prof_keys)
        out["ratio_exec"] = out["sim_s"] / out["exec_wall_s"] if out["exec_wall_s"] else None
        out["ratio_turn"] = out["sim_s"] / (out["exec_wall_s"] + out["render_s"]) if out["exec_wall_s"] else None
        out["stops"] = [s for r in self.rows for s in r["stops"]]
        out["stop_info"] = [r.get("stop") for r in self.rows if r.get("stop")]
        out["box_shift_mm"] = 1000 * max([abs((r.get("box") or {}).get("shift_m") or 0.0) for r in self.rows] + [0.0])
        # per execute: robot-vs-box / held-packet peaks and the box displacement during it
        out["per_exec"] = [{"box_peak": round(max([v for k, v in (r.get("contacts_peak_n") or {}).items()
                                                    if k.endswith("-box")] + [0.0]), 1),
                            "packet_env_peak": round(max([v for k, v in (r.get("contacts_peak_n") or {}).items()
                                                          if k.startswith("packet-") and k != "packet-table"]
                                                         + [0.0]), 1),
                            "box_moved_mm": round(1000 * float(r.get("box_shift_m") or 0.0), 1),
                            "stop": (r.get("stop") or {}).get("pair")} for r in self.rows]
        return out


SCENE_EXTRA: dict = {}


def scene_params(offset=-15, start_yaw=0, box_dynamic=True):
    return {"start_q": SIM_START_Q, "yaw_offset_deg": offset, "start_yaw_deg": [start_yaw, start_yaw],
            "nominal": True, "box_dynamic": box_dynamic, **SCENE_EXTRA}


def grasp(robot, rec, offset):
    """Expert phases up to the first lift (hover .. lift0) through the envelope."""
    obs = robot.reset({"name": "waffle_pick_place", "params": scene_params(offset)}, seed=offset + 100)
    env = yaw_env(robot)
    cur = {"st": obs.state}

    class Done(Exception):
        pass

    def act(vals, grip, phase=""):
        if phase in ("unrotate", "lift", "climb", "carry"):
            raise Done
        env.set_obstacles(robot.obstacles())
        acts, _ = env.filter([Action(ActionMode.EE_DELTA, vals, grip)], cur["st"])
        rep = rec.execute(acts)
        rec.observe()
        cur["st"] = rep.state_after
        return pose_to_matrix(rep.state_after.tcp_pos, rep.state_after.tcp_rotvec)

    ep = robot.episode
    w, z = ep["object_quat_wxyz"][0], ep["object_quat_wxyz"][3]
    try:
        run_scripted_yaw_pick_place(ep["object_pos"], 2 * np.arctan2(z, w),
                                    robot.server_info["scene_info"]["bin"]["center"], act,
                                    pose_to_matrix(obs.state.tcp_pos, obs.state.tcp_rotvec))
    except Done:
        pass
    return env, cur


def ee(robot, rec, env, cur, d, dyaw=0.0, check=True):
    env.set_obstacles(robot.obstacles())
    acts, ev = env.filter([Action(ActionMode.EE_DELTA, (*d, 0.0, 0.0, dyaw))], cur["st"])
    if not acts:
        return None, ev
    rep = rec.execute(acts)
    rec.observe()
    cur["st"] = rep.state_after
    return rep, ev


def joint_to(robot, rec, cur, target_pos):
    """Straight-line IK to a TCP target WITHOUT the envelope (contact tests)."""
    q = np.asarray(cur["st"].q, float)
    p0, rv = KIN.fk(q)
    n = max(1, int(np.ceil(np.linalg.norm(np.asarray(target_pos) - p0) / 0.005)))
    path = []
    for k in range(1, n + 1):
        qk = KIN.ik(p0 + (np.asarray(target_pos) - p0) * k / n, rv, q)
        if qk is None:
            break
        path.append(qk)
        q = qk
    a = Action(ActionMode.JOINT_ABS, tuple(path[-1]), q_target=tuple(path[-1]),
               q_path=tuple(tuple(x) for x in path[:-1]) or None)
    rep = rec.execute([a])
    rec.observe()
    cur["st"] = rep.state_after
    return rep


def scenario(name, robot, rec, args):
    if name == "expert":
        ok, goals = [], []
        for off, sy in [(-15, 0), (15, -20), (-40, 20), (30, 0)][:args.expert_scenes]:
            obs = robot.reset({"name": "waffle_pick_place", "params": scene_params(off, sy)}, seed=off + 100)
            env = yaw_env(robot)
            cur = {"st": obs.state}

            def act(vals, grip, phase=""):
                env.set_obstacles(robot.obstacles())
                acts, _ = env.filter([Action(ActionMode.EE_DELTA, vals, grip)], cur["st"])
                rep = rec.execute(acts)
                rec.observe()
                cur["st"] = rep.state_after
                return pose_to_matrix(rep.state_after.tcp_pos, rep.state_after.tcp_rotvec)
            ep = robot.episode
            w, z = ep["object_quat_wxyz"][0], ep["object_quat_wxyz"][3]
            try:
                run_scripted_yaw_pick_place(ep["object_pos"], 2 * np.arctan2(z, w),
                                            robot.server_info["scene_info"]["bin"]["center"], act,
                                            pose_to_matrix(obs.state.tcp_pos, obs.state.tcp_rotvec))
                g = robot.check_goal()
                ok.append(bool(g.success))
                goals.append({"message": g.message, **{k: g.metrics.get(k) for k in (
                    "inside_bin", "over_bin", "unloaded", "settled", "object_pos", "object_tilt_deg", "box_center",
                    "box_yaw", "box_tilt_deg", "box_shift_m")}})
                if not g.success:
                    print("   expert goal:", goals[-1], flush=True)
            except Exception as exc:  # noqa: BLE001 - recorded
                ok.append(False)
                goals.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
                print("   expert failed:", type(exc).__name__, str(exc)[:300], flush=True)
        return {"success": ok, "goals": goals}
    if name == "free_air":
        env, cur = grasp(robot, rec, -15)
        held0 = bool(rec.rows[-1].get("holding_pads"))
        for d, dy in [((0, 0, 0.03), 0), ((0, 0, 0), 25), ((0, 0, 0.05), -25), ((0, 0.04, 0.05), 0),
                      ((0.03, 0.03, 0.03), 20), ((0, 0, 0), -20)]:
            ee(robot, rec, env, cur, d, np.radians(dy))
        return {"held_after_grasp": held0, "held_end": bool(rec.rows[-1].get("holding_pads")),
                "stopped": bool(rec.summary()["stops"])}
    if name == "held_jam":
        env, cur = grasp(robot, rec, -15)
        ee(robot, rec, env, cur, (0, 0, 0.03))
        ee(robot, rec, env, cur, (0, 0, 0), np.radians(-20))
        # carry toward the box with the packet hanging below the rim height
        p = np.asarray(cur["st"].tcp_pos)
        joint_to(robot, rec, cur, (p[0], p[1], 0.20))
        for _ in range(4):
            rep = joint_to(robot, rec, cur, np.asarray(cur["st"].tcp_pos) + [0, 0.06, 0])
            if rep.stopped:
                break
        return {"stopped": bool(rec.summary()["stops"])}
    if name in ("box_push", "box_push_static"):
        obs = robot.reset({"name": "waffle_pick_place",
                           "params": scene_params(-15, box_dynamic=(name == "box_push"))}, seed=1)
        cur = {"st": obs.state}
        p = np.asarray(cur["st"].tcp_pos)
        joint_to(robot, rec, cur, (p[0], -0.20, 0.12))
        for _ in range(4):
            rep = joint_to(robot, rec, cur, np.asarray(cur["st"].tcp_pos) + [0, 0.05, 0])
            if rep.stopped:
                break
        return {"stopped": bool(rec.summary()["stops"])}
    if name == "mat_push":
        obs = robot.reset({"name": "waffle_pick_place", "params": scene_params(-15)}, seed=2)
        cur = {"st": obs.state}
        p = np.asarray(cur["st"].tcp_pos)
        joint_to(robot, rec, cur, (p[0] + 0.06, p[1] - 0.05, 0.10))
        joint_to(robot, rec, cur, (p[0] + 0.06, p[1] - 0.05, robot.spec.table_z - 0.03))
        return {"stopped": bool(rec.summary()["stops"])}
    if name == "frames":
        # the rendered frames after a fixed sequence (USD write-back on vs off must look the same)
        from PIL import Image
        out = Path(args.out).parent / f"frames_{args.label}"
        out.mkdir(parents=True, exist_ok=True)
        obs = robot.reset({"name": "waffle_pick_place", "params": scene_params(-15)}, seed=3)
        env = yaw_env(robot)
        cur = {"st": obs.state}
        Image.fromarray(np.asarray(robot.call("observe")["rgb"], np.uint8)).save(out / "0.png")
        for k, (d, dy) in enumerate([((0.05, 0.0, -0.05), 20), ((-0.04, 0.06, 0.0), -20), ((0, 0, 0.06), 0)], 1):
            ee(robot, rec, env, cur, d, np.radians(dy))
            Image.fromarray(np.asarray(robot.call("observe")["rgb"], np.uint8)).save(out / f"{k}.png")
        return {"frames": str(out)}
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("CONTROLR_ISAAC_TEST_PORT", "7821")))
    ap.add_argument("--physics", default="{}", help="JSON robot.params.physics (server start args)")
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--extra", default="{}", help="JSON merged into every variant (e.g. tcp_speed_m_s, settle)")
    ap.add_argument("--scenarios", default="expert")
    ap.add_argument("--plan", default="", help="variant:scen+scen,variant:scen ... (overrides --variants/--scenarios)")
    ap.add_argument("--safety", default="{}", help="JSON SafetyConfig overrides (e.g. box_force_stop_n)")
    ap.add_argument("--scene", default="{}", help="JSON task params merged into every scene (e.g. box_dynamic)")
    ap.add_argument("--expert-scenes", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(ROOT / "runs" / "isaac_profile.jsonl"))
    args = ap.parse_args()

    from controlr.robot.isaac.client import IsaacRobot
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    physics = json.loads(args.physics)
    extra = json.loads(args.extra)
    SCENE_EXTRA.update(json.loads(args.scene))
    robot = IsaacRobot({"port": args.port, "launch": True, "startup_timeout_s": 900, "physics": physics,
                        "log_path": str(Path(args.out).with_suffix(".server.log"))},
                       safety=SafetyConfig(**json.loads(args.safety)), rotation="yaw", tactile=False)
    print("server physics:", robot.server_info.get("physics"), flush=True)
    try:
        plan = ([(v.split(":")[0], v.split(":")[1].split("+")) for v in args.plan.split(",")] if args.plan else
                [(v, args.scenarios.split(",")) for v in args.variants.split(",")])
        for vname, scens in plan:
            for sname in scens:
                v = {**BASE, **VARIANTS[vname], **extra}
                robot.p.update(v)
                rec = Recorder(robot)
                t0 = time.perf_counter()
                res = scenario(sname, robot, rec, args)
                s = rec.summary()
                row = {"label": args.label, "physics": robot.server_info.get("physics"), "variant": vname,
                       "safety": json.loads(args.safety),
                       "settings": v, "scenario": sname, "wall_total_s": time.perf_counter() - t0, **s, **res}
                with open(args.out, "a") as f:
                    f.write(json.dumps(row, default=float) + "\n")
                print(f"{args.label:10s} {vname:9s} {sname:9s} sim {s['sim_s']:6.1f}s exec {s['exec_wall_s']:6.1f}s "
                      f"render {s['render_s']:5.1f}s ratio {s['ratio_exec'] or 0:.2f}/{s['ratio_turn'] or 0:.2f} | "
                      f"tgt {s['targets_s']:.1f} sim {s['simulate_s']:.1f} cont {s['contacts_s']:.1f} "
                      f"book {s['bookkeeping_s']:.1f} settle {s['settle_checks_s']:.1f} other {s['other_s']:.1f} | "
                      f"{res} stops {[(x.get('kind'), x.get('pair'), round(x.get('force', 0)), round(x.get('threshold', 0))) for x in s['stop_info']]} "
                      f"box {s['box_shift_mm']:.0f} mm", flush=True)
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
