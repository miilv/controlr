"""PROXY dev-eval on compute3: physics-only Isaac 5.1 RoboDojo episodes + Isaac 6.0 render on demand.

NOT the official harness (that is ``scripts/simlab/dev_eval.sh``: RoboDojo's own eval client, 5.1
rendering). Here every episode still goes through RoboDojo's own EvalEnv (reset + settle, ``take_action``
step accounting, ``step_lim``, ``run_reward()`` / ``get_score()`` latches), but there is no 5.1 renderer:
the policy gets RGB frames only when it asks (``observe()``: episode start, after each segment, on review),
rendered by the warm 6.0 service (``render_service.py``) from RoboDojo's own stage at the current state.
Results are booked like RoboDojo's ``_result.json`` and marked ``"proxy": ...``.

Policy interface (the integration stream plugs in its own; ``--policy module:Class``)::

    class Policy:
        def reset(self, info: dict) -> None          # task, layout_id, instruction, step_lim
        def next(self, obs: dict) -> dict             # {"actions": [joint action dicts]} | {"done": True}

``obs`` = the official contract: ``images`` (3 x 480x640 RGB), ``state`` (joints, flange poses, grippers),
``instruction``, plus ``steps_used`` / ``step_lim``.

    scripts/simlab/proxy_eval.sh --task stack_blocks --layouts L1.json ... --policy replay:TRAJ.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.runner import RoboDojoRunner, jsonable

PROXY_TAG = "proxy: isaac-5.1 physics-only + isaac-6.0 render on demand (not the official renderer)"


class RenderClient:
    def __init__(self, port: int | None = None, authkey: bytes | None = None, timeout_s: float = 300.0):
        port = port or int(os.environ["SIMLAB_RENDER_PORT"])
        authkey = authkey or bytes.fromhex(os.environ["SIMLAB_RENDER_AUTHKEY"])
        t0 = time.perf_counter()
        while True:
            try:
                self.conn = Client(("127.0.0.1", port), authkey=authkey)
                break
            except (ConnectionRefusedError, OSError):
                if time.perf_counter() - t0 > timeout_s:
                    raise
                time.sleep(2)
        self.render_s: list[float] = []

    def _call(self, req: dict) -> dict:
        self.conn.send(req)
        res = self.conn.recv()
        if not res.get("ok"):
            raise RuntimeError(f"render service: {res.get('error')}\n{res.get('trace', '')}")
        return res

    def load_stage(self, stage: str, cameras: dict) -> float:
        return self._call({"op": "load_stage", "stage": stage, "cameras": cameras})["s"]

    def render(self, gt: dict, subframes: int = 8) -> dict:
        t0 = time.perf_counter()
        res = self._call({"op": "render", "gt": {"objects": gt["objects"], "robot": gt["robot"],
                                                 "env_origin": gt.get("env_origin", [0, 0, 0])},
                          "subframes": subframes})
        self.render_s.append(time.perf_counter() - t0)
        res["images"] = {n: np.frombuffer(v["data"], np.uint8).reshape(v["shape"]) for n, v in res["images"].items()}
        return res

    def close(self, stop_service: bool = False) -> None:
        try:
            if stop_service:
                self._call({"op": "close"})
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


class ProxyEpisode:
    def __init__(self, task: str, render: RenderClient, workdir: Path):
        self.task = task
        self.r = render
        self.workdir = Path(workdir)
        self.run = RoboDojoRunner(task, physics_only=True, need_planner=False, workdir=str(self.workdir / "work"))
        self.n_obs = 0

    def reset(self, layout: dict) -> dict:
        t0 = time.perf_counter()
        info = self.run.load(layout)
        if not info["stable"]:
            return {"stable": False}
        stage = self.run.export_stage(self.workdir / "stage_current.usd")
        t_load = self.r.load_stage(str(stage), self.run.camera_spec())
        self.n_obs = 0
        return {"stable": True, "task": self.task, "layout_id": (layout.get("_simlab") or {}).get("id"),
                "instruction": info["instruction"][0], "step_lim": int(self.run.env.step_lim),
                "reset_s": time.perf_counter() - t0, "render_stage_s": t_load}

    def observe(self) -> dict:
        o = self.run.obs(0)                    # official state + instruction (no images: physics-only)
        res = self.r.render(self.run.gt(0))
        self.n_obs += 1
        rw = self.run.reward(0)
        return {"images": res["images"], "state": o["state"], "instruction": o["instruction"],
                "steps_used": rw["steps_used"], "step_lim": rw["step_lim"], "renderer": res["renderer"]}

    def step(self, action: dict) -> bool:
        return bool(self.run.step(action)[0])

    def result(self) -> dict:
        rw = self.run.reward(0)
        return {"success": rw["success"], "score": self.run.score(0), "steps_used": rw["steps_used"],
                "step_lim": rw["step_lim"], "observations": self.n_obs}


class ReplayPolicy:
    """Replays a recorded trajectory (motion.Trajectory JSON) segment by segment; asks for an
    observation after every segment. For smoke tests and latency numbers."""

    def __init__(self, traj_path: str):
        d = json.loads(Path(traj_path).read_text())
        self.actions, self.segments = d["actions"], d.get("segments") or [{"steps": len(d["actions"])}]

    def reset(self, info: dict) -> None:
        self.i, self.k = 0, 0

    def next(self, obs: dict) -> dict:
        if self.k >= len(self.segments) or self.i >= len(self.actions):
            return {"done": True}
        n = int(self.segments[self.k]["steps"])
        acts = self.actions[self.i:self.i + n]
        self.i, self.k = self.i + n, self.k + 1
        return {"actions": [{k: np.asarray(v, np.float32) for k, v in a.items()} for a in acts]}


def load_policy(spec: str):
    if spec.startswith("replay:"):
        return ReplayPolicy(spec.split(":", 1)[1])
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def run_eval(task: str, layout_files: list[str], policy, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rc = RenderClient()
    ep = ProxyEpisode(task, rc, out)
    details, t_all = {}, time.perf_counter()
    for i, lf in enumerate(layout_files):
        paths.assert_not_eval(lf)
        lay = json.loads(Path(lf).read_text())
        info = ep.reset(lay)
        if not info["stable"]:
            details[i] = {"layout": lf, "unstable": True}
            continue
        policy.reset(info)
        t0 = time.perf_counter()
        ended, obs_s = False, []
        while not ended:
            t1 = time.perf_counter()
            obs = ep.observe()
            obs_s.append(time.perf_counter() - t1)
            d = policy.next(obs)
            if d.get("done"):
                break
            for a in d["actions"]:
                if ep.step(a):
                    ended = True
                    break
        res = ep.result()
        res.update({"layout": lf, "layout_id": info.get("layout_id"), "wall_s": time.perf_counter() - t0,
                    "reset_s": info["reset_s"], "render_stage_s": info["render_stage_s"],
                    "observe_s_median": float(np.median(obs_s)) if obs_s else None})
        details[i] = res
        print(f"[proxy_eval] {task} {Path(lf).stem}: success={res['success']} score={res['score']:.2f} "
              f"steps={res['steps_used']}/{res['step_lim']} obs={res['observations']} "
              f"observe {res['observe_s_median']:.2f}s (render {np.median(rc.render_s[-len(obs_s):]):.2f}s) "
              f"reset {info['reset_s']:.1f}s + stage {info['render_stage_s']:.1f}s", flush=True)
    ok = [d for d in details.values() if not d.get("unstable")]
    result = {"proxy": PROXY_TAG, "task": task, "eval_time": len(ok),
              "success_rate": float(np.mean([d["success"] for d in ok])) if ok else 0.0,
              "score": float(np.mean([d["score"] for d in ok])) * 100 if ok else 0.0,
              "render_s_median": float(np.median(rc.render_s)) if rc.render_s else None,
              "wall_s": time.perf_counter() - t_all, "details": details}
    (out / "_result.json").write_text(json.dumps(jsonable(result), indent=1))
    rc.close()
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--policy", required=True, help="replay:<traj.json> or module:Class")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = run_eval(a.task, a.layouts, load_policy(a.policy), Path(a.out))
    print(f"[proxy_eval] {a.task}: success_rate {r['success_rate']:.2f} score {r['score']:.1f} over {r['eval_time']} "
          f"episodes; render median {r['render_s_median']:.2f}s ({PROXY_TAG})", flush=True)


if __name__ == "__main__":
    main()
