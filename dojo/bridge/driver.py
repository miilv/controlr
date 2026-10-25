"""Bridge driver: one RoboDojo episode of OUR full system behind a robot-harness strict server.

The driver plays the harness role that ``dojo.runtime.eval`` plays in the proxy (and RoboDojo's eval client plays
for an XPolicyLab policy): ``backend.reset()`` -> ``policy.reset(info)`` -> loop ``obs = backend.observe()``,
``policy.next(obs)``, ``backend.execute(actions)`` until the policy is done or the episode ended (RoboDojo's own
success latch or step limit, the reply's ``ended``). The policy is any proxy_eval policy: ``RuntimePolicy``
(perception + skills + twin) or ``AgentPolicy`` (the LLM agent on its program seam), unchanged; its own firewall
(``dojo.runtime.firewall.check_obs``, mode ``perception``) checks every observation a second time.

``info`` holds only the instruction and the step limit: no task name, layout id or seed exists on this path
(the server never sends them). The twin scene comes from the instruction (``twin.twin_scene``).

    python -m dojo.bridge.driver --server-id ID [--policy dojo.agent.policy:AgentPolicy] [--verify bundle --twin spawn]
        [--perception local|server] [--config '{"refine": true, ...}'] [--out DIR] [--close]

Outputs in ``<out>/<server-id>/``: ``bridge.json`` (end reason, steps, timings, call counts, events), ``report.json.gz``
(the policy's episode report), ``actions.json.gz`` (every executed action), ``calls.jsonl`` (one line per server call).
The verdict is never read here: the score is RoboDojo's ``_result.json``, written when the server closes.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .backend import RobotServerBackend
from .client import RobotServerClient
from .twin import TwinProcess, twin_scene


def load_policy(spec: str | None):
    if not spec:
        from dojo.runtime.policy import RuntimePolicy
        return RuntimePolicy
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def run_episode(policy, backend: RobotServerBackend, twin_factory=None, reset_timeout_s: float = 600.0,
                max_next: int = 100_000, calib_fold: int = 0, log=print) -> dict:
    """One episode. ``twin_factory(scene) -> TwinClient`` (verify=bundle): started for the instruction's scene.
    ``calib_fold``: which perception background / colour set to use (``info["dev_calib_fold"]``; any fold is
    legal behind the official client: neither half of our D0 holds an eval layout)."""
    t0 = time.perf_counter()
    ep = backend.reset(reset_timeout_s)
    instruction = ep["instruction"]
    scene = twin_scene(instruction)
    events = []
    twin_s = None
    if getattr(policy.cfg, "verify", "off") == "bundle":
        if twin_factory is None:
            if not os.environ.get("SIMLAB_TWIN_PORT"):
                raise RuntimeError("verify=bundle needs a twin: --twin spawn, or SIMLAB_TWIN_PORT/AUTHKEY of a running one")
        elif scene is None:
            events.append({"kind": "twin_unavailable", "reason": "no scene matches the instruction"})
        else:
            t1 = time.perf_counter()
            policy._tc = twin_factory(scene)
            twin_s = round(time.perf_counter() - t1, 1)
    policy.reset({"instruction": instruction, "step_lim": ep["step_lim"], "dev_calib_fold": int(calib_fold)})
    obs = backend.observe()
    end, n_next, t_policy, t_exec, t_obs = None, 0, 0.0, 0.0, 0.0
    try:
        while n_next < max_next:
            t1 = time.perf_counter()
            d = policy.next(obs)
            t_policy += time.perf_counter() - t1
            n_next += 1
            if d.get("done"):
                end = "policy_done"
                break
            t1 = time.perf_counter()
            r = backend.execute(d.get("actions") or [])
            t_exec += time.perf_counter() - t1
            if r["ended"]:
                end = "ended"
                break
            t1 = time.perf_counter()
            obs = backend.observe()
            t_obs += time.perf_counter() - t1
        else:
            end = "max_next"
    except Exception as e:  # noqa: BLE001 (the episode report is still written; the server scores what happened)
        end = f"driver_error: {type(e).__name__}: {e}"[:400]
        log(f"[bridge] {end}")
    rep = policy.episode_report()
    if getattr(policy, "bridge", None) is not None:
        policy.bridge.cancel()
    out = {"end": end, "instruction": instruction, "scene": scene, "step_lim": ep["step_lim"],
           "steps_used": backend.steps_used, "ended": backend.ended, "next_calls": n_next,
           "wall_s": round(time.perf_counter() - t0, 1), "policy_s": round(t_policy, 1), "exec_s": round(t_exec, 1),
           "observe_s": round(t_obs, 1), "twin_start_s": twin_s, "backend": dict(backend.stats),
           "server_calls": backend.c.n_calls, "server_wall_s": round(backend.c.wall_s, 1), "events": events,
           "policy_error": rep.get("error"), "policy_error_handled": (rep.get("error_handled") or "")[:400] or None,
           "policy": type(policy).__module__ + ":" + type(policy).__name__}
    return {"summary": out, "report": rep, "executed": list(backend.executed)}


def write_outputs(res: dict, out_dir: Path) -> None:
    from dojo.simlab.runner import jsonable
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bridge.json").write_text(json.dumps(jsonable(res["summary"]), indent=1, default=str))
    rep = dict(res["report"])
    rep.pop("look_frames", None)
    with gzip.open(out_dir / "report.json.gz", "wt") as f:
        json.dump(jsonable(rep), f, default=str)
    with gzip.open(out_dir / "actions.json.gz", "wt") as f:
        json.dump({"actions": [{k: np.asarray(v, np.float64).tolist() for k, v in a.items()} for a in res["executed"]]}, f)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server-id", required=True)
    ap.add_argument("--bench", default=None, help="robot-harness $BENCH (socket = $BENCH/run/<id>.sock)")
    ap.add_argument("--sock", default=None, help="explicit socket path (overrides --bench)")
    ap.add_argument("--policy", default=None, help="module:Class (default dojo.runtime.policy:RuntimePolicy)")
    ap.add_argument("--config", default="{}", help="RuntimeConfig fields as JSON (mode is always perception)")
    ap.add_argument("--verify", choices=("off", "bundle"), default="bundle")
    ap.add_argument("--twin", choices=("spawn", "env", "off"), default="spawn",
                    help="spawn: our twin service for the instruction's scene; env: SIMLAB_TWIN_PORT/AUTHKEY; off")
    ap.add_argument("--twin-envs", type=int, default=8)
    ap.add_argument("--hold-gpu-lock", action="store_true", help="the session holds the GPU lock (twin skips rdpy's)")
    ap.add_argument("--perception", choices=("local", "server"), default="local")
    ap.add_argument("--calib-fold", type=int, default=0, help="perception background / colour set (DOJO_BG_FOLDS/fold<k>)")
    ap.add_argument("--out", default="runs/bridge")
    ap.add_argument("--keep-frames", type=int, default=20, help="keep every k-th observation's PNGs (0: first only)")
    ap.add_argument("--wait-episode", type=float, default=900.0)
    ap.add_argument("--close", action="store_true", help="close the server after the episode (RoboDojo writes its result)")
    ap.add_argument("--allow-non-strict", action="store_true", help="dev only: accept a non-strict server")
    a = ap.parse_args(argv)

    from dojo.runtime.policy import RuntimeConfig
    cfg_d = json.loads(a.config)
    if cfg_d.get("mode", "perception") != "perception":
        raise SystemExit("behind a robot server the runtime runs in mode perception only (no GT on this path)")
    verify = a.verify if a.twin != "off" else "off"
    cfg = RuntimeConfig(**{**cfg_d, "mode": "perception", "verify": verify})
    out_dir = Path(a.out) / a.server_id
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DOJO_AGENT_LOG", str(out_dir / "agent"))      # AgentPolicy's transcript / calls / cells
    calls = (out_dir / "calls.jsonl").open("a")
    client = RobotServerClient(a.server_id, sock=a.sock, bench=a.bench,
                               log=lambda r: calls.write(json.dumps(r) + "\n"))
    backend = RobotServerBackend(client, strict=not a.allow_non_strict, frames_dir=out_dir / "frames",
                                 keep_every=a.keep_frames, log=print)
    info = backend.connect()
    print(f"[bridge] server {a.server_id}: {info.get('tier')}; commands {info.get('commands')}", flush=True)
    backend.wait_episode(a.wait_episode)
    pol = load_policy(a.policy)(cfg)
    if a.perception == "server":
        from .perception import ServerPerception
        local = None
        if os.environ.get("DOJO_PERCEPTION_PORT"):
            from dojo.runtime.perception_service import PerceptionClient
            local = PerceptionClient()
        pol._pc = ServerPerception(backend, local=local)
    twin = TwinProcess(a.twin_envs, out_dir, hold_gpu_lock=a.hold_gpu_lock) if (verify == "bundle" and a.twin == "spawn") else None
    try:
        res = run_episode(pol, backend, twin_factory=twin.start if twin else None, calib_fold=a.calib_fold)
        if twin is not None:
            res["summary"]["twin_boot_s"] = twin.boot_s
        res["summary"].update(server_id=a.server_id, perception=a.perception, verify=verify, config=cfg.__dict__)
        write_outputs(res, out_dir)
        s = res["summary"]
        print(f"[bridge] {a.server_id}: end={s['end']} steps={s['steps_used']}/{s['step_lim']} ended={s['ended']} "
              f"wall={s['wall_s']}s (policy {s['policy_s']} exec {s['exec_s']} obs {s['observe_s']}) "
              f"calls={s['server_calls']} -> {out_dir}", flush=True)
    finally:
        if twin is not None:
            twin.stop()
        if a.close:
            print(f"[bridge] close: {json.dumps(backend.close())[:300]}", flush=True)
        calls.close()


if __name__ == "__main__":
    main()
