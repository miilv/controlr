"""Twin service for the proxy bundle: a physics-only ``Twin`` (num_envs <= 16) on authenticated localhost.

Started by ``scripts/simlab/proxy_eval.sh --bundle`` next to the proxy episode env and the render service
(one scope, one GPU lock). A policy calls it through ``TwinClient`` (env: SIMLAB_TWIN_PORT,
SIMLAB_TWIN_AUTHKEY, set by the bundle script):

    tc = TwinClient()
    tc.load_from_estimate(layout_record, robot={"left_q": [...], "right_q": [...], "left_grip": 1, "right_grip": 1})
    outs = tc.verify({"layout": layout_record, "robot": {...}}, candidates, perturb={"pos_mm": 2, "yaw_deg": 1}, k=4)
    sid = tc.snapshot();  tc.verify({"snapshot_id": sid}, ...)        # snapshots stay in the service

Ops (pickled dicts): load_from_estimate, verify, snapshot, drop_snapshot, info, close. Candidates and
outcomes: see ``Twin.verify``.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from multiprocessing.connection import Client, Listener
from pathlib import Path


class TwinClient:
    def __init__(self, port: int | None = None, authkey: bytes | None = None, timeout_s: float = 600.0):
        port = port or int(os.environ["SIMLAB_TWIN_PORT"])
        authkey = authkey or bytes.fromhex(os.environ["SIMLAB_TWIN_AUTHKEY"])
        t0 = time.perf_counter()
        while True:
            try:
                self.conn = Client(("127.0.0.1", port), authkey=authkey)
                break
            except (ConnectionRefusedError, OSError):
                if time.perf_counter() - t0 > timeout_s:
                    raise
                time.sleep(2)

    def _call(self, req: dict):
        self.conn.send(req)
        res = self.conn.recv()
        if not res.get("ok"):
            raise RuntimeError(f"twin service: {res.get('error')}\n{res.get('trace', '')}")
        return res

    def info(self) -> dict:
        return self._call({"op": "info"})["info"]

    def load_from_estimate(self, layout: dict, robot: dict | None = None, settle_steps: int = 50) -> dict:
        return self._call({"op": "load_from_estimate", "layout": layout, "robot": robot,
                           "settle_steps": settle_steps})["result"]

    def verify(self, world: dict, candidates: list, perturb: dict | None = None, k: int = 1,
               record_traj: bool = False, seed: int = 0, keep=None) -> list[dict]:
        """``keep=(candidate, sample)`` | [(c, s), ...] | "nominal" (every (c, 0)): each kept rollout's outcome
        carries its own ``snapshot_id`` (drop the unused ones with ``drop_snapshot``);
        ``verify({"snapshot_id": sid}, ...)`` then starts EVERY env from it (chain through a grasp)."""
        return self._call({"op": "verify", "world": world, "candidates": candidates, "perturb": perturb, "k": k,
                           "record_traj": record_traj, "seed": seed, "keep": keep})["outcomes"]

    def snapshot(self) -> int:
        return self._call({"op": "snapshot"})["snapshot_id"]

    def drop_snapshot(self, sid: int) -> None:
        self._call({"op": "drop_snapshot", "snapshot_id": sid})

    def close(self, stop_service: bool = False) -> None:
        try:
            if stop_service:
                self._call({"op": "close"})
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


def serve(task: str, num_envs: int, port: int, authkey: bytes, ready: str | None = None) -> None:
    from dojo.simlab import paths
    from dojo.simlab.twin import Twin
    if num_envs > 16:
        raise SystemExit("bundle twin: num_envs <= 16")
    tw = Twin(task, num_envs=num_envs, physics_only=True, need_planner=False,
              workdir=str(paths.DATA / "simlab" / "work" / f"twin_service_{task}_{os.getpid()}"))
    snaps: dict[int, dict] = {}
    stats = {"calls": 0, "verify_s": 0.0, "rollouts": 0}
    with Listener(("127.0.0.1", port), authkey=authkey) as lst:
        if ready:
            Path(ready).write_text(str(os.getpid()))
        print(f"[twin_service] {task} x{tw.num_envs} on 127.0.0.1:{port}", flush=True)
        stop = False
        while not stop:
            conn = lst.accept()
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                t0 = time.perf_counter()
                stats["calls"] += 1
                try:
                    op = req.get("op")
                    if op == "info":
                        conn.send({"ok": True, "info": {"task": task, "num_envs": tw.num_envs, **stats}})
                    elif op == "load_from_estimate":
                        r = tw.load_from_estimate(req["layout"], req.get("robot"), int(req.get("settle_steps", 50)))
                        conn.send({"ok": True, "result": r})
                    elif op == "verify":
                        world = dict(req["world"])
                        if "snapshot_id" in world:
                            world = {"snapshot": snaps[int(world["snapshot_id"])]}
                        outs = tw.verify(world, req["candidates"], perturb=req.get("perturb"), k=int(req.get("k", 1)),
                                         record_traj=bool(req.get("record_traj")), seed=int(req.get("seed", 0)),
                                         keep=req.get("keep"))
                        for o in outs:                     # one snapshot per kept rollout
                            if o.get("kept"):
                                sid = max(snaps, default=0) + 1
                                snaps[sid] = tw.kept[(o["candidate"], o["sample"])]
                                o["snapshot_id"] = sid
                        stats["verify_s"] += time.perf_counter() - t0
                        stats["rollouts"] += len(outs)
                        conn.send({"ok": True, "outcomes": outs, "s": time.perf_counter() - t0})
                    elif op == "snapshot":
                        sid = max(snaps, default=0) + 1
                        snaps[sid] = tw.snapshot()
                        conn.send({"ok": True, "snapshot_id": sid})
                    elif op == "drop_snapshot":
                        snaps.pop(int(req["snapshot_id"]), None)
                        conn.send({"ok": True})
                    elif op == "close":
                        conn.send({"ok": True})
                        stop = True
                        break
                    else:
                        conn.send({"ok": False, "error": f"unknown op {op!r}"})
                except Exception as e:  # noqa: BLE001 (report, keep serving)
                    conn.send({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
            conn.close()
    print(f"[twin_service] stats {stats}", flush=True)
    tw.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    a = ap.parse_args(argv)
    serve(a.task, a.num_envs, int(os.environ["SIMLAB_TWIN_PORT"]), bytes.fromhex(os.environ["SIMLAB_TWIN_AUTHKEY"]),
          os.environ.get("SIMLAB_TWIN_READY"))


if __name__ == "__main__":
    main()
