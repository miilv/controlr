"""Watch-read G1 data: frame sequences of our dev layouts while the scene's own actors move (the robot holds still).

copy_demonstration (imitate_sorting_sequence): the demonstrator puts objects into its basket (our dev demos,
``dev_demo``, never the eval Traj); play_tic_tac_toe: the scripted opponent places pieces. Per layout: RoboDojo reset
+ settle of the settled dev record, then ``--steps`` hold actions, recording a frame every ``--every`` steps
(physics state + GT; images come from the Isaac 6.0 proxy render, ``render_d0_proxy.py``). Output in D0 layout:

    <out>/<task>/<id>/layout.json, cameras.json, frame_W<step>.json (+ PNGs after the render), stage.usd

    SIMLAB_GPU=0 scripts/simlab/rdpy.sh -m dojo.perception.watch_seq --task play_tic_tac_toe --ids 100000 100001 \
        --steps 600 --every 10 --out ~/dojo-percrnd-data/watch
GT is written for the scorer only; the perception function never reads it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run_task(task: str, ids: list[int], out: Path, steps: int, every: int) -> list[dict]:
    from dojo.simlab import paths
    from dojo.simlab.make_d0 import obs_record
    from dojo.simlab.runner import RoboDojoRunner, jsonable

    lay_dir = paths.DATA / "simlab" / "layouts" / task
    run = RoboDojoRunner(task, physics_only=True, workdir=str(out / "_work" / task))
    summary = []
    for i in ids:
        t0 = time.perf_counter()
        f = lay_dir / f"{i}.settled.json"
        if not f.exists():
            f = lay_dir / f"{i}.json"
        lay = json.loads(f.read_text())
        d = out / task / str(i)
        d.mkdir(parents=True, exist_ok=True)
        (d / "layout.json").write_text(json.dumps(lay))
        if not run.load(lay)["stable"]:
            summary.append({"layout": i, "status": "unstable"})
            continue
        n_frames, ended = 0, False
        for s in range(steps + 1):
            if s % every == 0:
                o = run.obs()
                tag = f"W{s:04d}"
                ob = obs_record(o, tag)
                ob["images"] = {cam: f"frame_{tag}_{cam}.png" for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
                ob["image_shape"] = {}
                rec = {"frame": tag, "step": s, "obs": ob, "gt": run.gt(),
                       "meta": {"renderer": None, "physics": "isaac-5.1 physics-only", "actors": "scene's own (robot holds)"}}
                (d / f"frame_{tag}.json").write_text(json.dumps(jsonable(rec)))
                n_frames += 1
            if s < steps:
                ended = any(run.step(run.hold_action()))
                if ended:
                    break
        run.export_stage(d / "stage.usd")
        if not (d / "cameras.json").exists():
            (d / "cameras.json").write_text(json.dumps(run.camera_spec(), indent=1))
        summary.append({"layout": i, "status": "ok", "frames": n_frames, "ended": ended, "s": round(time.perf_counter() - t0, 1)})
        print(f"[watch] {task}/{i}: {summary[-1]}", flush=True)
        (out / task / "watch_summary.json").write_text(json.dumps(summary, indent=1))
    run.close()
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ids", nargs="+", type=int, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run_task(a.task, a.ids, Path(a.out), a.steps, a.every)


if __name__ == "__main__":
    main()
