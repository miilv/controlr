"""Re-run D0 layouts on another machine with the FULL Isaac 5.1 renderer (genuine RoboDojo frames).

Input: a D0 root (``<task>/<id>/layout.json``, ``trajectory.json``, ``frame_<k>.json`` from the source
machine). For every layout: RoboDojo reset + settle from ``layout.json``, then the recorded action list
replayed bit-exactly, with frames saved at the same segment boundaries as D0 (frame 1 after the left-arm
look, frame 2 after the right-arm look, frame 3 after the both-arms move). Each ``frame_<k>.json`` is in
D0 format with ``meta.renderer = "isaac-5.1"`` plus ``compare_to_source``: object-pose differences from
the source machine's GT at the same frame (a cross-GPU replay check: G2b on D0 data).

    scripts/simlab/rdpy.sh -m dojo.simlab.rerender_d0 --src SRC_D0 --out OUT_D0 --task stack_blocks
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.layout import quat_angle_deg
from dojo.simlab.motion import Trajectory
from dojo.simlab.runner import RoboDojoRunner, jsonable, save_png

GROUPS = (("left_look",), ("right_look",), ("both_centre",))


def frame_boundaries(segments: list[dict]) -> list[int]:
    """Env-step counts after which frames 1..3 were taken (a failed move adds no segment)."""
    out, acc = [], 0
    for prefixes in GROUPS:
        acc += sum(int(sg["steps"]) for sg in segments if str(sg["name"]).startswith(prefixes))
        out.append(acc)
    return out


def _keyed(gt: dict) -> dict:
    """Objects by (type, category, index, label, clutter) + occurrence: RoboDojo's instance names carry a
    per-process counter, so they differ between machines that loaded other layouts before."""
    out, seen = {}, {}
    for o in gt["objects"]:
        if o.get("pos") is None:
            continue
        k = (o["type"], o["category"], o["category_idx"], o.get("label"), o.get("clutter"))
        n = seen.get(k, 0)
        seen[k] = n + 1
        out[k + (n,)] = o
    return out


def compare(gt_a: dict, gt_b: dict) -> dict:
    a, b = _keyed(gt_a), _keyed(gt_b)
    by_label = {}
    mm = deg = 0.0
    for name, o in a.items():
        p = b.get(name)
        if p is None:
            continue
        d = float(np.linalg.norm(np.subtract(o["pos"], p["pos"]))) * 1000
        r = quat_angle_deg(o["quat"], p["quat"])
        if o.get("label") and not o.get("clutter"):
            by_label[o["label"]] = {"mm": round(d, 4), "deg": round(r, 4)}
        mm, deg = max(mm, d), max(deg, r)
    qa = {arm: np.asarray(v["joint_pos"][:6]) for arm, v in gt_a["robot"].items()}
    qb = {arm: np.asarray(v["joint_pos"][:6]) for arm, v in gt_b["robot"].items()}
    return {"max_obj_mm": round(mm, 4), "max_obj_deg": round(deg, 4), "task_objects": by_label,
            "max_joint_rad": round(max(float(np.abs(qa[k] - qb[k]).max()) for k in qa if k in qb), 6)}


def save(run: RoboDojoRunner, src_dir: Path, d: Path, k: int, tag: str) -> dict:
    o = run.obs()
    for cam, img in o["images"].items():
        save_png(img, d / f"frame_{k}_{cam}.png")
    gt = run.gt()
    src = json.loads((src_dir / f"frame_{k}.json").read_text()) if (src_dir / f"frame_{k}.json").exists() else None
    rec = {"frame": k,
           "obs": {"instruction": o["instruction"], "state": {kk: v.tolist() for kk, v in o["state"].items()},
                   "images": {cam: f"frame_{k}_{cam}.png" for cam in o["images"]},
                   "image_shape": {cam: list(img.shape) for cam, img in o["images"].items()}},
           "gt": gt, "move": (src or {}).get("move", {}),
           "meta": {"renderer": tag, "physics": "isaac-5.1", "replayed_from": str(src_dir)}}
    if src is not None:
        rec["compare_to_source"] = compare(gt, src["gt"])
    (d / f"frame_{k}.json").write_text(json.dumps(jsonable(rec)))
    return rec.get("compare_to_source", {})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", required=True, help="one task per process (one Kit app)")
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--renderer-tag", default="isaac-5.1")
    a = ap.parse_args(argv)
    src_root, out_root = Path(a.src), Path(a.out)
    paths.assert_not_eval(out_root)
    for task in [a.task]:
        dirs = sorted(p for p in (src_root / task).iterdir() if p.is_dir() and p.name.isdigit()
                      and (not a.ids or int(p.name) in a.ids))
        run = RoboDojoRunner(task, workdir=str(out_root / "work" / task))
        summary = []
        for sd in dirs:
            t0 = time.perf_counter()
            d = out_root / task / sd.name
            d.mkdir(parents=True, exist_ok=True)
            lay = json.loads((sd / "layout.json").read_text())
            (d / "layout.json").write_text(json.dumps(lay, indent=1))
            tj = json.loads((sd / "trajectory.json").read_text())
            traj = Trajectory.from_json(tj)
            info = run.load(lay)
            if not info["stable"]:
                summary.append({"id": sd.name, "status": "unstable"})
                continue
            cmp = [save(run, sd, d, 0, a.renderer_tag)]
            bounds = frame_boundaries(tj.get("segments", []))
            done = 0
            for k, b in enumerate(bounds, start=1):
                while done < b and done < len(traj.actions):
                    run.step(traj.actions[done])
                    done += 1
                cmp.append(save(run, sd, d, k, a.renderer_tag))
            summary.append({"id": sd.name, "status": "ok", "s": round(time.perf_counter() - t0, 1),
                            "max_obj_mm_per_frame": [c.get("max_obj_mm") for c in cmp],
                            "max_joint_rad_per_frame": [c.get("max_joint_rad") for c in cmp]})
            print(f"[rerender] {task}/{sd.name}: obj diff vs source per frame "
                  f"{summary[-1]['max_obj_mm_per_frame']} mm ({summary[-1]['s']}s)", flush=True)
        (out_root / task).mkdir(parents=True, exist_ok=True)
        (out_root / task / "rerender_summary.json").write_text(json.dumps(summary, indent=1))
        run.close()


if __name__ == "__main__":
    main()
