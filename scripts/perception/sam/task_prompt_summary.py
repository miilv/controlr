"""Summarize task_prompt_eval JSONL without GPU packages.

python scripts/perception/sam/task_prompt_summary.py ARM.jsonl ... --out DIR
Writes frame/instance evidence CSV and a per-task/per-camera summary CSV.
Repeated checkpoints have separately named union arms; shared v2b rows deduplicate.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups = defaultdict(list)
    objects = []
    frames = []
    seen = set()
    for path in a.inputs:
        data = [json.loads(x) for x in open(path) if x.strip()]
        model = next(r["arm"] for r in data if r["arm"] not in ("union", "v2b"))
        for r in data:
            if r["layout_id"] >= 100025:
                continue
            arm = f"union_{model}" if r["arm"] == "union" else r["arm"]
            uid = (r["run"], r["frame"], r["cam"], arm)
            if uid in seen:
                continue
            seen.add(uid)
            prefix = {
                "task": r["task"],
                "layout_id": r["layout_id"],
                "run": r["run"],
                "gt_source": r.get("gt_source", "settled"),
                "cam": r["cam"],
                "frame": r["frame"],
                "arm": arm,
            }
            obs = [
                o
                for o in r["rows"]
                if not o["clutter"] and o["label"] != "camera_stand"
            ]
            # Camera stand is a scene fixture, never a task object. Other configured
            # supports/baskets stay in the denominator even if absent from the prompt bank.
            primary = [o for o in obs if o["px"] >= 30]
            groups[(r["task"], r["frame"], r["cam"], arm)].append((r, primary))
            frames.append(
                {
                    **prefix,
                    "n": len(primary),
                    "n_pred": r["n_pred"],
                    "phantoms": r["phantoms"],
                    "latency_s": r["latency_s"],
                }
            )
            objects.extend({**prefix, **o} for o in obs)
    summary = []
    for (task, phase, cam, arm), items in sorted(groups.items()):
        obs = [o for _, os_ in items for o in os_]
        n = len(obs)
        perlayout = [
            sum(o["iou"] >= 0.5 for o in os_) / len(os_) for _, os_ in items if os_
        ]
        if not n:
            continue
        summary.append(
            {
                "task": task,
                "frame": phase,
                "cam": cam,
                "arm": arm,
                "layouts": len(items),
                "n": n,
                "recall": sum(o["iou"] >= 0.5 for o in obs) / n,
                "mean_iou": statistics.mean(o["iou"] for o in obs),
                "category_recall": sum(o["cat_ok"] and o["iou"] >= 0.5 for o in obs)
                / n,
                "asset_recall": sum(o["class_ok"] and o["iou"] >= 0.5 for o in obs) / n
                if arm == "v2b" or arm.startswith("union")
                else "",
                "recall_layout_min": min(perlayout),
                "recall_layout_max": max(perlayout),
                "phantoms": sum(r["phantoms"] for r, _ in items),
                "phantoms_per_frame": statistics.mean(r["phantoms"] for r, _ in items),
                "median_s": statistics.median(r["latency_s"] for r, _ in items),
                "gt_source": items[0][0].get("gt_source", "settled"),
            }
        )
    by_class = defaultdict(list)
    for r in objects:
        by_class[
            (r["task"], r["frame"], r["cam"], r["key"], r["arm"], r["px"] < 30)
        ].append(r)
    class_summary = []
    for (task, phase, cam, key, arm, tiny), rs in sorted(by_class.items()):
        class_summary.append(
            {
                "task": task,
                "frame": phase,
                "cam": cam,
                "key": key,
                "arm": arm,
                "tiny_1_29px": tiny,
                "n": len(rs),
                "recall": sum(r["iou"] >= 0.5 for r in rs) / len(rs),
                "mean_iou": statistics.mean(r["iou"] for r in rs),
                "category_recall": sum(r["iou"] >= 0.5 and r["cat_ok"] for r in rs)
                / len(rs),
                "asset_recall": sum(r["iou"] >= 0.5 and r["class_ok"] for r in rs)
                / len(rs)
                if arm == "v2b" or arm.startswith("union")
                else "",
            }
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("summary", summary),
        ("class_summary", class_summary),
        ("objects", objects),
        ("frames", frames),
    ]:
        with (out / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
