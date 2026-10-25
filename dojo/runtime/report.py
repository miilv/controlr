"""Aggregate integ runs into markdown tables (per task x mode x verify, plus per-episode failures).

    python -m dojo.runtime.report ~/dojo-data/integ/runs/A_stack_off_1 ~/dojo-data/integ/runs/B_stack_off_1 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(dirs: list[str]) -> list[dict]:
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*_episodes.jsonl")):
            for ln in f.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    r["_run"] = Path(d).name
                    rows.append(r)
    return rows


def table(rows: list[dict]) -> str:
    keys = sorted({(r["task"], r["mode"], r["verify"]) for r in rows})
    out = ["| task | mode | verify | n | success | score | steps med / max (lim) | looks | wall s med | worker s med | "
           "verify s med | verified / skipped seg. med | failures |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        rs = [r for r in rows if (r["task"], r["mode"], r["verify"]) == k]
        n = len(rs)
        succ = sum(r["success"] for r in rs)
        cats: dict = {}
        for r in rs:
            if r.get("category"):
                cats[r["category"]] = cats.get(r["category"], 0) + 1
        steps = [r["steps_used"] for r in rs]
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {n} | {succ}/{n} | {np.mean([r['score'] for r in rs]) * 100:.1f} | "
                   f"{int(np.median(steps))} / {max(steps)} ({rs[0]['step_lim']}) | {sum(r.get('n_looks', 0) for r in rs)} | "
                   f"{np.median([r['wall_s'] for r in rs]):.0f} | "
                   f"{np.median([r['worker_s'] or 0 for r in rs]):.0f} | "
                   f"{_vmed(rs, 'wall_s', '.0f')} | {_vmed(rs, 'calls', '.0f')} / {_vmed(rs, 'skipped', '.0f')} | "
                   f"{', '.join(f'{c} {v}' for c, v in sorted(cats.items())) or '-'} |")
    return "\n".join(out)


def _vmed(rows: list[dict], key: str, fmt: str) -> str:
    """Median of a per-episode verify stat (``verify_stats[key]``); '-' when no episode has it."""
    v = [r["verify_stats"].get(key) or 0 for r in rows if isinstance(r.get("verify_stats"), dict)]
    return format(float(np.median(v)), fmt) if v else "-"


def failures(rows: list[dict]) -> str:
    out = ["| task | mode | verify | layout | score | category | detail |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["success"]:
            out.append(f"| {r['task']} | {r['mode']} | {r['verify']} | {r['layout_id']} | {r['score']:.2f} | "
                       f"{r.get('category')} | {str(r.get('why', ''))[:140].replace('|', '/')} |")
    return "\n".join(out)


def perception_table(rows: list[dict]) -> str:
    errs, first, ids, shapes, n, nf = [], [], [], [], 0, 0
    for r in rows:
        for lab, v in (r.get("perception") or {}).items():
            n += 1
            if not v.get("found"):
                nf += 1
                continue
            errs.append(v["xy_mm"])
            ids.append(v["id_ok"])
            shapes.append(v.get("shape_ok", True))
            if "xy_mm_first" in v:
                first.append(v["xy_mm_first"])
    if not n:
        return ""
    e = np.asarray(errs)
    s = (f"objects {n}, not found {nf}, xy med {np.median(e):.1f} mm, p90 {np.percentile(e, 90):.1f} mm, "
         f"> 20 mm {int((e > 20).sum())}, identity {np.mean(ids):.2f}, shape {np.mean(shapes):.2f}")
    if first:
        s += f"; before looks: xy med {np.median(first):.1f} p90 {np.percentile(first, 90):.1f} mm"
    return s


def gap_table(rows: list[dict]) -> str:
    """Mode gt vs mode perception on the SAME layouts (per task x verify): the perception gap."""
    out = ["| task | verify | paired n | A score | B score | gap | A succ | B succ | lost in B (layout: B category) |",
           "|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["task"], r["verify"]) for r in rows})
    for task, ver in keys:
        A = {r["layout_id"]: r for r in rows if (r["task"], r["verify"], r["mode"]) == (task, ver, "gt")}
        B = {r["layout_id"]: r for r in rows if (r["task"], r["verify"], r["mode"]) == (task, ver, "perception")}
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        sa = np.mean([A[i]["score"] for i in ids]) * 100
        sb = np.mean([B[i]["score"] for i in ids]) * 100
        lost = [f"{i}: {B[i].get('category')}" for i in ids if A[i]["success"] and not B[i]["success"]]
        out.append(f"| {task} | {ver} | {len(ids)} | {sa:.1f} | {sb:.1f} | {sa - sb:+.1f} | "
                   f"{sum(A[i]['success'] for i in ids)} | {sum(B[i]['success'] for i in ids)} | {'; '.join(lost) or '-'} |")
    return "\n".join(out)


def arm_table(arms: dict[str, list[dict]]) -> str:
    """Two arms (e.g. refine off / on) on the same layouts: totals plus the paired counts. ``rescued`` = failed
    in the first arm, succeeded in the second; ``lost`` = the reverse."""
    (na, ra), (nb, rb) = list(arms.items())[:2]
    out = [f"| task | paired n | {na} succ | {nb} succ | {na} score | {nb} score | rescued by {nb} | lost by {nb} |",
           "|---|---|---|---|---|---|---|---|"]
    for task in sorted({r["task"] for r in ra + rb}):
        A = {r["layout_id"]: r for r in ra if r["task"] == task}
        B = {r["layout_id"]: r for r in rb if r["task"] == task}
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        resc = [i for i in ids if not A[i]["success"] and B[i]["success"]]
        lost = [i for i in ids if A[i]["success"] and not B[i]["success"]]
        lost_s = ", ".join(f"{i}: {B[i].get('category')}" for i in lost)
        out.append(f"| {task} | {len(ids)} | {sum(A[i]['success'] for i in ids)} | {sum(B[i]['success'] for i in ids)} | "
                   f"{np.mean([A[i]['score'] for i in ids]) * 100:.1f} | {np.mean([B[i]['score'] for i in ids]) * 100:.1f} | "
                   f"{len(resc)} {resc if resc else ''} | {len(lost)} {lost_s} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--arm", action="append", default=[], help="NAME:DIR[,DIR...] (give two for a paired table)")
    a = ap.parse_args(argv)
    if a.arm:
        arms = {spec.split(":", 1)[0]: load(spec.split(":", 1)[1].split(",")) for spec in a.arm}
        print(arm_table(arms))
        print()
        for name, rows in arms.items():
            print(f"[{name}]")
            print(table(rows))
            print()
        if not a.dirs:
            return
    rows = load(a.dirs)
    print(table(rows))
    print()
    print(gap_table(rows))
    print()
    print(failures(rows))
    for mode in ("perception",):
        rs = [r for r in rows if r["mode"] == mode]
        for task in sorted({r["task"] for r in rs}):
            p = perception_table([r for r in rs if r["task"] == task])
            if p:
                print(f"\nperception {task}: {p}")


if __name__ == "__main__":
    main()
