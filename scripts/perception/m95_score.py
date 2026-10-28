"""Stream M: the grasp-time perception metric over saved integ runs (PERCEPTION_95.md; dojo.perception.grasp_metric).

    python3 scripts/perception/m95_score.py jobs RUNS_GLOB... --layouts-root ~/dojo-data/simlab/layouts > JOBS.jsonl
    scripts/simlab/rdpy.sh -u scripts/perception/m95_replay.py --task T --jobs JOBS.jsonl --out-dir REPLAYS
    python scripts/perception/m95_score.py score RUNS_GLOB... --replays REPLAYS --out OUT [--marks 429.json]
    python scripts/perception/m95_score.py table OUT/events.jsonl [--by task|run]

``jobs``: every perception-mode episode row with its report and actions (layout ids 100000-09 / 100020-49 only; a row's
layout path from another box is mapped to ``--layouts-root/<task>/<file>``). ``score``: the runtime's own belief at
each grasp / place (``grasp_metric.episode_events``) against the replay's GT at that step; writes ``events.jsonl``
(one scored event per line) and ``episodes.jsonl`` (per episode: events, clean counts, the replay's fidelity: final
score vs the episode's, and the dev GT probe's closes vs the replay's). ``table``: per task, events and clean % at
R = 5 / 10 / 15 cm, required-only and target-only, and the top failure reasons.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

SLICE = [(100000, 100009), (100020, 100049)]


def on_slice(lid) -> bool:
    try:
        i = int(lid)
    except (TypeError, ValueError):
        return False
    return any(a <= i <= b for a, b in SLICE)


def episodes(run_globs: list[str]):
    """(run dir, row, report path, actions path) of every perception-mode episode with both files."""
    seen = set()
    for pat in run_globs:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d) or d in seen:
                continue
            seen.add(d)
            for jf in sorted(glob.glob(os.path.join(d, "*_episodes.jsonl"))):
                tag = os.path.basename(jf)[: -len("_episodes.jsonl")]
                for ln in open(jf):
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if r.get("mode") != "perception" or "layout_id" not in r:
                        continue
                    rp = os.path.join(d, f"{tag}_{r['layout_id']}_report.json.gz")
                    ap = os.path.join(d, f"{tag}_{r['layout_id']}_actions.json.gz")
                    if os.path.exists(rp) and os.path.exists(ap):
                        yield d, r, rp, ap


def cmd_jobs(a) -> None:
    n = 0
    for d, r, rp, ap in episodes(a.runs):
        if not on_slice(r["layout_id"]):
            continue
        lay = r["layout"]
        if a.layouts_root:
            alt = os.path.join(os.path.expanduser(a.layouts_root), r["task"], os.path.basename(lay))
            if not os.path.exists(lay) or not lay.startswith(os.path.expanduser("~")):
                lay = alt
        if not os.path.exists(lay):
            print(f"no layout {lay}", file=sys.stderr)
            continue
        name = f"{os.path.basename(d)}_{r['layout_id']}"
        print(json.dumps({"name": name, "task": r["task"], "layout": lay, "actions": ap}))
        n += 1
    print(f"{n} jobs", file=sys.stderr)


def _probe_check(row: dict, frames: dict, actions: list, kin) -> dict:
    """The episode's dev GT probe (GT object nearest the TCP at each close, captured live) vs the replay's GT at the
    same step: does the replay reproduce what the episode's physics did?"""
    import numpy as np
    closes = ((row.get("dev_gt_probe") or {}).get("closes")) or []
    agree, n, worst = 0, 0, 0.0
    for c in closes:
        if c.get("label") is None or c.get("dist_mm") is None:
            continue
        f = frames.get(str(int(c["step"]) - 1))
        if f is None:
            continue
        a = actions[int(c["step"]) - 1]
        p = kin.tcp(c["arm"], np.asarray(a[f"{c['arm']}_arm_joint_state"], float).reshape(-1)[:6])[:3, 3]
        o = next((x for x in f["objects"] if x["label"] == c["label"]), None)
        if o is None:
            continue
        d = float(np.linalg.norm(np.asarray(o["pos"], float) - p)) * 1000
        n += 1
        worst = max(worst, abs(d - float(c["dist_mm"])))
        agree += abs(d - float(c["dist_mm"])) <= 2.0
    return {"probe_closes": n, "probe_agree_2mm": agree, "probe_worst_mm": round(worst, 1)}


def held_drift(frames: dict, track: dict, arm: str, k_grasp: int, gt_label: str, k_release: int) -> float | None:
    """mm: how far the GT object ``gt_label`` moved in the TCP frame between its grasp and the release (an object that
    stayed on the table, slipped or fell moved a lot; one carried in the jaws, a few mm)."""
    import numpy as np
    fg, fr = frames.get(str(k_grasp - 1)), frames.get(str(k_release - 1))
    if fg is None or fr is None:
        return None
    og = next((o for o in fg["objects"] if o["label"] == gt_label), None)
    orr = next((o for o in fr["objects"] if o["label"] == gt_label), None)
    if og is None or orr is None:
        return None
    Tg, Tr = np.asarray(track[arm][k_grasp], float), np.asarray(track[arm][k_release], float)
    pg = Tg[:3, :3].T @ (np.asarray(og["pos"], float) - Tg[:3, 3])
    pr = Tr[:3, :3].T @ (np.asarray(orr["pos"], float) - Tr[:3, 3])
    return float(np.linalg.norm(pr - pg)) * 1000


def cmd_score(a) -> None:
    from dojo.perception.grasp_metric import Footprints, Groups, episode_events, score_event, tcp_track
    from dojo.skills.kinematics import X5Kin

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    marks = json.load(open(a.marks)) if a.marks else {}
    kin = X5Kin()
    fp = Footprints()
    groups: dict = {}
    fe, fo = open(out / "events.jsonl", "w"), open(out / "episodes.jsonl", "w")
    n_ep = 0
    for d, r, rp, ap in episodes(a.runs):
        if not on_slice(r["layout_id"]):
            continue
        run = os.path.basename(d)
        name = f"{run}_{r['layout_id']}"
        rpath = Path(a.replays) / f"{name}.json.gz"
        if not rpath.exists():
            continue
        rep = json.load(gzip.open(rp, "rt"))
        acts = json.load(gzip.open(ap, "rt"))["actions"]
        rp_ = json.load(gzip.open(rpath, "rt"))
        frames = rp_["frames"]
        task = r["task"]
        g = groups.setdefault(task, Groups(task))
        mark = marks.get(f"{run}/{r['layout_id']}") or {}
        base = {"run": run, "task": task, "layout_id": r["layout_id"], "success": r.get("success"), "score": r.get("score"),
                "category": r.get("category"), "preset": r.get("preset"), "renderer": r.get("renderer", "isaac60"),
                "head": (r.get("code") or {}).get("head"), "agent_unavailable": bool(mark.get("excluded"))}
        ev_rows, other = [], Counter()
        moving = any(e.get("kind") == "track_moving" for e in rep.get("events") or [])
        track = tcp_track(acts, kin)
        last_grasp: dict = {}
        for ev, belief in episode_events(rep, acts, kin):
            if ev.kind not in ("grasp", "place"):
                other[ev.kind] += 1
                continue
            f = frames.get(str(ev.step - 1))
            if f is None:
                other["no_replay_frame"] += 1
                continue
            if ev.kind == "place" and ev.arm in last_grasp:
                d = held_drift(frames, track, ev.arm, *last_grasp[ev.arm], ev.step)
                if d is not None:
                    ev.meta["held_drift_mm"] = round(d, 1)
            s = score_event(ev, belief, f["objects"], task, footprints=fp, groups=g)
            if ev.kind == "grasp" and s["required"][0].get("gt_label"):
                last_grasp[ev.arm] = (ev.step, s["required"][0]["gt_label"])
            s.update(base)
            s["moving"] = moving
            ev_rows.append(s)
            fe.write(json.dumps(s) + "\n")
        fid = _probe_check(r, frames, acts, kin)
        epi = {**base, "events": len(ev_rows), "clean_R10": sum(s["clean_r"]["10"] for s in ev_rows),
               "clean_R5": sum(s["clean_r"]["5"] for s in ev_rows), "clean_R15": sum(s["clean_r"]["15"] for s in ev_rows),
               "clean_required": sum(s["clean_required"] for s in ev_rows), "other": dict(other),
               "replay_score": rp_.get("score"), "score_match": abs(float(rp_.get("score") or 0) - float(r.get("score") or 0)) < 1e-6,
               "replay_ended_at": rp_.get("ended_at"), "n_actions": len(acts), **fid}
        fo.write(json.dumps(epi) + "\n")
        n_ep += 1
        print(f"{name}: {len(ev_rows)} events, clean@10 {epi['clean_R10']}, score match {epi['score_match']}, "
              f"probe {fid['probe_agree_2mm']}/{fid['probe_closes']}", flush=True)
    fe.close()
    fo.close()
    if fp.missing:
        print(f"footprints: no mesh for {sorted(fp.missing)} (disc)", file=sys.stderr)
    print(f"{n_ep} episodes scored", file=sys.stderr)


def _reason_class(reason: str) -> str:
    w = reason.split()
    if w[0] in ("target", "held", "reference"):
        return f"{w[0]} {w[2]}"
    if w[0] == "neighbour":
        return "neighbour not found" if "not found" in reason else "neighbour off"
    return w[0]


def table(rows: list[dict], by: str = "task", radii=("5", "10", "15")) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(by)].append(r)
    lines = [f"| {by} | episodes | events (grasp / place) | clean R5 | clean R10 | clean R15 | required only | target only "
             "| top failures at R10 (class: n; examples) |", "|---|---|---|---|---|---|---|---|---|"]
    tot = []
    for k in sorted(groups, key=str):
        rs = groups[k]
        tot += rs
        lines.append(_row(str(k), rs, radii))
    lines.append(_row("**all**", tot, radii))
    return "\n".join(lines)


def _row(name: str, rs: list[dict], radii) -> str:
    n = len(rs)
    eps = len({(r["run"], r["layout_id"]) for r in rs})
    pct = lambda c: f"{100 * c / n:.0f} % ({c})" if n else "-"   # noqa: E731
    cls, ex = Counter(), defaultdict(Counter)
    for r in rs:
        for why in r["reasons"]["10"]:
            c = _reason_class(why)
            cls[c] += 1
            ex[c][why.split()[1]] += 1
    top = "; ".join(f"{c}: {m} ({', '.join(f'{lab} {v}' for lab, v in ex[c].most_common(3))})" for c, m in cls.most_common(4))
    return (f"| {name} | {eps} | {n} ({sum(r['kind'] == 'grasp' for r in rs)} / {sum(r['kind'] == 'place' for r in rs)}) | "
            + " | ".join(pct(sum(bool(r["clean_r"][t]) for r in rs)) for t in radii)
            + f" | {pct(sum(r['clean_required'] for r in rs))} | {pct(sum(r['clean_target'] for r in rs))} | {top or '-'} |")


def cmd_table(a) -> None:
    rows = [json.loads(ln) for f in a.events for ln in open(f) if ln.strip()]
    if a.exclude_unavailable:
        rows = [r for r in rows if not r.get("agent_unavailable")]
    if a.where:
        k, v = a.where.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]
    print(table(rows, a.by))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("runs", nargs="+")
    j.add_argument("--layouts-root", default="~/dojo-data/simlab/layouts")
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--replays", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--marks", default=None, help="c4_429_marks.json: episodes whose agentic route was unavailable")
    t = sub.add_parser("table")
    t.add_argument("events", nargs="+")
    t.add_argument("--by", default="task")
    t.add_argument("--where", default=None, help="KEY=VALUE filter")
    t.add_argument("--exclude-unavailable", action="store_true")
    a = ap.parse_args(argv)
    {"jobs": cmd_jobs, "score": cmd_score, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
