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


def family(run: str) -> str:
    """A run's round: the name without its task and repeat suffix (B_hang_mugs_breadth2 -> B_*_breadth2)."""
    import re
    for pat, fam in ((r"^c4_", "c4 (genuine 5.1)"), (r"^smoke51", "c4 (genuine 5.1)"), (r"^P3r3_", "P3r3"), (r"^P3n_", "P3n"),
                     (r"^P3_", "P3"), (r"^S79_", "S79"), (r"^TL_", "TL"), (r"^AGB?_", "AG"), (r"_breadth2$", "B_breadth2"),
                     (r"_breadth$", "B_breadth"), (r"^B_plug_heldpose", "B_heldpose")):
        if re.search(pat, run):
            return fam
    return "early (B_stack / B_build / B_tower ...)"


# newest first (the rounds' dates and code: c4 27 Oct 17-22Z submit_v2/v3 genuine 5.1; P3r3 27 Oct 12Z submit_v2 13aaf9a;
# TL 11Z; P3n 08Z eeb94de; P3 05Z 1f7e965; S79 03Z 48600bf; breadth2 26 Oct 21Z ed8ca42; heldpose 16Z; breadth 16Z
# f9b0d75; early rounds 25-26 Oct)
FAMILY_ORDER = ("c4 (genuine 5.1)", "P3r3", "TL", "P3n", "P3", "S79", "B_breadth2", "B_heldpose", "B_breadth", "AG",
                "early (B_stack / B_build / B_tower ...)")


def latest_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per task, the events of its newest round (``FAMILY_ORDER``) vs all older rounds."""
    best = {}
    for r in rows:
        k = FAMILY_ORDER.index(r.get("family")) if r.get("family") in FAMILY_ORDER else len(FAMILY_ORDER)
        best[r["task"]] = min(best.get(r["task"], 99), k)
    new = [r for r in rows if FAMILY_ORDER.index(r["family"]) == best[r["task"]]]
    old = [r for r in rows if FAMILY_ORDER.index(r["family"]) != best[r["task"]]]
    return new, old


def frame_at(frames: dict, k: int, back: int = 3):
    """The replay's frame after ``k`` actions, else the nearest earlier one within ``back`` steps (a replay recorded
    before the grasp rule took a closing run's last command: a ramp's last step); None beyond (re-run the replay)."""
    for j in range(k, max(-1, k - back - 1), -1):
        if str(j) in frames:
            return frames[str(j)]
    return None


def held_drift(frames: dict, track: dict, arm: str, k_grasp: int, gt_label: str, k_release: int) -> float | None:
    """mm: how far the GT object ``gt_label`` moved in the TCP frame between its grasp and the release (an object that
    stayed on the table, slipped or fell moved a lot; one carried in the jaws, a few mm)."""
    import numpy as np
    fg, fr = frame_at(frames, k_grasp - 1), frame_at(frames, k_release - 1)
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


def held_gap(f: dict, track: dict, arm: str, gt_label: str, k: int, fp) -> float | None:
    """mm: how far the TCP is outside the GT object's oriented bbox (its catalogue mesh) at the release: 0 when the
    jaws are on it (held, perhaps turned in the hand), large when the object stayed behind or fell."""
    import numpy as np

    from dojo.perception.grasp_metric import _quat_R, gt_rows_asset
    o = next((x for x in gt_rows_asset(f["objects"]) if x["label"] == gt_label), None)
    if o is None:
        return None
    V = fp.verts(o.get("key"))
    p = np.asarray(track[arm][k], float)[:3, 3]
    loc = _quat_R(o["quat"]).T @ (p - np.asarray(o["pos"], float))
    if V is None:
        return float(np.linalg.norm(loc)) * 1000
    lo, hi = V.min(0), V.max(0)
    return float(np.linalg.norm(np.maximum(0, np.maximum(lo - loc, loc - hi)))) * 1000


def cmd_score(a) -> None:
    from dojo.perception.grasp_metric import Footprints, Groups, episode_events, flatten_record, score_event, tcp_track
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
                "head": (r.get("code") or {}).get("head"), "agent_unavailable": bool(mark.get("excluded")),
                "date": __import__("time").strftime("%m-%d", __import__("time").gmtime(os.path.getmtime(ap))),
                "family": family(run)}
        ev_rows, other = [], Counter()
        moving = any(e.get("kind") == "track_moving" for e in rep.get("events") or [])
        track = tcp_track(acts, kin)
        rec0 = flatten_record((rep.get("records") or {}).get("initial")) if (rep.get("records") or {}).get("initial") else None
        last_grasp: dict = {}
        for ev, belief in episode_events(rep, acts, kin, footprints=fp):
            if ev.kind not in ("grasp", "place"):
                other[ev.kind] += 1
                continue
            f = frame_at(frames, ev.step - 1)
            if f is None:
                other["no_replay_frame"] += 1
                continue
            if ev.kind == "place" and ev.arm in last_grasp:
                d = held_drift(frames, track, ev.arm, *last_grasp[ev.arm], ev.step)
                if d is not None:
                    ev.meta["held_drift_mm"] = round(d, 1)
                gap = held_gap(f, track, ev.arm, last_grasp[ev.arm][1], ev.step, fp)
                if gap is not None:
                    ev.meta["held_gap_mm"] = round(gap, 1)
            s = score_event(ev, belief, f["objects"], task, footprints=fp, groups=g)
            if rec0 is not None:            # the same event against the frame-0 record (held objects: the runtime's)
                b0 = {**rec0, **{lab: o for lab, o in belief.items() if str(o.get("source", "")).endswith("held")}}
                s0 = score_event(ev, b0, f["objects"], task, footprints=fp, groups=g)
                s["frame0"] = {"clean_r": s0["clean_r"], "clean_required": s0["clean_required"],
                               "clean_target": s0["clean_target"], "target_cls": s0["required"][0]["cls"]}
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
               "replay_score": rp_.get("score"),
               "score_match": (None if (rp_.get("stopped_at") or len(acts)) < len(acts) and not rp_.get("ended_at")
                               else abs(float(rp_.get("score") or 0) - float(r.get("score") or 0)) < 1e-6),
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
    lines = [f"| {by} | episodes | events (grasp / place) | clean R5 | clean R10 | clean R15 | R10 w/o clutter | required only "
             "| target only | top failures at R10 (class: n; examples) |", "|---|---|---|---|---|---|---|---|---|---|"]
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
            + f" | {pct(sum(bool((r.get('clean_r_noclutter') or r['clean_r'])['10']) for r in rs))}"
            + f" | {pct(sum(r['clean_required'] for r in rs))} | {pct(sum(r['clean_target'] for r in rs))} | {top or '-'} |")


def cmd_table(a) -> None:
    rows = [json.loads(ln) for f in a.events for ln in open(f) if ln.strip()]
    if a.exclude_unavailable:
        rows = [r for r in rows if not r.get("agent_unavailable")]
    if a.where:
        k, v = a.where.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]
    if a.match_only:
        bad = set()
        for f in a.events:
            ep = Path(f).with_name("episodes.jsonl")
            if ep.exists():
                bad |= {(e["run"], e["layout_id"]) for e in map(json.loads, open(ep)) if e.get("score_match") is False}
        rows = [r for r in rows if (r["run"], r["layout_id"]) not in bad]
    if a.latest or a.old:
        new, old = latest_split(rows)
        rows = new if a.latest else old
        if a.latest:
            fams = defaultdict(set)
            for r in rows:
                fams[r["task"]].add(r["family"])
            print("rounds: " + "; ".join(f"{t} {', '.join(sorted(f))}" for t, f in sorted(fams.items())))
    if a.fails:
        for x in failure_rows(rows):
            print(json.dumps(x))
    elif a.taxonomy:
        print(taxonomy(rows))
    elif a.frame0:
        print(frame0_table(rows))
    elif a.gap:
        print(gap_table(rows))
    else:
        print(corr(rows) if a.corr else table(rows, a.by))


THIN = ("coin", "pen", "key", "card", "toothbrush", "screwdriver", "spoon", "knife", "fork", "chopstick", "mallet",
        "pencil", "marker", "nut", "bolt", "peg", "tube", "stick")


def failure_rows(rows: list[dict], R: str = "10") -> list[dict]:
    """One row per unclean event: what failed first (required objects before neighbours before phantoms) and its
    context: the belief's source for that object (record = never re-measured, refine_pick / refine_place = the wrist
    refine, held / released = carried by the jaws), the record's own occluded / hidden flags, thin objects."""
    out = []
    for r in rows:
        if r["clean_r"][R]:
            continue
        bad = [q for q in r["required"] if not q["ok"]]
        if bad:
            q = bad[0]
            perc = q.get("perc") or {}
            what = {"what": f"{q['role']} {q['cls']}", "label": q["label"], "gt_label": q.get("gt_label"),
                    "xy_mm": q.get("xy_mm"), "dz_mm": q.get("dz_mm"), "rot_deg": q.get("rot_deg"),
                    "source": q.get("source"), "occluded": perc.get("occluded"), "hidden_frac": perc.get("hidden_frac"),
                    "on": q.get("on")}
        else:
            nb = [n for n in r["neighbours"] if n["dist_m"] <= int(R) / 100 and not n["found"]]
            ph = [p for p in r["phantoms"] if p["dist_m"] <= int(R) / 100]
            if nb:
                n = nb[0]
                what = {"what": "neighbour " + ("not found" if n.get("xy_mm") is None else "off"), "label": n["label"],
                        "xy_mm": n.get("xy_mm"), "dist_m": n["dist_m"], "clutter": n.get("clutter"),
                        "belief_source": n.get("belief_source")}
            else:
                p = ph[0]
                what = {"what": "phantom", "label": p["label"], "xy_mm": p["nearest_gt_mm"], "dist_m": p["dist_m"],
                        "belief_source": p.get("source")}
        what["thin"] = any(t in str(what.get("label")) for t in THIN)
        out.append({"task": r["task"], "run": r["run"], "layout_id": r["layout_id"], "kind": r["kind"], "step": r["step"],
                    "arm": r["arm"], "success": r.get("success"), "reasons": r["reasons"][R], **what})
    return out


def taxonomy(rows: list[dict], R: str = "10") -> str:
    """Per task: unclean events by first failure, with the belief source of a failed required object."""
    f = failure_rows(rows, R)
    by = defaultdict(Counter)
    src = defaultdict(Counter)
    for x in f:
        by[x["task"]][x["what"]] += 1
        if x.get("source"):
            src[x["task"]][x["source"]] += 1
    lines = [f"| task | unclean events (R{R}) | first failure | belief source of a failed required object |", "|---|---|---|---|"]
    for t in sorted(by):
        lines.append(f"| {t} | {sum(by[t].values())} | {', '.join(f'{k} {v}' for k, v in by[t].most_common())} | "
                     f"{', '.join(f'{k} {v}' for k, v in src[t].most_common())} |")
    tot, ts = Counter(), Counter()
    for t in by:
        tot.update(by[t])
        ts.update(src[t])
    lines.append(f"| **all** | {sum(tot.values())} | {', '.join(f'{k} {v}' for k, v in tot.most_common())} | "
                 f"{', '.join(f'{k} {v}' for k, v in ts.most_common())} |")
    return "\n".join(lines)


RELEASED = ("released", "post_place", "held_dropped", "probe_released")


def blocker(x: dict) -> str:
    """The class that blocks an unclean event (its first failure), as the streams own them: not_held (the held state),
    in-hand pose, released (an object the robot let go earlier, believed where the release put it), target pose,
    reference pose, identity, neighbour / phantom."""
    w = x["what"]
    if w == "held not_held":
        return "not_held"
    if w.endswith("identity"):
        return "identity"
    if w.startswith("neighbour") or w == "phantom":
        return "released" if x.get("belief_source") in RELEASED else "neighbour"
    if x.get("source") in RELEASED:
        return "released"
    if w.startswith("held"):
        return "in-hand pose"
    if w.startswith("reference"):
        return "reference pose"
    return "target pose"


BLOCKERS = ("not_held", "in-hand pose", "released", "target pose", "reference pose", "identity", "neighbour")


def gap_table(rows: list[dict], goal: float = 0.95, R: str = "10") -> str:
    """Per task: events, clean at R, the clean events still missing to ``goal``, and the unclean events by blocker."""
    import math
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    fails = defaultdict(Counter)
    for x in failure_rows(rows, R):
        fails[x["task"]][blocker(x)] += 1
    lines = [f"| task | events | clean R{R} | events to {int(goal * 100)} % | unclean by blocker: " + " / ".join(BLOCKERS) + " |",
             "|---|---|---|---|---|"]
    tot_n = tot_c = tot_g = 0
    tot_f = Counter()
    order = sorted(by, key=lambda t: -(math.ceil(goal * len(by[t])) - sum(r["clean_r"][R] for r in by[t])))
    for t in order:
        n, c = len(by[t]), sum(bool(r["clean_r"][R]) for r in by[t])
        g = max(0, math.ceil(goal * n) - c)
        tot_n, tot_c, tot_g = tot_n + n, tot_c + c, tot_g + g
        tot_f.update(fails[t])
        lines.append(f"| {t} | {n} | {100 * c / n:.0f} % | {g} | " + " / ".join(str(fails[t].get(b, 0)) for b in BLOCKERS) + " |")
    if tot_n:
        lines.append(f"| **all** | {tot_n} | {100 * tot_c / tot_n:.0f} % | {max(0, math.ceil(goal * tot_n) - tot_c)} "
                     f"(sum per task {tot_g}) | " + " / ".join(str(tot_f.get(b, 0)) for b in BLOCKERS) + " |")
    return "\n".join(lines)


def frame0_table(rows: list[dict]) -> str:
    """The runtime's belief at the event vs the frame-0 record on the same events (what the looks, refines and
    re-reads between frame 0 and the event bought)."""
    by = defaultdict(list)
    for r in rows:
        if r.get("frame0"):
            by[r["task"]].append(r)
    lines = ["| task | events | target ok: frame-0 record -> runtime | clean R10: frame-0 -> runtime | fixed / broken (R10) |",
             "|---|---|---|---|---|"]
    tot = []
    for t in sorted(by) + ["**all**"]:
        rs = tot if t == "**all**" else by[t]
        if t != "**all**":
            tot += rs
        n = len(rs)
        if not n:
            continue
        a0, a1 = sum(r["frame0"]["clean_target"] for r in rs), sum(r["clean_target"] for r in rs)
        c0, c1 = sum(r["frame0"]["clean_r"]["10"] for r in rs), sum(r["clean_r"]["10"] for r in rs)
        fx = sum(r["clean_r"]["10"] and not r["frame0"]["clean_r"]["10"] for r in rs)
        br = sum(not r["clean_r"]["10"] and r["frame0"]["clean_r"]["10"] for r in rs)
        lines.append(f"| {t} | {n} | {a0} -> {a1} | {c0} -> {c1} | {fx} / {br} |")
    return "\n".join(lines)


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random success episode's value > a random failed one's), ties half (Mann-Whitney)."""
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def corr(rows: list[dict]) -> str:
    """Episode outcome vs the clean rate per threshold: which R separates successful from failed episodes best.
    Per variant: mean clean rate (success / failure), AUC of the clean rate for success, and "any unclean event" as a
    failure flag (share of failed episodes flagged = recall, share of flagged episodes that failed = precision)."""
    eps = defaultdict(list)
    for r in rows:
        eps[(r["run"], r["layout_id"])].append(r)
    variants = [("R5", lambda e: e["clean_r"]["5"]), ("R10", lambda e: e["clean_r"]["10"]),
                ("R15", lambda e: e["clean_r"]["15"]), ("required only", lambda e: e["clean_required"]),
                ("target only", lambda e: e["clean_target"])]
    lines = ["| variant | episodes (succ / fail) | clean rate succ / fail | AUC (clean rate -> success) | "
             "any-unclean flags: failed eps flagged (recall) | flagged eps that failed (precision) |",
             "|---|---|---|---|---|---|"]
    for name, f in variants:
        succ, fail, fl_fail, fl_all = [], [], 0, 0
        for evs in eps.values():
            rate = sum(bool(f(e)) for e in evs) / len(evs)
            flagged = rate < 1.0
            if evs[0].get("success"):
                succ.append(rate)
            else:
                fail.append(rate)
                fl_fail += flagged
            fl_all += flagged
        auc = _auc(succ, fail)
        mean = lambda v: f"{100 * sum(v) / len(v):.0f} %" if v else "-"   # noqa: E731
        lines.append(f"| {name} | {len(succ)} / {len(fail)} | {mean(succ)} / {mean(fail)} | "
                     f"{'-' if auc is None else f'{auc:.2f}'} | {fl_fail}/{len(fail)} | {fl_fail}/{fl_all} |")
    # neighbours / phantoms on their own: episodes whose required objects were always right, split by whether some
    # event had a neighbour / phantom problem within R (does the extra criterion predict a failure?)
    lines += ["", "| R | episodes, required always ok | of them with a neighbour / phantom problem: failed | without: failed |",
              "|---|---|---|---|"]
    for R in ("5", "10", "15"):
        clean_req = [evs for evs in eps.values() if all(e["clean_required"] for e in evs)]
        nb = [evs for evs in clean_req if any(not e["clean_r"][R] for e in evs)]
        ok = [evs for evs in clean_req if all(e["clean_r"][R] for e in evs)]
        fr = lambda v: f"{sum(not x[0].get('success') for x in v)}/{len(v)}"   # noqa: E731
        lines.append(f"| {R} cm | {len(clean_req)} | {fr(nb)} | {fr(ok)} |")
    return "\n".join(lines)


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
    t.add_argument("--corr", action="store_true", help="episode outcome vs the clean rate per threshold")
    t.add_argument("--taxonomy", action="store_true", help="unclean events per task by first failure and belief source")
    t.add_argument("--fails", action="store_true", help="one JSON line per unclean event (R10)")
    t.add_argument("--frame0", action="store_true", help="the runtime's belief vs the frame-0 record, same events")
    t.add_argument("--latest", action="store_true", help="per task only its newest round (FAMILY_ORDER)")
    t.add_argument("--gap", action="store_true", help="per task: events to 95 %% at R10 and what blocks them")
    t.add_argument("--old", action="store_true", help="per task every round but its newest")
    t.add_argument("--match-only", action="store_true", help="only episodes whose replay reproduced the score")
    a = ap.parse_args(argv)
    {"jobs": cmd_jobs, "score": cmd_score, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
