"""Stream C, X2: the report tables from the harness output (``x2_eval.py``): rt vs J, J vs S, per R, with and without
put_bottles and per task; the pre-registered rule (always-on is worth building only if S - J >= 3 points of clean@R10
with S's broken <= J's broken, verdicts against the per-writer p95 floor); writes by GT, offline-blind writes, the
online cost.

    python scripts/perception/bgperc/x2_report.py OUT [--arms J,S] [--md OUT.md]

Pairs rows by (run, step, kind, arm = the hand). S vs J at R: same | fixed (S clean, J not) | broken (J clean, S not)
| within noise (the flip's arm moved nothing past its writer's p95: its verdict against rt is "within noise").
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(p: Path) -> dict:
    out = {}
    if p.exists():
        for x in open(p):
            if x.strip():
                r = json.loads(x)
                out[(r["run"], r["step"], r["kind"], r["arm"])] = r
    return out


def verdict_vs(a: dict, b: dict, R: str) -> str:
    """b against a at R (a, b: rows of the same event)."""
    ca, cb = bool(a["clean_r"][R]), bool(b["clean_r"][R])
    if ca == cb:
        return "same"
    if b.get("verdict") == "within noise" or a.get("verdict") == "within noise":
        return "within noise"
    return "fixed" if cb else "broken"


def table(keys, rt, arms: dict, R: str) -> dict:
    out = {"events": len(keys), "rt": sum(bool(rt[k]["clean_r"][R]) for k in keys)}
    for name, ar in arms.items():
        out[name] = sum(bool(ar[k]["clean_r"][R]) for k in keys)
        out[f"{name}_vs_rt"] = dict(Counter(verdict_vs(rt[k], ar[k], R) for k in keys))
    if "J" in arms and "S" in arms:
        out["S_vs_J"] = dict(Counter(verdict_vs(arms["J"][k], arms["S"][k], R) for k in keys))
    return out


def pct(n, d):
    return f"{100.0 * n / d:.1f}" if d else "-"


def md(name: str, t: dict, arm_names) -> str:
    n = t["events"]
    cells = [name, str(n), f"{t['rt']} ({pct(t['rt'], n)})"]
    for a in arm_names:
        v = t[f"{a}_vs_rt"]
        cells += [f"{t[a]} ({pct(t[a], n)})", f"{v.get('fixed', 0)} / {v.get('broken', 0)} / {v.get('within noise', 0)}"]
    if "S_vs_J" in t:
        v = t["S_vs_J"]
        cells += [f"{v.get('fixed', 0)} / {v.get('broken', 0)} / {v.get('within noise', 0)}"]
    return "| " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arms", default="J,S")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    od = Path(a.out)
    names = [x for x in a.arms.split(",") if (od / x / "events.jsonl").exists()]
    rt = rows(od / "rt" / "events.jsonl")
    arms = {n: rows(od / n / "events.jsonl") for n in names}
    keys = sorted(set(rt).intersection(*[set(v) for v in arms.values()]) if arms else set(rt))
    lines = []
    head = ["set", "events", "rt clean"] + [x for n in names for x in (f"{n} clean", f"{n} vs rt fixed / broken / noise")]
    if "J" in arms and "S" in arms:
        head.append("S vs J fixed / broken / noise")
    for R in ("10", "5", "15"):
        lines.append(f"\n### clean@R{R} (paired events; episodes: {len({k[0] for k in keys})})\n")
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        on = [k for k in keys if "_gcon_" in k[0]]
        off = [k for k in keys if "_gcoff_" in k[0]]
        nb = [k for k in keys if rt[k]["task"] != "put_bottles_into_dustbin"]
        sets = [("all", keys), ("ON (gcon, primary)", on), ("OFF (gcoff)", off), ("all without put_bottles", nb),
                ("ON without put_bottles", [k for k in on if k in set(nb)])]
        by, by_on = defaultdict(list), defaultdict(list)
        for k in keys:
            by[rt[k]["task"]].append(k)
            if k in set(on):
                by_on[rt[k]["task"]].append(k)
        sets += sorted(by.items()) + [(f"{t} (ON)", v) for t, v in sorted(by_on.items())]
        for nm, ks in sets:
            lines.append(md(nm, table(ks, rt, arms, R), names))
    if "J" in arms and "S" in arms:
        t = table(keys, rt, arms, "10")
        d_pts = 100.0 * (t["S"] - t["J"]) / max(1, t["events"])
        bj, bs = t["J_vs_rt"].get("broken", 0), t["S_vs_rt"].get("broken", 0)
        lines.append(f"\n**Pre-registered rule** (always-on only if S - J >= 3 pts clean@R10 and S broken <= J broken): "
                     f"S - J = {d_pts:+.1f} pts, broken S {bs} vs J {bj} -> "
                     f"{'BUILD always-on' if d_pts >= 3 and bs <= bj else 'do not build always-on'}")
    for n in names:
        sp = od / n / "summary.json"
        if sp.exists():
            s = json.load(open(sp))
            c = s.get("online_cost") or {}
            rc = (s.get("all") or {}).get("recomputed") or {}
            lines.append(f"\n**{n}**: writes {c.get('writes')} (by GT {json.dumps(s.get('writes_by_gt'))}); ticks "
                         f"{c.get('ticks')} ({c.get('ticks_per_episode')} per episode, {c.get('periodic_ticks')} periodic), "
                         f"tick wall {c.get('tick_wall_s_per_episode')} s per episode, errors {c.get('errors')}, "
                         f"cache misses {c.get('dets_cache_misses')}; two-tick writes {c.get('two_tick_writes')} "
                         f"(pending < 10 steps: {c.get('two_tick_writes_lt10_steps')}); re-computed writes {rc.get('n')} "
                         f"(adopted {rc.get('adopted')}, offline-blind {rc.get('offline_blind')}); floor {s.get('floor')}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
