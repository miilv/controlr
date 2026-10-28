"""Tables for E1 (stream W): clean events per arm and R, fixed / broken vs the runtime's belief, existence verdicts vs
GT (dev), identity top-1, cost per hover, binned by the object's pixel size in the wrist view.

    python scripts/perception/wrist/e1_report.py E1.json [E1b.json ...]
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np


def arms_of(rows):
    return [a for a in rows[0]["scores"]] if rows else []


def clean_table(rows) -> list[str]:
    arms = arms_of(rows)
    out = ["| arm | R5 | R10 | R15 | required | target | R10 fixed / broken |", "|---|---|---|---|---|---|---|"]
    n = len(rows)
    for a in arms:
        c = {R: sum(bool(r["scores"][a]["clean_r"][R]) for r in rows) for R in ("5", "10", "15")}
        req = sum(bool(r["scores"][a]["clean_required"]) for r in rows)
        tgt = sum(bool(r["scores"][a]["clean_target"]) for r in rows)
        fx = sum(r["scores"][a]["clean_r"]["10"] and not r["scores"]["rt"]["clean_r"]["10"] for r in rows)
        br = sum(not r["scores"][a]["clean_r"]["10"] and r["scores"]["rt"]["clean_r"]["10"] for r in rows)
        out.append(f"| {a} | {c['5']}/{n} | {c['10']}/{n} | {c['15']}/{n} | {req}/{n} | {tgt}/{n} | +{fx} / -{br} |")
    return out


def reasons(rows, arm="rt", R="10") -> collections.Counter:
    c = collections.Counter()
    for r in rows:
        for x in r["scores"][arm]["reasons"][R]:
            w = x.split()
            kind = w[0] if w[0] in ("held", "target", "reference") else w[0]
            cls = "not found" if "not found" in x else ("phantom" if w[0] == "phantom" else
                                                        (w[2] if len(w) > 2 else "?"))
            if w[0] == "neighbour" and "not found" not in x:
                cls = "off"
            c[f"{kind} {cls}"] += 1
    return c


def verdicts(rows) -> collections.Counter:
    """(verdict, view, truth): truth = real when the belief is within 50 mm of a GT object of its group (dev)."""
    c = collections.Counter()
    for r in rows:
        for lab, o in r["check"].get("objects", {}).items():
            g = o.get("dev_gt_mm")
            c[(o.get("role"), o.get("verdict"), o.get("via") or "wrist", "real" if g is not None and g <= 50 else "phantom")] += 1
    return c


def identity(rows, margins=(0.0, 0.03, 0.05, 0.08)) -> list[str]:
    """Top-1 of the DINOv2 identity among the label group's candidates, and what a margin-gated change does."""
    items = []
    for r in rows:
        for lab, o in r["check"].get("objects", {}).items():
            idr = o.get("identity")
            if not idr or idr.get("n_keys", 1) < 2 or not o.get("dev_gt_key"):
                continue
            ids = {k.split(":", 1)[1]: v for k, v in idr["scores"].items() if k.startswith("id:")}
            items.append((o["key"], o["dev_gt_key"], ids, o.get("vis_px"), r["episode"], r["step"], lab))
    if not items:
        return ["(no identity rows)"]
    out = [f"objects with >= 2 candidates: {len(items)}; belief identity right: "
           f"{sum(b == g for b, g, *_ in items)}; DINOv2 top-1 right: {sum(max(s, key=s.get) == g for _, g, s, *_ in items)}"]
    out.append("| margin | changes | right | wrong | belief right after |")
    out.append("|---|---|---|---|---|")
    for m in margins:
        ch = right = wrong = after = 0
        for b, g, s, *_ in items:
            best = max(s, key=s.get)
            new = best if best != b and s[best] - s[b] >= m else b
            if new != b:
                ch += 1
                right += new == g
                wrong += new != g
            after += new == g
        out.append(f"| {m} | {ch} | {right} | {wrong} | {after}/{len(items)} |")
    return out


def flips(rows, margins=(0.03, 0.05, 0.08, 0.12)) -> list[str]:
    items = []
    for r in rows:
        for lab, o in r["check"].get("objects", {}).items():
            idr = o.get("identity") or {}
            if "flip_margin" not in idr or o.get("dev_rot_deg") is None or o.get("dev_rot_deg_flip") is None:
                continue
            need = o["dev_rot_deg_flip"] + 30 < o["dev_rot_deg"]        # the flip is closer to the truth
            harm = o["dev_rot_deg"] + 30 < o["dev_rot_deg_flip"]         # the flip moves away from it
            items.append((idr["flip_margin"], need, harm))
    out = [f"flip candidates: {len(items)}; flips needed (GT): {sum(n for _, n, _ in items)}",
           "| flip margin | flips | right | wrong |", "|---|---|---|---|"]
    for m in margins:
        f = [(n, h) for fm, n, h in items if fm >= m]
        out.append(f"| {m} | {len(f)} | {sum(n for n, _ in f)} | {sum(h for _, h in f)} |")
    return out


def cost(rows) -> list[str]:
    t = np.asarray([r["wall_s"] for r in rows], float)
    st = collections.defaultdict(list)
    for r in rows:
        for k, v in (r["check"].get("time_s") or {}).items():
            st[k].append(v)
    out = [f"wall per hover: median {np.median(t):.2f} s, p90 {np.percentile(t, 90):.2f} s, max {t.max():.2f} s, "
           f"> 1 s: {int((t > 1).sum())}/{len(t)}"]
    out.append("stages (median / max s): " + ", ".join(f"{k} {np.median(v):.3f} / {max(v):.2f}" for k, v in st.items()))
    return out


def by_pixels(rows, bins=(0, 2000, 10000, 30000, 1e9)) -> list[str]:
    """Wrist-visible verdict precision by the object's visible pixels."""
    c = collections.defaultdict(collections.Counter)
    for r in rows:
        for lab, o in r["check"].get("objects", {}).items():
            px = o.get("vis_px") or 0
            if (o.get("via") or "wrist") != "wrist" or o.get("verdict") == "unseen":
                continue
            b = next(i for i in range(len(bins) - 1) if bins[i] <= px < bins[i + 1])
            g = o.get("dev_gt_mm")
            c[b][(o.get("verdict"), "real" if g is not None and g <= 50 else "phantom")] += 1
    out = []
    for b in sorted(c):
        out.append(f"vis px {int(bins[b])}-{int(bins[b + 1]) if bins[b + 1] < 1e9 else 'inf'}: "
                   + ", ".join(f"{v}/{t} {n}" for (v, t), n in sorted(c[b].items())))
    return out


def main():
    for p in sys.argv[1:]:
        d = json.load(open(p))
        rows = d["rows"]
        print(f"## {p}  (sam {d.get('sam')}, embed {d.get('embed')}, head {d.get('head')}, hover {d.get('hover')})")
        print("\n".join(clean_table(rows)))
        print("rt reasons R10:", dict(reasons(rows)))
        last = arms_of(rows)[-1]
        print(f"{last} reasons R10:", dict(reasons(rows, last)))
        print("verdicts (role, verdict, view, truth):")
        for k, v in sorted(verdicts(rows).items(), key=lambda kv: -kv[1]):
            print("  ", k, v)
        print("\n".join(by_pixels(rows)))
        print("\n".join(identity(rows)))
        print("\n".join(flips(rows)))
        print("\n".join(cost(rows)))
        errs = [r for r in rows if r.get("error")]
        if errs:
            print(f"errors: {len(errs)} (first: {errs[0]['error'][:200]})")


if __name__ == "__main__":
    main()
