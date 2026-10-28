"""Per-task table of an E2 run (``e2_release.py``): clean@R10 per arm over the events every arm scored, fixed / broken
against the runtime's belief, and the realised fraction of the oracle's gain; plus the reads' outcomes.

    python scripts/perception/wrist/e2_report.py E2.json [E2b.json ...] [--arm search] [--R 10]
(several files: their episodes pooled, the later file's rows replace an episode already seen)
"""
from __future__ import annotations

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--arm", default="search")
    ap.add_argument("--R", default="10")
    a = ap.parse_args()
    d = {"rows": [], "reads": []}
    for p in a.paths:
        di = json.load(open(p))
        eps = {r["episode"] for r in di["rows"]}
        d["rows"] = [r for r in d["rows"] if r["episode"] not in eps] + di["rows"]
        d["reads"] = [x for x in d["reads"] if x["episode"] not in eps] + di.get("reads", [])
        d["arms"] = di["arms"]
    arms, R, w = d["arms"], a.R, a.arm
    rows = [r for r in d["rows"] if not any(r["scores"][an]["reasons"][R] == ["not scored"] for an in arms)]
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        c = by[r["task"]]
        c["n"] += 1
        rt = r["scores"]["rt"]["clean_r"][R]
        for an in arms:
            v = r["scores"][an]["clean_r"][R]
            c[an] += v
            c[f"{an}+"] += v and not rt
            c[f"{an}-"] += rt and not v
    print(f"| task | events | rt | oracle | {w} | oracle fixed / broken | {w} fixed / broken | realised |")
    print("|---|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    for t in sorted(by):
        c = by[t]
        tot.update(c)
        real = f"{c[w + '+'] - c[w + '-']}/{c['oracle+'] - c['oracle-']}" if c["oracle+"] else "-"
        print(f"| {t} | {c['n']} | {c['rt']} | {c['oracle']} | {c[w]} | +{c['oracle+']} / -{c['oracle-']} | "
              f"+{c[w + '+']} / -{c[w + '-']} | {real} |")
    print(f"| all | {tot['n']} | {tot['rt']} | {tot['oracle']} | {tot[w]} | +{tot['oracle+']} / -{tot['oracle-']} | "
          f"+{tot[w + '+']} / -{tot[w + '-']} | {tot[w + '+'] - tot[w + '-']}/{tot['oracle+'] - tot['oracle-']} |")
    reads = [x for x in d.get("reads", []) if x.get("src") == w]
    out = collections.Counter((x["episode"].rsplit("_", 1)[0], x.get("via") if x.get("ok") else f"no:{x.get('why')}")
                              for x in reads)
    print("\nreads by task / outcome:")
    for (t, o), n in sorted(out.items()):
        print(f"  {t:24s} {o:24s} {n}")
    tm = sorted(x["wall_s"] for x in reads)
    if tm:
        print(f"wall per read: median {tm[len(tm) // 2]:.2f} s, max {tm[-1]:.2f} s, n {len(tm)}")


if __name__ == "__main__":
    main()
