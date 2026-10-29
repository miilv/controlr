"""Paired comparison of two segmenters' ``eval.py pred`` rows on the same frames (the no-regression check of an
incremental model, e.g. v3 vs v2b on v2b's own classes). DEV ONLY (reads GT rows).

    python scripts/perception/seg/compare_rows.py rows_v2b.jsonl rows_v3.jsonl \\
        --classes ~/dojo-data/segtrain/models/v2b/classes.json [--top 15]

An object is "right" when its matched prediction has IoU >= 0.5 and the right asset class. Per subset (all cams,
head, wrists; objects whose asset is in ``--classes``): n, right A, right B, fixed (B right, A not), broken (A right,
B not); then the classes with the most broken objects net.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: str) -> dict:
    out = {}
    for line in open(Path(path).expanduser()):
        r = json.loads(line)
        if r.get("frame_row"):
            continue
        out[(r["set"], r["task"], r["episode"], r["frame"], r["cam"], r["gi"])] = r
    return out


def right(r: dict) -> bool:
    return bool(r["class_ok"]) and r["iou"] >= 0.5


def found(r: dict) -> bool:
    return r["iou"] >= 0.5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--classes", required=True, help="classes.json: only objects whose asset is in it")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    keep = set(json.loads(Path(a.classes).expanduser().read_text())["names"])
    A, B = load(a.a), load(a.b)
    common = [k for k in A if k in B and A[k]["key"] in keep]
    print(f"{len(common)} paired objects (A {len(A)}, B {len(B)} rows); assets in {a.classes}\n")
    print("| subset | n | found A | found B | right A | right B | fixed | broken |")
    print("|---|---|---|---|---|---|---|---|")
    for name, f in (("all cams", lambda k: True), ("head", lambda k: k[4] == "cam_head"),
                    ("wrists", lambda k: k[4] != "cam_head"),
                    ("task objects, head", lambda k: k[4] == "cam_head" and not A[k]["clutter"])):
        ks = [k for k in common if f(k)]
        ra, rb = sum(right(A[k]) for k in ks), sum(right(B[k]) for k in ks)
        fa, fb = sum(found(A[k]) for k in ks), sum(found(B[k]) for k in ks)
        fix = sum(right(B[k]) and not right(A[k]) for k in ks)
        brk = sum(right(A[k]) and not right(B[k]) for k in ks)
        print(f"| {name} | {len(ks)} | {fa} | {fb} | {ra} | {rb} | {fix} | {brk} |")
    for s in sorted({k[0] for k in common}):
        ks = [k for k in common if k[0] == s and k[4] == "cam_head"]
        print(f"| {s}, head | {len(ks)} | {sum(found(A[k]) for k in ks)} | {sum(found(B[k]) for k in ks)} | "
              f"{sum(right(A[k]) for k in ks)} | {sum(right(B[k]) for k in ks)} | "
              f"{sum(right(B[k]) and not right(A[k]) for k in ks)} | {sum(right(A[k]) and not right(B[k]) for k in ks)} |")
    net = Counter()
    for k in common:
        net[A[k]["key"]] += int(right(A[k])) - int(right(B[k]))
    print(f"\nClasses with the most objects broken net (A right, B not, minus fixed), top {a.top}:")
    for key, n in net.most_common(a.top):
        if n <= 0:
            break
        ks = [k for k in common if A[k]["key"] == key]
        print(f"- {key}: -{n} of {len(ks)} (B predicted: {Counter(B[k]['pred_key'] for k in ks).most_common(3)})")


if __name__ == "__main__":
    main()
