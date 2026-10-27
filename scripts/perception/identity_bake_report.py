"""Tables for identity_bake.py rows: top-1 identity per object kind, ours (B1 / belt read) vs each embedding arm.

    identity_bake_report.py ROWS.jsonl [--emb EMB.npz] [--min-area 0] [--arms dinov2_b/ma_lit ...] [--md]

Units: every head blob that covers >= 30 % of one GT object's silhouette (``gt``), merged blobs (a second object
covers >= 20 %) apart. Ours = the best hypothesis' key (no hypothesis = wrong); ``reported`` = the reads belt_read
passes to the tracker (confidence >= MIN_RIDER_CONF). Fusion arms (``fuse:<arm>:<w>``): embedding similarity minus
w x our loss for the keys B1 refined (others count loss 1.0). Gallery arm (``--emb``): 1-NN over the observed crops
of the OTHER layouts (leave-one-layout-out), per model and crop.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

import numpy as np


def kind(key):
    return key.split("/")[1] if key else "-"


def load(path, min_area=0):
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r.get("gt") and r["area"] >= min_area and r.get("second_overlap", 0) < 0.2 and r.get("scores")]


def pred_b1(r):
    return (r.get("b1") or {}).get("key")


def pred_arm(r, arm):
    if arm == "b1":
        return pred_b1(r)
    if arm == "b1_reported":
        b = r.get("b1") or {}
        return b.get("key") if b.get("reported") else None
    if arm.startswith("fuse:"):
        _, a, w = arm.split(":")
        s = r["scores"].get(a) or {}
        ls = r.get("b1_losses") or {}
        if not s:
            return None
        return max(s, key=lambda k: s[k] - float(w) * ls.get(k, 1.0))
    if arm.startswith("ens:"):  # mean similarity of several arms ("ens:a,b,c")
        parts = [r["scores"].get(x) or {} for x in arm[4:].split(",")]
        ks = set.intersection(*[set(p) for p in parts]) if parts and all(parts) else set()
        return max(ks, key=lambda k: sum(p[k] for p in parts)) if ks else None
    if arm.startswith("restrict:"):  # the embedding's best among the keys B1 refined
        a = arm.split(":", 1)[1]
        s = r["scores"].get(a) or {}
        ls = r.get("b1_losses") or {}
        ks = [k for k in s if k in ls] or list(s)
        return max(ks, key=lambda k: s[k]) if ks else None
    s = r["scores"].get(arm) or {}
    return next(iter(s), None)


def gallery_preds(rows, emb_path):
    """{(layout, step, blob): {model/crop: predicted key}} by 1-NN over the other layouts' labelled crops."""
    z = np.load(emb_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    idx = {(m["layout"], m["step"], m["blob"]): i for i, m in enumerate(meta)}
    out = collections.defaultdict(dict)
    lab = np.array([m["gt"] or "" for m in meta])
    lay = np.array([m["layout"] for m in meta])
    for name in z.files:
        if name == "meta":
            continue
        E = z[name].astype(np.float32)
        S = E @ E.T
        for r in rows:
            i = idx.get((r["layout"], r["step"], r["blob"]))
            if i is None:
                continue
            ok = (lay != r["layout"]) & (lab != "")
            if not ok.any():
                continue
            j = np.flatnonzero(ok)[np.argmax(S[i, ok])]
            out[(r["layout"], r["step"], r["blob"])]["gal:" + name.replace("__", "/")] = lab[j]
    return out


def table(rows, arms, gal=None):
    kinds = sorted({kind(r["gt"]) for r in rows})
    res = {}
    for arm in arms:
        c = collections.Counter()
        n = collections.Counter()
        conf = collections.Counter()
        for r in rows:
            k = kind(r["gt"])
            p = gal[(r["layout"], r["step"], r["blob"])].get(arm) if (gal is not None and arm.startswith("gal:")) else pred_arm(r, arm)
            n[k] += 1
            if p == r["gt"]:
                c[k] += 1
            else:
                conf[(k, kind(p))] += 1
        res[arm] = {"kinds": {k: (c[k], n[k]) for k in kinds}, "total": (sum(c.values()), sum(n.values())), "conf": conf}
    return kinds, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--emb")
    ap.add_argument("--min-area", type=int, default=0)
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--layouts", nargs="*")
    ap.add_argument("--steps-mod", type=int, nargs=2, help="keep rows with step/10 %% A == B")
    ap.add_argument("--reported-only", action="store_true", help="only blobs belt_read reports")
    ap.add_argument("--confusions", type=int, default=6)
    a = ap.parse_args(argv)
    rows = load(a.rows, a.min_area)
    if a.layouts:
        rows = [r for r in rows if r["layout"] in a.layouts]
    if a.steps_mod:
        rows = [r for r in rows if (r["step"] // 10) % a.steps_mod[0] == a.steps_mod[1]]
    if a.reported_only:
        rows = [r for r in rows if (r.get("b1") or {}).get("reported")]
    all_arms = sorted({k for r in rows for k in r["scores"]})
    arms = a.arms or (["b1", "b1_reported"] + all_arms)
    gal = None
    if a.emb:
        gal = gallery_preds(rows, a.emb)
        arms = arms + sorted({k for v in gal.values() for k in v})
    kinds, res = table(rows, arms, gal)
    print(f"n = {len(rows)} blobs, layouts {sorted({r['layout'] for r in rows})}, min area {a.min_area}")
    print("| arm | " + " | ".join(kinds) + " | all |")
    print("|---|" + "---|" * (len(kinds) + 1))
    for arm in arms:
        t = res[arm]
        cells = [f"{t['kinds'][k][0]}/{t['kinds'][k][1]}" for k in kinds]
        print(f"| {arm} | " + " | ".join(cells) + f" | **{t['total'][0]}/{t['total'][1]}** ({t['total'][0] / max(1, t['total'][1]):.2f}) |")
    for arm in arms[: max(0, a.confusions)]:
        print(arm, "confusions:", ", ".join(f"{g}->{p} {v}" for (g, p), v in res[arm]["conf"].most_common(8)))


if __name__ == "__main__":
    sys.exit(main())
