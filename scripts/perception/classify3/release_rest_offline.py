"""CLASSIFY3 (dev only): container tests (and, where the code has it, a container-floor release rest) evaluated offline
on every release of recorded sort-into-containers camera rows, against physics-only replays of the same actions.

    python scripts/perception/classify3/release_rest_offline.py --rows ROWS --layouts LAYOUT_ROOT --replay REPLAY \\
        --catalog CATALOG [--out out.json]

A release is a ``post_place`` event (the runtime's post-place read after the jaws opened on an object); its release
pose (xy, orientation) is the label's next ``belief_write``. For each release the ported classify skill's own container
test (``in_bin``: the root position inside the basket's box, below its rim + 10 mm) is evaluated on
* **legacy**: the belief the skill read next (the first ``dev_belief_probe`` snapshot after the release);
* **container** (only with a ``dojo.runtime.belief.drop_rest`` that takes ``container=``: the prototype of commit
  197e79b4, removed after this measurement): the same release pose (as logged; raised to 5 mm over the rim if lower)
  with ``drop_rest(container=True)`` over the three baskets and the other believed objects;
each with the root test and the alternative tests ``TESTS`` (bbox centre, lowest point), compared with **physics**: the matched GT instance (frame 0, <= 50 mm) in the replay's next frame after the release.
Counts: true / false placement claims per rule. GT is read for scoring only.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.perception.classify3.stage_audit import BASKET_HALF, SLICES, TABLE_Z, basket_of, match  # noqa: E402

BASKET_TOP = 0.842


def asset(catalog: str, key: str):
    from dojo.skills.world import AssetInfo
    d = Path(catalog) / key
    g = np.load(d / "geom.npz")
    info = json.loads((d / "info.json").read_text())
    return AssetInfo(key=key, bbox_min=np.asarray(info["bbox_min"], float), bbox_max=np.asarray(info["bbox_max"], float),
                     mesh=(np.asarray(g["vertices"], float), np.asarray(g["faces"], int)))


def obj(label, key, pos, quat, catalog, cache):
    from dojo.skills.world import ObjectState
    if key not in cache:
        cache[key] = asset(catalog, key)
    typ, cat, idx = key.split("/")
    return ObjectState(label, typ, cat, int(idx), np.array(pos, float), np.array(quat, float), cache[key])


def in_bin(pos, b) -> bool:
    """classify_into_bins_p2's ``in_bin`` on the basket box (catalogue bbox at the basket's pose)."""
    x, y, z = (float(v) for v in pos[:3])
    return bool(abs(x - b[0]) <= BASKET_HALF[0] and abs(y - b[1]) <= BASKET_HALF[1]
                and TABLE_Z - 0.01 <= z <= BASKET_TOP + 0.01)


TESTS = {"centre": lambda o: [o.center_w[0], o.center_w[1], o.center_w[2]],
         "bottom": lambda o: [o.center_w[0], o.center_w[1], float(o.aabb()[0][2])]}


def row_releases(row: str, layouts: str, replay: str, catalog: str, cache: dict) -> list:
    import inspect

    from dojo.runtime.belief import drop_rest
    has_container = "container" in inspect.signature(drop_rest).parameters
    ep = json.loads(open(glob.glob(os.path.join(row, "*_episodes.jsonl"))[0]).readline())
    if int(ep["layout_id"]) not in SLICES:
        raise SystemExit(f"{row}: outside our slices")
    rep = json.load(gzip.open(glob.glob(os.path.join(row, "*_report.json.gz"))[0]))
    lay = json.load(open(os.path.join(layouts, ep["task"], os.path.basename(ep["layout"]))))
    gt0 = {o["label"]: o for lst in lay["Rigid"].values() for o in lst}
    bk = sorted(lay["Geometry"]["basket"], key=lambda o: o["default_pos"][0])
    baskets = [tuple(o["default_pos"][:2]) for o in bk]
    snaps = rep["dev_belief_probe"]["snaps"]
    bel0 = {k: v for k, v in snaps[0]["objects"].items() if v["key"].startswith("Rigid/")}
    gt_of = {b: g for g, b in match({g: o["default_pos"] for g, o in gt0.items()},
                                     {b: v["pos"] for b, v in bel0.items()}).items()}
    rp = json.load(gzip.open(os.path.join(replay, os.path.basename(row.rstrip("/")) + ".json.gz")))
    frames = {int(k): v for k, v in rp["frames"].items()}
    ev = rep["events"]
    out = []
    for i, e in enumerate(ev):
        if e.get("kind") != "post_place":
            continue
        lab, step = e["label"], int(e["step"])
        w = next((x for x in ev[i:] if x.get("kind") == "belief_write" and x.get("label") == lab
                  and int(x.get("step", -1)) >= step), None)
        if w is None or lab not in bel0:
            continue
        a7 = np.asarray(w["after7"], float)
        key = bel0[lab]["key"]
        sn = [s for s in snaps if s["step"] <= step]
        others = []
        for k, v in (sn[-1]["objects"] if sn else {}).items():
            if k != lab and v["key"].startswith("Rigid/"):
                others.append(obj(k, v["key"], v["pos"], v["quat"], catalog, cache))
        fx = [obj(f"basket{j}", f"Geometry/basket/{int(o['category_idx']):05d}", o["default_pos"], o["default_ori"],
                  catalog, cache) for j, o in enumerate(bk)]
        o = obj(lab, key, a7[:3], a7[3:], catalog, cache)
        o.pos[2] += max(0.0, (BASKET_TOP + 0.005) - float(o.aabb()[0][2]))  # let go where logged, never below the rim
        pos_c, ev_c = drop_rest(o, others, fx, container=True) if has_container else (o.pos, None)
        pos_c = o.pos if pos_c is None else pos_c
        oc = obj(lab, key, pos_c, o.quat, catalog, cache)
        nb = basket_of([a7[0], a7[1], 0.80], baskets)          # the basket its release xy is over (None: none)
        # legacy: what the skill read next, the first belief snapshot after the release (the logged write can precede
        # the drop rule)
        nxt = next((sp for sp in snaps if sp["step"] > step and lab in sp["objects"]), None)
        leg = np.asarray(nxt["objects"][lab]["pos"], float) if nxt else a7[:3]
        legq = np.asarray(nxt["objects"][lab]["quat"], float) if nxt else a7[3:]
        ol = obj(lab, key, leg, legq, catalog, cache)
        g = gt_of.get(lab)
        k = min((k for k in frames if k > step), default=max(frames))
        gp = next((fo["pos"] for fo in frames[k]["objects"] if fo["label"] == g), None) if g else None
        out.append({"row": os.path.basename(row.rstrip("/")), "label": lab, "gt": g, "step": step, "basket": nb,
                    "legacy_in": nb is not None and in_bin(leg, baskets[nb]),
                    "container_in": has_container and nb is not None and in_bin(pos_c, baskets[nb]),
                    "container_how": (ev_c or {}).get("how"),
                    # alternative container tests on the same beliefs: the bbox centre / the bbox bottom below the rim
                    **{f"{rule}_{t}_in": nb is not None and in_bin(test(ob_), baskets[nb])
                       for rule, ob_ in (("legacy", ol), ("container", oc)) for t, test in TESTS.items()
                       if rule == "legacy" or has_container},
                    "physics_in": None if gp is None or nb is None else basket_of(gp, baskets) == nb,
                    "legacy_z": round(float(leg[2]), 4), "write_z": round(float(a7[2]), 4), "container_z": round(float(pos_c[2]), 4),
                    "post_place_ok": e.get("ok"), "writer": w.get("writer")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cache: dict = {}
    rel = []
    for d in sorted(glob.glob(os.path.join(a.rows, "c4b_*_1000??"))):
        rel += row_releases(d, a.layouts, a.replay, a.catalog, cache)
    summ = {}
    for task in ("classifysg", "classifylangsg"):
        rs = [r for r in rel if f"c4b_{task}_" in r["row"] and r["basket"] is not None and r["physics_in"] is not None]
        s = {"releases_over_a_basket": len(rs), "physics_in": sum(r["physics_in"] for r in rs)}
        rules = ("legacy", "legacy_centre", "legacy_bottom")
        if any("container_bottom_in" in r for r in rs):     # the prototype rest rule is in this checkout
            rules += ("container", "container_centre", "container_bottom")
        for rule in rules:
            s[rule] = {"claim_in": sum(r[f"{rule}_in"] for r in rs),
                       "true_in": sum(r[f"{rule}_in"] and r["physics_in"] for r in rs),
                       "false_in": sum(r[f"{rule}_in"] and not r["physics_in"] for r in rs),
                       "false_out": sum((not r[f"{rule}_in"]) and r["physics_in"] for r in rs)}
        summ[task] = s
    res = {"summary": summ, "releases": rel}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
