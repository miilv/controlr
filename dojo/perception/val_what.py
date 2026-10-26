"""DEV calibration of ``identity`` (what IS there) on D0 wrist looks: for every visible object with same-category twins
in its episode, refine from GT perturbed (dx, dy ~ U(-20, 20) mm, yaw ~ U(-10, 10) deg; neighbours = GT), then rank
the candidates by observed colour. GT builds the priors and scores; nothing else.

Candidate sets: ``twins`` = the episode's objects of the same category; ``group`` (stack_blocks) = every asset of the
episode's size group (5). Colour maps: none, the 6.0 map and the 5.1 map (``color_calib*.json``, fold held out by
episode). Same states rendered by both renderers (``~/dojo-data/d0`` 6.0 proxy, ``~/dojo-data/d0_51`` Isaac 5.1).

    DOJO_CODE=<code> python dojo/perception/val_what.py OUT.json stack_blocks 1,2 25 ~/dojo-data/d0 ~/dojo-data/d0_51
"""
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("DOJO_CODE", os.path.expanduser("~/dojo-perception")))
from scipy.spatial.transform import Rotation  # noqa: E402

from dojo.perception import d0, identity  # noqa: E402
from dojo.perception.cameras import pose_to_T, wrist_camera  # noqa: E402
from dojo.perception.refine import refine, refine_identity  # noqa: E402

GROUPS = {"stack_blocks": [[5, 8, 11, 14, 17], [6, 9, 12, 15, 18], [7, 10, 13, 16, 19]]}
CAT = Path(os.environ.get("DOJO_CATALOG", os.path.expanduser("~/dojo-perception-data/catalog")))


def maps():
    out = {"none": {0: None, 1: None}}
    for tag, f in (("m60", "color_calib.json"), ("m51", "color_calib_51.json")):
        p = CAT / f
        if p.is_file():
            d = json.loads(p.read_text())
            out[tag] = {k: np.asarray(d[f"fold{k}"], float) for k in (0, 1)}
    return out


def main():
    out_path, tasks, frames, nmax = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), int(sys.argv[4])
    roots = [Path(os.path.expanduser(r)) for r in sys.argv[5:]]
    M = maps()
    rows = []
    for root in roots:
        rng = np.random.default_rng(0)  # same priors for both renderers
        for t in tasks:
            for ep in d0.episodes(root, t)[:nmax]:
                fold = zlib.crc32(ep.name.encode()) % 2
                for k in frames:
                    kk = int(k) if k.isdigit() else k
                    if not (ep / f"frame_{k}.json").exists() or not all((ep / f"frame_{k}_{c}.png").exists() for c in d0.CAMS):
                        continue
                    obs, gt = d0.load_obs(ep, kk), d0.load_gt(ep, kk)
                    objs = [o for o in gt["objects"] if not o.get("clutter") and o.get("label") and o.get("pos") is not None
                            and o["category"] not in ("camera_stand", "dustbin")]
                    for arm in ("left", "right"):
                        q = np.asarray(obs["state"][f"{arm}_arm_joint_state"])
                        if np.abs(q).max() < 0.02:
                            continue
                        cam = wrist_camera(arm, q)
                        for o in objs:
                            twins = [x for x in objs if x["category"] == o["category"]]
                            if len(twins) < 2:
                                continue
                            uv, z = cam.project(np.array([o["pos"]]))
                            if z[0] <= 0.03 or not (60 < uv[0, 0] < 580 and 60 < uv[0, 1] < 420):
                                continue
                            key = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
                            T = pose_to_T(o["pos"], o["quat"])
                            d = rng.uniform(-0.02, 0.02, 2)
                            a = np.deg2rad(rng.uniform(-10, 10))
                            P = T.copy()
                            P[:2, 3] += d
                            P[:3, :3] = Rotation.from_euler("z", a).as_matrix() @ T[:3, :3]
                            prior = list(P[:3, 3]) + list(Rotation.from_matrix(P[:3, :3]).as_quat()[[3, 0, 1, 2]])
                            oth = [(f"{x['type']}/{x['category']}/{x['category_idx']:05d}", list(x["pos"]) + list(x["quat"]))
                                   for x in gt["objects"] if x is not o and x.get("pos") is not None
                                   and x["category"] not in ("camera_stand",)]
                            if os.environ.get("VAL_SIZE"):  # size decision at the hover (refine_identity, keep_look)
                                g = next((g for g in GROUPS.get(t, []) if o["category_idx"] in g), None)
                                if g is None:
                                    continue
                                series = g.index(o["category_idx"])  # texture slot within the size group
                                sizes = [gg[series] for gg in GROUPS[t]]
                                wrong = [i for i in sizes if i != o["category_idx"]][int(rng.integers(0, 2))]
                                reps = [GROUPS[t][j][(series + 1) % 5] for j in range(3) if GROUPS[t][j][series] != wrong]
                                ids = [f"Rigid/block/{wrong:05d}"] + [f"Rigid/block/{i:05d}" for i in reps]
                                ri = refine_identity(ids, obs["images"][f"cam_{arm}_wrist"], obs["state"], prior, arm=arm,
                                                     others=oth)
                                rows.append({"root": root.name, "task": t, "ep": ep.name, "frame": k, "arm": arm,
                                             "label": o["label"], "key": key, "start": ids[0], "chosen": ri["obj_id"],
                                             "correct": ri["obj_id"] == key, "identity_ok": ri["identity_ok"],
                                             "margin": ri["identity_margin"], "ok": ri["ok"], "iou": ri["iou"],
                                             "scores": ri["identity_scores"], "rank": {}})
                                continue
                            r = refine(key, obs["images"][f"cam_{arm}_wrist"], obs["state"], prior, arm=arm, others=oth)
                            cands = {"twins": [{"label": x["label"], "key": f"{x['type']}/{x['category']}/{x['category_idx']:05d}"}
                                               for x in twins]}
                            for g in GROUPS.get(t, []):
                                if o["category_idx"] in g:
                                    cands["group"] = [{"label": None, "key": f"{o['type']}/{o['category']}/{i:05d}"} for i in g]
                            row = {"root": root.name, "task": t, "ep": ep.name, "frame": k, "arm": arm, "label": o["label"],
                                   "key": key, "ok": r["ok"], "iou": r["iou"], "why": r.get("why"),
                                   "obs_fit": r.get("obs_fit"), "obs_prior": r.get("obs_prior"), "rank": {}}
                            for src in ("fit", "prior"):
                                rgb = (r.get(f"obs_{src}") or {}).get("rgb")
                                for cs, cl in cands.items():
                                    for mt, mm in M.items():
                                        rk = identity.rank(rgb, cl, mm[fold])
                                        if rk:
                                            p = identity.pick(rk)
                                            row["rank"][f"{src}/{cs}/{mt}"] = {"best": p["key"], "correct": p["key"] == key,
                                                                               "margin": p["margin"], "top": rk[:3]}
                            rows.append(row)
                    print(f"[val_what] {root.name} {t} {ep.name} f{k}: rows {len(rows)}", flush=True)
    json.dump(rows, open(out_path, "w"))
    summarize(rows)


def summarize(rows):
    from collections import defaultdict
    if rows and "chosen" in rows[0]:
        for root in sorted({r["root"] for r in rows}):
            S = [r for r in rows if r["root"] == root]
            adopt = [r for r in S if r["identity_ok"]]
            print(f"{root:8s} size at the hover: n {len(S)} top1 {np.mean([r['correct'] for r in S]):.3f} | identity_ok "
                  f"{len(adopt)} ({len(adopt) / max(len(S), 1):.2f}), precision {np.mean([r['correct'] for r in adopt]) if adopt else float('nan'):.3f}"
                  f" | wrong adopted {sum(1 for r in adopt if not r['correct'])}")
        return
    acc = defaultdict(list)
    for r in rows:
        for v, x in r["rank"].items():
            acc[(r["root"], v)].append((x["correct"], x["margin"]))
    for (root, v), xs in sorted(acc.items()):
        c = np.array([a for a, _ in xs], float)
        m = np.array([b for _, b in xs], float)
        hi = m >= 0.15
        print(f"{root:8s} {v:22s} n {len(xs):4d} top1 {c.mean():.3f} | margin>=0.15: {hi.mean():.2f} of rows, "
              f"precision {c[hi].mean() if hi.any() else float('nan'):.3f} | margin<0.05 precision "
              f"{c[m < 0.05].mean() if (m < 0.05).any() else float('nan'):.3f}")


if __name__ == "__main__":
    main()
