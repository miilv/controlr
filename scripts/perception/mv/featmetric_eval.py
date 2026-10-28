"""DEV-ONLY E3 (research R3), discrete stage: multi-view DINOv2 feature-metric scoring of B1's frame-0 poses against
their yaw turns (and, with --cands, other sources' candidates), over head + wrist views (+ D0's look frames with
--views x). Writes every candidate's distance so margins are chosen offline; eval_g1 before / after per margin.

    python scripts/perception/mv/featmetric_eval.py --out DIR [--views home|x] [--model dinov2_b] [--n-yaw 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

CALIB = {"d0_51": "color_calib_51.json", "d0_ext": "color_calib.json", "d0_coin": "color_calib.json"}


def scene_files(roots: list[str]):
    from dojo.perception.mv3d_eval import scene_list

    return scene_list(roots)


def run_scene(ep: Path, fm, b1dump: Path, views_mode: str, n_yaw: int, gt_cand: bool = False) -> dict:
    from dojo.perception.cameras import R_to_quat_wxyz, pose_to_T
    from dojo.perception.featmetric import yaw_turns
    from dojo.perception.mv3d_eval import SET_ALIAS, _b1_record, scene_views
    from dojo.perception.render import load_asset, robot_mask

    root_name, task, epn = ep.parent.parent.name, ep.parent.name, ep.name
    mf = b1dump / {v: k for k, v in SET_ALIAS.items()}.get(root_name, root_name) / task / epn / "meta.json"
    meta = json.loads(mf.read_text())
    calib = json.loads((Path(os.path.expanduser("~/dojo-perception-data/catalog")) / CALIB.get(root_name, "color_calib.json")).read_text())
    fm.set_color(calib.get(f"fold{zlib.crc32(epn.encode()) % 2}"))
    vs = scene_views(ep, views_mode)
    views = [{"name": f"{v['name']}@{v['frame']}", "img": v["img"], "cam": v["cam"],
              "robot": robot_mask(v["cam"], v["state"], dilate_px=2)} for v in vs]
    rec = _b1_record(meta)
    gt_objs = None
    if gt_cand:  # DEV DIAGNOSTIC: the GT pose as one more candidate (is the right answer the feature-metric minimum?)
        from dojo.perception import d0
        from dojo.skills.artroot import row_to_asset

        gt_objs = d0.load_gt(ep, 0)["objects"]
    objs = []
    t0 = time.time()
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec.get(typ) or {}).items():
            for j, r in enumerate(lst):
                key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
                try:
                    a = load_asset(key)
                except Exception:
                    continue
                T = pose_to_T(r["default_pos"], r["default_ori"])
                ctr = T[:3, :3] @ ((a.V.min(0) + a.V.max(0)) / 2) + T[:3, 3]
                cands = yaw_turns(T, n_yaw, ctr)
                has_gt = False
                gi = r.get("_gt_i")
                if gt_objs is not None and gi is not None and 0 <= gi < len(gt_objs) and gt_objs[gi].get("pos") is not None:
                    g = row_to_asset(gt_objs[gi])
                    if f"{g['type']}/{g['category']}/{int(g['category_idx']):05d}" == key:  # same asset: a pose question
                        cands.append(pose_to_T(g["pos"], g["quat"]))
                        has_gt = True
                d, info = fm.score(key, cands, views)
                objs.append({"typ": typ, "cat": cat, "j": j, "label": r.get("label"), "gt_i": r.get("_gt_i"), "has_gt": has_gt,
                             "d": [None if not np.isfinite(x) else round(float(x), 5) for x in d], "views": info,
                             "cands": [{"pos": [round(float(v), 5) for v in c[:3, 3]],
                                        "ori": [round(float(v), 6) for v in R_to_quat_wxyz(c[:3, :3])]} for c in cands]})
    return {"set": root_name, "task": task, "episode": epn, "objects": objs, "s": round(time.time() - t0, 2)}


def apply(rows: list[dict], margin: float, b1dump: Path) -> dict:
    """eval_g1 before / after switching each object to its lowest-distance candidate when it beats B1's by ``margin``."""
    import copy
    from collections import Counter

    from dojo.perception import d0
    from dojo.perception.mv3d_eval import SET_ALIAS, _b1_record, _score, root_dir

    tot = Counter()
    switched = 0
    for r in rows:
        if "error" in r:
            continue
        meta = json.loads((b1dump / {v: k for k, v in SET_ALIAS.items()}.get(r["set"], r["set"]) / r["task"] / r["episode"] / "meta.json").read_text())
        rec0 = _b1_record(meta)
        rec1 = copy.deepcopy(rec0)
        for o in r["objects"]:
            d = np.array([np.inf if x is None else x for x in o["d"]])
            if o.get("has_gt"):
                d = d[:-1]  # the GT candidate is a diagnostic, never applied
            if not np.isfinite(d[0]):
                continue
            k = int(np.argmin(d))
            if k != 0 and d[k] < d[0] - margin:
                e = rec1[o["typ"]][o["cat"]][o["j"]]
                e["default_pos"], e["default_ori"] = o["cands"][k]["pos"], o["cands"][k]["ori"]
                switched += 1
        gt = d0.load_gt(root_dir(r["set"]) / r["task"] / r["episode"], 0)  # DEV-ONLY
        m = {"set": r["set"], "task": r["task"], "episode": r["episode"], "frame": 0}
        c0, _ = _score(rec0, gt, r["task"], m)
        c1, _ = _score(rec1, gt, r["task"], m)
        for kk in c0:
            tot[("before", c0[kk])] += 1
            tot[("after", c1.get(kk))] += 1
            if c0[kk] != c1.get(kk):
                tot[(c0[kk], c1.get(kk))] += 1
    within = lambda s: tot[(s, "ok")] + tot[(s, "identity")]  # noqa: E731
    return {"margin": margin, "switched": switched, "within_before": within("before"), "within_after": within("after"),
            "changes": {f"{a}->{b}": v for (a, b), v in tot.items() if a not in ("before", "after")}}


def gt_diag(rows: list[dict], b1dump: Path) -> dict:
    """With the GT pose among the candidates: how often the feature metric ranks GT below B1's pose, split by how far
    B1's pose is from GT (eval_g1 class of that object)."""
    from collections import Counter

    from dojo.perception import d0
    from dojo.perception.mv3d_eval import _b1_record, _score, root_dir, SET_ALIAS

    c = Counter()
    for r in rows:
        if "error" in r or not any(o.get("has_gt") for o in r["objects"]):
            continue
        meta = json.loads((b1dump / {v: k for k, v in SET_ALIAS.items()}.get(r["set"], r["set"]) / r["task"] / r["episode"] / "meta.json").read_text())
        gt = d0.load_gt(root_dir(r["set"]) / r["task"] / r["episode"], 0)
        cls, _ = _score(_b1_record(meta), gt, r["task"], {"set": r["set"], "task": r["task"], "episode": r["episode"], "frame": 0})
        bypos = {k[2]: v for k, v in cls.items()}
        for o in r["objects"]:
            if not o.get("has_gt") or o["d"][0] is None or o["d"][-1] is None:
                continue
            k = tuple(np.round(gt["objects"][o["gt_i"]]["pos"], 4))
            cl = bypos.get(k, "?")
            c[(cl, "n")] += 1
            c[(cl, "gt_better")] += o["d"][-1] < o["d"][0]
            c[(cl, "gt_better_by_0.02")] += o["d"][-1] < o["d"][0] - 0.02
    return {"gt_diag": {f"{a}:{b}": v for (a, b), v in sorted(c.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=["~/dojo-data/d0_51", "~/dojo-data/d0_ext"])
    ap.add_argument("--b1dump", default="~/dojo-data/pbake-mv/c4/sam/b1/dump")
    ap.add_argument("--views", default="home")
    ap.add_argument("--model", default="dinov2_b")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-yaw", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply-only", action="store_true")
    ap.add_argument("--gt-cand", action="store_true", help="DEV diagnostic: add the GT pose as a candidate (never applied)")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    b1dump = Path(a.b1dump).expanduser()
    fr = out / "frames.jsonl"
    if not a.apply_only:
        from dojo.perception.featmetric import FeatureMetric

        fm = FeatureMetric(a.model, size=a.size, layer=a.layer)
        eps = scene_files([os.path.expanduser(r) for r in a.roots])
        if a.limit:
            eps = eps[:a.limit]
        with open(fr, "w") as fh:
            for ep in eps:
                try:
                    r = run_scene(ep, fm, b1dump, a.views, a.n_yaw, gt_cand=a.gt_cand)
                except Exception as e:
                    import traceback

                    r = {"file": str(ep), "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(r.get("task"), r.get("episode"), r.get("error") or (len(r["objects"]), r["s"]), flush=True)
    rows = [json.loads(line) for line in open(fr)]
    res = [apply(rows, m, b1dump) for m in (0.0, 0.01, 0.02, 0.04, 0.08)]
    res.append(gt_diag(rows, b1dump))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
