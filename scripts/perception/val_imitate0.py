"""Frame-0 B1 on recorded imitate sequences (watch51): per GT mover, found / category / asset / xy error."""
import json, os, sys, time, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(os.environ.get("WATCH_ROOT") or os.path.expanduser("~/dojo-data/memory/watch51/imitate_sorting_sequence"))
TASK = "imitate_sorting_sequence"


def run(ep):
    from dojo.perception import d0
    from dojo.perception.inverse_layout import EstimatorConfig, estimate
    from dojo.perception.segment import load_backgrounds, with_scene
    import cv2
    fold = int(os.environ.get("FOLD", "0"))
    d = Path(os.environ["DOJO_BG_FOLDS"]) / f"fold{fold}"
    bgs = load_backgrounds(d)
    m = d / "cam_head_clean_unseen.png"
    if m.exists():
        bgs["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    # as the perception service: a scene background keyed by the scene's Dynamic / Geometry assets (set_scene)
    from dojo.perception.inverse_layout import task_map
    keys = sorted({k for dd in task_map()[TASK]["labels"].values() if dd.get("type") in ("Dynamic", "Geometry")
                   for k in dd.get("candidates") or []})
    bgs = with_scene(bgs, d, keys)
    cal = json.loads(Path(os.environ["DOJO_COLOR_CALIB"]).read_text())
    kw = json.loads(os.environ.get("DOJO_CFG", "{}"))
    cfg = EstimatorConfig(use_wrists=True, color_M=cal[f"fold{fold}"], **kw)
    ep = Path(ep)
    k = os.environ.get("FRAME", "W0000")
    obs = d0.load_obs(ep, k)
    t0 = time.time()
    rec = estimate(obs["images"], obs["state"], TASK, obs["instruction"], bgs=bgs, cfg=cfg)
    dt = time.time() - t0
    gt = d0.load_gt(ep, k)
    return ep.name, rec, gt, dt


def score(rec, gt):
    preds = []
    for typ in ("Rigid",):
        for cat, lst in (rec.get(typ) or {}).items():
            for r in lst:
                preds.append({"label": r["label"], "key": f"{typ}/{cat}/{int(r['category_idx']):05d}", "cat": cat,
                              "pos": np.asarray(r["default_pos"]), "conf": (r.get("_perception") or {}).get("confidence")})
    G = [o for o in gt["objects"] if o["type"] == "Rigid"]
    C = np.array([[np.linalg.norm(p["pos"][:2] - np.asarray(o["pos"][:2])) for p in preds] for o in G]) if preds else np.zeros((len(G), 0))
    rows = []
    ri, ci = linear_sum_assignment(C) if preds else ([], [])
    mt = dict(zip(ri, ci))
    for i, o in enumerate(G):
        key = f"Rigid/{o['category']}/{o['category_idx']:05d}"
        r = {"label": o["label"], "gt": key, "side": "demo" if o["label"].startswith("aim") else "own"}
        if i in mt and C[i, mt[i]] < 0.05:
            p = preds[mt[i]]
            r.update(found=True, pred=p["key"], xy_mm=round(1000 * C[i, mt[i]], 1), cat_ok=p["cat"] == o["category"],
                     key_ok=p["key"] == key, conf=p["conf"])
        else:
            r.update(found=False, cat_ok=False, key_ok=False)
        rows.append(r)
    n_phantom = len(preds) - sum(1 for r in rows if r["found"])
    return rows, n_phantom


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    ids = os.environ.get("IDS", "").split()
    eps = sorted(p for p in ROOT.iterdir() if p.is_dir() and (not ids or p.name in ids))
    W = int(os.environ.get("WORKERS", "4"))
    allrows = []
    with ProcessPoolExecutor(W) as ex, open(out / "rows.jsonl", "w") as f:
        for name, rec, gt, dt in ex.map(run, [str(e) for e in eps]):
            rows, ph = score(rec, gt)
            (out / f"rec_{name}.json").write_text(json.dumps(rec, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
            for r in rows:
                r["ep"] = name
                f.write(json.dumps(r) + "\n")
            allrows += rows
            fd = sum(r["found"] for r in rows); co = sum(r["cat_ok"] for r in rows); ko = sum(r["key_ok"] for r in rows)
            print(f"{name}: found {fd}/10 cat {co} key {ko} phantoms {ph} {dt:.0f}s | " +
                  " ".join(f"{r['label']}:{'-' if not r['found'] else ('K' if r['key_ok'] else ('C' if r['cat_ok'] else 'x'))}" for r in rows), flush=True)
    for side in ("own", "demo"):
        rs = [r for r in allrows if r["side"] == side]
        xy = [r["xy_mm"] for r in rs if r["found"]]
        print(side, f"n {len(rs)} found {sum(r['found'] for r in rs)} cat {sum(r['cat_ok'] for r in rs)} key {sum(r['key_ok'] for r in rs)} xy med {np.median(xy):.1f} p90 {np.percentile(xy,90):.1f}")
