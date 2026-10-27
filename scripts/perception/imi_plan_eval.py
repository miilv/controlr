"""Imitate end to end, offline (GT for scoring only): frame-0 record (cached rec_<id>.json from val_imitate0) ->
departure reads over the recorded sequence (watch.read_sources, as copy_demonstration calls them) -> the demonstrated
order + the twin the skill would take for each (asset, else category). Per layout: order exact, twins right, plan right.
usage: imi_plan_eval.py REC_DIR [ID ...]   (env WATCH_ROOT, DOJO_BG_FOLDS, DOJO_COLOR_CALIB)"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(os.environ.get("WATCH_ROOT") or os.path.expanduser("~/dojo-data/memory/watch51/imitate_sorting_sequence"))
TASK = "imitate_sorting_sequence"
BASE = np.array([0.0, -0.45])          # midpoint of our arm bases (kin.base): the skill's demo side = farther half


def bgs_for():
    from dojo.perception.segment import load_backgrounds, with_scene
    from dojo.perception.inverse_layout import task_map
    d = Path(os.environ["DOJO_BG_FOLDS"]) / "fold0"
    b = load_backgrounds(d)
    m = d / "cam_head_clean_unseen.png"
    if m.exists():
        b["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    keys = sorted({k for dd in task_map()[TASK]["labels"].values() if dd.get("type") in ("Dynamic", "Geometry")
                   for k in dd.get("candidates") or []})
    return with_scene(b, d, keys)


def main():
    from dojo.perception import d0
    from dojo.perception.inverse_layout import EstimatorConfig
    from dojo.perception.watch import watch_read
    recd = Path(sys.argv[1])
    ids = sys.argv[2:] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    bgs = bgs_for()
    cal = json.loads(Path(os.environ["DOJO_COLOR_CALIB"]).read_text())
    cfg = EstimatorConfig(use_wrists=True, color_M=cal["fold0"])
    tot = {"order": 0, "twins": 0, "plan": 0, "n": 0, "dep_seen": 0, "dep_n": 0, "twin_ok": 0}
    out = []
    for lid in ids:
        ep = ROOT / lid
        rec = json.loads((recd / f"rec_{lid}.json").read_text())
        srcs = [{"asset": f"Rigid/{c}/{int(r['category_idx']):05d}", "label": r["label"], "pos": r["default_pos"],
                 "quat": r["default_ori"], "conf": float((r.get("_perception") or {}).get("confidence", 0.5))}
                for c, lst in (rec.get("Rigid") or {}).items() for r in lst]
        geo = {r["label"]: r for c, lst in (rec.get("Geometry") or {}).items() for r in lst}
        demo_b = max((r for r in geo.values() if "basket" in str(r.get("label", "")) or True), key=lambda r: r["default_pos"][1])
        region = {"kind": "container", "container": {"asset": f"Geometry/basket/{int(demo_b['category_idx']):05d}",
                                                     "pos": demo_b["default_pos"], "quat": demo_b["default_ori"]},
                  "candidates": sorted({s["asset"] for s in srcs}), "sources": srcs}
        order_s = sorted(srcs, key=lambda s: -np.linalg.norm(np.asarray(s["pos"][:2]) - BASE))
        demo = [s["label"] for s in order_s[:len(srcs) // 2]]
        frames = sorted((p.stem[len("frame_"):] for p in ep.glob("frame_W*.json")), key=lambda t: int(t[1:]))
        g0 = d0.load_gt(ep, frames[0])
        G = {o["label"]: o for o in g0["objects"] if o["type"] == "Rigid"}
        def near(s):
            o = min(G.values(), key=lambda o: np.hypot(o["pos"][0] - s["pos"][0], o["pos"][1] - s["pos"][1]))
            return o["label"] if np.hypot(o["pos"][0] - s["pos"][0], o["pos"][1] - s["pos"][1]) < 0.05 else None
        smap = {s["label"]: near(s) for s in srcs}
        prior, gone, gt_gone, gone_any, false_gone = None, [], [], [], []
        for t in frames:
            obs = d0.load_obs(ep, t)
            r = watch_read({"images": obs["images"], "state": obs["state"]}, region, TASK, bgs=bgs, cfg=cfg, prior=prior)
            prior = r
            if os.environ.get("TRACE"):
                with open(recd / f"trace_{lid}.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": t, "src": [(e["label"], smap.get(e["label"]), e["fg"], e["n_px"]) for e in r["sources"]],
                                         "base": r.get("src_base"), "gone": [e["label"] for e in r.get("gone") or []]}) + "\n")
            gt = d0.load_gt(ep, t)
            for e in r.get("gone") or []:
                if e["label"] not in gone_any:
                    gone_any.append(e["label"])
                    # a false departure: the GT object at the source's place has not left it (moved < 20 mm, 3-D: a
                    # lift is vertical) by this read; a source on no GT object (a phantom) is not counted
                    o0 = G.get(smap.get(e["label"]) or "")
                    ot = next((o for o in gt["objects"] if o0 is not None and o["label"] == o0["label"]), None)
                    if o0 is not None and ot is not None and np.linalg.norm(np.subtract(ot["pos"], o0["pos"])) < 0.02:
                        false_gone.append((e["label"], smap.get(e["label"]), t))
                if e["label"] in demo and e["label"] not in gone:
                    gone.append(e["label"])
            for o in gt["objects"]:
                if o["type"] == "Rigid" and o["label"] in G and o["label"] not in gt_gone and \
                        np.hypot(o["pos"][0] - G[o["label"]]["pos"][0], o["pos"][1] - G[o["label"]]["pos"][1]) > 0.03:
                    gt_gone.append(o["label"])
        gt_order = [l for l in gt_gone if l.startswith("aim")]
        got = [smap.get(l) for l in gone]
        # twins as copy_demonstration picks them: same asset, else same category, among our side's free sources
        by = {s["label"]: s for s in srcs}
        used, twins = set(), []
        for l in gone:
            k = by[l]["asset"]
            free = [s["label"] for s in order_s[len(srcs) // 2:] if s["label"] not in used]
            c = [m for m in free if by[m]["asset"] == k] or [m for m in free if by[m]["asset"].split("/")[1] == k.split("/")[1]]
            t_ = c[0] if c else None
            if t_:
                used.add(t_)
            twins.append(t_)
        tw_ok = [bool(a and smap.get(t_) and a.startswith("aim") and smap.get(t_) == "t" + a[3:]) for a, t_ in zip(got, twins)]
        order_ok = got == gt_order and len(gt_order) == 5
        plan_ok = order_ok and all(tw_ok) and len(tw_ok) == 5
        base = (prior or {}).get("src_base") or {}
        row = {"layout": lid, "gt_order": gt_order, "read_order": got, "order_ok": order_ok, "twins_ok": tw_ok,
               "plan_ok": plan_ok, "n_src": len(srcs), "false_gone": false_gone,
               "demo_base": {l: base.get(l) for l in demo}}
        out.append(row)
        tot["n"] += 1; tot["order"] += order_ok; tot["plan"] += plan_ok; tot["twins"] += all(tw_ok) and len(tw_ok) == 5
        tot["dep_n"] += len(gt_order); tot["dep_seen"] += sum(1 for x in got if x in gt_order); tot["twin_ok"] += sum(tw_ok)
        tot["false_gone"] = tot.get("false_gone", 0) + len(false_gone)
        print(json.dumps(row), flush=True)
    (recd / "plan_eval.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print("TOTAL", json.dumps(tot))


if __name__ == "__main__":
    main()
