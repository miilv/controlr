"""Where B1 loses an object on one frame: its head blobs, the segmenter's masks, the series count, the seg-propose trace
and, per GT object, the blobs / masks over its pixels and the nearest prediction. DEV ONLY (reads GT after the
estimate); refuses the measurement ids (100005-09, 100020-29).

    with_seg_server.sh python scripts/perception/sam/b1_trace.py --folds ~/dojo-perception-data/bg_folds_51 \\
        --calib ~/dojo-perception-data/catalog/color_calib_51.json --cfg "$(cat cfg.json)" [--attr '{"k": v}'] EP_DIR...

EP_DIR: a frame-0 directory (``frame_0.json`` + PNGs + ``inst_0_cam_head.png``: seg_render / d0 layout), named by its
layout id (``100000``, ``100030_v1``); its parent is the task. One JSON line per frame on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np

MEASUREMENT = set(range(100005, 100010)) | set(range(100020, 100030))


def layout_id(ep: Path) -> int:
    m = re.match(r"(\d{6})", ep.name)
    if not m:
        raise SystemExit(f"{ep}: no layout id in the directory name")
    return int(m.group(1))


def trace(ep: Path, folds: str, calib: str, cfg_over: dict, attrs: dict) -> dict:
    from PIL import Image

    from dojo.perception import d0, seg_twin
    from dojo.perception import inverse_layout as IL
    from dojo.perception.eval_g1 import fold_backgrounds

    cap: dict = {}
    blobs0, client0 = IL.blobs, seg_twin.client

    def blobs(v, name):
        out = blobs0(v, name)
        if name == "cam_head":
            cap["blobs"] = out
        return out

    class Spy:
        def __init__(self, c):
            self.c = c

        def detect(self, img, conf=0.25):
            d = self.c.detect(img, conf=conf)
            cap.setdefault("dets", d)
            return d

        def __getattr__(self, k):
            return getattr(self.c, k)

    IL.blobs = blobs
    seg_twin.client = lambda: (lambda c: Spy(c) if c is not None else None)(client0())
    try:
        os.environ["DOJO_BG_FOLDS"] = folds
        obs = d0.load_obs(ep, 0)
        kw = {"color_M": json.loads(Path(calib).read_text())[f"fold{zlib.crc32(ep.name.encode()) % 2}"], **cfg_over}
        cfg = IL.EstimatorConfig(use_wrists=True, **kw)
        for k, v in attrs.items():
            setattr(cfg, k, v)
        rec = IL.estimate(obs["images"], obs["state"], obs["task"], obs["instruction"], bgs=fold_backgrounds(ep), cfg=cfg)
    finally:
        IL.blobs, seg_twin.client = blobs0, client0
    gt = d0.load_gt(ep, 0)["objects"]                       # DEV ONLY, after the estimate
    inst = np.asarray(Image.open(ep / "inst_0_cam_head.png"))
    p = rec.get("_perception", {})
    bl, dets = cap.get("blobs", []), cap.get("dets") or []
    preds = [{"key": f"{t}/{c}/{int(r['category_idx']):05d}", "pos": [round(float(v), 3) for v in r["default_pos"][:2]],
              "conf": (r.get("_perception") or {}).get("confidence")}
             for t in ("Rigid",) for c, lst in (rec.get(t) or {}).items() for r in lst]
    rows = []
    for i, o in enumerate(gt):
        if o.get("clutter") or o.get("type") != "Rigid":
            continue
        m = inst == i + 1
        near = min(((float(np.hypot(q["pos"][0] - o["pos"][0], q["pos"][1] - o["pos"][1])), q["key"]) for q in preds),
                   default=None)
        rows.append({"key": f"Rigid/{o['category']}/{int(o['category_idx']):05d}", "px": int(m.sum()),
                     "blobs": [[j, int((b.mask[m] > 0).sum())] for j, b in enumerate(bl) if (b.mask[m] > 0).any()],
                     "dets": [[d.key, round(float(d.score), 2), int(((d.mask > 0) & m).sum())] for d in dets
                              if ((d.mask > 0) & m).any()],
                     "nearest": [round(near[0] * 1000), near[1]] if near else None})
    return {"ep": ep.name, "instruction": obs["instruction"], "series": p.get("series"),
            "seg": (p.get("sam") or {}).get("seg"), "unexplained": p.get("unexplained"),
            "blobs": [{"i": j, "area": int(b.area), "c": [round(float(x), 1) for x in b.centroid]} for j, b in enumerate(bl)],
            "dets": [{"key": d.key, "score": round(float(d.score), 3), "area": int(d.mask.sum())} for d in dets],
            "preds": preds, "gt": rows}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("eps", nargs="+")
    ap.add_argument("--folds", required=True, help="background folds (bg_folds_51 for genuine 5.1 frames)")
    ap.add_argument("--calib", required=True, help="colour calibration json (color_calib_51.json for 5.1)")
    ap.add_argument("--cfg", default="{}", help="EstimatorConfig fields as JSON (e.g. the bake's cfg)")
    ap.add_argument("--attr", default="{}", help="attributes set on the config after construction (a probe)")
    a = ap.parse_args(argv)
    for e in a.eps:
        ep = Path(e).expanduser()
        if layout_id(ep) in MEASUREMENT:
            raise SystemExit(f"{ep}: a measurement id (100005-09, 100020-29) is for scoring only, never traced")
        print(json.dumps(trace(ep, a.folds, a.calib, json.loads(a.cfg), json.loads(a.attr)), default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
