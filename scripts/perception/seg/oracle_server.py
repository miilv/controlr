"""A segmenter server that answers with the GT instance masks (an ORACLE: DEV ONLY, never a runtime component). It
speaks seg_twin's wire format, so ``seg_propose`` runs unchanged against it: the ceiling of the proposer with a perfect
twin-trained segmenter (every visible object, exact mask, right asset class).

    python scripts/perception/seg/oracle_server.py --gt ~/dojo-data/segtrain/eval_gt --sock ~/dojo-data/segtrain/oracle.sock \\
        [--min-px 30] [--drop-clutter]

Images are matched by content (``sam_masks.image_key`` of the frame's head RGB, as d0.load_obs reads it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SETS = {"d0_51": Path.home() / "dojo-data/d0_51", "d0_ext": Path.home() / "dojo-data/d0_ext"}


class Oracle:
    def __init__(self, gt_root: Path, min_px: int, drop_clutter: bool):
        from PIL import Image

        from dojo.perception.d0 import _read_rgb
        from dojo.perception.sam_masks import image_key

        self.by_key = {}
        for mp in sorted(gt_root.glob("*/*/*/gt_*.json")):
            m = json.loads(mp.read_text())
            if "cam_head" not in m["cams"]:
                continue
            d = SETS[m["set"]] / m["task"] / m["episode"]
            k = m["frame"]
            img = _read_rgb(d / f"frame_{k}_cam_head.png")
            objs = json.loads((d / f"frame_{k}.json").read_text())["gt"]["objects"]
            self.by_key[image_key(img)] = (mp.parent / f"gt_{k}_cam_head.png", objs)
        self.min_px, self.drop_clutter = min_px, drop_clutter
        self.Image = Image
        self.n = 0
        print(f"[oracle] {len(self.by_key)} head frames", flush=True)

    def handle(self, req, img):
        from dojo.perception.sam_masks import pack
        from dojo.perception.seg_twin import asset_key, instances

        hit = self.by_key.get(req["key"])
        if hit is None:
            return {"dets": []}
        p, objs = hit
        codes = np.asarray(self.Image.open(p)).astype(np.int64)
        out = []
        for i, m in instances(codes, self.min_px).items():
            if i >= len(objs) or (self.drop_clutter and objs[i].get("clutter")):
                continue
            ys, xs = np.nonzero(m)
            out.append((*pack(m), asset_key(objs[i]), 1.0, [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]))
        self.n += 1
        return {"dets": out}

    def info(self):
        return {"oracle": True, "frames": len(self.by_key), "served": self.n}


def main(argv=None):
    from dojo.perception.sam_masks import serve

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sock", required=True)
    ap.add_argument("--min-px", type=int, default=30)
    ap.add_argument("--drop-clutter", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8.0)
    a = ap.parse_args(argv)
    M = Oracle(Path(a.gt).expanduser(), a.min_px, a.drop_clutter)
    serve(os.path.expanduser(a.sock), ["oracle"], a.max_hours, M=M)


if __name__ == "__main__":
    main()
