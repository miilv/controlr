"""DEV: the plane-sweep height map on genuine-5.1 conveyor frames (perception2's belt51 copy: arms parked at the scan
pose, head + both wrists at the same instant, GT per frame). Per object on the belt: decided pixels, proud share,
detected (>= 30 % of its decided raised pixels proud), top-surface height error.

    DOJO_GL=0 python scripts/perception/stereo_belt.py --root ~/dojo-data/perception2/memcopy/belt51 --out OUT [--every 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dojo.perception.stereo_eval import run_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2, help="every n-th W frame")
    ap.add_argument("--sweep", default='{"agg":5,"fill":true,"min_blob":30,"lr":true}')
    ap.add_argument("--viz", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(a.sweep)
    n = 0
    with open(out / "rows.jsonl", "w") as fh:
        for ep in sorted(Path(a.root).expanduser().glob("*/*")):
            if not ep.is_dir() or int(ep.name) not in set(range(100000, 100010)) | set(range(100020, 100050)):
                continue
            ks = sorted(p.stem[len("frame_"):] for p in ep.glob("frame_W*.json"))[:: a.every]
            for k in ks:
                r = run_frame(ep, k, 1.0, out if n < a.viz else None, kw)
                n += 1
                r["frame"] = k
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(ep.name, k, [(o["category"], o.get("px_known"), o.get("proud_frac"), o.get("detected"), o.get("top_err_mm"))
                                   for o in r["objects"] if o["category"] not in ("camera_stand",)], flush=True)


if __name__ == "__main__":
    main()
