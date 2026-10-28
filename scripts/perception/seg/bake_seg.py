"""Twin-trained segmenter, end to end: B1 with / without ``seg_propose`` on the held-out frame-0 frames, scored with
eval_g1's scorer through the SAM bake-off's driver (``scripts/perception/sam/bake.py``: same frames, same scorer, same
summary). DEV ONLY (the scorer reads GT after the estimate).

    DOJO_SEG_ADDR=SOCK python scripts/perception/seg/bake_seg.py run --out DIR [--cfg '{"seg_propose": "on"}'] \\
        [--sets d0_51,d0_ext] [--workers 6]
    python scripts/perception/sam/bake.py summary DIR [BASE_DIR]

Frames: ids 100000-09 and 100020-24 only (the segmenter's held-out evaluation ids; its training draws are variants of
100030-49). CPU only by default (B1's CPU silhouettes, DOJO_GL=0: no GPU process; ``--gl gpu`` uses the box's EGL).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/perception/sam"))
import bake  # noqa: E402  (the SAM bake-off's driver: frames, run_frame, scorer; importable by the worker processes)

CPU_GL_ENV = {"LIBGL_ALWAYS_SOFTWARE": "1", "EGL_DEVICE_ID": "1", "PYOPENGL_PLATFORM": "egl",
              "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json", "LP_NUM_THREADS": "1"}


def held_out(i: int) -> bool:
    return 100000 <= i <= 100009 or 100020 <= i <= 100024


def transitions(dir_a: str, dir_b: str) -> None:
    """Per-object failure class in run B (baseline) -> run A, as a matrix, plus the per-task fixed / broken lists."""
    from collections import Counter, defaultdict

    from dojo.perception.eval_g1 import TOLERANCE, failure_class

    def load(d):
        out = {}
        for line in open(Path(d) / "objects.jsonl"):
            r = json.loads(line)
            k = (r["set"], r["task"], r["episode"], r["label"], tuple(round(x, 4) for x in r["gt_pos"]))
            out[k] = failure_class(r, TOLERANCE.get(r["task"], (None, None, "")))
        return out

    A, B = load(dir_a), load(dir_b)
    keys = sorted(set(A) & set(B))
    cls = ("ok", "identity", "overshoot", "gross", "missed")
    m = Counter((B[k], A[k]) for k in keys)
    print(f"{len(keys)} objects; rows = baseline ({dir_b}), columns = {dir_a}\n")
    print("| baseline \\ arm | " + " | ".join(cls) + " |")
    print("|---" * (len(cls) + 1) + "|")
    for b in cls:
        print(f"| {b} | " + " | ".join(str(m[(b, a)]) for a in cls) + " |")
    by = defaultdict(lambda: [0, 0])
    for k in keys:
        good_b, good_a = B[k] in ("ok", "identity"), A[k] in ("ok", "identity")
        if good_a and not good_b:
            by[k[1]][0] += 1
        if good_b and not good_a:
            by[k[1]][1] += 1
    print("\nwithin tolerance, fixed / broken per task: " + ", ".join(f"{t} {f}/{b}" for t, (f, b) in sorted(by.items())))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--cfg", default="")
    r.add_argument("--sets", default="d0_51,d0_ext")
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--episodes", nargs="*", help="only these TASK/ID frames (debugging)")
    t = sub.add_parser("transitions")
    t.add_argument("dir")
    t.add_argument("base")
    r.add_argument("--gl", choices=("none", "llvmpipe", "gpu"), default="none",
                   help="none: B1's CPU silhouettes (DOJO_GL=0; ~5x faster than llvmpipe); llvmpipe: Mesa's software GL; "
                        "gpu: the box's EGL (a GPU process)")
    a = ap.parse_args(argv)
    if a.cmd == "transitions":
        return transitions(a.dir, a.base)
    if a.gl == "llvmpipe":
        os.environ.update(CPU_GL_ENV)
    elif a.gl == "none":
        os.environ["DOJO_GL"] = "0"
    sets = a.sets.split(",")
    jobs = [(s, ep, json.loads(a.cfg) if a.cfg else None, str(a.out), False) for s, ep in bake.frames(sets, a.tasks)
            if held_out(int(Path(ep).name))
            and (not a.episodes or f"{Path(ep).parent.name}/{Path(ep).name}" in a.episodes)]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({"cfg": a.cfg, "sets": sets, "tasks": a.tasks, "n_frames": len(jobs),
                                                   "seg_addr": os.environ.get("DOJO_SEG_ADDR", ""), "argv": sys.argv,
                                                   "gl": a.gl}))
    print(len(jobs), "frames", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(a.out / "objects.jsonl", "w") as fo, \
            open(a.out / "frames.jsonl", "w") as ff:
        for i, res in enumerate(ex.map(bake.run_frame, jobs)):
            ff.write(json.dumps(res["frame"]) + "\n")
            for row in res["objects"]:
                fo.write(json.dumps(row) + "\n")
            if res["frame"].get("error"):
                print("ERR", res["frame"]["task"], res["frame"]["episode"], res["frame"]["error"][:300], flush=True)
            if i % 20 == 0:
                print(i, res["frame"]["task"], res["frame"]["episode"], round(res["frame"]["runtime_s"], 1),
                      round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
