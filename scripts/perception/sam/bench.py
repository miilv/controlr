"""SAM speed / VRAM per model, alone in the process (run in the GPU slot with no SAM server up).

    python scripts/perception/sam/bench.py IMG [IMG ...]     (head frames)

Per model: load time, peak VRAM, and per frame: SAM 3.1 text (image + 8 prompts), point proposals at 5 clicks
(SAM 2.1: 3 masks each; SAM 3.1: the tracker's interactive head, one mask each).
"""
import sys
import time

import cv2
import numpy as np
import torch

from dojo.perception import sam_masks

PROMPTS = ["block", "bowl", "bottle", "t-block", "coin", "piggy bank", "laptop", "headset"]


def main():
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in sys.argv[1:]]
    uv = np.array([[320, 240], [330, 240], [310, 240], [320, 250], [320, 230]], float)
    for name in ("sam2.1_small", "sam2.1_large", "sam3.1"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        M = sam_masks._Models([name])
        load = time.time() - t0
        tt, tp = [], []
        for k, img in enumerate(imgs):
            key = sam_masks.image_key(img)
            if name == "sam3.1":
                t1 = time.time()
                M.handle({"op": "text", "prompts": PROMPTS, "thr": 0.4, "key": key}, img)
                tt.append(time.time() - t1)
            t1 = time.time()
            M.handle({"op": "points", "model": name, "uv": uv.tolist(), "key": key}, img)
            tp.append(time.time() - t1)
        warm = slice(1, None) if len(imgs) > 1 else slice(None)
        print(f"{name}: load {load:.1f} s, peak VRAM {(torch.cuda.max_memory_allocated() - base) / 1e9:.2f} GB, "
              f"5-click proposal {np.median(tp[warm]) * 1000:.0f} ms/object"
              + (f", text {len(PROMPTS)} prompts {np.median(tt[warm]) * 1000:.0f} ms/frame" if tt else ""), flush=True)
        del M
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
