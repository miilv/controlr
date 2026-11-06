"""Twin-trained segmenter: train a YOLO-seg model on export_yolo's dataset, duty-cycled to a box-wide power budget.

    YOLO_CONFIG_DIR=~/dojo-data/segtrain/ultra_cfg python scripts/perception/seg/train.py \\
        --data ~/dojo-data/segtrain/yolo_v1/data.yaml --model ~/dojo-data/segtrain/models/yolo11s-seg.pt \\
        --epochs 60 --batch 16 --workers 6 --project ~/dojo-data/segtrain/runs --name v1 \\
        [--power-target 270 --power-max 420]

POWER (compute4 rule: box-wide median <= 300 W, max <= 450 W): after every training batch the trainer reads the GPU's
total board power (NVML; every process on the GPU, not only ours) and sleeps. The sleep grows by 30 % while the 20-s
mean is above ``--power-target`` and shrinks by 15 % while it is 40 W below; any reading above ``--power-max`` stalls
the trainer until the power is back under the target. Every batch's reading and sleep go to ``<run>/power_train.csv``.
"""
from __future__ import annotations

import argparse
import collections
import os
import time
from pathlib import Path


class Throttle:
    def __init__(self, target_w: float, max_w: float, log: Path, sleep0: float = 0.3, window_s: float = 20.0,
                 power_fn=None, sleep_fn=time.sleep):
        self.target, self.max = float(target_w), float(max_w)
        self.sleep = float(sleep0)
        self.window = float(window_s)
        self.hist: collections.deque = collections.deque()
        self.log = log
        self.power_fn = power_fn or _nvml_power()
        self.sleep_fn = sleep_fn
        self.n = 0
        self.stall_s = 0.0
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            if not log.exists():
                log.write_text("t,power_w,mean_w,sleep_s,stall_s\n")

    def mean(self, now: float) -> float:
        while self.hist and now - self.hist[0][0] > self.window:
            self.hist.popleft()
        return sum(p for _, p in self.hist) / max(len(self.hist), 1)

    def step(self, now: float | None = None) -> float:
        """One batch boundary: read the power, adapt the sleep, sleep. Returns the seconds slept."""
        now = time.time() if now is None else now
        p = self.power_fn()
        self.hist.append((now, p))
        m = self.mean(now)
        stall = 0.0
        while p > self.max and stall < 30.0:      # over the hard cap: wait until the box is back under the target
            self.sleep_fn(1.0)
            stall += 1.0
            p = self.power_fn()
            if p <= self.target:
                break
        if m > self.target:
            self.sleep = min(self.sleep * 1.3 + 0.02, 8.0)
        elif m < self.target - 40.0:
            self.sleep = max(self.sleep * 0.85, 0.02)
        self.sleep_fn(self.sleep)
        self.stall_s += stall
        self.n += 1
        if self.log is not None and self.n % 5 == 0:
            with self.log.open("a") as f:
                f.write(f"{now:.1f},{p:.1f},{m:.1f},{self.sleep:.3f},{stall:.0f}\n")
        return self.sleep + stall


def _nvml_power():
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0


def auto_batch_budget(free: int, total: int, headroom_gb: float):
    """YOLO auto-batch target and allocator cap, preserving shared VRAM reserve."""
    import math

    if not math.isfinite(headroom_gb) or headroom_gb < 6:
        raise ValueError("auto batch headroom must be at least 6 GiB")
    reserve = headroom_gb * 2**30
    if not 0 < free <= total or free <= reserve:
        raise RuntimeError("insufficient free VRAM after requested headroom")
    return (free - reserve) / free, (free - reserve) / total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-headroom-gb", type=float, default=None,
                    help="reserve this much free VRAM; with --batch -1 size automatically, "
                         "then reuse the resolved integer for matched training arms")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--power-target", type=float, default=270.0)
    ap.add_argument("--power-max", type=float, default=420.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--no-amp-check", action="store_true",
                    help="AMP on without Ultralytics' check (it downloads a YOLO model to run the check)")
    ap.add_argument("--no-font-check", action="store_true",
                    help="skip Ultralytics' label-font check (matplotlib + a font download; an inference-only "
                         "environment without matplotlib's dependencies; plots are off anyway)")
    a = ap.parse_args(argv)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(a.project).expanduser().parent / "ultra_cfg"))
    from ultralytics import YOLO

    if a.no_amp_check:
        import ultralytics.engine.trainer as _tr

        _tr.check_amp = lambda *args, **kw: True
    if a.no_font_check:
        import ultralytics.data.utils as _du

        _du.check_font = lambda *args, **kw: None

    run = Path(a.project).expanduser() / a.name
    thr = Throttle(a.power_target, a.power_max, run / "power_train.csv")
    model = YOLO(str(run / "weights" / "last.pt") if a.resume else os.path.expanduser(a.model))
    model.add_callback("on_train_batch_end", lambda trainer: thr.step())
    batch = a.batch
    if a.batch_headroom_gb is not None:
        import torch

        free, total = torch.cuda.mem_get_info()
        fraction, limit = auto_batch_budget(free, total, a.batch_headroom_gb)
        batch = fraction if a.batch < 1 else a.batch
        torch.cuda.set_per_process_memory_fraction(limit)
        print(f"[train] batch={batch}; reserve={a.batch_headroom_gb} GiB", flush=True)

        def check_headroom(trainer):
            if torch.cuda.mem_get_info()[0] < 6 * 2**30:
                raise RuntimeError("shared GPU free VRAM dropped below 6 GiB")

        model.add_callback("on_train_batch_end", check_headroom)

        def record_batch(trainer):
            import json

            (run / "resolved_batch.json").write_text(json.dumps({"batch": int(trainer.batch_size),
                "headroom_gib": a.batch_headroom_gb, "auto": a.batch < 1}) + "\n")

        model.add_callback("on_train_start", record_batch)
    t0 = time.time()
    # augmentations: geometry stays mild (the cameras are fixed; wrist views vary by the arm pose); no mosaic in the
    # last 10 epochs; colour jitter small (one renderer, one lighting)
    model.train(data=os.path.expanduser(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=batch, workers=a.workers,
                project=str(Path(a.project).expanduser()), name=a.name, exist_ok=True, patience=a.patience,
                hsv_h=0.005, hsv_s=0.3, hsv_v=0.25, degrees=0.0, translate=0.1, scale=0.3, fliplr=0.0, mosaic=1.0,
                close_mosaic=a.close_mosaic, overlap_mask=True, plots=False, amp=True, resume=a.resume, cache=False)
    print(f"[train] done in {(time.time() - t0) / 3600:.2f} h; throttle slept {thr.stall_s:.0f} s stalled, "
          f"last sleep {thr.sleep:.2f} s/batch", flush=True)


if __name__ == "__main__":
    main()
