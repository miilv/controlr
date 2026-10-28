"""Incremental classes for a twin-trained YOLO-seg checkpoint (segmenter v3 = v2b + new assets).

    python scripts/perception/seg/expand_head.py --model ~/dojo-data/segtrain/models/v2b/best.pt \\
        --classes ~/dojo-data/segtrain/yolo_v3/classes.json --out ~/dojo-data/segtrain/models/v3_init.pt

Every weight of ``--model`` is kept: its classes keep their ids and their classification-logit rows. The new classes
(the names after the model's own in ``--classes``, an ``export_yolo.py --base-classes`` export) get new rows: zero
weights and the head's prior bias (Ultralytics' ``Detect.bias_init``), so before training they never fire and the
old classes' scores are unchanged. ``train.py --model <out>`` then fine-tunes with every weight loaded (the class
count matches the data); without this, a different class count re-initialises the whole classification layer.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def expand(yolo, names: list[str]) -> tuple[int, int]:
    """Grow the Segment / Detect head of ``yolo`` (an ultralytics YOLO) from its classes to ``names`` (which must
    start with them, in order). Returns (old, new) class counts."""
    import torch

    old = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    if names[:len(old)] != old:
        bad = next(i for i, (a, b) in enumerate(zip(old, names)) if a != b) if len(names) >= len(old) else len(names)
        raise SystemExit(f"--classes must start with the model's {len(old)} classes in order (first mismatch at {bad})")
    head = yolo.model.model[-1]
    n_old, n_new = int(head.nc), len(names)
    for i, seq in enumerate(head.cv3):
        conv = seq[-1]
        new = torch.nn.Conv2d(conv.in_channels, n_new, 1).to(conv.weight.device, conv.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:n_old] = conv.weight
            new.bias.fill_(math.log(5 / n_new / (640 / float(head.stride[i])) ** 2))
            new.bias[:n_old] = conv.bias
        seq[-1] = new
    head.nc = n_new
    head.no = n_new + head.reg_max * 4
    yolo.model.nc = n_new
    yolo.model.yaml["nc"] = n_new
    yolo.model.names = {i: n for i, n in enumerate(names)}
    return n_old, n_new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from ultralytics import YOLO

    y = YOLO(str(Path(a.model).expanduser()))
    names = json.loads(Path(a.classes).expanduser().read_text())["names"]
    n_old, n_new = expand(y, names)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    y.save(str(out))
    print(f"[expand] {n_old} -> {n_new} classes -> {out}", flush=True)


if __name__ == "__main__":
    main()
