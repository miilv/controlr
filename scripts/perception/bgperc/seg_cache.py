"""Stream C, X2: the frozen segmenter's detections on every stored frame, computed once and cached by the frame's
content hash, so every arm (rt / J / S, all replayed offline on the same stored frames) reads identical detections.

    SEG_WEIGHTS=.../v2b/best.pt SEG_CLASSES=.../v2b/classes.json \\
        python scripts/perception/bgperc/seg_cache.py SEQ_ROOT OUT_DIR [--valid VALID.json] [--names a,b]

Per sequence ``OUT_DIR/<name>.json.gz``: {"header": {weights_sha1, imgsz, conf, classes_sha1, model}, "frames":
{"<step:04d>_<cam>.<fmt>": {"sha1": sha1 of the stored file's bytes, "dets": [{"key", "score", "box", "mask"}]}}}
(``<fmt>``: the sequence's ``meta.json`` "fmt", png = lossless; a jpg sequence's frame-0 PNG is not cached) where "mask"
is the mask cropped to its own bounding box, as base64 of ``np.packbits`` with its crop shape (``decode_mask``). ``OUT_DIR/index.json``:
per sequence its file and that file's sha1 (the cache hash ``valid.json`` records).
"""
import argparse
import base64
import glob
import gzip
import hashlib
import json
import os
import time

import numpy as np


def encode_mask(m: np.ndarray) -> dict:
    """Lossless: the crop is the mask's own bounding box (the upsampled mask can spill past the detector's box)."""
    m = np.asarray(m, bool)
    ys, xs = np.nonzero(m)
    if not len(ys):
        return {"xy": [0, 0], "shape": [0, 0], "bits": ""}
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = m[y0:y1, x0:x1]
    return {"xy": [x0, y0], "shape": list(crop.shape), "bits": base64.b64encode(np.packbits(crop).tobytes()).decode()}


def decode_mask(d: dict, H: int = 480, W: int = 640) -> np.ndarray:
    h, w = d["shape"]
    crop = np.unpackbits(np.frombuffer(base64.b64decode(d["bits"]), np.uint8))[: h * w].reshape(h, w).astype(bool)
    out = np.zeros((H, W), bool)
    x0, y0 = d["xy"]
    out[y0:y0 + h, x0:x0 + w] = crop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--valid", default=None)
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    import cv2

    from dojo.perception.seg_twin import _SegModel
    w, c = os.environ["SEG_WEIGHTS"], os.environ["SEG_CLASSES"]
    cl = json.load(open(c))
    classes = cl if isinstance(cl, list) else cl.get("classes") or cl.get("names")
    model = _SegModel(w, classes)
    header = {"weights_sha1": hashlib.sha1(open(w, "rb").read()).hexdigest(),
              "classes_sha1": hashlib.sha1(open(c, "rb").read()).hexdigest(), "imgsz": model.imgsz,
              "conf": model.conf, "model": os.path.basename(os.path.dirname(w))}
    os.makedirs(a.out, exist_ok=True)
    names = sorted(os.listdir(a.root))
    if a.valid:
        ok = {r["name"] for r in json.load(open(a.valid)) if r.get("valid")}
        names = [n for n in names if n in ok]
    if a.names:
        names = [n for n in names if n in set(a.names.split(","))]
    idx_path = os.path.join(a.out, "index.json")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    for n in names:
        dst = os.path.join(a.out, f"{n}.json.gz")
        if os.path.exists(dst) and n in index:
            continue
        t0, frames = time.time(), {}
        fmt = json.load(open(os.path.join(a.root, n, "meta.json"))).get("fmt", "png")
        for f in sorted(glob.glob(os.path.join(a.root, n, "frames", f"*.{fmt}"))):
            raw = open(f, "rb").read()
            img = np.ascontiguousarray(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1])
            dets = [{"key": k, "score": round(s, 4), "box": [round(v, 1) for v in b], "mask": encode_mask(m)}
                    for m, k, s, b in model.detect(img)]
            frames[os.path.basename(f)] = {"sha1": hashlib.sha1(raw).hexdigest(), "dets": dets}
        with gzip.open(dst, "wt") as fh:
            json.dump({"header": header, "frames": frames}, fh)
        index[n] = {"file": os.path.basename(dst), "sha1": hashlib.sha1(open(dst, "rb").read()).hexdigest(),
                    "frames": len(frames)}
        json.dump(index, open(idx_path, "w"), indent=1)
        print(f"[seg_cache] {n}: {len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    json.dump(index, open(idx_path, "w"), indent=1)
    with open(os.path.join(a.out, "header.json"), "w") as fh:
        json.dump(header, fh, indent=1)


if __name__ == "__main__":
    main()
