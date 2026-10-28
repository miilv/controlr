"""Twin-trained instance segmenter (perception stream S): labels, metrics and the detector service.

The model (an instance segmenter whose class is the catalogue asset, ``Type/category/idx``) is trained only on our own
simulator's renders of NEW layout draws (``dojo.simlab.seg_render``); this module holds what the label export, the
training, the evaluation and the runtime proposer share:

- labels: an instance code image (``seg_render``'s ``inst_<j>_<cam>.png`` from Isaac's annotator, or ``gt_code_map``:
  GT poses + catalogue meshes + the arms, rendered with our calibrated cameras) -> per-object masks -> YOLO polygons;
- metrics: mask IoU, greedy IoU matching of predictions to GT instances;
- the detector service: ``python -m dojo.perception.seg_twin serve --sock S --weights W`` (torch + ultralytics, one
  GPU process; the same wire format and server loop as ``sam_masks``), and ``client()`` (``DOJO_SEG_ADDR``; None
  without a server, so every flag that uses it is a no-op then).

GT is read only by the label / eval tools (dev), never by the estimator.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

ENV_ADDR = "DOJO_SEG_ADDR"
ROBOT_CODE = 65534
OTHER_CODE = 65533
MIN_PX = 20


def asset_key(o: dict) -> str:
    """Catalogue key of a GT row / layout prediction (``type``, ``category``, ``category_idx``)."""
    return f"{o['type']}/{o['category']}/{int(o['category_idx']):05d}"


def category_of(key: str) -> str:
    return key.split("/")[1] if key.count("/") >= 2 else key


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------

def gt_camera(gt: dict, name: str):
    """The exact camera of a frame from its DEV-ONLY GT record (``gt["cameras"][name]``: K, T_world_cam_cv)."""
    from .cameras import Camera

    c = gt["cameras"][name]
    return Camera(name, np.asarray(c["K"], float), np.asarray(c["T_world_cam_cv"], float), int(c["width"]), int(c["height"]))


def gt_code_map(cam, objects: list[dict], state: dict | None = None, robot: bool = True, table: bool = True,
                renderer=None) -> tuple[np.ndarray, list[bool]]:
    """Instance code image of GT ``objects`` seen by ``cam`` (0 = nothing, i + 1 = objects[i], ROBOT_CODE = an arm,
    OTHER_CODE = the table slab), with occlusion between them, rendered from the GT poses and the catalogue meshes
    (articulations in their asset frame). Returns the codes and, per object, whether it could be rendered (a catalogue
    mesh exists). DEV ONLY."""
    from dojo.skills.artroot import row_to_asset

    from .cameras import pose_to_T
    from .render import load_asset, robot_items

    items, ok = [], []
    for i, o in enumerate(objects):
        if o.get("pos") is None or o.get("quat") is None:
            ok.append(False)
            continue
        try:
            a = load_asset(asset_key(o))
            r = row_to_asset(o)
            items.append({"id": i + 1, "key": asset_key(o), "V": a.Vg, "F": a.Fg, "T": pose_to_T(r["pos"], r["quat"])})
            ok.append(True)
        except Exception:  # noqa: BLE001 (no catalogue mesh: not rendered)
            ok.append(False)
    if robot and state is not None:
        items += [{**it, "id": ROBOT_CODE} for it in robot_items(state, closed_jaws=False)]
    if table:   # the table occludes what stands below its top (put_bottles' dustbin on the floor)
        items.append(_table_item())
    sr = renderer or _renderer(cam.w, cam.h)
    _, ids, _ = sr.render(cam, items)
    return ids.astype(np.int64).clip(0, 65535).astype(np.uint16), ok


# the default scene's table (env_cfg/scene/default.yml: 1.4 x 1.1 m, top at z 0.765); a conveyor scene's belt is a
# fixture object of its own
TABLE_BOX = ((-0.70, 0.70), (-0.60, 0.50), (0.715, 0.7645))


def _table_item() -> dict:
    (x0, x1), (y0, y1), (z0, z1) = TABLE_BOX
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    return {"id": OTHER_CODE, "key": "seg:table", "V": V, "F": F, "T": np.eye(4)}


_SR: dict = {}


def _renderer(w: int, h: int):
    from .render import SceneRenderer

    r = _SR.get((w, h))
    if r is None:
        r = _SR[(w, h)] = SceneRenderer(w, h)
    return r


def instances(codes: np.ndarray, min_px: int = MIN_PX) -> dict[int, np.ndarray]:
    """{object index: bool mask} for every object code with at least ``min_px`` visible pixels."""
    codes = np.asarray(codes)
    out = {}
    u, c = np.unique(codes, return_counts=True)
    for v, n in zip(u.tolist(), c.tolist()):
        if v == 0 or v >= OTHER_CODE or n < min_px:
            continue
        out[int(v) - 1] = codes == v
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def mask_polygon(mask: np.ndarray, min_area: float = 4.0, eps_px: float = 0.7) -> np.ndarray | None:
    """One closed polygon (N, 2) pixel coordinates for a possibly fragmented mask: every outer contour of at least
    ``min_area`` px, simplified by ``eps_px``, joined by zero-width bridges between their closest points (a fragmented,
    occluded object stays one instance, as YOLO-seg's label format wants). None for an empty mask."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2).astype(float) for c in cs if cv2.contourArea(c) >= min_area
          or len(cs) == 1]
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return None
    cs.sort(key=lambda c: -len(c))
    poly = cs[0]
    for c in cs[1:]:
        d = np.linalg.norm(poly[:, None, :] - c[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        # walk: poly[..i], c[j..] + c[..j] (closed loop), back to poly[i..]
        loop = np.concatenate([c[j:], c[: j + 1]])
        poly = np.concatenate([poly[: i + 1], loop, poly[i:]])
    return poly


def yolo_line(cls: int, poly: np.ndarray, w: int, h: int) -> str:
    xy = np.clip(poly / [w, h], 0.0, 1.0)
    return " ".join([str(int(cls))] + [f"{v:.5f}" for v in xy.ravel()])


# --------------------------------------------------------------------------------------------
# Matching / metrics
# --------------------------------------------------------------------------------------------

@dataclass
class Det:
    mask: np.ndarray       # uint8 0/1, image size
    key: str               # class = catalogue asset key
    score: float
    box: tuple = ()        # x0, y0, x1, y1 (pixels)
    alts: list = field(default_factory=list)   # [(key, prob)] runner-up classes, best first


def match_greedy(gt: dict[int, np.ndarray], preds: list[Det], thr: float = 0.0) -> dict[int, tuple[int, float]]:
    """GT index -> (prediction index, IoU): greedy one-to-one by IoU (pairs with IoU <= ``thr`` stay unmatched)."""
    pairs = []
    for gi, gm in gt.items():
        for pi, p in enumerate(preds):
            v = iou(gm, p.mask)
            if v > thr:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    out, used = {}, set()
    for v, gi, pi in pairs:
        if gi in out or pi in used:
            continue
        out[gi] = (pi, v)
        used.add(pi)
    return out


# --------------------------------------------------------------------------------------------
# Detector service (GPU, torch + ultralytics)
# --------------------------------------------------------------------------------------------

class _SegModel:
    """``handle(req, img)`` / ``info()`` for ``sam_masks.serve``: op ``detect`` -> masks + class keys + scores."""

    def __init__(self, weights: str, classes: list[str] | None = None, imgsz: int | None = None, conf: float = 0.25):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        t0 = time.time()
        self.model = YOLO(weights)
        names = self.model.names
        self.classes = classes or [names[i] for i in range(len(names))]
        # the input size it was trained at (a model trained at 960 px for small objects must also infer at 960)
        train_args = (getattr(self.model, "ckpt", None) or {}).get("train_args") or {}
        self.imgsz, self.conf = int(imgsz or train_args.get("imgsz") or 640), conf
        self.timing: dict = {}
        self.load_s = round(time.time() - t0, 1)

    def detect(self, img: np.ndarray, conf: float | None = None, max_det: int = 100) -> list[tuple]:
        import cv2

        r = self.model.predict(img[..., ::-1].copy(), imgsz=self.imgsz, conf=conf or self.conf, max_det=max_det,
                               retina_masks=True, verbose=False)[0]
        out = []
        if r.masks is None:
            return out
        ms = r.masks.data.cpu().numpy() > 0.5
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        bx = r.boxes.xyxy.cpu().numpy()
        H, W = img.shape[:2]
        for m, c, s, b in zip(ms, cls, sc, bx):
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            out.append((m, self.classes[int(c)], float(s), [float(v) for v in b]))
        return out

    def info(self) -> dict:
        return {"timing": {k: (n, round(t / max(n, 1), 4)) for k, (n, t) in self.timing.items()},
                "peak_vram_gb": round(self.torch.cuda.max_memory_allocated() / 1e9, 2), "load_s": self.load_s,
                "n_classes": len(self.classes), "imgsz": self.imgsz}

    def handle(self, req, img):
        from .sam_masks import pack

        t0 = time.time()
        if req["op"] != "detect":
            raise ValueError(req["op"])
        res = [(*pack(m), k, s, b) for m, k, s, b in self.detect(img, req.get("conf"), int(req.get("max_det", 100)))]
        n, tot = self.timing.get("detect", (0, 0.0))
        self.timing["detect"] = (n + 1, tot + time.time() - t0)
        return {"dets": res}


class SegClient:
    """Talks to ``serve`` over a unix socket (the sam_masks wire format). One per process (``client()``)."""

    def __init__(self, addr: str):
        from .sam_masks import SamClient

        self._c = SamClient(addr)

    def detect(self, img: np.ndarray, conf: float = 0.25, max_det: int = 100) -> list[Det]:
        from .sam_masks import unpack

        r = self._c._call({"op": "detect", "conf": float(conf), "max_det": int(max_det)}, img)
        return [Det(unpack(m, s), k, float(sc), tuple(b)) for m, s, k, sc, b in r["dets"]]

    def info(self) -> dict:
        return self._c.info()


_CLIENT: dict = {}


def client() -> SegClient | None:
    """The process's client, or None (``DOJO_SEG_ADDR`` unset, or no server: callers run without the detector)."""
    addr = os.environ.get(ENV_ADDR, "")
    if not addr:
        return None
    c = _CLIENT.get(addr)
    if c is None:
        try:
            c = SegClient(addr)
        except Exception as e:  # pragma: no cover - depends on a live server
            import warnings

            warnings.warn(f"segmenter server at {addr} unreachable ({e}); seg flags ignored")
            c = False
        _CLIENT[addr] = c
    return c or None


def main(argv=None):
    import argparse
    import json

    from .sam_masks import serve

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--sock", required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--classes", default=None, help="classes.json of the export (else the model's own names)")
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--imgsz", type=int, default=None, help="inference size (default: the model's training size)")
    s.add_argument("--max-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    classes = json.load(open(a.classes))["names"] if a.classes else None
    M = _SegModel(a.weights, classes, imgsz=a.imgsz, conf=a.conf)
    print(f"[seg] loaded {a.weights}: {len(M.classes)} classes in {M.load_s} s", flush=True)
    serve(os.path.expanduser(a.sock), ["seg"], a.max_hours, M=M)


if __name__ == "__main__":
    main()
