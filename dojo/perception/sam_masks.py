"""SAM mask proposals for the inverse layout (perception flags ``sam_fg`` / ``sam_propose``, default off).

Two models answer through one GPU process (``serve``), so CPU estimator workers share one copy of the weights:

- **SAM 3.1** (Meta's ``sam3`` code; the checkpoint is Comfy-Org's fp16 repack of ``facebook/sam3.1``, whose keys are
  Meta's own ``detector.*`` / ``tracker.*``): text (concept) prompts through the detector, point prompts through the
  tracker's interactive head on a one-frame "video".
- **SAM 2.1** (hiera small / large, Apache-2.0): point prompts, three masks per point (REAL_RSI's proposer pattern:
  prompts at the predicted object centre plus a few offsets, every mask scored by a model fit).

The estimator uses them in two ways (``inverse_layout.EstimatorConfig``):

- ``sam_fg``: the head view's foreground combined with SAM 3.1's text masks for the scene's vocabulary (the label
  groups' category and catalogue names, the instruction's target phrase; never the task name): ``union`` adds
  objects the static-background model misses (low contrast), ``trim`` cuts foreground blobs that touch a mask to the
  mask (a shadow attached to an object), ``gate`` keeps background-model foreground only near a SAM mask (cast
  shadows, clutter outside the vocabulary) plus the masks themselves, ``replace`` uses the masks alone.
- ``sam_propose``: after the joint refinement every chosen object is re-fitted inside each SAM mask proposed at its
  head-view centre (+ offsets); the fit that agrees best with its mask (IoU) wins, B1's own pose included.

The client is a no-op when ``DOJO_SAM_ADDR`` is unset or the server is unreachable (B1 runs unchanged).

    python -m dojo.perception.sam_masks serve --sock ~/dojo-data/sam/sam.sock [--models sam3.1,sam2.1_small]
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

ENV_ADDR = "DOJO_SAM_ADDR"
MODELS_DIR = os.environ.get("DOJO_SAM_MODELS", os.path.expanduser("~/dojo-data/models"))
SAM31_CKPT = "sam3.1/sam3.1_multiplex_fp16.safetensors"
SAM2_CKPT = {"sam2.1_small": ("sam2.1/small/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
             "sam2.1_large": ("sam2.1/large/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml")}


# --------------------------------------------------------------------------------------------
# Vocabulary (text prompts) from the config's label groups: what B1 itself looks for
# --------------------------------------------------------------------------------------------

def _noun(s: str) -> str:
    s = re.sub(r"[_\s]+", " ", str(s)).strip().lower()
    return s


def vocabulary(groups: list[dict], instruction: str = "", max_names: int = 2) -> list[str]:
    """Text prompts for the scene: each label group's category names (``pen_holder`` -> "pen holder") and up to
    ``max_names`` catalogue caption names per category, plus the instruction's target phrase ("Pick up the X by 10
    cm"). One-letter categories (``t``) keep only their caption names ("T-block")."""
    from .inverse_layout import asset_info, target_phrase

    out: list[str] = []

    def add(p):
        p = _noun(p)
        if p and p not in out:
            out.append(p)

    phrase = target_phrase(instruction) if instruction else None
    if phrase:
        add(phrase)
    for g in groups:
        cats: dict[str, list[str]] = {}
        for k in g["candidates"]:
            cats.setdefault(k.split("/")[1], []).append(k)
        if len(cats) > 12 and phrase:  # an open vocabulary (general_pickup's 85 categories): the phrase is the prompt
            continue
        for cat, keys in cats.items():
            if len(cat) > 1:
                add(cat)
            names = []
            for k in keys[:4]:
                try:
                    names += [n for n in (asset_info(k).get("names") or []) if n not in names]
                except Exception:
                    pass
            for n in names[:max_names]:
                add(n)
    return out


# --------------------------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------------------------

def pack(mask: np.ndarray) -> tuple[bytes, tuple]:
    m = np.asarray(mask, bool)
    return np.packbits(m.ravel()).tobytes(), m.shape


def unpack(data: bytes, shape: tuple) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(np.frombuffer(data, np.uint8), count=n).reshape(shape).astype(np.uint8)


def image_key(img: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()[:20]


@dataclass
class SamMask:
    mask: np.ndarray  # uint8 0/1, image size
    score: float
    prompt: str = ""
    point: int = -1  # index of the prompting point (points requests)


class SamClient:
    """Talks to ``serve`` over a unix socket. Thread-unsafe; one per process (``client()``)."""

    def __init__(self, addr: str, timeout_s: float = 60.0):
        from multiprocessing.connection import Client

        self.addr = addr
        self.conn = Client(addr, family="AF_UNIX")
        self.timeout_s = timeout_s

    def _call(self, req: dict, img: np.ndarray) -> dict:
        req = dict(req, key=image_key(img), shape=img.shape)
        self.conn.send(req)
        r = self.conn.recv()
        if r.get("need_image"):
            self.conn.send({"image": np.ascontiguousarray(img, np.uint8).tobytes()})
            r = self.conn.recv()
        if "error" in r:
            raise RuntimeError(f"sam server: {r['error']}")
        return r

    def text(self, img: np.ndarray, prompts: list[str], thr: float = 0.4) -> list[SamMask]:
        r = self._call({"op": "text", "prompts": list(prompts), "thr": float(thr)}, img)
        return [SamMask(unpack(m, s), sc, p) for m, s, sc, p in r["masks"]]

    def points(self, img: np.ndarray, uv: np.ndarray, model: str = "sam2.1_small", multimask: bool = True) -> list[SamMask]:
        """One positive click per point (each its own object); SAM 2.1 returns 3 masks per point when ``multimask``."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        r = self._call({"op": "points", "model": model, "uv": uv.tolist(), "multimask": bool(multimask)}, img)
        return [SamMask(unpack(m, s), sc, "", int(i)) for m, s, sc, i in r["masks"]]

    def info(self) -> dict:
        self.conn.send({"op": "info"})
        return self.conn.recv()


class LocalSam:
    """The server's models in this process, with ``SamClient``'s ``points`` interface (offline evals: one GPU process
    for SAM + DINOv2 + renders, no socket). ``models``: e.g. ["sam2.1_small"]."""

    def __init__(self, models: list[str], M=None):
        self.M = M if M is not None else _Models(models)

    def points(self, img: np.ndarray, uv: np.ndarray, model: str = "sam2.1_small", multimask: bool = True) -> list[SamMask]:
        uv = np.asarray(uv, float).reshape(-1, 2)
        r = self.M.handle({"op": "points", "model": model, "uv": uv.tolist(), "multimask": bool(multimask),
                           "key": image_key(img)}, np.ascontiguousarray(img, np.uint8))
        return [SamMask(unpack(m, s), sc, "", int(i)) for m, s, sc, i in r["masks"]]


_CLIENT: dict = {}


def client() -> SamClient | None:
    """The process's client, or None (``DOJO_SAM_ADDR`` unset, or no server: the caller runs plain B1)."""
    addr = os.environ.get(ENV_ADDR, "")
    if not addr:
        return None
    c = _CLIENT.get(addr)
    if c is None:
        try:
            c = SamClient(addr)
        except Exception as e:  # pragma: no cover - depends on a live server
            import warnings

            warnings.warn(f"SAM server at {addr} unreachable ({e}); SAM flags ignored")
            c = False
        _CLIENT[addr] = c
    return c or None


# --------------------------------------------------------------------------------------------
# Foreground combination (sam_fg)
# --------------------------------------------------------------------------------------------

def combine_fg(fg: np.ndarray, known: np.ndarray, masks: list[np.ndarray], mode: str, dilate_px: int = 5,
               trim_cover: float = 0.5) -> np.ndarray:
    """The head view's foreground with SAM masks: ``union`` fg | masks; ``trim`` masks | every foreground component
    that the masks cover by at least ``trim_cover`` cut to the masks' neighbourhood (an object SAM outlines plus its
    cast shadow), other components kept (objects SAM's vocabulary missed, blobs of several objects of which SAM
    outlines one); ``gate`` masks | (fg near a mask); ``replace`` masks only. Always inside ``known``."""
    import cv2

    fg = (np.asarray(fg) > 0).astype(np.uint8)
    sam = np.zeros_like(fg)
    for m in masks:
        sam |= (np.asarray(m) > 0).astype(np.uint8)
    near = cv2.dilate(sam, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)) if dilate_px else sam
    if mode == "union":
        out = fg | sam
    elif mode == "trim":
        n, lab = cv2.connectedComponents(fg, connectivity=8)
        area = np.bincount(lab.ravel(), minlength=n)
        cover = np.bincount(lab[(sam > 0) & (fg > 0)].ravel(), minlength=n)
        touch = cover >= trim_cover * np.maximum(area, 1)
        touch &= cover > 0
        touch[0] = False
        out = sam | (fg & ~touch[lab]) | (fg & touch[lab] & near)
    elif mode == "gate":
        out = sam | (fg & near)
    elif mode == "replace":
        out = sam
    else:
        raise ValueError(f"sam_fg mode {mode!r}")
    return (out & (np.asarray(known) > 0)).astype(np.uint8)


def text_masks_for(img: np.ndarray, groups: list[dict], instruction: str, thr: float = 0.4,
                   max_frac: float = 0.08) -> list[SamMask]:
    """SAM 3.1 text masks for the scene vocabulary; masks bigger than ``max_frac`` of the image (the table, a wall)
    are dropped. Empty without a server."""
    c = client()
    if c is None:
        return []
    voc = vocabulary(groups, instruction)
    if not voc:
        return []
    ms = c.text(img, voc, thr=thr)
    lim = max_frac * img.shape[0] * img.shape[1]
    return [m for m in ms if 0 < int(m.mask.sum()) <= lim]


# --------------------------------------------------------------------------------------------
# Server (GPU, torch): python -m dojo.perception.sam_masks serve
# --------------------------------------------------------------------------------------------

class _Models:
    def __init__(self, names: list[str]):
        import torch

        self.torch = torch
        self.names = names
        self.lock = threading.Lock()
        self.sam31 = None
        self.sam2: dict = {}
        self._img_key: dict = {}
        self.timing: dict = {}
        t0 = time.time()
        if "sam3.1" in names:
            self._load_sam31()
        for n in names:
            if n in SAM2_CKPT:
                self._load_sam2(n)
        self.load_s = round(time.time() - t0, 1)

    # ---- SAM 3.1
    def _load_sam31(self):
        import tempfile

        torch = self.torch
        from safetensors.torch import load_file
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        ck = os.path.join(MODELS_DIR, SAM31_CKPT)
        pt = ck.replace(".safetensors", ".pt")
        if not os.path.exists(pt):  # Meta's builder reads a torch checkpoint; same keys
            torch.save(load_file(ck), pt)
        pred = build_sam3_multiplex_video_predictor(checkpoint_path=pt, use_fa3=False, use_rope_real=False,
                                                    async_loading_frames=False)
        self.sam31 = pred
        self.sam31_proc = Sam3Processor(pred.model.detector, confidence_threshold=0.3)
        self.sam31_dir = tempfile.mkdtemp(prefix="sam31_frame_")
        self._cap = {}
        anp = pred.model.tracker.add_new_points

        def cap(*a, **k):  # the multiplex output cache lags one call: take the tracker's own masks
            r = anp(*a, **k)
            self._cap["r"] = r
            return r

        pred.model.tracker.add_new_points = cap

    def sam31_text(self, img, prompts, thr):
        torch = self.torch
        from PIL import Image

        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = self.sam31_proc.set_image(Image.fromarray(img))
            for p in prompts:
                self.sam31_proc.reset_all_prompts(st)
                st = self.sam31_proc.set_text_prompt(p, st)
                ms = st["masks"].squeeze(1).cpu().numpy()
                sc = st["scores"].float().cpu().numpy()
                for m, s in zip(ms, sc):
                    if s >= thr:
                        out.append((m, float(s), p))
        return out

    def sam31_points(self, img, uv):
        import cv2

        torch = self.torch
        H, W = img.shape[:2]
        cv2.imwrite(os.path.join(self.sam31_dir, "0.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
        m_ = self.sam31.model
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = m_.init_state(resource_path=self.sam31_dir, async_loading_frames=False)
            for i, (u, v) in enumerate(uv):
                self._cap.clear()
                m_.add_prompt(st, frame_idx=0, points=torch.tensor([[u / W, v / H]], dtype=torch.float32),
                              point_labels=torch.tensor([1], dtype=torch.int32), obj_id=i + 1)
                _, oids, _, vr = self._cap["r"]
                j = list(oids).index(i + 1)
                logit = vr[j, 0].float()
                out.append(((logit > 0).cpu().numpy(), float(torch.sigmoid(logit.max()).item()), i))
        return out

    # ---- SAM 2.1
    def _load_sam2(self, name):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ck, cfg = SAM2_CKPT[name]
        self.sam2[name] = SAM2ImagePredictor(build_sam2(cfg, os.path.join(MODELS_DIR, ck), device="cuda"))

    def sam2_points(self, name, img, key, uv, multimask):
        torch = self.torch
        p = self.sam2[name]
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self._img_key.get(name) != key:
                p.set_image(img)
                self._img_key[name] = key
            for i, (u, v) in enumerate(uv):
                ms, sc, _ = p.predict(point_coords=np.array([[u, v]], np.float32), point_labels=np.array([1]),
                                      multimask_output=multimask)
                for m, s in zip(ms, sc):
                    out.append((m > 0, float(s), i))
        return out

    def info(self) -> dict:
        return {"timing": {k: (n, round(t / max(n, 1), 4)) for k, (n, t) in self.timing.items()},
                "peak_vram_gb": round(self.torch.cuda.max_memory_allocated() / 1e9, 2), "load_s": self.load_s}

    def handle(self, req, img):
        t0 = time.time()
        op = req["op"]
        with self.lock:
            if op == "text":
                res = [(*pack(m), s, p) for m, s, p in self.sam31_text(img, req["prompts"], req.get("thr", 0.4))]
            elif op == "points":
                name = req.get("model", "sam2.1_small")
                if name == "sam3.1":
                    raw = self.sam31_points(img, req["uv"])
                else:
                    raw = self.sam2_points(name, img, req["key"], req["uv"], req.get("multimask", True))
                res = [(*pack(m), s, i) for m, s, i in raw]
            else:
                raise ValueError(op)
            self.torch.cuda.synchronize()
        k = op if op == "text" else f"points:{req.get('model')}"
        n, tot = self.timing.get(k, (0, 0.0))
        self.timing[k] = (n + 1, tot + time.time() - t0)
        return {"masks": res}


def serve(sock: str, models: list[str], max_hours: float = 6.0, M=None, ready: threading.Event | None = None) -> None:
    """Answer SAM requests on a unix socket until killed (or ``max_hours``). One thread per client; the GPU is used by
    one request at a time. ``M``: a models object with ``handle(req, img)`` / ``info()`` (tests pass a fake)."""
    from multiprocessing.connection import Listener

    if M is None:
        M = _Models(models)
        print(f"[sam] loaded {models} in {M.load_s} s, vram {M.torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    if os.path.exists(sock):
        os.unlink(sock)
    lst = Listener(sock, family="AF_UNIX")
    os.chmod(sock, 0o600)
    if ready is not None:
        ready.set()
    images: dict = {}
    t_end = time.time() + max_hours * 3600

    def run(conn):
        try:
            while True:
                req = conn.recv()
                if req.get("op") == "info":
                    conn.send({"models": models, **M.info()})
                    continue
                key = req["key"]
                img = images.get(key)
                if img is None:
                    conn.send({"need_image": True})
                    img = np.frombuffer(conn.recv()["image"], np.uint8).reshape(req["shape"])
                    if len(images) > 64:
                        images.pop(next(iter(images)))
                    images[key] = img
                try:
                    conn.send(M.handle(req, img))
                except Exception as e:  # report, keep serving
                    import traceback

                    conn.send({"error": f"{type(e).__name__}: {e} | {traceback.format_exc()[-400:]}"})
        except (EOFError, ConnectionResetError, BrokenPipeError):
            pass

    while time.time() < t_end:
        conn = lst.accept()
        threading.Thread(target=run, args=(conn,), daemon=True).start()


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--sock", required=True)
    s.add_argument("--models", default="sam3.1,sam2.1_small")
    s.add_argument("--max-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(os.path.expanduser(a.sock), [m for m in a.models.split(",") if m], a.max_hours)


if __name__ == "__main__":
    main()
