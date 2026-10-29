"""Glyph reads of make_kong's tiles by a crop classifier (stream S; flag ``EstimatorConfig.glyph_read``, default off).

The classifier (``scripts/perception/seg/kong_cls.py``: ResNet-18 on 128 px tile crops, 42 glyphs + "back") was
trained only on our renders of 100030-49 variants. It runs in one GPU process (``serve``, the ``sam_masks`` wire
format); ``client(category)`` talks to the model for an asset category (``MODEL_OF``: numerals / operators ->
``DOJO_GLYPH_ADDR_DIGIT``, mahjong -> ``DOJO_GLYPH_ADDR_KONG``; the old ``DOJO_GLYPH_ADDR`` only as a checked fallback) and is
None without a server (the flag is then a no-op).

A slot tile's read (``read``): the tile's silhouette at B1's pose in every view -> a square crop around it (as the
classifier was trained) -> class probabilities; the view with the largest top-1 margin wins. It is a read only when
the top class is a glyph among the label's candidates and its margin clears that view kind's margin (``head`` /
``wrist``, chosen on validation draws for a precision target, stored with the model). Anything else -- a back, a
grazing view, a low margin, no server -- is no read: the entry's identity stays "unknown" (``identity_marks``).
"""
from __future__ import annotations

import os
import time

import numpy as np

ENV_ADDR = "DOJO_GLYPH_ADDR"
MIN_PX = 40
# classify_mask: no read when this share (or more) of the 3-px ring around the mask is an arm: a tile half hidden by
# the robot reads as another glyph (a 3 as a 9). Chosen on the digit val draws (100045-48): reads with ring < 0.3 kept
# >= 98 % precision, ring >= 0.3 fell to ~85 %.
ROBOT_RING_MAX = 0.3
RING_PX = 3


def crop_square(img: np.ndarray, bbox: tuple, size: int = 128, margin: float = 0.25) -> np.ndarray:
    """A square crop around ``bbox`` (x0, y0, x1, y1, inclusive pixels) grown by ``margin`` per side, border-replicated
    past the image, resized to ``size`` (the training crops of ``kong_crops.py``)."""
    import cv2

    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 8) * (0.5 + margin)
    a, b, c, d = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    H, W = img.shape[:2]
    pad = max(0, -a, -b, c - W, d - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        a, b, c, d = a + pad, b + pad, c + pad, d + pad
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA if c - a > size else cv2.INTER_LINEAR)


def mask_bbox(mask: np.ndarray) -> tuple | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def decide(probs: np.ndarray, n_glyph: int, margin: float) -> tuple[int, float, bool, str]:
    """(top class, top-1 margin, read?, why) for one crop's probabilities (index n_glyph = "back")."""
    order = np.argsort(probs)[::-1]
    top, m = int(order[0]), float(probs[order[0]] - probs[order[1]])
    if top == n_glyph:
        return top, m, False, "back"
    if m < margin:
        return top, m, False, "low_margin"
    return top, m, True, "read"


def read(views: dict, sm, T: np.ndarray, candidates: list[str], client_=None, held: bool = False) -> dict:
    """The glyph read of one tile at pose ``T`` (shape ``sm``) over ``views`` (ViewMasks with ``img``, ``cam``,
    ``known``). ``held``: the tile is in the jaws (the draw read): the model's ``held_<view>`` margins apply.
    Returns {"read", "key", "class", "margin", "view", "why", "views": per-view summary}. The reader is the one for
    the candidates' category (``client(category)``); a server of another model gives no read ("wrong model")."""
    cat = candidates[0].split("/")[1] if candidates and candidates[0].count("/") >= 2 else None
    c = client_ or client(cat)
    if c is None:
        return {"read": False, "why": "no glyph server"}
    boxes, names, imgs = [], [], []
    for name, v in views.items():
        if "@" in name:
            continue
        sil = sm.sil(v.cam, T)
        vis = sil & (v.known > 0) if getattr(v, "known", None) is not None else sil
        if int(vis.sum()) < MIN_PX:
            continue
        bb = mask_bbox(sil)
        if bb is None:
            continue
        boxes.append(bb)
        names.append(name)
        imgs.append(v.img)
    if not boxes:
        return {"read": False, "why": "not visible"}
    out = {"read": False, "why": "low_margin", "views": {}}
    best = None
    for name, img, bb in zip(names, imgs, boxes):
        try:
            r = c.classify(img, [bb])
        except WrongModel:
            return {"read": False, "why": "wrong model"}
        P, n_glyph, margins = np.asarray(r["probs"][0], float), int(r["n_glyph"]), r["margins"]
        kind = "head" if name == "cam_head" else "wrist"
        # the tile in the jaws (the draw read) has its own, stricter margin when the model carries one
        thr = margins.get(f"held_{kind}", margins.get(kind, 1.0)) if held else margins.get(kind, 1.0)
        top, m, ok, why = decide(P, n_glyph, float(thr))
        key = None if top == n_glyph else class_name(r, top)
        if ok and key not in candidates:
            ok, why = False, "not a candidate"
        out["views"][name] = {"class": top, "margin": round(m, 3), "why": why}
        if ok and (best is None or m > best[1]):
            best = (key, m, name, top)
    if best is not None:
        out.update(read=True, key=best[0], margin=round(best[1], 3), view=best[2], why="read", **{"class": best[3]})
    else:
        whys = {v["why"] for v in out["views"].values()}
        out["why"] = "back" if whys == {"back"} else ("not a candidate" if "not a candidate" in whys else "low_margin")
    return out


def class_name(reply: dict, top: int) -> str:
    """A class index's name: the model's own names when it carries them (the digit / operator model: "0".."9", "plus",
    ...), else make_kong's asset key."""
    names = reply.get("names")
    return names[top] if names and top < len(names) else f"Rigid/mahjong/{top:05d}"


def robot_ring(mask: np.ndarray, robot: np.ndarray, px: int = RING_PX) -> float:
    """Share of the ``px``-wide ring around ``mask`` covered by ``robot`` (bool: the arms' pixels in the view)."""
    m = np.asarray(mask) > 0
    d = m.copy()
    for axis in (0, 1):            # square (2 px + 1) dilation, separable; no cv2 needed
        g = d.copy()
        for k in range(1, px + 1):
            g |= np.roll(d, k, axis) | np.roll(d, -k, axis)
        d = g
    ring = d & ~m
    return float(np.asarray(robot, bool)[ring].mean()) if ring.any() else 0.0


def classify_mask(img: np.ndarray, mask: np.ndarray, view: str = "cam_head", held: bool = False, client_=None,
                  robot: np.ndarray | None = None, robot_ring_max: float = ROBOT_RING_MAX,
                  category: str | None = None) -> dict:
    """One glyph read of an object outlined by ``mask`` in ``img`` (a detector's / segmenter's mask: no fitted pose
    needed, e.g. the number tiles B1 has no geometry for). ``view`` picks the margin kind (head / wrist, held_*).
    ``robot``: the arms' pixels in this view (bool, e.g. ``~known`` of a B1 view or ``render.robot_mask``); with it a
    mask bordered by an arm (``robot_ring`` >= ``robot_ring_max``) is not read. Pass it: the margins are calibrated
    without such crops, and without it a tile half hidden by the gripper can be read wrong at a high margin.
    Returns {"read", "name", "class", "margin", "why"}; no read below the margin, on a back, for a mask cut by the
    image border (a cut digit is another digit: 8 -> 3, the margins are calibrated without those), next to an arm,
    or without a server. ``category``: the object's asset category picks the reader (``client(category)``; a server of
    another model gives no read, "wrong model")."""
    m_ = np.asarray(mask) > 0
    bb = mask_bbox(m_)
    if bb is None or int(m_.sum()) < MIN_PX:
        return {"read": False, "why": "not visible"}
    if m_[0].any() or m_[-1].any() or m_[:, 0].any() or m_[:, -1].any():
        return {"read": False, "why": "edge"}
    if robot is not None and robot_ring(m_, robot) >= robot_ring_max:
        return {"read": False, "why": "robot"}
    c = client_ or client(category)
    if c is None:
        return {"read": False, "why": "no glyph server"}
    try:
        r = c.classify(img, [bb])
    except WrongModel:
        return {"read": False, "why": "wrong model"}
    P, n_glyph, margins = np.asarray(r["probs"][0], float), int(r["n_glyph"]), r["margins"]
    kind = "head" if view == "cam_head" else "wrist"
    thr = margins.get(f"held_{kind}", margins.get(kind, 1.0)) if held else margins.get(kind, 1.0)
    top, m, ok, why = decide(P, n_glyph, float(thr))
    return {"read": ok, "name": None if top == n_glyph else class_name(r, top), "class": top, "margin": round(m, 3),
            "why": why}


# --------------------------------------------------------------------------------------------
# Server (GPU, torch) / client
# --------------------------------------------------------------------------------------------

class _GlyphModel:
    def __init__(self, path: str):
        import torch
        import torchvision

        self.torch = torch
        ck = torch.load(path, map_location="cpu")
        self.n_glyph = int(ck.get("n_glyph", 42))
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, self.n_glyph + 1)
        m.load_state_dict(ck["state"])
        self.model = m.eval().to("cuda")
        self.margins = dict(ck.get("margins") or {"head": ck.get("margin", 1.0), "wrist": ck.get("margin", 1.0)})
        self.names = ck.get("names")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.timing: dict = {}

    def info(self) -> dict:
        return {"timing": {k: (n, round(t / max(n, 1), 4)) for k, (n, t) in self.timing.items()},
                "n_glyph": self.n_glyph, "margins": self.margins}

    def handle(self, req, img):
        t0 = time.time()
        if req["op"] != "glyph":
            raise ValueError(req["op"])
        torch = self.torch
        crops = np.stack([crop_square(img, tuple(b)) for b in req["boxes"]]).astype(np.float32) / 255.0
        x = (torch.from_numpy(crops).permute(0, 3, 1, 2).cuda() - self.mean) / self.std
        with torch.inference_mode():
            p = torch.softmax(self.model(x), 1).cpu().numpy()
        n, tot = self.timing.get("glyph", (0, 0.0))
        self.timing["glyph"] = (n + 1, tot + time.time() - t0)
        return {"probs": p.tolist(), "n_glyph": self.n_glyph, "margins": self.margins, "names": self.names}


class GlyphClient:
    def __init__(self, addr: str, model: str | None = None):
        from .sam_masks import SamClient

        self._c = SamClient(addr, loud=True)     # unreachable / gone / erroring -> SideServerDown
        self.model = model       # the model this address must serve ("digit" / "kong"); None = unchecked (legacy)

    def classify(self, img: np.ndarray, boxes: list) -> dict:
        r = self._c._call({"op": "glyph", "boxes": [list(map(int, b)) for b in boxes]}, img)
        if self.model is not None and not model_fits(self.model, r):
            raise WrongModel(f"glyph server at this address does not serve the {self.model} model")
        return r


class WrongModel(RuntimeError):
    """The server's reply does not fit the model its address is for (a misconfigured env fails closed)."""


# one category -> model map: the operators are separate categories but live in the digit model
MODEL_OF = {"number": "digit", "plus": "digit", "minus": "digit", "multiplication": "digit", "division": "digit",
            "equal": "digit", "mahjong": "kong"}
ENV_ADDR_OF = {"digit": "DOJO_GLYPH_ADDR_DIGIT", "kong": "DOJO_GLYPH_ADDR_KONG"}


def model_fits(model: str, reply: dict) -> bool:
    """Does a server reply come from ``model``? The digit model carries its class names ("0".."9", the operators);
    make_kong's tile model carries none (its classes are mahjong asset indices)."""
    names = reply.get("names")
    if model == "digit":
        return bool(names) and "0" in names
    if model == "kong":
        return not names
    return False


_CLIENT: dict = {}


def client(category: str | None = None) -> GlyphClient | None:
    """The glyph reader for an asset ``category`` (``MODEL_OF``: numerals and operators -> the digit model at
    ``DOJO_GLYPH_ADDR_DIGIT``, mahjong tiles -> the kong model at ``DOJO_GLYPH_ADDR_KONG``; any other category: None).
    The old single ``DOJO_GLYPH_ADDR`` is a fallback, used only when its replies fit the category's model (checked on
    every reply: a digit model never reads a mahjong tile). ``category`` None: the legacy unchecked client. None only when
    no address is set; a set address that does not answer raises ``sam_masks.SideServerDown``."""
    if category is None:
        model, addr = None, os.environ.get(ENV_ADDR, "")
    else:
        model = MODEL_OF.get(category)
        if model is None:
            return None
        addr = os.environ.get(ENV_ADDR_OF[model], "") or os.environ.get(ENV_ADDR, "")
    if not addr:
        return None
    key = (addr, model)
    c = _CLIENT.get(key)
    if c is None:
        c = GlyphClient(addr, model)     # a SET address that does not answer raises SideServerDown (infra, not "no read")
        _CLIENT[key] = c
    return c


def main(argv=None):
    import argparse

    from .sam_masks import serve

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--sock", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--max-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    M = _GlyphModel(os.path.expanduser(a.model))
    print(f"[glyph] loaded {a.model}: {M.n_glyph} glyphs, margins {M.margins}", flush=True)
    serve(os.path.expanduser(a.sock), ["glyph"], a.max_hours, M=M)


if __name__ == "__main__":
    main()
