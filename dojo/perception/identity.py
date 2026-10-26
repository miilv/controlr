"""What IS there: rank interchangeable candidates (same category / asset group) by how well each explains the
observed pixels, for held_look's ``what`` and the post-place refine's ``best``.

Candidates are ``[{"label", "key", "pose7"}]`` (the runtime's belief: the asked object first, with its EXPECTED
pose, then its twins with their belief poses). The observed colour comes from the pixels at the expected pose (the
object between the jaws, or the refined object); each candidate is scored by the catalogue colour of its asset,
mapped to the renderer's look (``color_calib``). Twins with the same asset key cannot be told apart by appearance:
when the best key is shared, ``label`` is None and only ``key`` is reported.

Score: 1 - ``appearance_distance`` (0..1, higher = better). ``margin`` = best - runner-up among DISTINCT keys.
``conf`` = calibrated P(the best key is right | margin) (``CONF_*`` tables, fitted by ``val_what.py``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# P(correct | margin) by margin bin (upper edge, P), Laplace-smoothed and made monotone. Fitted by val_what.py on
# stack_blocks wrist looks, 25 layouts x frames 1-2, the same states rendered by Isaac 6.0 (proxy, 6.0 colour map) and
# Isaac 5.1 (5.1 map), twins and 5-asset groups: 154 ok fits (margin >= 0.15: 132/132 right), 27 not-ok fits read at
# the prior (margin >= 0.15: 17/19 right).
CONF_FIT: list[tuple[float, float]] = [(0.05, 0.5), (0.1, 0.7), (0.15, 0.9), (0.25, 0.97), (1.01, 0.99)]
CONF_PRIOR: list[tuple[float, float]] = [(0.05, 0.5), (0.1, 0.6), (0.15, 0.7), (1.01, 0.85)]
# held_look (in-jaw view, mean colour, lightness weight 1, catalogue colours unmapped): 90 synthetic Isaac 6.0 frames,
# 10 stack layouts x D0 look poses, every block in turn in the jaws. Twins: margin >= 0.25 67/68 right, below 14/22.
CONF_HELD: list[tuple[float, float]] = [(0.25, 0.6), (1.01, 0.97)]
HELD_W_L = 1.0  # in the jaws the lighting is the gripper's own: lightness separates light wood from brown wood
EMPTY_FG_FRAC = 0.25  # foreground below this share of the expected silhouette: nothing there

_M_CACHE: dict[str, np.ndarray | None] = {}


def load_color_M(path: str | None = None, which: str = "all") -> np.ndarray | None:
    """The catalogue -> renderer colour map (4x3) from ``color_calib.json`` (``DOJO_COLOR_CALIB``), or None."""
    p = path or os.environ.get("DOJO_COLOR_CALIB", "")
    k = f"{p}:{which}"
    if k not in _M_CACHE:
        M = None
        if p and Path(p).is_file():
            d = json.loads(Path(p).read_text())
            if which in d:
                M = np.asarray(d[which], float)
        _M_CACHE[k] = M
    return _M_CACHE[k]


def ref_rgb(key: str, color_M=None) -> np.ndarray:
    from .render import load_asset

    c = np.asarray(load_asset(key).info["mean_color"], float)
    if color_M is not None:
        return np.clip(np.r_[c, 1.0] @ np.asarray(color_M, float), 0.0, 1.0)
    return c


def rank(obs_rgb, candidates: list[dict], color_M=None, ref=ref_rgb, w_l: float = 0.3) -> list[dict]:
    """[{"label", "key", "score"}] best first; empty when there is no observed colour."""
    from .inverse_layout import appearance_distance

    if obs_rgb is None:
        return []
    o = np.asarray(obs_rgb, float)
    refs: dict[str, np.ndarray] = {}
    out = []
    for c in candidates:
        k = c["key"]
        if k not in refs:
            refs[k] = ref(k, color_M)
        out.append({"label": c.get("label"), "key": k, "score": round(1.0 - appearance_distance(o, refs[k], w_l=w_l), 4)})
    return sorted(out, key=lambda r: -r["score"])


def conf_from_margin(margin: float, table=None) -> float:
    table = table or CONF_FIT
    for edge, p in table:
        if margin < edge:
            return p
    return table[-1][1]


def pick(ranked: list[dict], table=None) -> dict | None:
    """{"label" (None when twins share the best key), "key", "conf", "margin"} from ``rank``'s output."""
    if not ranked:
        return None
    best = ranked[0]
    other_keys = [r for r in ranked if r["key"] != best["key"]]
    margin = best["score"] - other_keys[0]["score"] if other_keys else 1.0
    same = [r for r in ranked if r["key"] == best["key"]]
    return {"label": best["label"] if len(same) == 1 else None, "key": best["key"],
            "conf": round(conf_from_margin(margin, table), 3), "margin": round(float(margin), 4)}


def asked_verdict(what: dict | None, asked: dict, present: bool | None, conf_min: float = 0.7,
                  margin_min: float = 0.15) -> bool | None:
    """Is the asked candidate the one there? False only when something is there and a DIFFERENT key wins clearly;
    True when the asked key wins (or ties with a same-key twin); None when undecidable."""
    if present is None:
        return None
    if present is False:
        return False
    if what is None:
        return None
    if what["key"] == asked["key"]:
        return True
    if what["conf"] >= conf_min and what["margin"] >= margin_min:
        return False
    return None


def refine_what(candidates: list[dict], wrist_frame, proprio: dict, arm: str, others=None, held=None,
                color_M=None, **kw) -> dict:
    """The post-place refine of ``candidates[0]`` (the asked object, at its expected pose) plus what IS there:
    ``best`` {label, key, pose7, iou, conf, margin} | None (= empty), ``asked_present`` bool | None, ``scored`` top 3.
    ``others`` = every other known object (the twins included) as (key, pose7), like ``refine``'s."""
    from .refine import refine

    asked = candidates[0]
    M = color_M if color_M is not None else load_color_M()
    r = refine(asked["key"], wrist_frame, proprio, list(asked["pose7"]), arm=arm, others=others or None, held=held, **kw)
    op = r.get("obs_prior") or {}
    fg_frac = (op.get("fg_px", 0) / op["sil_px"]) if op.get("sil_px") else None
    if r.get("ok"):
        present, src = True, r.get("obs_fit") or {}
    elif fg_frac is not None and fg_frac < EMPTY_FG_FRAC:
        present, src = False, {}
    else:
        present, src = None, op
    ranked = rank(src.get("rgb"), candidates, M)
    best = pick(ranked, CONF_FIT if present else CONF_PRIOR)
    if best is not None:
        best.update({"pose7": r["pose"] if r.get("ok") else None, "iou": r.get("iou")})
    if present is False:
        best = None
    return {**r, "best": best, "asked_present": asked_verdict(best, asked, present), "scored": ranked[:3],
            "fg_frac_at_prior": None if fg_frac is None else round(fg_frac, 3)}
