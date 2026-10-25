"""perception's episode tracker (``dojo.perception.tracker.Tracker``) over the runtime's belief.

The runtime keeps its own belief between perception calls (frame 0 / look fuse / search / agentic, then refines,
post-place measurements, twin predictions), so the tracker is not carried from call to call. Each B1 re-read of the
scene (the look-fused record, the re-estimate at a confirmed divergence) goes through one tracker update seeded with
the belief: rule 2 (jitter: a reading within 4 mm keeps the belief pose) and rule 3 (only touched objects move: an
untouched object keeps its pose unless the reading is confident, >= 0.8, and at least as confident as the belief)
decide which re-read poses the belief takes. "Touched" comes from the runtime's own record of what the robot moved
(``RuntimeProvider.touched``: picked, released, or moved by a twin prediction since the object's last good reading),
not from the TCP-proximity guess the tracker makes in the J1 path.

``filter_readings`` -> ({label: accepted?}, tracker events). Readings carry the belief's labels (the runtime matches
re-estimates to the belief by position first), so pairing is label-first; any ambiguity (a label paired with another
label's reading) leaves that label's reading un-accepted.
"""

from __future__ import annotations

import numpy as np


def _obj(label: str, category: str, key: str, pos, quat, conf: float, size=None) -> dict:
    return {"label": label, "category": category, "asset": key, "pos": [float(v) for v in pos],
            "quat": [float(v) for v in quat], "confidence": float(conf), **({"size": list(size)} if size is not None else {})}


def filter_readings(belief: dict, readings: dict, touched: set) -> tuple[dict, list[dict]]:
    """``belief``: {label: ObjectState}; ``readings``: {label: ObjectState-like (pos, quat, confidence, category,
    asset.key)} for the labels a re-read proposes to move; ``touched``: labels the robot moved since their last
    good reading."""
    from dojo.perception.tracker import Tracker

    labs = [lab for lab in readings if lab in belief]
    if not labs:
        return {}, []
    t = Tracker()
    t.update([_obj(lab, belief[lab].category, belief[lab].asset.key, belief[lab].pos, belief[lab].quat,
                   getattr(belief[lab], "confidence", 0.9)) for lab in labs])
    for o in t.objects:
        if o["label"] in touched:
            t.touched.add(o["_tid"])
    objs, _last, events = t.update([_obj(lab, readings[lab].category, readings[lab].asset.key, readings[lab].pos,
                                         readings[lab].quat, getattr(readings[lab], "confidence", 0.5)) for lab in labs])
    out = {}
    for o in objs:
        lab = o.get("label")
        if lab not in readings or lab in out:
            continue
        out[lab] = (o.get("source") != "prior"
                    and float(np.linalg.norm(np.asarray(o["pos"], float) - np.asarray(readings[lab].pos, float))) < 1e-6)
    for lab in labs:
        out.setdefault(lab, False)
    return out, events


__all__ = ["filter_readings"]
