# CLASSIFY3 (2026-11-07): classify_into_bins_p2 with a choice of container test (our own wrapper; p2 itself unchanged).
"""classify_into_bins_p3: p2 (``classify_into_bins_p2``) run with its container test chosen by ``contain``.

run(robot, contain="bottom", **p2 arguments)

``contain``:
  * ``"root"``: p2's own test, the object's root position inside the container's box and below its rim + 10 mm;
  * ``"bottom"`` (default here): the root's xy inside the box and the object's lowest point (its box's bottom, ``aabb``)
    below the rim + 10 mm. On the camera the believed pose of an object let go into a basket keeps its in-hand
    orientation (standing, on its side), so its root sits 15-110 mm above the rim while it rests on the floor: on 20
    recorded camera rows (journal 2026-11-07-dojo-classify3) 25 of 96 releases that physics had inside read "not in"
    under the root test, and the skill re-did them (retries, fallbacks, nudges) until the step budget ran out.
Everything else (grounding, stages, retries, fallbacks, home) is p2's.
"""
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONTAIN = ("root", "bottom")


def load_p2():
    """A fresh private instance of p2 (its module globals are ours to rebind; nothing shared with other runs)."""
    p = os.path.join(HERE, "..", "classify_into_bins_p2", "skill_robodojo.py")
    spec = importlib.util.spec_from_file_location(f"cib_p2_for_p3_{os.getpid()}_{time.time_ns()}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def in_bin_bottom(objs, k, c):
    """``k`` is in container ``c``: its root's xy inside ``c``'s box, its lowest point below ``c``'s rim + 10 mm and
    its root above ``c``'s floor - 10 mm (p2's z band, with the object's bottom for the upper bound)."""
    lo, hi = np.asarray(objs[c]["aabb"][0], float), np.asarray(objs[c]["aabb"][1], float)
    p = np.asarray(objs[k]["pos"], float)
    ab = objs[k].get("aabb")
    z_low = float(ab[0][2]) if ab else float(p[2])
    return bool(lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and z_low <= hi[2] + 0.01 and p[2] >= lo[2] - 0.01)


def run(robot, contain="bottom", **kw):
    if contain not in CONTAIN:
        return {"ok": False, "verified": False, "error": f"contain must be one of {CONTAIN}, not {contain!r}"}
    m = load_p2()
    if contain == "bottom":
        m.in_bin = in_bin_bottom
    r = m.run(robot, **kw)
    if isinstance(r, dict):
        r.setdefault("details", {})["contain"] = contain
    return r
