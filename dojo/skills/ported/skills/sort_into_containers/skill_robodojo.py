# Our own wrapper (2026-11-07) around the vendored sort mechanics of classify_into_bins_p2 (p2 itself unchanged).
"""sort_into_containers: sort the scene's objects by category into its containers (a fixed category -> container map from
the instruction, or a free sort), with p2's grounding, stages, retries, fallbacks and home, and two general options.

run(robot, contain="root", park_other="off", **p2 arguments)

``contain``: how "this object is in that container" is judged from the world state:
  * ``"root"`` (p2): the object's root position inside the container's box and below its rim + 10 mm;
  * ``"bottom"``: the root's xy inside the box and the object's lowest point (its box's bottom, ``aabb``) below the
    rim + 10 mm. A believed object let go into a container keeps its in-hand orientation (standing, on its side), so
    its root can sit well above the rim while it rests on the floor; the root test then reads a placed object as not
    placed and the skill re-does it (journal 2026-11-07-dojo-classify3: 23 of 96 releases physics had inside).
``park_other``: ``"off"`` (p2) | ``"home"``: before every stage that carries with one arm (direct, drop_held, nudge),
the other arm goes home (v3's checked ``home_safe``) when it is away from home with open jaws | ``"blocked"``: only
after such a stage left the object neither placed nor held and one of its transits was refused because of the other
arm: that arm goes home and the stage runs once more (the idle arm was the commonest obstacle of a refused carry in
G3; the proactive park also moved where nothing blocked, journal 2026-11-07-dojo-classify3).
"""
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONTAIN = ("root", "bottom")
PARK = ("off", "home", "blocked")
CARRY_STAGES = ("direct", "drop_held", "nudge")


def load_p2():
    """A fresh private instance of p2 (its module globals are ours to rebind; nothing shared with other runs)."""
    p = os.path.join(HERE, "..", "classify_into_bins_p2", "skill_robodojo.py")
    spec = importlib.util.spec_from_file_location(f"cib_p2_for_sort_{os.getpid()}_{time.time_ns()}", p)
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


def blocked_by_other_arm(r) -> bool:
    """A stage result one of whose transits was refused because the other arm was in the way."""
    return isinstance(r, dict) and any(t.get("refused") and (t.get("direct_block") or {}).get("what") == "other_arm"
                                       for t in r.get("transits") or [] if isinstance(t, dict))


def parking(load_v3, parks: list, mode: str = "home"):
    """``load_v3`` whose instances home the idle arm around a one-arm carry stage: before it (``mode="home"``) or,
    after a refusal the other arm caused, before one more try (``"blocked"``). Every park is appended to ``parks``."""
    def load():
        V3 = load_v3()
        run0 = V3.run

        def park(robot, stage, arm, kw, why):
            other = "right" if arm == "left" else "left"
            e = robot.obs(cameras=[])["eef"][other]
            if e.get("at_home") or float(e.get("grip_cmd", 1.0)) < 0.5:
                return False
            h = V3.home_safe(robot, [other], "auto", 0.04, 7.0)
            parks.append({"stage": stage, "obj": kw.get("obj"), "arm": other, "why": why,
                          "ok": bool(h.get("ok", True)) if isinstance(h, dict) else None})
            return True

        def run(robot, stage, *a, **kw):
            arm = kw.get("arm")
            one_arm = stage in CARRY_STAGES and arm in ("left", "right")
            if one_arm and mode == "home":
                park(robot, stage, arm, kw, "before")
            r = run0(robot, stage, *a, **kw)
            if (one_arm and mode == "blocked" and isinstance(r, dict) and not r.get("in_container")
                    and not r.get("held") and blocked_by_other_arm(r) and park(robot, stage, arm, kw, "blocked")):
                r = run0(robot, stage, *a, **kw)
            return r
        V3.run = run
        return V3
    return load


def run(robot, contain="root", park_other="off", **kw):
    if contain not in CONTAIN:
        return {"ok": False, "verified": False, "error": f"contain must be one of {CONTAIN}, not {contain!r}"}
    if park_other not in PARK:
        return {"ok": False, "verified": False, "error": f"park_other must be one of {PARK}, not {park_other!r}"}
    m = load_p2()
    if contain == "bottom":
        m.in_bin = in_bin_bottom
    parks: list = []
    if park_other != "off":
        m.load_v3 = parking(m.load_v3, parks, park_other)
    r = m.run(robot, **kw)
    if isinstance(r, dict):
        d = r.setdefault("details", {})
        d["contain"], d["park_other"], d["parks"] = contain, park_other, parks
    return r
