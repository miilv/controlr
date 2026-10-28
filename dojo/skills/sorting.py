"""Objects into containers by category with OUR library (``library.arrange`` "put" steps): the alternative to the ported
classify_into_bins_v3b for "sort the objects by category into the baskets" / "put X objects into the left basket, ...".

``sort_groups(instruction, world)`` -> [(container label, [object labels])] or None:
* containers: objects whose category names a basket / bin, ordered left to right (world x);
* items: every other rigid object (clutter included: the instruction says "the objects"), grouped by category;
* a fixed map when the instruction has "<category words> objects into the <left|middle|right> <container>" clauses (each
  phrase must name exactly one category of the scene, every category must be named), else the free assignment that
  puts each group into a different container at the least total distance (group centroid to container centre).

``sort_into_bins(sk, groups)`` puts every object into its group's container with ``library.arrange``'s "put" steps (our
pick / place_oriented; ``library.put_into`` drops into containers off the table, not into baskets on it).
Reads the WorldState only (categories, poses), never labels' text: a label is a handle.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

POS_WORDS = {"left": 0, "middle": 1, "center": 1, "centre": 1, "right": 2}
CONTAINER_RE = r"basket|\bbin\b|_bin\b|bin_"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("_", " ").strip().lower())


def parse_map(text: str) -> list[tuple[str, int]]:
    pat = re.compile(r"([a-z][a-z_]*(?:\s+[a-z][a-z_]*){0,2})\s+objects?\s+into\s+the\s+(left|middle|center|centre|right)\s+"
                     r"(?:basket|bin|box|container)")
    out = []
    for phrase, pos in pat.findall((text or "").lower()):
        words = [w for w in phrase.split() if w not in ("put", "and", "then", "the", "all", "place")]
        out.append((" ".join(words), POS_WORDS[pos]))
    return out


def sort_groups(instruction: str, world) -> list[tuple[str, list[str]]] | None:
    objs = [o for o in world.objects.values() if o.category != "camera_stand"]
    conts = sorted([o for o in objs if re.search(CONTAINER_RE, o.category.lower())], key=lambda o: float(o.center_w[0]))
    items = [o for o in objs if o.obj_type == "Rigid" and o not in conts]
    if len(conts) < 2 or len(items) < 2:
        return None
    groups: dict[str, list] = {}
    for o in items:
        groups.setdefault(o.category, []).append(o)
    want = parse_map(instruction)
    if want:
        gmap = {}
        for phrase, idx in want:
            hit = [g for g in groups if _norm(g) == _norm(phrase) or _norm(phrase).endswith(_norm(g))]
            if len(hit) != 1 or idx >= len(conts):
                return None
            gmap[hit[0]] = conts[idx]
        if set(gmap) != set(groups):
            return None
    else:
        cats = sorted(groups)
        if len(cats) > len(conts):
            return None
        cent = {g: np.mean([o.center_w[:2] for o in groups[g]], axis=0) for g in cats}
        best, best_d = None, 1e9
        for perm in itertools.permutations(conts, len(cats)):
            d = sum(float(np.linalg.norm(cent[g] - c.center_w[:2])) for g, c in zip(cats, perm))
            if d < best_d:
                best, best_d = dict(zip(cats, perm)), d
        gmap = best
    out = [(c.label, [o.label for o in sorted(groups[g], key=lambda o: float(np.linalg.norm(o.center_w[:2] - c.center_w[:2])))])
           for g, c in gmap.items()]
    return sorted(out, key=lambda gc: float(np.linalg.norm(world.objects[gc[0]].center_w[:2])))


def inside(world, label: str, container: str) -> bool:
    """The object's centre over the container's footprint and below its rim + 2 cm (believed or true poses)."""
    o, c = world.objects.get(label), world.objects.get(container)
    if o is None or c is None:
        return False
    mn, mx = c.aabb()
    p = o.center_w
    return bool(mn[0] <= p[0] <= mx[0] and mn[1] <= p[1] <= mx[1] and p[2] <= mx[2] + 0.02)


def sort_into_bins(sk, groups: list, passes: int = 2, min_left: int = 120) -> dict:
    """Every object of each (container, objects) group put into its container with ``library.arrange``'s "put" steps
    (placed upright inside, the container an obstacle for the approach). A second pass takes the objects that are not
    inside their container (a slipped grasp) while ``min_left`` steps remain. ok = every object inside."""
    from dojo.skills import library as L
    want = {o: c for c, objects in groups for o in objects}
    log = []
    for k in range(int(passes)):
        w = sk.refresh()
        todo = [o for o, c in want.items() if not inside(w, o, c)]
        if not todo or sk.ex.ended or sk.ex.steps_left < min_left:
            break
        w.notes.pop("arrange", None)                     # arrange's own "done" memory: this pass re-tries them
        r = L.arrange(sk, [("put", o, want[o]) for c, objects in groups for o in objects if o in todo]) or {}
        log.append({"pass": k, "todo": todo, "stage": r.get("stage"), "note": str(r.get("note") or "")[:160]})
    w = sk.refresh()
    left = [o for o, c in want.items() if not inside(w, o, c)]
    return {"ok": not left, "stage": "done" if not left else "sort", "not_inside": left, "passes": log,
            "groups": [{"container": c, "n": len(objs)} for c, objs in groups]}


SKILLS = {"sort_into_bins": sort_into_bins}

__all__ = ["sort_groups", "sort_into_bins", "inside", "parse_map", "SKILLS"]
