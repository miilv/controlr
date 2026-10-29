"""Pour from several bottles into several bowls, one pair after the other, in the order the instruction gives (stream N;
pour_by_language's instruction: "Pour the liquid from the first red bottle into the first black bowl, from the second
turquoise bottle into the second white bowl, and from the third violet bottle into the third brown bowl. Then reset
the robot arm.").

The pairs and their order come from the instruction's clauses ("from the <colour> bottle into the <colour> bowl"); each
colour word is matched to one object of that kind by colour (an assignment over the named colours, so "violet" and
"turquoise" need no exact RGB): perception's colour name when present, else the asset's mean catalogue colour. Each
pour is stream P's port of robot-harness's ``pour_bottle`` (``dojo.skills.ported``: grasp the neck, roll over the bowl,
drain, untilt, put the bottle back, both arms home). Pours run strictly in order (lead's ruling): the next one starts
only after the previous one ran its whole procedure; a pour that could not run stops the sequence.

Route "pour_order", off by default (``RuntimeConfig.new_routes``). Route-only (``library.ROUTE_ONLY``): it calls the
ported skill, so the LLM agent's manual does not offer it.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

NAMED_RGB = {"red": (230, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 230), "yellow": (240, 230, 30),
             "orange": (250, 110, 0), "cyan": (20, 230, 230), "turquoise": (64, 224, 208), "purple": (140, 40, 200),
             "violet": (143, 80, 255), "pink": (250, 140, 190), "white": (240, 240, 240), "black": (20, 20, 20),
             "brown": (110, 70, 40), "grey": (128, 128, 128), "gray": (128, 128, 128)}
CLAUSE = re.compile(r"from\s+the\s+(?:(?:first|second|third|fourth)\s+)?([a-z]+)\s+bottle\s+into\s+the\s+"
                    r"(?:(?:first|second|third|fourth)\s+)?([a-z]+)\s+bowl")


def _rgb(o):
    c = getattr(o, "colour", None)
    if c and c in NAMED_RGB:
        return np.asarray(NAMED_RGB[c], float)
    rgb = getattr(o.asset, "rgb", None)
    return None if rgb is None else np.asarray(rgb, float)[:3]


def match_colours(words: list[str], objs: list) -> dict | None:
    """{colour word: object label}: the assignment of the named colours to distinct objects with the least total
    colour distance; None when a word is no colour we know or an object has no colour."""
    if any(w not in NAMED_RGB for w in words) or len(objs) < len(words):
        return None
    rgbs = [_rgb(o) for o in objs]
    if any(c is None for c in rgbs):
        return None
    best, best_cost = None, float("inf")
    for perm in itertools.permutations(range(len(objs)), len(words)):
        cost = sum(float(np.linalg.norm(rgbs[j] - np.asarray(NAMED_RGB[w], float))) for w, j in zip(words, perm))
        if cost < best_cost:
            best, best_cost = perm, cost
    return {w: objs[j].label for w, j in zip(words, best)}


def route(text: str, world, routes) -> tuple | None:
    """("pour_in_order", {"pairs": [[bottle, bowl], ...]}, why) when the route is on and every clause's colours match."""
    if "pour_order" not in set(routes or ()) or not re.search(r"\bpour\b", text):
        return None
    clauses = CLAUSE.findall(text)
    if not clauses:
        return None
    objs = [o for o in world.task_objects() if o.obj_type == "Rigid"]
    bottles = [o for o in objs if "bottle" in o.category.lower()]
    bowls = [o for o in objs if "bowl" in o.category.lower()]
    mb = match_colours([c[0] for c in clauses], bottles)
    mw = match_colours([c[1] for c in clauses], bowls)
    if mb is None or mw is None or len(set(mb.values())) < len(clauses):
        return None
    pairs = [[mb[b], mw[w]] for b, w in clauses]
    return ("pour_in_order", {"pairs": pairs}, f"{len(pairs)} pours in order: {pairs}")


CENTRE_X = 0.05     # m: a bottle this close to x = 0 is on both arms' side (pour_bottle_lean.CENTRE_X)


def alternate_arms(xs: list) -> list:
    """The arm per pour (bottle x in order) that alternates the most between consecutive pours: a bottle right of
    +CENTRE_X is the right arm's, left of -CENTRE_X the left's, one in between either; ties keep the vendored skill's
    choice (right for x >= 0). The overlap pace hands a bottle to the other arm only across an arm switch."""
    import itertools as it
    free = [i for i, x in enumerate(xs) if abs(x) <= CENTRE_X]
    best, best_key = None, None
    for pick in it.product(("right", "left"), repeat=len(free)):
        arms = ["right" if x > CENTRE_X else "left" if x < -CENTRE_X else None for x in xs]
        for i, a in zip(free, pick):
            arms[i] = a
        key = (sum(a != b for a, b in zip(arms, arms[1:])), sum(a == ("right" if xs[i] >= 0 else "left")
                                                                for i, a in enumerate(arms)))
        if best_key is None or key > best_key:
            best, best_key = arms, key
    return best


def pour_in_order(sk, pairs: list, reserve: int | None = None) -> dict:
    """Pour each [bottle, bowl] in ``pairs`` in order through the ported ``pour_bottle``; stop at the first pour that
    did not run its whole procedure (a later pour must never come before an earlier one)."""
    from dojo.skills.ported import ported
    done = []
    held = None                     # overlap: the bottle the other arm grasped and lifted during the previous pour
    plan_arms = None
    if getattr(sk, "ported_pour_pace", "v1") == "overlap":
        try:
            w = sk.refresh()
            plan_arms = alternate_arms([float(w.obj(b).pos[0]) for b, _ in pairs])
        except Exception:  # noqa: BLE001 (no world: the skill picks its arms itself)
            plan_arms = None
    for i, (bottle, bowl) in enumerate(pairs):
        sk.refresh()
        from dojo.skills.planner import pour_bottle_call
        pace = getattr(sk, "ported_pour_pace", "v1")
        skill, args = pour_bottle_call(getattr(sk, "ported_pour_aim", "v1"), pace)
        if skill != "pour_bottle" and i < len(pairs) - 1:
            # a lean pour does not home between pours: the checker banks nothing at home (its arms-home condition is
            # "either arm", and the idle arm never leaves), the next pour starts from the retreat (the skill homes
            # the other arm itself when it switches arms), and the episode's finish homes both at the end
            args["home"] = False
        if pace == "overlap":
            # the other arm grasps and lifts the next pair's bottle while this pour runs; that pour then starts with
            # it in hand. The pours themselves stay strictly in order
            if i < len(pairs) - 1:
                args["companion"] = [f"@{pairs[i + 1][0]}", f"@{pairs[i + 1][1]}"]
            if held is not None:
                args["pre_held"] = held
            elif plan_arms:
                args["arm_pref"] = plan_arms[i]
        r = ported(sk, skill, {"container": f"@{bottle}", "target": f"@{bowl}", **args}, reserve=reserve)
        held = r.get("companion")
        sk.log(f"[pour_order] {i + 1}/{len(pairs)} {bottle} -> {bowl}: ok {r.get('ok')} verified {r.get('verified')} "
               f"steps {r.get('steps')} {str(r.get('error') or '')[:120]}")
        if not r.get("ok"):
            return {"ok": False, "stage": f"pour {i + 1}", "note": f"done {done}; {bottle} -> {bowl}: {r.get('status')}"}
        done.append([bottle, bowl])
        if r.get("status") in ("ended", "step_budget"):
            break
    return {"ok": len(done) == len(pairs), "stage": "poured", "note": f"done {done}"}


ROUTE_SKILLS = {"pour_in_order": pour_in_order}

__all__ = ["route", "match_colours", "pour_in_order", "ROUTE_SKILLS", "CLAUSE"]
