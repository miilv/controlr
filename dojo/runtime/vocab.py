"""Instruction -> perception vocabulary without recognising the task: the catalogue fallback.

``supervisor.VOCABULARIES`` (instruction regexes) is the fast path. When no entry matches (a paraphrase, a phrase the
table did not anticipate), ``catalog_vocabulary`` picks the catalogue candidate set (``catalog/tasks.json`` key)
whose OBJECTS the instruction names:

* every label of every candidate set lists its candidate categories (``Rigid/<category>/<idx>`` keys);
* each category has English aliases: its own name split on ``_`` (singular), plus ``ALIASES`` below, a lexicon
  over catalogue categories (not over tasks);
* the instruction's phrases (singular words, longest 1-3 grams first, non-overlapping) that name a category select
  it; a set scores each phrase by how discriminative it is across sets (idf), see ``score_sets``;
* ties go to the shortest name (the std set before its ``_random`` twin, which shares the instruction).

Nothing here reads the task id: the input is the instruction and the public catalogue only.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

# catalogue category -> English words / phrases (singular) an instruction may use for it
ALIASES: dict[str, tuple[str, ...]] = {
    "block": ("block", "wooden block", "board"), "cube": ("block", "cube"), "small_cube": ("block", "cube"),
    "triangular_prism": ("set square", "prism"), "cube_cushion": ("mat", "pad", "cushion"), "number": ("number", "digit"),
    "basket_grasp": ("basket",), "action_camera": ("camera",), "toy_car": ("car", "toy car", "toy"),
    "chocolate_bar": ("chocolate",), "wooden_toy": ("toy",), "piggy_bank": ("coin bank", "piggy bank", "bank"),
    "vertical_coin_stand": ("holder", "coin holder", "stand"), "factory_bolt": ("screw", "bolt"), "factory_nut": ("nut",),
    "egg_basket": ("basket",), "egg_holder": ("egg holder",), "oil_pen": ("pen",), "pen_holder": ("pen holder",),
    "Top_Long": ("clothes", "cloth", "shirt", "garment"), "cup_holder": ("mug rack", "rack", "mug holder"),
    "key_slot": ("keyhole", "key slot", "lock"), "test_tube": ("tube", "test tube"), "test_tube_slot": ("rack", "tube rack"),
    "mahjong": ("tile", "kong", "mahjong"), "bread": ("bread", "slice"), "bread_shelf": ("shelf",), "toaster": ("toaster", "toast"),
    "alarm": ("alarm clock", "alarm", "clock"), "mousemat": ("mouse pad", "mousepad", "mat"), "monitor": ("monitor", "stand"),
    "electric_toothbrush": ("toothbrush",), "photo": ("image", "photo", "picture"),
    "mallet_stand": ("mallet stand",), "stack_base": ("peg", "base"), "stackingblocks": ("stacking toy", "piece", "ring"),
    "checkerboard": ("board", "tic-tac-toe", "tic tac toe", "grid"), "chessman": ("piece", "marker"),
    "socket": ("power strip", "socket", "outlet", "strip"), "charger": ("charger", "plug"), "sphere": ("ball",),
    "wine_bottle": ("bottle",), "wine_bowl": ("bowl",), "wuliangye": ("bottle", "liquid"), "goblet": ("cup", "glass", "goblet"),
    "SpringButton": ("button",), "number_card": ("number card", "card"), "t": ("t-shaped block", "t block"),
    "t_cushion": ("t-shaped pad", "pad"), "dustbin": ("dustbin", "bin", "trash can"),
    "division": ("operator", "equation"), "equal": ("equation", "equal"), "minus": ("operator", "equation"),
    "multiplication": ("operator", "equation"), "plus": ("operator", "equation"),
    "matryoshka_dolls": ("nesting doll", "doll", "matryoshka"), "headset": ("headphone", "headset"),
    "headset_stand": ("headphone stand", "headset stand"), "laptop_stand": ("laptop stand",),
    "vertical_laptop_storage_rack": ("laptop stand", "laptop rack"), "toolbox": ("toolbox", "tool box"),
    "hammer": ("hammer", "tool"), "pliers": ("plier", "tool"), "tape_measure": ("tape measure", "tool"), "wrench": ("wrench", "tool"),
    "broom_shovel": ("dustpan", "pan", "shovel"), "earbuds": ("earbud",), "paper_ball": ("paper ball", "ball"),
    "brick": ("brick", "block"), "conveyor": ("conveyor",), "garage": ("garage",),
}
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _sing(w: str) -> str:
    for suf, rep in (("ies", "y"), ("ches", "ch"), ("shes", "sh"), ("xes", "x"), ("sses", "ss"), ("s", "")):
        if len(w) > 3 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


def _words(text: str) -> list[str]:
    return [_sing(w) for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text or "").lower())]


def _grams(words: list[str], n: int = 3) -> set[str]:
    out = set()
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            out.add(" ".join(words[i:i + k]))
    return out


def category_terms(cat: str) -> set[str]:
    base = " ".join(_words(cat.replace("_", " ")))
    terms = {base} | {" ".join(_words(a)) for a in ALIASES.get(cat, ())}
    return {t for t in terms if t}


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    """{set: {label: [categories]}} from catalogue tasks.json (or the compact form)."""
    d = json.loads(Path(path).read_text())
    out = {}
    for name, entry in d.items():
        labels = entry.get("labels", entry) if isinstance(entry, dict) else {}
        labs = {}
        for lab, v in labels.items():
            if isinstance(v, list):
                labs[lab] = sorted(set(v))
            else:
                cats = {c["name"] for c in v.get("categories") or []} | {k.split("/")[1] for k in v.get("candidates") or []
                                                                          if k.count("/") == 2}
                labs[lab] = sorted(cats)
        out[name] = labs
    return out


def catalog_path() -> str | None:
    for d in [*os.environ.get("DOJO_CATALOG", "").split(os.pathsep), "~/dojo-perception-data/catalog",
              "/root/dojo-data/perception/catalog"]:
        if d and Path(os.path.expanduser(d), "tasks.json").exists():
            return str(Path(os.path.expanduser(d), "tasks.json"))
    return None


def _spans(words: list[str], vocab: set[str], n: int = 3) -> list[str]:
    """Instruction phrases that name some catalogue term, longest first, non-overlapping ("toy car" uses up "toy")."""
    used, out = set(), []
    for k in range(n, 0, -1):
        for i in range(len(words) - k + 1):
            g = " ".join(words[i:i + k])
            if g in vocab and not used & set(range(i, i + k)):
                used |= set(range(i, i + k))
                out.append(g)
    return out


def score_sets(instruction: str, sets: dict) -> list[tuple[float, str, dict]]:
    """[(score, set name, detail)] best first. Per instruction phrase that names one of a set's categories: the
    phrase's idf over the sets (how discriminative the WORD is), x 1.0 when it is the category's own name, 0.8 for an
    alias; times 0.75 + 0.25 x the share of the set's labels the named categories cover; + 0.5 when a number word
    equals a named category's label count. Phrases and categories pair one to one (greedy by weight), and a set whose
    named categories sit on fewer labels than there are of them (a pool set) is scaled by labels / categories."""
    words = _words(instruction)
    nums = {NUMBERS[w] for w in words if w in NUMBERS} | {int(w) for w in words if w.isdigit() and int(w) <= 20}
    cats_of = {name: {c for cs in labs.values() for c in cs} for name, labs in sets.items()}
    terms = {c: category_terms(c) for cs in cats_of.values() for c in cs}
    n_sets = len(sets)
    # object mentions: the longest phrases naming any catalogue term ("toy car" is one mention); a set without that
    # term may still match the phrase's head noun (its last word), at 0.8
    spans = _spans(words, {t for ts in terms.values() for t in ts})

    def cands(cats, g):
        full = [(c, 1.0 if " ".join(_words(c.replace("_", " "))) == g else 0.8) for c in cats if g in terms[c]]
        if full or " " not in g:
            return full
        h = g.split()[-1]
        return [(c, 0.8 * (1.0 if " ".join(_words(c.replace("_", " "))) == h else 0.8)) for c in cats if h in terms[c]]
    sets_of = {g: {name for name, cs in cats_of.items() if cands(cs, g)} for g in spans}
    out = []
    for name, labs in sets.items():
        cats = cats_of[name]
        pairs = []                                    # (weight, phrase, category)
        for g in spans:
            if not sets_of[g]:
                continue
            w = math.log(1 + n_sets / len(sets_of[g]))
            pairs += [(w * f, g, c) for c, f in cands(cats, g)]
        used_g, best = set(), {}                      # one phrase <-> one category (greedy by weight)
        for w, g, c in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
            if g not in used_g and c not in best:
                used_g.add(g)
                best[c] = w
        m, sc = set(best), sum(best.values())
        if not m or not labs:
            continue
        hit_labels = [lab for lab, cs in labs.items() if set(cs) & m]
        cover = len(hit_labels) / len(labs)
        # k distinct objects named but all on fewer labels (a pool set, e.g. one label with 85 candidates): implausible
        distinct = min(1.0, len(hit_labels) / len(m))
        counts = {c: sum(1 for cs in labs.values() if c in cs) for c in m}
        bonus = 0.5 if nums and any(v in nums for v in counts.values()) else 0.0
        out.append((round(sc * (0.75 + 0.25 * cover) * distinct + bonus, 4), name,
                    {"matched": sorted(m), "phrases": list(spans), "cover": round(cover, 3), "count_bonus": bonus}))
    out.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    return out


def catalog_vocabulary(instruction: str, path: str | None = None, min_score: float = 1.0) -> tuple[str | None, dict]:
    """(catalogue set name or None, detail) for an instruction, from the catalogue alone."""
    p = path or catalog_path()
    if p is None:
        return None, {"why": "no catalogue tasks.json"}
    ranked = score_sets(instruction, _load(p))
    if not ranked or ranked[0][0] < min_score:
        return None, {"why": "no catalogue object named", "top": [(s, n) for s, n, _ in ranked[:3]]}
    s, name, det = ranked[0]
    return name, {**det, "score": s, "runner_up": [(s2, n2) for s2, n2, _ in ranked[1:3]]}


__all__ = ["ALIASES", "catalog_vocabulary", "score_sets", "category_terms", "catalog_path"]
