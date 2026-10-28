# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/colour_order.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""Shared helpers for the instruction-ordered tier-A composites (stack_in_order, pour_in_order, cover_uncover).

Grounding is from the instruction string only: colour words and their order, plus nouns (block, bottle, bowl, cup).
A colour word is matched to an object through the catalogue model id in the privileged instance name
(`<category>_<model id>_<n>`, RoboDojo layout_manager) and a colour table copied from the PUBLIC task code
(gen_instruction colour maps and the task configs). That table is a disclosed prior; it is tier A only, because the
model id comes from the privileged pose reply. Task labels (the checker's names) are never used to choose.
"""
import importlib.util
import os
import re

# catalogue category -> model id -> colour word (public RoboDojo task code: instruction colour maps and configs)
CATALOGUE_COLOURS = {
    "cube": {1: "blue", 2: "green", 3: "red", 4: "red", 5: "blue", 6: "green", 7: "yellow", 8: "orange", 9: "cyan"},
    "wine_bottle": {0: "red", 1: "turquoise", 2: "violet"},
    "wine_bowl": {0: "black", 1: "white", 2: "brown"},
}
NOUN_CATS = {"block": ("cube", "block"), "cube": ("cube", "block"), "bottle": ("bottle",), "bowl": ("bowl",),
             "cup": ("cup",), "mug": ("mug", "cup"), "glass": ("glass",), "vase": ("vase",)}
ORDINALS = ("first", "second", "third", "fourth", "fifth", "next", "last")
INST_RE = re.compile(r"^(?P<cat>.+?)_(?P<model>\d+)_(?P<n>\d+)$")


def colour_words(table=None):
    t = table or CATALOGUE_COLOURS
    return sorted({c for m in t.values() for c in m.values()}, key=len, reverse=True)


def parse_inst(inst):
    """'cube_3_0' -> ('cube', 3); None when the name has no catalogue suffix."""
    m = INST_RE.match(inst or "")
    return (m.group("cat"), int(m.group("model"))) if m else None


def colour_of(inst, rec=None, table=None):
    t = table or CATALOGUE_COLOURS
    p = parse_inst(inst)
    if p is None:
        return None
    cat = (rec or {}).get("category") or p[0]
    return (t.get(cat) or t.get(p[0]) or {}).get(p[1])


def colours_in(text, table=None):
    """Colour words of `text` in order of appearance (whole words, lower case)."""
    words = colour_words(table)
    pat = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")
    return pat.findall((text or "").lower())


def cat_matches(rec, inst, noun):
    keys = NOUN_CATS.get(noun, (noun,))
    hay = " ".join(str(s or "") for s in (rec.get("category"), inst)).lower()
    return any(k in hay for k in keys)


def ground(objs, colour, noun, table=None, skip=()):
    """The one object of `noun`'s category whose catalogue colour is `colour`; raises ValueError otherwise."""
    hits = [k for k, v in objs.items() if k not in skip and "error" not in v and cat_matches(v, k, noun)
            and colour_of(k, v, table) == colour]
    if len(hits) != 1:
        cands = sorted((k, colour_of(k, v, table)) for k, v in objs.items() if cat_matches(v, k, noun))
        raise ValueError(f"colour {colour!r} {noun}: {len(hits)} matches {hits}; candidates {cands}")
    return hits[0]


def parse_pairs(text, sources=("bottle", "cup", "mug", "glass", "jug"), targets=("bowl", "cup", "vase", "glass", "basket", "mug")):
    """'from the first red bottle into the first black bowl, from the second ...' -> [(red, bottle, black, bowl), ...]."""
    ords = r"(?:(?:" + "|".join(ORDINALS) + r")\s+)?"
    pat = re.compile(r"from\s+the\s+" + ords + r"([a-z]+)\s+(" + "|".join(sources) + r")s?\b\s+into\s+the\s+" + ords +
                     r"([a-z]+)\s+(" + "|".join(targets) + r")s?\b")
    return [tuple(m) for m in pat.findall((text or "").lower())]


def load_sibling(name, fname="skill_robodojo.py"):
    """A fresh module instance of skills/<name>/<fname> (read-only reuse of another skill's helpers)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, name, fname)
    spec = importlib.util.spec_from_file_location(f"sib_{name}_{os.getpid()}_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
