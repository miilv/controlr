"""Digits shown on objects (press_by_number's number cards): B1's glyph evidence, the record / J1 fields, and the one
rule that arbitrates B1's read against the agentic perceiver's explicit reading. Stdlib only: the J1 client side, the
runtime and the bridge import it.

A digit is never derived from an asset index nobody read. B1 picks a card's asset by textured render-and-compare
(``inverse_layout._slot_identity``); when no texture could be scored (no GL, a flat or hidden glyph) or the runner-up
ties, the chosen index is a placeholder (the first candidate), and the old export turned it into a digit anyway
(robot-harness REVIEW-G2-B1-PERCEPTION finding 2: ``digit=5 alternative_margin=0.0 ambiguous=True``).

Record fields (``_perception`` of a digit-category entry; ``inverse_layout.estimate``, ``agentic.record``):
  digit         int | None  the digit the object shows; None = unknown
  digit_conf    float       identity confidence 0..1, separate from the pose's ``confidence`` (B1: margin over the
                            runner-up / ``ambiguity_margin``, capped at 1; agent: its confidence for the entry)
  digit_source  str         "texture" (B1's glyph read) | "config" (one candidate of that geometry) | "agent" (the
                            agentic perceiver's explicit ``value``) | "agent+texture" (both, agreeing) | "j1" (a J1
                            server's ``value``) | "unread" (digit None)
  digit_unread  str         why unknown: "no_texture" | "tied" | "not_read" (identity never read) | "no_value" |
                            "agent_rejected"
J1 objects carry the digit as ``attributes.digit`` (B1, as the schema expects) or ``value`` (agent output), plus
``digit_conf`` / ``digit_source`` / ``digit_unread`` (stripped by the J1 schema, kept in our logs).

THE RULE (``resolve``), one object with a B1 reading and maybe an explicit agent ``value``:
  1. the agent's value is ACCEPTED when it is a digit (int 0-9, or its string), one a candidate shows where the label's
     candidates are known (``valid``; the record merge checks them, ``agentic.record.apply_digit``), and the agent's
     confidence for the entry is >= ``AGENT_MIN_CONF`` (the agent's own ``confidence`` when it wrote one, else
     ``AGENT_DEFAULT_CONF``, j1.answer_to_output's default for an agent reading); else it is ignored and logged;
  2. an accepted value is the digit: it agrees with B1's read ("agent+texture", the larger confidence), replaces B1's
     unread placeholder ("agent"), or OVERRIDES a different B1 read ("agent", logged as a conflict with both digits and
     B1's confidence). The agent made it with B1's digit and margin in its hint and the frames in front of it: an
     explicit different value is a deliberate correction, and B1's texture compare is the side known to fail under
     occlusion (J1 press L4: a hovering arm made the card read 1 at margin 0.003, truth 5);
  3. without an accepted value, B1's read stands (or the digit is unknown).
The asset index follows the resolved digit (``idx_for_digit``), so the twin, the planner and the J1 output agree.
"""
from __future__ import annotations

# category -> digit shown by asset index (catalog.CATEGORY_ATTRIBUTES: Geometry/number_card/0000<i> shows i)
DIGIT_OF = {"number_card": lambda idx: int(idx)}

AGENT_MIN_CONF = 0.5
AGENT_DEFAULT_CONF = 0.8


def digit_of(category: str | None, idx) -> int | None:
    f = DIGIT_OF.get(str(category or ""))
    return None if f is None or idx is None else f(int(idx))


def category_of_key(key: str | None) -> str | None:
    parts = str(key or "").split("/")
    return parts[1] if len(parts) == 3 else None


def idx_for_digit(category: str, digit: int, idxs=None) -> int | None:
    """The asset index of ``category`` that shows ``digit``: among ``idxs`` (the label's candidates) when given."""
    if category not in DIGIT_OF:
        return None
    for i in (idxs if idxs is not None else range(100)):
        if digit_of(category, i) == digit:
            return int(i)
    return None


def asset_for_digit(key: str | None, digit: int) -> str | None:
    """The catalogue key of ``key``'s category that shows ``digit`` (``key`` itself when it already does, or when the
    category has no digits / no asset shows it)."""
    cat = category_of_key(key)
    if cat not in DIGIT_OF or digit_of(cat, key.split("/")[2]) == digit:
        return key
    idx = idx_for_digit(cat, digit)
    return key if idx is None else f"{key.split('/')[0]}/{cat}/{idx:05d}"


def as_digit(v) -> int | None:
    """An int 0-9 from an int or a one-digit string; anything else (bool, float, words) is not a digit."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 0 <= v <= 9 else None
    if isinstance(v, str) and v.strip().isdigit() and len(v.strip()) == 1:
        return int(v.strip())
    return None


def glyph_evidence(key: str, texture, margin, min_margin: float = 0.0, full_margin: float = 0.05) -> dict:
    """B1's glyph identity read of one object (``key`` = the chosen candidate). Read only when the chosen candidate's
    texture was scored (``texture`` not None) and its cost beats the best alternative by MORE than ``min_margin``
    (``margin`` None: no alternative). ``confidence`` = min(1, margin / ``full_margin``) when read, else 0."""
    if texture is None:
        why = "no_texture"
    elif margin is not None and margin <= min_margin:
        why = "tied"
    else:
        why = "texture"
    read = why == "texture"
    conf = (1.0 if margin is None else min(1.0, max(0.0, float(margin)) / full_margin)) if read else 0.0
    return {"key": key, "read": read, "why": why, "texture": None if texture is None else round(float(texture), 4),
            "margin": None if margin is None else round(float(margin), 4), "confidence": round(conf, 3)}


def record_fields(category: str, idx, glyph: dict | None, key: str, n_same_geometry: int) -> dict:
    """``_perception`` digit fields of a B1 record entry ({} for a category without digits). ``glyph``: the entry's
    ``glyph_evidence`` (ignored when it is about another key); ``n_same_geometry``: candidates the geometry cannot tell
    apart (1 = the config fixes the asset)."""
    if category not in DIGIT_OF:
        return {}
    if glyph is not None and glyph.get("key") == key:
        if glyph.get("read"):
            return {"digit": digit_of(category, idx), "digit_conf": glyph.get("confidence"), "digit_source": "texture"}
        return {"digit": None, "digit_source": "unread", "digit_unread": glyph.get("why") or "not_read"}
    if n_same_geometry <= 1:
        return {"digit": digit_of(category, idx), "digit_conf": 1.0, "digit_source": "config"}
    return {"digit": None, "digit_source": "unread", "digit_unread": "not_read"}


def from_record(category: str, p: dict) -> tuple[int | None, float | None, str | None]:
    """(digit, confidence, source) of a perception record entry's ``_perception`` ``p``. A digit-category entry
    without recorded digit fields (a label found by the runtime's search, an older record) is unread: its asset index
    was chosen, not read. (None, None, None) for other categories."""
    if category not in DIGIT_OF:
        return None, None, None
    if "digit_source" not in p:
        return None, None, "unread"
    d = as_digit(p.get("digit"))
    c = p.get("digit_conf")
    return d, (None if c is None else float(c)), (p["digit_source"] if d is not None else "unread")


def b1_reading(o: dict | None) -> dict | None:
    """B1's reading of a J1-style object (``attributes.digit`` / ``value``, ``digit_conf``, ``digit_unread``):
    {"digit", "conf", "source", "unread"}; None when the object is not a digit object."""
    if not o:
        return None
    d = as_digit((o.get("attributes") or {}).get("digit"))
    if d is None:
        d = as_digit(o.get("value"))
    if d is not None:
        c = o.get("digit_conf")
        return {"digit": d, "conf": 1.0 if c is None else float(c), "source": o.get("digit_source") or "texture",
                "unread": None}
    if o.get("digit_unread") or category_of_key(o.get("asset")) in DIGIT_OF or o.get("category") in DIGIT_OF:
        return {"digit": None, "conf": None, "source": "unread", "unread": o.get("digit_unread") or "not_read"}
    return None


def resolve(b1: dict | None, agent_value=None, agent_conf: float | None = None, valid=None) -> dict:
    """THE RULE (module docstring). ``b1``: ``b1_reading`` of the paired B1 object (None: no B1 object);
    ``agent_value`` / ``agent_conf``: the agent's explicit reading (None: it gave none); ``valid``: the digits the
    candidates show (None: any 0-9). -> {"digit", "conf", "source", "unread", "event"}; ``event`` (a dict, or None) is
    what the caller logs: the conflict, the agent's correction of an unread card, or why its value was ignored."""
    a = as_digit(agent_value)
    ac = AGENT_DEFAULT_CONF if agent_conf is None else float(agent_conf)
    event = None
    accepted = False
    if agent_value is not None:
        if a is None or (valid is not None and a not in set(valid)):
            event = {"rule": "agent_invalid", "agent": agent_value}
        elif ac < AGENT_MIN_CONF:
            event = {"rule": "agent_low_conf", "agent": a, "agent_conf": round(ac, 3), "min_conf": AGENT_MIN_CONF}
        else:
            accepted = True
    bd = (b1 or {}).get("digit")
    bc = (b1 or {}).get("conf")
    if accepted:
        if bd is None:
            ev = {"rule": "agent_reads", "agent": a, "agent_conf": round(ac, 3)}
            if b1 is not None:
                ev["b1_unread"] = b1.get("unread")
            return {"digit": a, "conf": round(ac, 3), "source": "agent", "unread": None, "event": ev}
        if bd == a:
            return {"digit": a, "conf": round(max(ac, bc if bc is not None else 0.0), 3), "source": "agent+texture",
                    "unread": None, "event": None}
        return {"digit": a, "conf": round(ac, 3), "source": "agent", "unread": None,
                "event": {"rule": "agent_over_b1", "agent": a, "agent_conf": round(ac, 3), "b1": bd,
                          "b1_conf": None if bc is None else round(float(bc), 3)}}
    if bd is not None:
        if event is not None:
            event["kept"] = bd
        return {"digit": bd, "conf": bc, "source": (b1 or {}).get("source") or "texture", "unread": None, "event": event}
    unread = "agent_rejected" if event is not None else ((b1 or {}).get("unread") or "not_read")
    return {"digit": None, "conf": None, "source": "unread", "unread": unread, "event": event}


def keep_agent_digit(new: dict, old: dict) -> dict | None:
    """A later B1 re-read ``new`` of an object whose tracked state ``old`` carries the agent's accepted reading (``value``
    with ``digit_source`` agent / agent+texture; J1-style dicts, paired by ``tracker.Tracker``) goes through THE RULE
    again, so the next B1-gated call does not silently replace the correction with B1's digit. Edits ``new`` (the digit
    as ``value``, B1's ``attributes.digit`` dropped: the J1 schema would prefer it); returns the rule's event."""
    a = as_digit(old.get("value"))
    if old.get("digit_source") not in ("agent", "agent+texture") or a is None \
            or new.get("digit_source") in ("agent", "agent+texture"):  # ``new`` is the tracked state itself, not a re-read
        return None
    r = resolve(b1_reading(new), a, old.get("digit_conf"))
    attrs = {k: v for k, v in (new.get("attributes") or {}).items() if k != "digit"}
    new.pop("attributes", None)
    if attrs:
        new["attributes"] = attrs
    new.pop("digit_unread", None)
    new.update(value=r["digit"], digit_source=r["source"], digit_conf=r["conf"])
    new["asset"] = asset_for_digit(new.get("asset"), r["digit"]) or new.get("asset")
    return r["event"]


__all__ = ["DIGIT_OF", "AGENT_MIN_CONF", "AGENT_DEFAULT_CONF", "digit_of", "category_of_key", "idx_for_digit",
           "asset_for_digit", "as_digit", "glyph_evidence", "record_fields", "from_record", "b1_reading", "resolve",
           "keep_agent_digit"]
