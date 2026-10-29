"""Glyph identity reads persist across a run's later records (scan / agentic / look-fused / search re-estimates).

A slot's identity is READ once (``glyph_read``: a numeral's digit, an operator, a mahjong tile) when a view shows its
glyph. A later re-estimate of the same label often cannot read it (the scan's arm hides it, the agentic perceiver
starts from the label's first candidate): before, that later record put the label's placeholder back and the read was
lost (5.1 solve_equation 100005: frame 0 read t1 "minus", the scan record had "division" / unknown, and the skill
stopped at "values unknown"). Rule, per label (the semantics of ``RuntimePolicy._merge_glyph``):
- a later record WITHOUT a read of the label keeps the earlier read (asset, identity "read", digit fields; ``carried``);
- an explicit later read without a classifier read (``identity`` "read", e.g. the agentic perceiver) counts as a read:
  same asset agrees, another asset contradicts;
- a later read that AGREES (same asset) refreshes it;
- a later read that CONTRADICTS it makes the label unknown (identity "unknown", digit None / "unread"), and the
  earlier read is dropped.
Only SLOT entries (``_perception.slot``: a label the config pins to a pose or stacks on a base) carrying a classifier
read (``_perception.glyph_read.read``) start a persisted read, so this is a no-op unless ``EstimatorConfig.glyph_read``
is on. A free object's label is not an identity: each re-estimate assigns the free labels to tiles anew (in 100005 the
scan's num_5 sat where frame 0's num_6 was), so persisting by label there would carry a digit onto another tile. No
config relation, no task keying: labels only. Pure python.
"""
from __future__ import annotations

import copy

RECORD_TYPES = ("Rigid", "Geometry", "Articulation", "Dynamic")
DIGIT_FIELDS = ("digit", "digit_conf", "digit_source", "digit_unread")


def _entries(record: dict):
    for typ in RECORD_TYPES:
        for cat, lst in list((record.get(typ) or {}).items()):
            for r in list(lst):
                yield typ, cat, r


def _is_read(r: dict) -> bool:
    g = (r.get("_perception") or {}).get("glyph_read")
    return bool(isinstance(g, dict) and g.get("read") and (r.get("_perception") or {}).get("identity") != "unknown")


def _asset(typ: str, cat: str, r: dict) -> str:
    return f"{typ}/{cat}/{int(r['category_idx']):05d}"


def _move(record: dict, typ: str, cat: str, r: dict, new_typ: str, new_cat: str) -> None:
    lst = record[typ][cat]
    lst.remove(r)
    if not lst:
        del record[typ][cat]
    record.setdefault(new_typ, {}).setdefault(new_cat, []).append(r)


def persist_reads(record: dict, reads: dict) -> tuple[dict, dict, list]:
    """Apply the earlier ``reads`` ({label: {"asset", "perception"}}) to ``record`` (a copy is returned) and update
    them with its own reads. Returns (record, reads, events)."""
    rec = copy.deepcopy(record)
    reads = dict(reads)
    events = []
    for typ, cat, r in list(_entries(rec)):
        lab = r.get("label")
        if not lab:
            continue
        p = r.setdefault("_perception", {})
        prev = reads.get(lab)
        if _is_read(r) and (p.get("slot") or prev is not None):
            key = _asset(typ, cat, r)
            if prev is not None and prev["asset"] != key:
                p["identity"] = "unknown"
                p["glyph_read"] = {**p["glyph_read"], "read": False, "why": "disagree", "earlier": prev["asset"]}
                if p.get("digit") is not None or prev["perception"].get("digit") is not None:
                    p.update(digit=None, digit_conf=None, digit_source="unread", digit_unread="disagree")
                reads.pop(lab, None)
                events.append({"label": lab, "why": "disagree", "earlier": prev["asset"], "later": key})
                continue
            reads[lab] = {"asset": key, "perception": {k: p.get(k) for k in ("identity", "glyph_read", *DIGIT_FIELDS)
                                                         if k in p}}
            continue
        if prev is None:
            continue
        g = p.get("glyph_read")
        if isinstance(g, dict) and g.get("read"):          # read but marked unknown: leave it unknown
            continue
        if p.get("identity") == "read":                     # another reader's explicit read (the agentic perceiver)
            if _asset(typ, cat, r) != prev["asset"]:
                p["identity"] = "unknown"
                if p.get("digit") is not None or prev["perception"].get("digit") is not None:
                    p.update(digit=None, digit_conf=None, digit_source="unread", digit_unread="disagree")
                reads.pop(lab, None)
                events.append({"label": lab, "why": "disagree", "earlier": prev["asset"], "later": _asset(typ, cat, r)})
            continue
        ntyp, ncat, nidx = prev["asset"].split("/")
        was = _asset(typ, cat, r)
        r["category_idx"] = int(nidx)
        if (ntyp, ncat) != (typ, cat):
            _move(rec, typ, cat, r, ntyp, ncat)
            r["category"] = ncat
        for k, v in prev["perception"].items():
            p[k] = copy.deepcopy(v)
        p["glyph_read"] = {**(p.get("glyph_read") or {}), "carried": True}
        events.append({"label": lab, "why": "carried", "asset": prev["asset"], "was": was})
    return rec, reads, events
