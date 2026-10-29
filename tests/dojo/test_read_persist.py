"""Glyph identity reads persist across a run's later records (dojo.perception.read_persist): the 5.1 solve_equation
100005 case (frame 0 read t1 "minus"; the scan record put the placeholder "division" back, identity unknown)."""
from dojo.perception.read_persist import persist_reads


def _rec(t1_cat, t1_read, t0_digit=None):
    p1 = {"identity": "read" if t1_read else "unknown", "slot": True}
    if t1_read:
        p1["glyph_read"] = {"read": True, "name": t1_cat, "conf": 0.5, "n_views": 1}
    p0 = {"slot": True}
    if t0_digit is not None:
        p0.update(identity="read", digit=t0_digit, digit_conf=1.0, digit_source="glyph",
                  glyph_read={"read": True, "name": str(t0_digit), "conf": 1.0})
    return {"Rigid": {t1_cat: [{"category": t1_cat, "category_idx": 0, "label": "t1", "_perception": p1}],
                      "number": [{"category": "number", "category_idx": t0_digit or 0, "label": "t0", "_perception": p0}]}}


def test_a_read_survives_a_later_record_without_one():
    rec0, reads, ev = persist_reads(_rec("minus", True, t0_digit=9), {})
    assert ev == [] and reads["t1"]["asset"] == "Rigid/minus/00000" and reads["t0"]["asset"] == "Rigid/number/00009"
    scan = _rec("division", False)                                     # the scan: placeholder back, no read
    rec1, reads, ev = persist_reads(scan, reads)
    t1 = rec1["Rigid"]["minus"][0]
    assert "division" not in rec1["Rigid"] and t1["category"] == "minus" and t1["_perception"]["identity"] == "read"
    assert t1["_perception"]["glyph_read"]["carried"] is True
    t0 = rec1["Rigid"]["number"][0]
    assert t0["category_idx"] == 9 and t0["_perception"]["digit"] == 9 and t0["_perception"]["digit_source"] == "glyph"
    assert {e["label"] for e in ev} == {"t0", "t1"} and all(e["why"] == "carried" for e in ev)
    assert scan["Rigid"]["division"][0]["category"] == "division"         # the input record is not modified


def test_an_agreeing_read_refreshes_and_a_contradiction_makes_it_unknown():
    _, reads, _ = persist_reads(_rec("minus", True), {})
    rec, reads, ev = persist_reads(_rec("minus", True), reads)
    assert ev == [] and rec["Rigid"]["minus"][0]["_perception"]["identity"] == "read"
    rec, reads, ev = persist_reads(_rec("plus", True), reads)
    t1 = rec["Rigid"]["plus"][0]["_perception"]
    assert t1["identity"] == "unknown" and t1["glyph_read"]["why"] == "disagree" and "t1" not in reads
    rec, reads, ev = persist_reads(_rec("division", False), reads)          # nothing left to carry after a conflict
    assert rec["Rigid"]["division"][0]["_perception"]["identity"] == "unknown" and ev == []


def test_records_without_classifier_reads_are_untouched():
    rec = {"Rigid": {"block": [{"category": "block", "category_idx": 3, "label": "b0",
                                "_perception": {"identity": "read", "confidence": 0.9}}]}}
    out, reads, ev = persist_reads(rec, {})
    assert out == rec and reads == {} and ev == []


def test_an_explicit_read_by_another_reader_agrees_or_contradicts():
    _, reads, _ = persist_reads(_rec("minus", True), {})
    agent = {"Rigid": {"plus": [{"category": "plus", "category_idx": 0, "label": "t1", "_perception": {"identity": "read"}}]}}
    out, reads2, ev = persist_reads(agent, reads)
    assert out["Rigid"]["plus"][0]["_perception"]["identity"] == "unknown" and ev[0]["why"] == "disagree"
    agent = {"Rigid": {"minus": [{"category": "minus", "category_idx": 0, "label": "t1", "_perception": {"identity": "read"}}]}}
    out, reads3, ev = persist_reads(agent, reads)
    assert out["Rigid"]["minus"][0]["_perception"]["identity"] == "read" and ev == [] and "t1" in reads3


def test_free_labels_are_not_identities_and_never_persist():
    def free(idx, digit):
        return {"Rigid": {"number": [{"category": "number", "category_idx": idx, "label": "num_6",
                                      "_perception": {"digit": digit, "digit_conf": 0.5, "digit_source": "glyph",
                                                      "glyph_read": {"read": True, "name": str(digit)}}}]}}
    _, reads, ev = persist_reads(free(5, 5), {})
    assert reads == {} and ev == []
    out, reads, ev = persist_reads(free(7, 7), reads)                    # the scan put num_6 on another tile
    assert out["Rigid"]["number"][0]["_perception"]["digit"] == 7 and ev == []
