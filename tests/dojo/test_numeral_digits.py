"""RoboDojo's 3-D numerals (Rigid/number): the digit comes only from the crop classifier (``glyph_read``), never from
B1's texture compare; without a classifier run the record carries no digit fields (the default is unchanged)."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import digits


def test_record_fields_for_numerals_only_from_the_classifier():
    assert digits.record_fields("number", 17, {"key": "Rigid/number/00017", "read": True}, "Rigid/number/00017", 30) == {}
    r = digits.record_fields("number", 17, None, "Rigid/number/00017", 30, classifier={"read": True, "name": "7", "conf": 1.0})
    assert r == {"digit": 7, "digit_conf": 1.0, "digit_source": "glyph"}
    r = digits.record_fields("number", 17, None, "Rigid/number/00017", 30, classifier={"read": False, "why": "back"})
    assert r == {"digit": None, "digit_source": "unread", "digit_unread": "back"}
    assert digits.from_record("number", {}) == (None, None, None)                      # no classifier run: as before
    assert digits.from_record("number", {"digit": 7, "digit_conf": 1.0, "digit_source": "glyph"}) == (7, 1.0, "glyph")
    assert digits.from_record("number", {"digit": None, "digit_source": "unread"}) == (None, None, "unread")
    assert "number" not in digits.DIGIT_OF                                             # agent / bridge paths unchanged


def _h(key="Rigid/number/00013"):
    cands = [f"Rigid/number/{i:05d}" for i in range(30)]
    sm = SimpleNamespace(sil=lambda cam, T: np.ones((4, 4), np.uint8))
    return SimpleNamespace(key=key, per_view={"geom_group": cands}, sm=lambda: sm, T=lambda: np.eye(4))


def _views(*names):
    return {n: SimpleNamespace(cam=None, img=None, known=None) for n in names}


def test_glyph_read_numeral_needs_two_agreeing_views_for_full_confidence(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import glyph_read
    from dojo.perception import inverse_layout as il

    def fake(reads):
        it = iter(reads)
        return lambda img, mask, view="cam_head", held=False, client_=None: next(it)

    h = _h()
    monkeypatch.setattr(glyph_read, "classify_mask", fake([{"read": True, "name": "6", "margin": 0.6}] * 2), raising=False)
    il._glyph_read_numeral(h, _views("cam_head", "cam_left_wrist"))
    assert h.key == "Rigid/number/00016" and h.per_view["glyph_read"]["conf"] == 1.0   # group 1 kept, digit 6
    h = _h()
    monkeypatch.setattr(glyph_read, "classify_mask", fake([{"read": True, "name": "6", "margin": 0.6},
                                                           {"read": False, "why": "edge"}]), raising=False)
    il._glyph_read_numeral(h, _views("cam_head", "cam_left_wrist"))
    assert h.per_view["glyph_read"]["conf"] == 0.5 and h.per_view["glyph_read"]["n_views"] == 1   # one view: confirm
    h = _h()
    monkeypatch.setattr(glyph_read, "classify_mask", fake([{"read": True, "name": "6", "margin": 0.6},
                                                           {"read": True, "name": "9", "margin": 0.7}]), raising=False)
    il._glyph_read_numeral(h, _views("cam_head", "cam_left_wrist"))
    assert h.key == "Rigid/number/00013" and h.per_view["glyph_read"] == {
        "views": {"cam_head": 6, "cam_left_wrist": 9}, "unread": {}, "read": False, "why": "disagree"}
    h = _h()
    monkeypatch.setattr(glyph_read, "classify_mask", fake([{"read": True, "name": "plus", "margin": 0.9}]), raising=False)
    il._glyph_read_numeral(h, _views("cam_head"))
    assert h.per_view["glyph_read"]["read"] is False and h.key == "Rigid/number/00013"
    h3 = SimpleNamespace(key="Rigid/block/00001", per_view={}, sm=lambda: None, T=lambda: np.eye(4))
    il._glyph_read_numeral(h3, {})
    assert "glyph_read" not in h3.per_view                                             # other categories untouched


def test_merge_view_read_confirms_first_reads_and_drops_disagreement():
    m = digits.merge_view_read
    assert m(None, None, None, 7) == (7, 0.5, "glyph", "first")                       # a first read: one view
    assert m(None, None, "unread", 7) == (7, 0.5, "glyph", "first")
    assert m(7, 0.5, "glyph", 7) == (7, 1.0, "glyph", "confirmed")                    # a second view agrees
    assert m(7, 0.5, "glyph", 1) == (None, None, "unread", "disagree")                # disagreement: unknown
    assert m(7, 1.0, "glyph", None) == (7, 1.0, "glyph", None)                         # no read: unchanged


def test_the_refine_hook_merges_a_wrist_digit_read_into_the_belief():
    from types import SimpleNamespace

    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    pol = RuntimePolicy(RuntimeConfig())
    pol.report = {"events": []}
    pol.bridge = SimpleNamespace(steps_used=12)
    o = SimpleNamespace(digit=7, digit_conf=0.5, digit_source="glyph")
    prov = SimpleNamespace(belief=SimpleNamespace(objects={"digit_0": o}))
    pol._merge_glyph(prov, "digit_0", {"read": True, "name": "7", "margin": 0.6})
    assert (o.digit, o.digit_conf, o.digit_source) == (7, 1.0, "glyph")
    pol._merge_glyph(prov, "digit_0", {"read": True, "name": "1", "margin": 0.6})
    assert (o.digit, o.digit_source) == (None, "unread") and pol.report["events"][-1]["why"] == "disagree"
