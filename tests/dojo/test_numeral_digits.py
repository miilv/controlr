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


def test_glyph_read_numeral_sets_the_digit_within_the_texture_group(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import glyph_read
    from dojo.perception import inverse_layout as il

    cands = [f"Rigid/number/{i:05d}" for i in range(30)]
    h = SimpleNamespace(key="Rigid/number/00013", per_view={"geom_group": cands}, sm=lambda: None, T=lambda: np.eye(4))
    monkeypatch.setattr(glyph_read, "read", lambda views, sm, T, c, **k: {"read": True, "name": "6", "margin": 0.4})
    il._glyph_read_numeral(h, {})
    assert h.key == "Rigid/number/00016" and h.per_view["glyph_read"]["conf"] == 1.0   # group 1 kept, digit 6
    h2 = SimpleNamespace(key="Rigid/number/00013", per_view={"geom_group": cands}, sm=lambda: None, T=lambda: np.eye(4))
    monkeypatch.setattr(glyph_read, "read", lambda views, sm, T, c, **k: {"read": True, "name": "plus", "margin": 0.9})
    il._glyph_read_numeral(h2, {})
    assert h2.key == "Rigid/number/00013" and h2.per_view["glyph_read"] == {"read": False, "name": "plus", "margin": 0.9,
                                                                          "why": "not a digit"}
    h3 = SimpleNamespace(key="Rigid/block/00001", per_view={}, sm=lambda: None, T=lambda: np.eye(4))
    il._glyph_read_numeral(h3, {})
    assert "glyph_read" not in h3.per_view                                             # other categories untouched
