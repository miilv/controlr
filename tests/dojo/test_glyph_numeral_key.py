"""glyph_numeral_key: a numeral's mesh follows the glyph read (D-C17) instead of the fitted geometry's digit.
Found on dev 100000 (D-C19 rehearsal): 9 of 20 glyph-read numerals kept another digit's asset (arrange_largest_number
2/4, its _random 3/4, solve_equation 4/12) because the read digit was looked up only inside the fitted geometry's
group. Every digit has its own mesh; the texture groups share it. Synthetic: no assets, no frames."""
from types import SimpleNamespace

import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
from dojo.perception import inverse_layout as IL  # noqa: E402

CANDS = [f"Rigid/number/{i:05d}" for i in range(50)]


def _h(key="Rigid/number/00038", tag="up"):
    geom = [k for k in CANDS if int(k[-5:]) % 10 == int(key[-1])]
    return IL.Hypothesis(0, key, tag, 0.1, 0.2, 0.0, 0.1, {"geom_group": geom})


@pytest.fixture
def read3(monkeypatch):
    monkeypatch.setattr(IL, "_glyph_views", lambda sil_of, views, cat: {"read": True, "name": "3", "conf": 1.0})
    monkeypatch.setattr(IL, "shape_models", lambda k: (SimpleNamespace(tag="up"),))


def test_default_keeps_the_fitted_mesh(read3):
    h = _h()
    IL._glyph_read_numeral(h, {})
    assert h.key == "Rigid/number/00038" and h.per_view["glyph_read"]["name"] == "3" and "glyph_mesh" not in h.per_view


def test_label_mode_takes_the_read_digits_asset_in_the_same_texture_group(read3):
    h = _h()
    IL._glyph_read_numeral(h, {}, CANDS)
    assert h.key == "Rigid/number/00033"                       # group 3 kept, digit 3's mesh
    assert h.per_view["glyph_mesh"] == {"from": "Rigid/number/00038", "to": "Rigid/number/00033", "switched": True}
    assert h.per_view["geom_group"] == [f"Rigid/number/{10 * g + 3:05d}" for g in range(5)]
    h = _h("Rigid/number/00048")                               # no group-4 three among the candidates: another group's
    IL._glyph_read_numeral(h, {}, [k for k in CANDS if k != "Rigid/number/00043"])
    assert h.key == "Rigid/number/00003"


def test_a_read_the_geometry_already_shows_or_a_missing_tag_changes_nothing(read3, monkeypatch):
    h = _h("Rigid/number/00023")
    IL._glyph_read_numeral(h, {}, CANDS)
    assert h.key == "Rigid/number/00023" and "glyph_mesh" not in h.per_view
    monkeypatch.setattr(IL, "shape_models", lambda k: (SimpleNamespace(tag="side"),))
    h = _h()
    IL._glyph_read_numeral(h, {}, CANDS)
    assert h.key == "Rigid/number/00038" and h.per_view["glyph_mesh"]["switched"] is False


def test_config_default_and_values():
    assert IL.EstimatorConfig().glyph_numeral_key == "geom"
