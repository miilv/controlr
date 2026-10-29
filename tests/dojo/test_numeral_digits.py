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
        return lambda img, mask, view="cam_head", held=False, client_=None, category=None: next(it)

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
        "views": {"cam_head": "6", "cam_left_wrist": "9"}, "unread": {}, "read": False, "why": "disagree"}
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
    o = SimpleNamespace(digit=7, digit_conf=0.5, digit_source="glyph", category="number", identity_known=True)
    prov = SimpleNamespace(belief=SimpleNamespace(objects={"digit_0": o}))
    pol._merge_glyph(prov, "digit_0", {"read": True, "name": "7", "margin": 0.6})
    assert (o.digit, o.digit_conf, o.digit_source) == (7, 1.0, "glyph")
    pol._merge_glyph(prov, "digit_0", {"read": True, "name": "1", "margin": 0.6})
    assert (o.digit, o.digit_source) == (None, "unread") and pol.report["events"][-1]["why"] == "disagree"


def test_the_arm_mask_goes_to_a_reader_that_takes_it(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import glyph_read
    from dojo.perception import inverse_layout as il
    seen = []

    def v2(img, mask, view="cam_head", held=False, client_=None, robot=None, category=None):
        seen.append(robot)
        return {"read": False, "why": "robot"}

    monkeypatch.setattr(glyph_read, "classify_mask", v2, raising=False)
    h = _h()
    views = {"cam_head": SimpleNamespace(cam=None, img=None, known=None, robot=np.ones((4, 4), np.uint8))}
    il._glyph_read_numeral(h, views)
    assert seen and seen[0].dtype == bool and h.per_view["glyph_read"]["why"] == "robot"


def test_the_report_keeps_the_digit_fields():
    from dojo.runtime.policy import _slim
    rec = {"Rigid": {"number": [{"label": "num_3", "_perception": {
        "confidence": 0.8, "digit": 3, "digit_conf": 1.0, "digit_source": "glyph", "glyph_read": {"read": True},
        "texture": {"big": "diagnostic"}}}]}}
    p = _slim(rec)["Rigid"]["number"][0]["_perception"]
    assert p == {"confidence": 0.8, "digit": 3, "digit_conf": 1.0, "digit_source": "glyph", "glyph_read": {"read": True}}


def test_an_operator_wrist_read_sets_a_placeholder_identity_and_never_trusts_the_placeholder():
    """solve_equation's t1 (an operator pinned in a slot): its record asset is the label's first candidate, a
    placeholder ("unknown"). A wrist read names the operator: identity known; a read against a known identity: unknown."""
    from types import SimpleNamespace

    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    pol = RuntimePolicy(RuntimeConfig())
    pol.report = {"events": []}
    pol.bridge = SimpleNamespace(steps_used=20)
    o = SimpleNamespace(category="division", idx=0, obj_type="Rigid", asset=None, identity_known=False,
                        digit=None, digit_conf=None, digit_source=None)
    prov = SimpleNamespace(belief=SimpleNamespace(objects={"t1": o}), assets=None)
    pol._merge_glyph(prov, "t1", {"read": True, "name": "plus", "margin": 0.9})
    assert (o.category, o.identity_known) == ("plus", True) and o.asset.key == "Rigid/plus/00000"
    pol._merge_glyph(prov, "t1", {"read": True, "name": "plus", "margin": 0.9})
    assert o.identity_known is True and pol.report["events"][-1]["why"] == "confirmed"
    pol._merge_glyph(prov, "t1", {"read": True, "name": "minus", "margin": 0.9})
    assert o.identity_known is False and pol.report["events"][-1]["why"] == "disagree"


def _slot_world(monkeypatch, cands, reads):
    """A support slot on a mat: the label's candidates ``cands``, the classifier answering ``reads`` per view."""
    from types import SimpleNamespace

    from dojo.perception import glyph_read
    from dojo.perception import inverse_layout as il

    box = SimpleNamespace(Vs=np.zeros((3, 3)), Fs=np.zeros((1, 3), int), extents=np.array([0.06, 0.06, 0.008]))
    monkeypatch.setattr(il, "task_map", lambda: {"t": {"labels": {"t1": {"placement": {"relative_plane": "mat1"}},
                                                                  "mat1": {"placement": {}}}}})
    monkeypatch.setattr(il, "support_slot", lambda pl: ("mat1", "slot"))
    monkeypatch.setattr(il, "support_slot_pose", lambda *a, **k: ([0.0, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(il, "_hyp_pose", lambda h: ([0.0, 0.0, 0.79], [1.0, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(il, "load_asset", lambda key: box)
    full = np.ones((8, 8), np.uint8)
    monkeypatch.setattr(il, "silhouette", lambda cam, V, F, out=None: full.copy())
    it = iter(reads)
    monkeypatch.setattr(glyph_read, "classify_mask",
                        lambda img, mask, view="cam_head", held=False, client_=None, robot=None, category=None: next(it),
                        raising=False)
    mat = SimpleNamespace(label_group=0, key="Geometry/cube_cushion/00001", loss=0.1, per_view={},
                          sm=lambda: SimpleNamespace(sil=lambda cam, T, out=None: out), T=lambda: np.eye(4))
    cam = SimpleNamespace(h=8, w=8)
    views = {n: SimpleNamespace(cam=cam, img=None, fg=np.ones((8, 8), np.uint8), known=np.ones((8, 8), np.uint8),
                                robot=np.zeros((8, 8), np.uint8)) for n in ("cam_head", "cam_left_wrist")}
    groups = [{"labels": ["mat1"], "candidates": ["Geometry/cube_cushion/00001"]}, {"labels": ["t1"], "candidates": cands}]
    return il, groups, [mat], views


def test_a_slot_operator_is_read_and_a_placeholder_never_counts(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception.identity_marks import identity_known
    ops = ["Rigid/division/00000", "Rigid/plus/00000", "Rigid/minus/00000", "Rigid/multiplication/00000"]
    il, groups, chosen, views = _slot_world(monkeypatch, ops, [{"read": True, "name": "plus", "margin": 0.9}] * 2)
    rec = il._place_support_slots([groups[1]], chosen, groups, views, "t", {}, glyph=True)[0][2]
    assert rec["category"] == "plus" and rec["_perception"]["identity"] == "read"
    assert rec["_perception"]["glyph_read"]["conf"] == 1.0
    il, groups, chosen, views = _slot_world(monkeypatch, ops, [{"read": False, "why": "robot"}] * 2)
    rec = il._place_support_slots([groups[1]], chosen, groups, views, "t", {}, glyph=True)[0][2]
    assert rec["category"] == "division" and "identity" not in rec["_perception"]          # the placeholder ...
    from dojo.perception.identity_marks import identity_marks
    record = {"Rigid": {"division": [rec]}}
    identity_marks(record, "t", {"t1": {"candidates": ops}})
    assert not identity_known(rec["_perception"])                                            # ... stays unknown


def test_a_slot_numeral_gets_its_digit_fields(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    nums = [f"Rigid/number/{i:05d}" for i in range(30)]
    il, groups, chosen, views = _slot_world(monkeypatch, nums, [{"read": True, "name": "7", "margin": 0.8},
                                                                {"read": False, "why": "edge"}])
    p = il._place_support_slots([groups[1]], chosen, groups, views, "t", {}, glyph=True)[0][2]
    assert p["category_idx"] == 7 and p["_perception"]["digit"] == 7 and p["_perception"]["digit_conf"] == 0.5
    assert p["_perception"]["digit_source"] == "glyph" and p["_perception"]["identity"] == "read"
    il, groups, chosen, views = _slot_world(monkeypatch, nums, [])
    p = il._place_support_slots([groups[1]], chosen, groups, views, "t", {}, glyph=False)[0][2]
    assert "digit_source" not in p["_perception"] and "glyph_read" not in p["_perception"]   # off: as before


def test_a_slot_mahjong_read_reaches_the_record_and_the_report(monkeypatch):
    """make_kong's slot tiles: _glyph_read_slot's read is kept (slimmed) in the record, and the report keeps
    identity / slot / glyph_read."""
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from types import SimpleNamespace

    from dojo.perception import glyph_read
    from dojo.perception import inverse_layout as il
    from dojo.runtime.policy import _slim
    monkeypatch.setattr(glyph_read, "read", lambda views, sm, T, cands, **k: {
        "read": True, "key": "Rigid/mahjong/00005", "class": 5, "margin": 0.7, "view": "cam_head", "why": "read",
        "views": {"cam_head": {"class": 5, "margin": 0.7, "why": "read"}}})
    monkeypatch.setattr(il, "_slot_shape", lambda key, placement, tag: SimpleNamespace(tag="up"))
    h = SimpleNamespace(key="Rigid/mahjong/00000", tag="up", per_view={}, sm=lambda: None, T=lambda: np.eye(4))
    g = {"candidates": [f"Rigid/mahjong/{i:05d}" for i in range(42)]}
    il._glyph_read_slot(h, g, {}, None, {}, SimpleNamespace(glyph_read="on"))
    assert h.key == "Rigid/mahjong/00005" and h.per_view["identity"] == "read"
    rec = {"Rigid": {"mahjong": [{"label": "mahjong0_0", "_perception": {
        "identity": "read", "slot": True, "glyph_read": il._slim_glyph(h.per_view["glyph_read"]), "texture": {"x": 1}}}]}}
    p = _slim(rec)["Rigid"]["mahjong"][0]["_perception"]
    assert p["identity"] == "read" and p["slot"] is True and p["glyph_read"]["key"] == "Rigid/mahjong/00005"
    assert "texture" not in p


def test_config_slot_numerals_and_operators_are_read_and_placeholders_stay_unknown(monkeypatch):
    """solve_equation's t0 / t2 / t4 (numerals) and t1 (an operator) pinned on the mats go through the config-slot
    path: the classifier reads them there; a texture-based identity never stands without a classifier read."""
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from types import SimpleNamespace

    from dojo.perception import glyph_read
    from dojo.perception import inverse_layout as il

    def world(reads):
        it = iter(reads)
        monkeypatch.setattr(glyph_read, "classify_mask",
                            lambda img, mask, view="cam_head", held=False, client_=None, robot=None, category=None: next(it),
                            raising=False)
        cam = SimpleNamespace(h=8, w=8)
        return {n: SimpleNamespace(cam=cam, img=None, known=None, robot=np.zeros((8, 8), np.uint8))
                for n in ("cam_head", "cam_left_wrist")}

    def hyp(key):
        sm = SimpleNamespace(sil=lambda cam, T: np.ones((8, 8), np.uint8))
        return SimpleNamespace(key=key, per_view={"identity": "read"}, sm=lambda: sm, T=lambda: np.eye(4))

    ops = {"candidates": ["Rigid/division/00000", "Rigid/plus/00000", "Rigid/minus/00000", "Rigid/multiplication/00000"]}
    h = hyp("Rigid/division/00000")
    il._glyph_read_config_slot(h, ops, world([{"read": True, "name": "minus", "margin": 0.9}] * 2))
    assert h.key == "Rigid/minus/00000" and h.per_view["identity"] == "read"
    h = hyp("Rigid/division/00000")                       # a texture "read" never stands without the classifier
    il._glyph_read_config_slot(h, ops, world([{"read": False, "why": "low_margin"}] * 2))
    assert h.key == "Rigid/division/00000" and "identity" not in h.per_view
    nums = {"candidates": [f"Rigid/number/{i:05d}" for i in range(50)]}
    h = hyp("Rigid/number/00000")
    il._glyph_read_config_slot(h, nums, world([{"read": True, "name": "4", "margin": 0.8}] * 2))
    assert h.key == "Rigid/number/00004" and h.per_view["glyph_read"]["conf"] == 1.0
    f = digits.record_fields("number", 4, None, h.key, 50, classifier=h.per_view["glyph_read"])
    assert f == {"digit": 4, "digit_conf": 1.0, "digit_source": "glyph"}
    assert il._glyph_slot_group(nums["candidates"]) and il._glyph_slot_group(ops["candidates"])
    assert not il._glyph_slot_group(["Rigid/mahjong/00001", "Rigid/mahjong/00002"])     # kong: its own path
