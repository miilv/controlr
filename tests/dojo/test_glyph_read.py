"""dojo.perception.glyph_read: the tile classifier's read rule, the per-view crop / read over views with a fake client,
the server wire format with a fake model, and the estimator flag (default off; a read sets the asset and "read",
anything else leaves the tile unread). No torch, no GPU, no weights."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import glyph_read as gr


def test_decide_reads_only_a_confident_glyph():
    n = 4  # 4 glyphs + back (index 4)
    assert gr.decide(np.array([0.9, 0.05, 0.03, 0.01, 0.01]), n, 0.5) == (0, pytest.approx(0.85), True, "read")
    assert gr.decide(np.array([0.5, 0.4, 0.05, 0.03, 0.02]), n, 0.5)[2:] == (False, "low_margin")
    assert gr.decide(np.array([0.05, 0.02, 0.02, 0.01, 0.9]), n, 0.5)[2:] == (False, "back")


class _Sm:
    def __init__(self, box):
        self.box = box

    def sil(self, cam, T):
        m = np.zeros((48, 64), np.uint8)
        x0, y0, x1, y1 = self.box[cam]
        m[y0:y1, x0:x1] = 1
        return m


class _Client:
    def __init__(self, probs_by_shape):
        self.calls = 0
        self.probs = probs_by_shape

    def classify(self, img, boxes):
        self.calls += 1
        return {"probs": [self.probs[int(img[0, 0, 0])]], "n_glyph": 4, "margins": {"head": 0.5, "wrist": 0.3}}


def _views():
    def mk(tag):
        return SimpleNamespace(cam=tag, img=np.full((48, 64, 3), tag, np.uint8), known=np.ones((48, 64), np.uint8))

    return {"cam_head": mk(0), "cam_left_wrist": mk(1), "cam_head@1": mk(2)}


def test_read_takes_the_most_confident_view_among_candidates():
    sm = _Sm({0: (10, 10, 20, 20), 1: (5, 5, 30, 30), 2: (0, 0, 10, 10)})
    probs = {0: [0.55, 0.40, 0.03, 0.01, 0.01],      # head: below the head margin (0.5)
             1: [0.02, 0.85, 0.08, 0.03, 0.02],      # wrist: glyph 1 above the wrist margin (0.3)
             2: [1.0, 0.0, 0.0, 0.0, 0.0]}           # an extra frame view (name with "@"): ignored
    c = _Client(probs)
    r = gr.read(_views(), sm, np.eye(4), ["Rigid/mahjong/00000", "Rigid/mahjong/00001"], client_=c)
    assert r["read"] and r["key"] == "Rigid/mahjong/00001" and r["view"] == "cam_left_wrist" and c.calls == 2
    assert r["views"]["cam_head"]["why"] == "low_margin"
    r2 = gr.read(_views(), sm, np.eye(4), ["Rigid/mahjong/00000"], client_=c)      # glyph 1 is not a candidate
    assert not r2["read"] and r2["why"] in ("not a candidate", "low_margin")
    tiny = _Sm({0: (10, 10, 12, 12), 1: (5, 5, 7, 7), 2: (0, 0, 1, 1)})              # < MIN_PX everywhere
    assert gr.read(_views(), tiny, np.eye(4), ["Rigid/mahjong/00001"], client_=c) == {"read": False, "why": "not visible"}


def test_crop_square_pads_past_the_border():
    pytest.importorskip("cv2")
    img = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    assert gr.crop_square(img, (0, 0, 10, 10), size=32).shape == (32, 32, 3)
    assert gr.crop_square(img, (60, 40, 63, 47), size=16).shape == (16, 16, 3)


class _FakeModel:
    def handle(self, req, img):
        return {"probs": [[0.1, 0.8, 0.05, 0.03, 0.02] for _ in req["boxes"]], "n_glyph": 4, "margins": {"head": 0.5}}

    def info(self):
        return {"fake": True}


def test_client_server_roundtrip(tmp_path, monkeypatch):
    from dojo.perception import sam_masks

    sock = str(tmp_path / "glyph.sock")
    ready = threading.Event()
    threading.Thread(target=sam_masks.serve, args=(sock, ["glyph"], 0.01), kwargs={"M": _FakeModel(), "ready": ready},
                     daemon=True).start()
    assert ready.wait(5)
    monkeypatch.setenv(gr.ENV_ADDR, sock)
    gr._CLIENT.clear()
    c = gr.client()
    r = c.classify(np.zeros((8, 8, 3), np.uint8), [(1, 1, 4, 4), (2, 2, 6, 6)])
    assert len(r["probs"]) == 2 and r["n_glyph"] == 4
    gr._CLIENT.clear()
    monkeypatch.delenv(gr.ENV_ADDR)
    assert gr.client() is None


def test_estimator_flag_default_off_and_read_rule(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    assert il.EstimatorConfig().glyph_read == "off"
    g = {"candidates": ["Rigid/mahjong/00000", "Rigid/mahjong/00001"]}
    h = SimpleNamespace(per_view={"identity": "read"}, key="Rigid/mahjong/00000", tag="up", sm=lambda: None, T=lambda: None)
    assert il._glyph_read_slot(h, g, {}, None, {}, il.EstimatorConfig()) is h and h.per_view == {"identity": "read"}
    cfg = il.EstimatorConfig(glyph_read="kong_cls")
    monkeypatch.setattr(gr, "read", lambda views, sm, T, cands: {"read": False, "why": "back"})
    h = SimpleNamespace(per_view={"identity": "read"}, key="Rigid/mahjong/00000", tag="up", sm=lambda: None, T=lambda: None)
    il._glyph_read_slot(h, g, {}, None, {}, cfg)
    assert "identity" not in h.per_view and h.per_view["glyph_read"]["why"] == "back"     # no read: unread
    monkeypatch.setattr(gr, "read", lambda views, sm, T, cands: {"read": True, "key": "Rigid/mahjong/00001"})
    monkeypatch.setattr(il, "_slot_shape", lambda key, placement, tag: SimpleNamespace(tag="up"))
    h = SimpleNamespace(per_view={}, key="Rigid/mahjong/00000", tag="up", sm=lambda: None, T=lambda: None)
    il._glyph_read_slot(h, g, {}, None, {}, cfg)
    assert h.key == "Rigid/mahjong/00001" and h.per_view["identity"] == "read"
    other = {"candidates": ["Rigid/number_card/00001", "Rigid/number_card/00002"]}
    h = SimpleNamespace(per_view={"identity": "read"}, key="Rigid/number_card/00001", tag="up", sm=lambda: None, T=lambda: None)
    assert il._glyph_read_slot(h, other, {}, None, {}, cfg).per_view == {"identity": "read"}   # only mahjong tiles


def test_classify_mask_reads_names_and_margins():
    class C:
        def classify(self, img, boxes):
            return {"probs": [[0.02, 0.9, 0.03, 0.03, 0.02]], "n_glyph": 4, "margins": {"head": 0.5, "held_wrist": 0.95},
                    "names": ["0", "6", "9", "plus"]}

    m = np.zeros((48, 64), np.uint8)
    m[10:20, 10:20] = 1
    img = np.zeros((48, 64, 3), np.uint8)
    r = gr.classify_mask(img, m, "cam_head", client_=C())
    assert r["read"] and r["name"] == "6"
    r = gr.classify_mask(img, m, "cam_left_wrist", held=True, client_=C())     # the stricter held margin
    assert not r["read"] and r["why"] == "low_margin"
    assert gr.classify_mask(img, np.zeros_like(m), client_=C())["why"] == "not visible"
    cut = np.zeros_like(m)
    cut[0:12, 30:40] = 1                                                         # cut by the frame: never read
    assert gr.classify_mask(img, cut, "cam_left_wrist", client_=C())["why"] == "edge"
    assert gr.class_name({}, 3) == "Rigid/mahjong/00003"
