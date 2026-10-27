"""dojo.perception.embed_id helpers and the belt read's embedding pick (no torch, no catalogue: stubs)."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import belt
from dojo.perception.embed_id import composite, fuse, key_scores, masked, square_box


def test_square_box_is_square_padded_and_has_a_minimum_side():
    x0, y0, x1, y1 = square_box((100, 50, 40, 20), pad=0.25)
    assert x1 - x0 == pytest.approx(y1 - y0) == pytest.approx(60.0)
    assert (x0 + x1) / 2 == pytest.approx(120.0) and (y0 + y1) / 2 == pytest.approx(60.0)
    x0, y0, x1, y1 = square_box((10, 10, 4, 6), pad=0.25, min_side=32)
    assert x1 - x0 == pytest.approx(32.0)


def test_masked_and_composite_keep_only_the_object_pixels():
    rgb = np.full((4, 4, 3), 200, np.uint8)
    a = np.zeros((4, 4), np.uint8)
    a[1:3, 1:3] = 1
    m = masked(rgb, a, fill=10)
    assert (m[1:3, 1:3] == 200).all() and m[0, 0, 0] == 10
    bg = np.full((4, 4, 3), 7, np.uint8)
    c = composite(rgb, a, bg)
    assert (c[1:3, 1:3] == 200).all() and c[0, 0, 0] == 7


def test_key_scores_take_each_keys_best_view():
    sim = np.array([0.1, 0.9, 0.3, 0.5, 0.4])
    ks = key_scores(sim, ["a", "a", "b", "b", "b"], ["a", "b", "c"])
    assert ks == {"a": 0.9, "b": 0.5}
    assert key_scores(sim, ["a", "a", "b", "b", "b"], ["b"], agg="top3")["b"] == pytest.approx(0.4)


def test_fuse_counts_an_unfitted_key_at_loss_one():
    sims = {"a": 0.70, "b": 0.60, "c": 0.80}
    r = fuse(sims, {"a": 0.2, "b": 0.1}, 0.5)
    assert [k for k, _ in r] == ["a", "b", "c"]      # 0.70 - 0.10, 0.60 - 0.05, 0.80 - 0.50
    assert fuse(sims, {"a": 0.2, "b": 0.1}, 0.0)[0][0] == "c"


class _StubIdentity:
    def __init__(self, sims):
        self.sims = sims
        self.calls = 0

    def scores(self, img, cam, mask, bbox, centroid, keys, bg=None):
        self.calls += 1
        return {k: v for k, v in self.sims.items() if k in keys}


def _blob(area):
    return SimpleNamespace(area=area, mask=np.ones((2, 2), np.uint8), bbox=(0, 0, 2, 2), centroid=np.array([1.0, 1.0]))


def test_embed_pick_prefers_the_embedding_and_fits_a_key_we_did_not_propose(monkeypatch):
    import dojo.perception.embed_id as E

    stub = _StubIdentity({"Rigid/camera/00002": 0.80, "Rigid/phone/00000": 0.55, "Rigid/garage/00016": 0.50})
    monkeypatch.setattr(E, "identity", lambda *a, **k: stub)
    fitted = []

    def fake_fit(key, gi, bi, b, views, cfg, groups):
        fitted.append(key)
        return SimpleNamespace(key=key, loss=0.4, blob=bi)

    monkeypatch.setattr(belt, "_fit_key", fake_fit)
    cfg = SimpleNamespace(embed_model="dinov2_s", color_M=None, embed_lit=False, embed_crop="composite",
                          embed_weight=0.25, embed_min_px=40)
    groups = [{"candidates": ["Rigid/camera/00002", "Rigid/garage/00016", "Rigid/phone/00000"]}]
    views = {"cam_head": SimpleNamespace(img=None, cam=None)}
    ours = {0: [SimpleNamespace(key="Rigid/phone/00000", loss=0.3, blob=0)],
            1: [SimpleNamespace(key="Rigid/phone/00000", loss=0.3, blob=1)]}
    best, picks = belt._embed_pick([_blob(500), _blob(20)], ours, [0], groups, views, {}, cfg)
    # blob 0: camera 0.80 - 0.25 x 1.0 = 0.55 vs phone 0.55 - 0.25 x 0.3 = 0.475: the camera, fitted on the blob
    assert picks[0]["key"] == "Rigid/camera/00002" and picks[0]["ours"] == "Rigid/phone/00000"
    assert fitted == ["Rigid/camera/00002"] and [h.key for h in best[0]] == ["Rigid/phone/00000", "Rigid/camera/00002"]
    assert 1 not in picks and len(best[1]) == 1 and stub.calls == 1     # below embed_min_px: ours stays, no crop scored
    assert len(ours[0]) == 1                                             # the caller's lists are not mutated
    # a strong enough fit of ours wins under a larger weight
    cfg.embed_weight = 1.0
    _, picks = belt._embed_pick([_blob(500)], {0: ours[0]}, [0], groups, views, {}, cfg)
    assert picks[0]["key"] == "Rigid/phone/00000"


def test_estimator_config_keeps_the_embedding_off_by_default():
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig

    assert EstimatorConfig().embed_model is None


def test_merge_fragments_joins_close_blobs_given_the_same_asset(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    import dojo.perception.inverse_layout as IL

    sm = SimpleNamespace(height=0.1, asset=SimpleNamespace(extents=np.array([0.13, 0.13, 0.24])))
    monkeypatch.setattr(IL, "shape_models", lambda key: (sm,))
    monkeypatch.setattr(belt, "_fit_key", lambda key, gi, bi, b, views, cfg, groups: SimpleNamespace(key=key, loss=0.3, blob=bi, area=b.area))

    class Cam:  # 1 px = 1 mm on the plane
        def ray_plane(self, uv, z):
            return np.c_[uv / 1000.0, np.full(len(uv), z)]

    def blob(x0, x1):
        m = np.zeros((10, 400), np.uint8)
        m[2:8, x0:x1] = 1
        ys, xs = np.nonzero(m)
        return IL.Blob("cam_head", m, int(m.sum()), np.array([xs.mean(), ys.mean()]), (x0, 2, x1 - x0, 6))

    blobs_in = [blob(0, 20), blob(60, 80), blob(300, 320)]          # centres 10, 70 and 310 mm along the belt
    picks = {i: {"key": "Rigid/garage/00016"} for i in range(3)}
    best = {i: [SimpleNamespace(key="Rigid/phone/00000", loss=0.5, blob=i)] for i in range(3)}
    groups = [{"candidates": ["Rigid/garage/00016", "Rigid/phone/00000"]}]
    out_b, out_best, out_picks = belt._merge_fragments(blobs_in, best, picks, [0], groups,
                                                       {"cam_head": SimpleNamespace(cam=Cam())}, SimpleNamespace())
    assert sorted(out_picks) == [0, 2] and out_picks[0]["merged"] == 2       # 60 mm apart <= 0.5 x 240 mm: one object
    assert out_b[0].area == blobs_in[0].area + blobs_in[1].area and out_best[0][0].key == "Rigid/garage/00016"
    assert 1 not in out_best and "merged" not in out_picks[2] and blobs_in[0].area == 120   # inputs untouched


def test_service_cfg_takes_estimator_overrides_from_the_environment(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.runtime.perception_service import _Perception

    monkeypatch.delenv("DOJO_COLOR_CALIB", raising=False)
    monkeypatch.delenv("DOJO_PERCEPTION_CFG", raising=False)
    assert _Perception().cfg(0).embed_model is None
    monkeypatch.setenv("DOJO_PERCEPTION_CFG", '{"embed_model": "dinov2_s", "embed_merge": true}')
    c = _Perception().cfg(0)
    assert c.embed_model == "dinov2_s" and c.embed_merge is True
