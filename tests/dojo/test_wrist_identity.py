"""inverse_layout wrist_identity "embed" (default off): a chosen table object that a wrist view sees large gets the learned
identity of that crop; a category that beats the fitted one by the margin is re-fitted at the head blob and replaces
it when the fused score is higher (imitate: our watch read as an action camera from the head's ~300 px)."""
from types import SimpleNamespace

import numpy as np
import pytest

AC, WATCH = "Rigid/action_camera/00002", "Rigid/watch/00000"


class _SM:
    def __init__(self, key, box):
        self.key, self.tag, self.box, self.yaw_period = key, "t0", box, 2 * np.pi

    def sil(self, cam, T, out=None):
        m = np.zeros((cam.h, cam.w), np.uint8) if out is None else out
        y0, y1, x0, x1 = self.box[cam.name]
        m[y0:y1, x0:x1] = 1
        return m

    def pose(self, x, y, yaw, z=0.0):
        return np.eye(4)


def _setup(monkeypatch, sims, wrist_box=(10, 60, 10, 70)):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import embed_id
    from dojo.perception import inverse_layout as IL
    box = {"cam_head": (2, 6, 2, 6), "cam_left_wrist": wrist_box}
    sms = {AC: _SM(AC, box), WATCH: _SM(WATCH, box)}
    monkeypatch.setattr(IL, "shape_by_tag", lambda key, tag: sms[key])
    monkeypatch.setattr(IL, "_geometry_groups", lambda g, prior: {("ac",): [sms[AC]], ("w",): [sms[WATCH]]})
    monkeypatch.setattr(IL, "_score_hyp", lambda h, *a, **k: h)
    monkeypatch.setattr(IL, "coarse_fit", lambda sm, b, views, step: [(0.1, -0.2, 0.0, 0.4)])
    monkeypatch.setattr(IL, "region_ok", lambda *a, **k: True)
    monkeypatch.setattr(IL, "_centre_xy", lambda h, g=None: (h.x, h.y))
    calls = []
    emb = SimpleNamespace(scores=lambda img, cam, mask, bbox, cen, keys, bg=None: calls.append((cam.name, int(mask.sum())))
                          or dict(sims))
    monkeypatch.setattr(embed_id, "identity", lambda *a, **k: emb)

    def vm(name, h, w):
        return SimpleNamespace(cam=SimpleNamespace(name=name, h=h, w=w), img=np.zeros((h, w, 3), np.uint8),
                               fg=np.ones((h, w), np.uint8), robot=np.zeros((h, w), np.uint8), bg=None)
    views = {"cam_head": vm("cam_head", 8, 8), "cam_left_wrist": vm("cam_left_wrist", 80, 100)}
    y0, y1, x0, x1 = wrist_box
    views["cam_left_wrist"].fg[:] = 0
    views["cam_left_wrist"].fg[y0:y1, x0:x1] = 1                  # the object's own pixels: the table is background
    h = IL.Hypothesis(0, AC, "t0", 0.1, -0.2, 0.0, 0.3)
    cfg = IL.EstimatorConfig(wrist_identity="embed")
    return IL, views, h, cfg, calls


def test_wrist_identity_replaces_a_category_the_wrist_crop_reads_clearly(monkeypatch):
    IL, views, h, cfg, calls = _setup(monkeypatch, {AC: 0.41, WATCH: 0.66})
    out, info = IL._wrist_identity([h], views, cfg, [], [(0, {}, {})])
    assert calls == [("cam_left_wrist", 50 * 60)]
    assert out[0].key == WATCH and info["checked"] == 1
    assert info["changed"] == [{"from": AC, "to": WATCH, "xy": [0.1, -0.2], "view": "cam_left_wrist", "px": 3000}]
    assert out[0].per_view["wrist_identity"]["from"] == AC


def test_wrist_identity_keeps_the_fit_under_the_margin_or_on_a_small_crop(monkeypatch):
    IL, views, h, cfg, calls = _setup(monkeypatch, {AC: 0.50, WATCH: 0.60})
    out, info = IL._wrist_identity([h], views, cfg, [], [(0, {}, {})])
    assert out[0] is h and info["changed"] == [] and h.per_view["wrist_identity"]["top"] == "watch"
    IL, views, h, cfg, calls = _setup(monkeypatch, {AC: 0.41, WATCH: 0.66}, wrist_box=(10, 30, 10, 40))   # 600 px < 800
    out, info = IL._wrist_identity([h], views, cfg, [], [(0, {}, {})])
    assert out[0] is h and calls == [] and info["small"][0]["px"] == 600


def test_wrist_identity_is_off_by_default():
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig
    assert EstimatorConfig().wrist_identity == "off"


def test_wrist_identity_rejects_a_refit_that_moves_onto_another_object(monkeypatch):
    IL, views, h, cfg, calls = _setup(monkeypatch, {AC: 0.41, WATCH: 0.66})
    monkeypatch.setattr(IL, "coarse_fit", lambda sm, b, views, step: [(0.15, -0.2, 0.0, 0.2)])   # 50 mm away
    out, info = IL._wrist_identity([h], views, cfg, [], [(0, {}, {})])
    assert out[0] is h and info["changed"] == [] and "beat" in h.per_view["wrist_identity"]["why"]
