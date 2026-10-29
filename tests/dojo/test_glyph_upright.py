"""glyph_upright (EstimatorConfig, default off): a runtime wrist crop of a flat glyph (the digit model's numerals /
operators) is turned so that world +y points up before the classifier reads it; the look-over rolls the wrist camera
180 deg (a 6 read as a 9). Also: a glyph server answering an error is an infra failure through both read paths (the
frame-0 views and the runtime wrist read), never a silent "no read" (arrange_random 100021 at 3ad78f28: every read
"error: RuntimeError"). No torch, no GPU, no URDF."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import glyph_read as gr
from dojo.perception.sam_masks import SideServerDown

K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])


class Camera:
    """cameras.Camera's pinhole (that module needs scipy): T = camera -> world, OpenCV axes."""

    def __init__(self, name, K, T):
        self.name, self.K, self.T_wc = name, K, T

    def project(self, X):
        Xc = (np.asarray(X, float).reshape(-1, 3) - self.T_wc[:3, 3]) @ self.T_wc[:3, :3]
        z = Xc[:, 2]
        return Xc[:, :2] / z[:, None] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]], z


def head_camera():
    pytest.importorskip("scipy")
    from dojo.perception.cameras import head_camera as hc
    return hc()


def _down_cam(x_img, y_img, at=(0.0, -0.15, 0.95)):
    """A camera looking straight down (optical axis world -z) whose image right / down are the world axes given."""
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_img, y_img, (0.0, 0.0, -1.0)
    T[:3, 3] = at
    assert np.allclose(np.cross(T[:3, 0], T[:3, 1]), T[:3, 2])
    return Camera("cam_left_wrist", K, T)


UPRIGHT = _down_cam((1, 0, 0), (0, -1, 0))          # world +y up in the image
ROLLED = _down_cam((-1, 0, 0), (0, 1, 0))           # the look-over at wrist yaw 0: world +y down
QUARTER = _down_cam((0, 1, 0), (1, 0, 0))           # world +y to the right
TILE = (0.0, -0.15, 0.78)


def test_upright_deg_measures_the_roll_of_world_y():
    assert gr.upright_deg(UPRIGHT, TILE) == pytest.approx(0.0, abs=1e-6)
    assert abs(gr.upright_deg(ROLLED, TILE)) == pytest.approx(180.0, abs=1e-6)
    assert gr.upright_deg(QUARTER, TILE) == pytest.approx(90.0, abs=1e-6)
    assert gr.upright_deg(UPRIGHT, (0.0, -0.15, 1.2)) is None                         # behind the camera


def test_the_head_camera_sees_the_tables_y_nearly_up():
    for x in (-0.3, 0.0, 0.3):                      # the training roll of the head crops
        assert abs(gr.upright_deg(head_camera(), (x, -0.15, 0.78))) < 20.0


def _scene():
    """A 9 x 9 tile in a 240 x 320 image, its "top" 3 rows bright: the top is where world +y is in UPRIGHT."""
    img = np.zeros((240, 320, 3), np.uint8)
    m = np.zeros((240, 320), np.uint8)
    m[100:109, 150:159] = 1
    img[100:109, 150:159] = 60
    img[100:103, 150:159] = 250
    return img, m


def test_turn_upright_turns_counter_clockwise_about_the_mask():
    img, m = _scene()
    im0, m0 = gr.turn_upright(img, m, 0.0)
    assert im0.shape[0] == im0.shape[1] and m0.sum() == m.sum()
    ys, xs = np.nonzero(im0[..., 0] == 250)
    my, mx = np.nonzero(m0)
    assert ys.mean() < my.mean()                    # unturned: the bright rows on top
    im1, m1 = gr.turn_upright(img, m, 180.0)
    ys, xs = np.nonzero(im1[..., 0] > 200)
    my, mx = np.nonzero(m1)
    assert ys.mean() > my.mean()                    # upside down: at the bottom
    im2, m2 = gr.turn_upright(img, m, 90.0)         # counter-clockwise: the top goes to the left
    ys, xs = np.nonzero(im2[..., 0] > 200)
    my, mx = np.nonzero(m2)
    assert xs.mean() < mx.mean() and abs(ys.mean() - my.mean()) < 2
    # the classifier's crop of the turned mask (+25 % per side) stays inside the patch
    x0, y0, x1, y1 = gr.mask_bbox(gr.turn_upright(img, m, 45.0)[1])
    half = max(x1 - x0, y1 - y0, 8) * 0.75
    assert (x0 + x1) / 2 - half >= 0 and (x0 + x1) / 2 + half < im2.shape[1]
    edge = np.zeros_like(m)
    edge[2:12, 300:318] = 1                          # near the border: the patch is border-replicated
    im3, m3 = gr.turn_upright(img, edge, 30.0)
    assert im3.shape[0] == im3.shape[1] and m3.any()


class _Rec:
    """A fake reader: remembers what it was sent, reads glyph 1 ("6") confidently."""

    def __init__(self):
        self.sent = []

    def classify(self, img, boxes):
        self.sent.append((img, boxes))
        return {"probs": [[0.02, 0.9, 0.03, 0.03, 0.02]], "n_glyph": 4, "margins": {"head": 0.5, "wrist": 0.5},
                "names": ["0", "6", "9", "plus"]}


def test_classify_mask_turns_only_when_asked():
    img, m = _scene()
    c = _Rec()
    r = gr.classify_mask(img, m, "cam_left_wrist", client_=c)
    assert r["read"] and "turned" not in r and c.sent[-1][0] is img and c.sent[-1][1] == [(150, 100, 158, 108)]
    r = gr.classify_mask(img, m, "cam_left_wrist", client_=c, upright_deg=180.0)
    sent, (bb,) = c.sent[-1]
    assert r["read"] and r["turned"] == 180.0 and sent.shape[0] == sent.shape[1] < 100
    x0, y0, x1, y1 = bb
    assert sent[y1, (x0 + x1) // 2, 0] == 250 and sent[y0, (x0 + x1) // 2, 0] == 60      # the bright top now below
    cut = np.zeros_like(m)
    cut[0:12, 30:40] = 1                             # the edge check is on the view's own mask, before any turn
    assert gr.classify_mask(img, cut, "cam_left_wrist", client_=c, upright_deg=180.0)["why"] == "edge"


def _service_read(monkeypatch, key, cam, cfg):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import cameras, refine
    from dojo.runtime import perception_service as ps

    img, m = _scene()
    calls = []

    def classify(img_, mask, view="cam_head", category=None, upright_deg=None):
        calls.append({"view": view, "category": category, "upright_deg": upright_deg})
        return {"read": True, "name": "6", "margin": 0.95, "why": "read",
                **({"turned": round(upright_deg, 1)} if upright_deg is not None else {})}

    monkeypatch.setattr(gr, "classify_mask", classify)
    monkeypatch.setattr(cameras, "cameras_from_state", lambda state, model=None: {"cam_left_wrist": cam,
                                                                                  "cam_head": head_camera()})
    monkeypatch.setattr(refine, "_shape_for", lambda k, R, tag: SimpleNamespace(sil=lambda c, T: m))
    if cfg is None:
        monkeypatch.delenv("DOJO_PERCEPTION_CFG", raising=False)
    else:
        monkeypatch.setenv("DOJO_PERCEPTION_CFG", cfg)
    f = {"state": {}, "images": {"cam_left_wrist": img, "cam_head": img}}
    out = ps._Perception()._glyph_view_read(f, key, "left", np.r_[TILE, 1.0, 0.0, 0.0, 0.0], 0)
    return out, calls


def test_the_flag_is_off_by_default_and_turns_only_digit_model_wrist_reads(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig

    assert EstimatorConfig().glyph_upright == "off"
    out, calls = _service_read(monkeypatch, "Rigid/number/00006", ROLLED, '{"glyph_read": "on"}')
    assert out["read"] and "turned" not in out and [c["upright_deg"] for c in calls] == [None]
    out, calls = _service_read(monkeypatch, "Rigid/number/00006", ROLLED, '{"glyph_read": "on", "glyph_upright": "on"}')
    assert len(calls) == 1 and abs(calls[0]["upright_deg"]) == pytest.approx(180.0, abs=1e-6)
    assert abs(out["turned"]) == 180.0
    out, calls = _service_read(monkeypatch, "Rigid/plus/00000", UPRIGHT, '{"glyph_read": "on", "glyph_upright": "on"}')
    assert calls[0]["upright_deg"] == pytest.approx(0.0, abs=1e-6)                       # operators: the digit model
    out, calls = _service_read(monkeypatch, "Rigid/mahjong/00003", ROLLED, '{"glyph_read": "on", "glyph_upright": "on"}')
    assert {c["view"] for c in calls} == {"cam_left_wrist", "cam_head"} and all(c["upright_deg"] is None for c in calls)


class _Broken:
    def info(self):
        return {}

    def handle(self, req, img):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


@pytest.fixture
def broken_digit_server(tmp_path, monkeypatch):
    from dojo.perception import sam_masks

    sock = str(tmp_path / "glyph_digit.sock")
    ready = threading.Event()
    threading.Thread(target=sam_masks.serve, args=(sock, ["glyph"], 0.01), kwargs={"M": _Broken(), "ready": ready},
                     daemon=True).start()
    assert ready.wait(5)
    monkeypatch.setenv(gr.ENV_ADDR_OF["digit"], sock)
    gr._CLIENT.clear()
    yield sock
    gr._CLIENT.clear()


def test_an_erroring_glyph_server_is_infra_in_both_read_paths(broken_digit_server, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import cameras, inverse_layout as il, refine
    from dojo.runtime import perception_service as ps

    img, m = _scene()
    with pytest.raises(SideServerDown, match="illegal memory"):
        gr.classify_mask(img, m, "cam_head", category="number")
    views = {"cam_head": SimpleNamespace(cam="h", img=img, known=np.ones_like(m), robot=None)}
    with pytest.raises(SideServerDown):                       # frame 0 / records: inverse_layout._glyph_views
        il._glyph_views(lambda cam: m.astype(bool), views, "number")
    monkeypatch.setattr(cameras, "cameras_from_state", lambda state, model=None: {"cam_left_wrist": UPRIGHT})
    monkeypatch.setattr(refine, "_shape_for", lambda k, R, tag: SimpleNamespace(sil=lambda c, T: m))
    from dojo.perception import render
    monkeypatch.setattr(render, "robot_mask", lambda cam, state, closed_jaws=False: np.zeros_like(m))
    f = {"state": {}, "images": {"cam_left_wrist": img}}
    monkeypatch.setenv("DOJO_PERCEPTION_CFG", '{"glyph_read": "on", "glyph_upright": "on"}')
    with pytest.raises(SideServerDown):                       # runtime wrist reads: the service's refine
        ps._Perception()._glyph_view_read(f, "Rigid/number/00006", "left", np.r_[TILE, 1.0, 0.0, 0.0, 0.0], 0)


def test_the_look_over_rolls_the_real_wrist_camera_upside_down():
    """The runtime look (regrasp._look_move: hand straight down 18 cm over the tile, wrist yaw 0 tried first) on the
    URDF's wrist camera: world +y points down in the image. Needs the X5 URDF (DOJO_X5_DIR); skipped without it."""
    pytest.importorskip("scipy")
    import math

    from dojo.perception.cameras import wrist_camera
    from dojo.skills.geometry import T_from
    from dojo.skills.kinematics import X5Kin, down_R

    kin = X5Kin()
    for arm, x in (("left", -0.1), ("right", 0.1)):
        tile = np.array([x, -0.15, 0.78])
        r = kin.ik(arm, T_from(tile + [0.0, 0.0, 0.18], down_R(0.0)), np.zeros(6))
        assert r.ok
        try:
            cam = wrist_camera(arm, r.q)
        except FileNotFoundError:
            pytest.skip("no X5 URDF")
        assert abs(gr.upright_deg(cam, tile)) == pytest.approx(180.0, abs=10.0)
        r = kin.ik(arm, T_from(tile + [0.0, 0.0, 0.18], down_R(math.pi)), np.zeros(6))
        if r.ok:                                     # the other heading sees it upright
            assert abs(gr.upright_deg(wrist_camera(arm, r.q), tile)) < 10.0
