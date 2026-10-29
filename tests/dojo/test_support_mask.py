"""inverse_layout support_mask / wrist_pool (default off): a support robot standing at its home pose in the head view
has its pixels unknown (not foreground) in the head and home wrist views; the twin segmenter's home wrist masks become
class-agnostic proposal blobs. imitate's demonstrator Franka merged with the far objects into one head blob."""
from types import SimpleNamespace

import numpy as np
import pytest


def _views(head_img):
    def vm(img):
        h, w = img.shape[:2]
        return SimpleNamespace(img=img, fg=np.ones((h, w), np.uint8), known=np.ones((h, w), np.uint8),
                               robot=np.zeros((h, w), np.uint8))
    return {"cam_head": vm(head_img), "cam_left_wrist": vm(np.zeros((6, 8, 3), np.uint8))}


HOME = {"left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6)}
MOVED = {"left_arm_joint_state": np.full(6, 0.3), "right_arm_joint_state": np.zeros(6)}


def _model(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    ref = np.zeros((6, 8, 3), np.uint8)
    ref[1:3, 2:5] = (240, 240, 235)                               # the robot's pixels at its home
    mh = np.zeros((6, 8), bool)
    mh[1:3, 2:5] = True
    mw = np.zeros((6, 8), bool)
    mw[4:6, 0:2] = True
    monkeypatch.setattr(IL, "_support_models", lambda: (("franka_home", ref, {"cam_head": mh, "cam_left_wrist": mw}),))
    return IL, ref, mh, mw


def test_support_mask_makes_the_robot_unknown_where_it_stands_at_home(monkeypatch):
    IL, ref, mh, mw = _model(monkeypatch)
    img = ref.copy()
    img[0, 0] = (90, 50, 30)                                      # the table elsewhere: not part of the match
    v = _views(img)
    info = IL._support_mask(v, HOME)
    assert info["applied"] == "franka_home" and set(info["views"]) == {"cam_head", "cam_left_wrist"}
    assert not v["cam_head"].known[mh].any() and not v["cam_head"].fg[mh].any() and v["cam_head"].robot[mh].all()
    assert v["cam_head"].known[~mh].all() and v["cam_head"].fg[~mh].all()
    assert not v["cam_left_wrist"].known[mw].any()
    v = _views(img)
    info = IL._support_mask(v, MOVED)                             # our arm moved: the wrist mask no longer fits
    assert info["views"] == ["cam_head"] and v["cam_left_wrist"].known.all()


def test_support_mask_needs_the_robot_at_its_home(monkeypatch):
    IL, ref, mh, mw = _model(monkeypatch)
    away = ref.copy()
    away[mh] = (90, 50, 30)                                       # the robot moved off: table there
    v = _views(away)
    assert IL._support_mask(v, HOME) == {"why": "no support robot at its home"}
    assert v["cam_head"].known.all() and v["cam_head"].fg.all()


def test_the_shipped_franka_model_loads():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    IL._support_models.cache_clear()
    ms = {name: (ref, masks) for name, ref, masks in IL._support_models()}
    ref, masks = ms["franka_home"]
    assert ref.shape == (480, 640, 3) and set(masks) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    assert 5000 < int(masks["cam_head"].sum()) < 20000


class _Seg:
    def __init__(self, masks):
        self.masks = masks

    def detect(self, img, conf=0.25, max_det=100):
        return [SimpleNamespace(mask=m.astype(np.uint8), key="Rigid/watch/00000", score=0.4) for m in self.masks]


def test_wrist_pool_blobs_come_from_home_wrist_masks_on_known_pixels(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL, seg_twin
    big = np.zeros((40, 60), bool)
    big[5:25, 10:30] = True                                       # 400 px
    small = np.zeros((40, 60), bool)
    small[30:35, 40:45] = True                                    # 25 px: below MIN_BLOB_PX for a wrist
    monkeypatch.setattr(seg_twin, "client", lambda: _Seg([big, small]))
    known = np.ones((40, 60), np.uint8)
    known[5:25, 10:15] = 0                                        # an arm over part of it
    vm = SimpleNamespace(known=known)
    images = {"cam_left_wrist": np.zeros((40, 60, 3), np.uint8)}
    cfg = SimpleNamespace(seg_min_score=0.25)
    bl, info = IL._wrist_seg_blobs(images, {"cam_left_wrist": vm}, HOME, cfg)
    assert info["blobs"] == 1 and bl[0].view == "cam_left_wrist" and bl[0].area == 300
    assert bl[0].bbox == (15, 5, 15, 20)
    bl, info = IL._wrist_seg_blobs(images, {"cam_left_wrist": vm}, MOVED, cfg)
    assert bl == [] and info["why"] == "arms moved"
    monkeypatch.setattr(seg_twin, "client", lambda: None)
    assert IL._wrist_seg_blobs(images, {"cam_left_wrist": vm}, HOME, cfg)[1]["why"] == "no segmenter server"


def test_flags_default_off():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception.inverse_layout import EstimatorConfig
    c = EstimatorConfig()
    assert c.support_mask is False and c.wrist_pool is False
