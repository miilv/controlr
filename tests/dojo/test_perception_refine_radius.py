"""refine()'s caller-given search radius and the empty-hand robot mask (pure python + numpy, no assets)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from dojo.perception import refine as R  # noqa: E402

TRUTH = (0.03, -0.02)


def _sq(x, y):
    m = np.zeros((60, 80), np.uint8)
    u, v = int(round(40 + 500 * x)), int(round(30 + 500 * y))
    m[max(v - 5, 0):v + 5, max(u - 5, 0):u + 5] = 1
    return m


class _Shape:
    tag, zmin, R0 = "up", 0.0, np.eye(3)

    def pose(self, x, y, yaw):
        T = np.eye(4)
        T[:2, 3] = (x, y)
        return T

    def sil(self, cam, T):
        return _sq(T[0, 3], T[1, 3])


def _fake_refine(obj_id, img, proprio, prior, arm="left", **kw):
    near = np.hypot(prior[0] - TRUTH[0], prior[1] - TRUTH[1]) < 0.012
    pose = [TRUTH[0], TRUTH[1], prior[2], 1.0, 0.0, 0.0, 0.0] if near else list(prior)
    return {"pose": pose, "ok": bool(near), "iou": 0.9 if near else 0.1, "moved_mm": 0.0, "turned_deg": 0.0}


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr(R, "refine", _fake_refine)
    monkeypatch.setattr(R, "wrist_camera", lambda arm, q: SimpleNamespace(name="cam_left_wrist"))
    monkeypatch.setattr(R, "shape_models", lambda key: [_Shape()])
    monkeypatch.setattr(R, "_shape_for", lambda key, Rm, tag: _Shape())
    monkeypatch.setattr(R, "local_foreground", lambda img, cam, sil, rob, **kw: (_sq(*TRUTH), np.ones_like(sil)))
    import dojo.perception.render as render

    monkeypatch.setattr(render, "robot_mask", lambda cam, st, dilate_px=0, closed_jaws=True: np.zeros((60, 80), np.uint8))


def test_radius_search_finds_the_object_off_the_prior(scene):
    # Memory, 5.1 conveyor: the head-view track 3-6 cm off at the end of the approach
    prior = [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]
    out = R._radius_refine("Rigid/can/00000", None, {"left_arm_joint_state": np.zeros(6)}, prior, "left", 0.04, {})
    assert out["ok"] and out["pose"][:2] == list(TRUTH)
    assert out["moved_mm"] == pytest.approx(1000 * np.hypot(*TRUTH), abs=0.1)
    assert out["radius_search"]["used"] and out["radius_search"]["tried"]


def test_radius_search_keeps_an_ok_prior_and_reports_a_miss(scene):
    st = {"left_arm_joint_state": np.zeros(6)}
    near = [TRUTH[0] + 0.005, TRUTH[1], 0.78, 1.0, 0.0, 0.0, 0.0]
    out = R._radius_refine("Rigid/can/00000", None, st, near, "left", 0.04, {})
    assert out["ok"] and out["radius_search"] == {"radius_mm": 40.0, "used": False}
    far = [-0.06, 0.06, 0.78, 1.0, 0.0, 0.0, 0.0]  # the object is outside the radius: the prior's own (not ok) result
    out = R._radius_refine("Rigid/can/00000", None, st, far, "left", 0.02, {})
    assert not out["ok"] and out["pose"] == far and not out["radius_search"]["used"]


def test_refine_takes_the_radius_path_only_with_a_radius(monkeypatch):
    seen = []
    monkeypatch.setattr(R, "_radius_refine", lambda *a, **kw: seen.append(a[5]) or {"ok": True})
    R.refine("Rigid/can/00000", None, {}, [0, 0, 0.78, 1, 0, 0, 0], search_radius=0.05)
    assert seen == [0.05]


def test_empty_hand_masks_no_held_box_between_preshaped_jaws(monkeypatch):
    import dojo.perception.render as render
    from dojo.perception import robot_model

    links = ("link6", "link7", "link8")
    V = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0.05]], float)
    F = np.array([[0, 1, 2]])
    model = SimpleNamespace(fk_world=lambda arm, q, g=None: {k: np.eye(4) for k in links})
    monkeypatch.setattr(robot_model.X5Model, "default", classmethod(lambda cls: model))
    monkeypatch.setattr(render, "robot_link_meshes", lambda: {k: (V, F) for k in links})
    st = {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": [0.44]}  # a pre-grasp opening below half
    held = [it for it in render.robot_items(st) if it["link"] in ("held_box", "jaw_sweep")]
    assert held  # unknown hand: commanded below half open is a grip, the held box is masked
    assert not [it for it in render.robot_items(st, closed_jaws=False) if it["link"] in ("held_box", "jaw_sweep")]
