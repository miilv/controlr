"""The camera garment read (``dojo.perception.garment``): the template fit, a synthetic head frame read end to end, and
the runtime provider's one-read cache (CPU only)."""

from __future__ import annotations

import math

import numpy as np
import pytest


def _tshirt(rng):
    body = rng.uniform([-0.15, -0.2], [0.15, 0.14], (3000, 2))
    sl = rng.uniform([0.15, 0.02], [0.28, 0.1], (800, 2))
    return np.vstack([body, sl, sl * [-1, 1]])


def test_template_fit_recovers_yaw_and_shift():
    pytest.importorskip("scipy")
    from dojo.perception.garment import fit_template
    rng = np.random.default_rng(0)
    T = _tshirt(rng)
    th, t = math.radians(12), np.array([0.03, -0.13])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    obs = T @ R.T + t + rng.normal(0, 0.002, T.shape)
    f = fit_template(obs, {"xy": T})
    assert abs(math.degrees(f["yaw"] - th)) < 2.5
    assert np.linalg.norm(f["t"] - t) < 0.01


def test_read_garment_from_a_synthetic_head_frame():
    pytest.importorskip("scipy")
    cv2 = pytest.importorskip("cv2")
    from dojo.perception.cameras import head_camera
    from dojo.perception.garment import CLOTH_Z, read_garment
    rng = np.random.default_rng(1)
    T = _tshirt(rng)
    tpl = {"Garment/Top_Long/00004": {"xy": T, "kp": {"left_sleeve": np.array([-0.27, 0.06, 0.0]),
                                                      "right_hem": np.array([0.12, -0.19, 0.0])}}}
    th, t = math.radians(-14), np.array([-0.01, -0.12])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    P = np.c_[T @ R.T + t, np.full(len(T), CLOTH_Z)]
    cam = head_camera()
    bg = np.zeros((cam.h, cam.w, 3), np.uint8)
    bg[:] = (140, 80, 40)                                   # a brown table
    img = bg.copy()
    uv, _ = cam.project(P)
    for u, v in uv:
        cv2.circle(img, (int(u), int(v)), 3, (15, 15, 15), -1)
    r = read_garment(img, tpl, cam=cam, bg=bg)
    assert r is not None and r["asset"] == "Garment/Top_Long/00004"
    yaw = 2 * math.atan2(r["quat"][3], r["quat"][0])
    assert abs(math.degrees(yaw - th)) < 3.0
    for k, x in tpl["Garment/Top_Long/00004"]["kp"].items():
        truth = R @ x[:2] + t
        assert np.linalg.norm(np.asarray(r["keypoints"][k][:2]) - truth) < 0.015, k


def test_runtime_provider_reads_the_garment_once():
    from dojo.runtime.belief import RuntimeProvider

    class Bridge:
        n = 0

        def observe(self):
            Bridge.n += 1
            return {"images": {}, "state": {}, "steps_used": 0}
    calls = []
    prov = RuntimeProvider(Bridge(), "", "Fold the clothes neatly.", "perception")
    assert prov.garment("target") is None                   # no reader: nothing
    prov.garment_reader = lambda obs: calls.append(obs) or {"asset": "Garment/Top_Long/00004", "keypoints": {"a": [0, 0, 0]},
                                                          "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "iou": 0.5}
    g1, g2 = prov.garment("target"), prov.garment("target")
    assert g1 is g2 and len(calls) == 1 and g1["live"] is False
    assert any(e["kind"] == "garment_read" for e in prov.events)


def test_shipped_templates_cover_both_garment_sets():
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).resolve().parents[2] / "dojo" / "perception" / "data" / "garment_templates.json")
                     .read_text())
    keys = set(raw)
    assert {f"Garment/Top_Long/{i:05d}" for i in (1, 4, 9, 6, 7, 11)} <= keys      # standard and random sets
    assert all(isinstance(v.get("seed"), int) and v["seed"] >= 100000 for v in raw.values())   # our dev layouts only
    assert all(len(v["xy"]) >= 1000 and "left_sleeve" in v["kp"] for v in raw.values())
def test_update_hems_follows_a_shifted_hem_corner():
    pytest.importorskip("scipy")
    cv2 = pytest.importorskip("cv2")
    from dojo.perception.cameras import head_camera
    from dojo.perception.garment import CLOTH_Z, update_hems
    rng = np.random.default_rng(2)
    T = _tshirt(rng)
    cam = head_camera()
    bg = np.zeros((cam.h, cam.w, 3), np.uint8)
    bg[:] = (140, 80, 40)

    def frame(P2):
        img = bg.copy()
        uv, _ = cam.project(np.c_[P2, np.full(len(P2), CLOTH_Z)])
        for u, v in uv:
            cv2.circle(img, (int(u), int(v)), 3, (15, 15, 15), -1)
        return img
    t = np.array([0.0, -0.12])
    P0 = T + t
    P1 = P0.copy()
    lower_left = (T[:, 0] < -0.05) & (T[:, 1] < -0.1)        # the bottom-left of the body slides 3 cm right, 2 cm up
    P1[lower_left] += [0.03, 0.02]
    read0 = {"pos": [t[0], t[1], 0.765], "quat": [1.0, 0, 0, 0],
             "keypoints": {"left_hem": [-0.15 + t[0], -0.2 + t[1], 0.773], "right_hem": [0.15 + t[0], -0.2 + t[1], 0.773]}}
    r = update_hems(frame(P0), frame(P1), read0, cam=cam, bg=bg)
    assert r is not None
    dl = np.asarray(r["keypoints"]["left_hem"][:2]) - np.asarray(read0["keypoints"]["left_hem"][:2])
    dr = np.asarray(r["keypoints"]["right_hem"][:2]) - np.asarray(read0["keypoints"]["right_hem"][:2])
    assert np.linalg.norm(dl - [0.03, 0.02]) < 0.012, dl
    assert np.linalg.norm(dr) < 0.008, dr


def test_runtime_provider_garment_update_uses_the_first_frame():
    from dojo.runtime.belief import RuntimeProvider

    class Bridge:
        n = 0

        def observe(self):
            Bridge.n += 1
            return {"images": {}, "state": {}, "steps_used": Bridge.n}
    prov = RuntimeProvider(Bridge(), "", "Fold the clothes neatly.", "perception")
    prov.garment_reader = lambda obs: {"keypoints": {"left_hem": [0, 0, 0]}, "pos": [0, 0, 0], "quat": [1, 0, 0, 0]}
    seen = []
    prov.garment_updater = lambda o0, o1, read: seen.append((o0["steps_used"], o1["steps_used"])) or {
        "keypoints": {"left_hem": [0.01, 0.02, 0.0], "right_hem": [0.0, 0.0, 0.0]}, "shift_mm": {"left_hem": 22.4}}
    assert prov.garment_update("target", ["left_hem"]) is None      # no first read yet
    prov.garment("target")
    g = prov.garment_update("target", ["left_hem"])
    assert g == {"keypoints": {"left_hem": [0.01, 0.02, 0.0]}, "live": False}
    assert seen == [(1, 2)]                                         # the first read's frame, then the current one
