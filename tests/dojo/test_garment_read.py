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
    from dojo.perception.garment import TEMPLATES, load_templates
    import json
    raw = json.loads(TEMPLATES.read_text())
    keys = set(raw)
    assert {f"Garment/Top_Long/{i:05d}" for i in (1, 4, 9, 6, 7, 11)} <= keys      # standard and random sets
    assert all(isinstance(v.get("seed"), int) and v["seed"] >= 100000 for v in raw.values())   # our dev layouts only
    t = load_templates()
    assert all(len(v["xy"]) >= 1000 and "left_sleeve" in v["kp"] for v in t.values())
