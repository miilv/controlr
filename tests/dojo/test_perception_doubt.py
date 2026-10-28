"""perception.doubt (b1_doubt): the annotation rule, the measurement through the mv3d server (a fake model), and its
consumers behind their flags: the agentic gate's risk floor and the look macro. No torch, no GPU, no assets."""
import threading
import time

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from dojo.perception import doubt, mv3d  # noqa: E402
from dojo.perception.cameras import TABLE_Z, head_camera  # noqa: E402


def _rec():
    return {"Rigid": {"block": [
        {"label": "a", "category": "block", "category_idx": 0, "default_pos": [0.0, -0.1, TABLE_Z + 0.02], "default_ori": [1, 0, 0, 0]},
        {"label": "b", "category": "block", "category_idx": 0, "default_pos": [0.1, -0.1, TABLE_Z + 0.02], "default_ori": [1, 0, 0, 0]},
        {"label": "c", "category": "block", "category_idx": 0, "default_pos": [0.2, -0.1, TABLE_Z + 0.02], "default_ori": [1, 0, 0, 0]}]},
        "Geometry": {"mat": [{"label": "m", "category": "mat", "category_idx": 0, "default_pos": [0, 0, TABLE_Z],
                              "default_ori": [1, 0, 0, 0], "_perception": {"place_tag": "fixed"}}]}}


def test_annotate_or_and():
    rec = _rec()
    a, b, c = rec["Rigid"]["block"]
    d = {id(a): 0.70, id(b): 0.30, id(c): 0.30}
    dh = {id(a): 2.0, id(b): -25.0, id(c): 3.0}
    assert doubt.annotate(rec, d, dh, feat_d=0.55, dh_mm=15, mode="or") == ["a", "b"]
    assert a["_perception"]["b1_doubt"]["why"] == ["appearance"] and b["_perception"]["b1_doubt"]["why"] == ["height"]
    assert "b1_doubt" not in rec["Geometry"]["mat"][0]["_perception"]  # fixed fixtures are never doubted
    d[id(b)] = 0.6
    assert doubt.annotate(rec, d, dh, feat_d=0.55, dh_mm=15, mode="and") == ["b"]
    assert [x[0] for x in doubt.doubted(rec)] == ["b"]
    assert doubt.mark_for_look(rec)["Rigid"]["block"][1]["_perception"]["doubt_look"] is True
    assert "doubt_look" not in rec["Rigid"]["block"][1]["_perception"]  # a copy


class _Fake:
    def handle(self, req):
        if req.get("op") == "featdist":  # the first object looks wrong, the rest right
            return {"d": [0.8] + [0.2] * (len(req["objects"]) - 1), "timings": {}}
        cams = [mv3d.Camera(f"v{i}", np.array(K), np.array(T), int(w), int(h)) for i, (K, T, w, h) in enumerate(req["cams"])]
        return {"depths": [np.nan_to_num(mv3d.table_depth(c)).astype(np.float16) for c in cams], "timings": {}}

    def info(self):
        return {"model": "fake"}


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import os

    sock = str(tmp_path / "mv3d.sock")
    threading.Thread(target=mv3d.serve, kwargs={"sock": sock, "M": _Fake(), "max_hours": 0.05}, daemon=True).start()
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.02)
    monkeypatch.setenv(mv3d.ENV_ADDR, sock)
    mv3d._CLIENT.clear()
    yield sock
    mv3d._CLIENT.clear()


def test_b1_doubt_through_the_server(server, monkeypatch):
    pytest.importorskip("trimesh")
    from dojo.perception import render

    V = np.array([[x, y, z] for x in (-0.015, 0.015) for y in (-0.015, 0.015) for z in (-0.02, 0.0)], float)
    F = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                  [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int64)

    class _A:
        Vs, Fs = V, F
    monkeypatch.setattr(render, "load_asset", lambda key: _A())
    monkeypatch.setattr(render, "_GL", False)
    rec = _rec()
    head = head_camera()
    imgs = {"cam_head": np.zeros((480, 640, 3), np.uint8)}
    summ = doubt.b1_doubt(rec, imgs, {"cam_head": head}, {}, feat_d=0.55, dh_mm=15)
    # the measured map is the bare table: every block standing 2 cm proud is a height mismatch, "a" also looks wrong
    assert summ["labels"] == ["a", "b", "c"]
    assert rec["Rigid"]["block"][0]["_perception"]["b1_doubt"]["why"] == ["appearance", "height"]
    assert rec["_perception"]["b1_doubt"]["thr"]["mode"] == "or"


def test_b1_doubt_skips_without_server(monkeypatch):
    monkeypatch.delenv(mv3d.ENV_ADDR, raising=False)
    mv3d._CLIENT.clear()
    rec = _rec()
    assert "skipped" in doubt.b1_doubt(rec, {}, {}, {})
    assert "b1_doubt" not in rec["Rigid"]["block"][0].get("_perception", {})


def test_gate_risk_floor_is_off_by_default(monkeypatch):
    from dojo.perception.agentic.j1 import risk
    from dojo.perception.agentic.record import needs_agent, record_objects

    rec = _rec()
    for r in rec["Rigid"]["block"] + rec["Geometry"]["mat"]:
        r["_perception"] = {**r.get("_perception", {}), "confidence": 0.95, "residual": 0.1}
    rec["Rigid"]["block"][1]["_perception"]["b1_doubt"] = {"doubt": True, "why": ["appearance"]}
    objs = {o["label"]: o for o in record_objects(rec)}
    assert objs["b"]["flags"].get("b1_doubt") is True and "b1_doubt" not in objs["a"]["flags"]
    monkeypatch.delenv("AGENTIC_DOUBT_RISK", raising=False)
    base = risk(objs["b"])
    assert base < 0.2 and needs_agent(rec, thr=0.2)[0] is False
    monkeypatch.setenv("AGENTIC_DOUBT_RISK", "0.5")
    assert risk(objs["b"]) == pytest.approx(0.5) and risk(objs["a"]) == pytest.approx(base)
    needs, why = needs_agent(rec, thr=0.2)
    assert needs and "'b'" in why


def test_look_macro_flags_and_plans_doubted_objects(monkeypatch):
    from dojo.perception import look, render
    from dojo.runtime.estimate import flagged

    rec = _rec()
    for r in rec["Rigid"]["block"]:
        r["_perception"] = {"confidence": 0.95}
    rec["Rigid"]["block"][2]["_perception"]["b1_doubt"] = {"doubt": True, "why": ["height"]}
    assert flagged(rec, 0.7) == []
    assert flagged(rec, 0.7, doubt=True) == [{"label": "c", "why": ["doubt:height"]}]
    monkeypatch.setattr(look, "flange_for_camera", lambda T: np.r_[T[:3, 3], 1.0, 0, 0, 0])
    monkeypatch.setattr(render, "load_asset", lambda key: (_ for _ in ()).throw(KeyError(key)))
    assert look.look_plan({}, rec) == []
    plan = look.look_plan({}, doubt.mark_for_look(rec))
    assert plan and plan[0]["purpose"] == "doubt" and plan[0]["label"] == "c" and plan[0]["arm"] == "right"
