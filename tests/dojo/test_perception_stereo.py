"""dojo.perception.stereo: plane-sweep height map, the record height check and its slot-base consumer, on a synthetic
two-camera scene (a textured table and a textured plate 5 cm above it). No GPU, no assets.

The perception extra (scipy, opencv) is needed; without it the module is skipped (CI installs only dev).
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from dojo.perception import cameras  # noqa: E402
from dojo.perception.cameras import TABLE_Z, Camera  # noqa: E402
from dojo.perception.inverse_layout import reject_slot_bases  # noqa: E402
from dojo.perception.stereo import View, plane_homography, plane_sweep, scaled_camera  # noqa: E402

PLATE = (-0.21, -0.09, -0.06, 0.06, 0.05)  # x0, x1, y0, y1, height above the table
HEIGHTS = tuple(np.round(np.r_[np.arange(-0.01, 0.07, 0.002)], 4))


def _look_at(eye, target, K) -> Camera:
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return Camera("cam_left_wrist", K, T)


def _tex(seed: int):
    rng = np.random.default_rng(seed)
    f = rng.uniform(250, 900, 24)
    a = rng.uniform(0, np.pi, 24)
    ph = rng.uniform(0, 2 * np.pi, 24)

    def t(X):
        s = sum(np.sin(fi * (np.cos(ai) * X[:, 0] + np.sin(ai) * X[:, 1]) + pi) for fi, ai, pi in zip(f, a, ph))
        return np.clip(0.5 + 0.08 * s, 0, 1)
    return t


def _render(cam: Camera, plate=PLATE) -> np.ndarray:
    """Grey image (as RGB uint8): the plate's top where the ray meets it inside its extent, else the table."""
    vv, uu = np.mgrid[0:cam.h, 0:cam.w]
    uv = np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5]
    x0, x1, y0, y1, hp = plate
    Xp = np.nan_to_num(cam.ray_plane(uv, TABLE_Z + hp))
    on = (Xp[:, 0] > x0) & (Xp[:, 0] < x1) & (Xp[:, 1] > y0) & (Xp[:, 1] < y1)
    Xt = np.nan_to_num(cam.ray_plane(uv, TABLE_Z))
    g = np.where(on, _tex(1)(Xp), _tex(0)(Xt))
    img = (g.reshape(cam.h, cam.w) * 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)


@pytest.fixture(scope="module")
def scene():
    head = cameras.head_camera()
    K = cameras.intrinsics(cameras.WRIST["focal_length"], cameras.WRIST["horizontal_aperture"])
    wrist = _look_at(np.array([-0.32, -0.30, 1.05]), np.array([-0.15, 0.0, TABLE_Z]), K)
    return head, wrist, _render(head), _render(wrist)


def _masks(cam: Camera, plate=PLATE, margin=0.015):
    vv, uu = np.mgrid[0:cam.h, 0:cam.w]
    uv = np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5]
    x0, x1, y0, y1, hp = plate
    Xp = np.nan_to_num(cam.ray_plane(uv, TABLE_Z + hp))
    inner = (Xp[:, 0] > x0 + margin) & (Xp[:, 0] < x1 - margin) & (Xp[:, 1] > y0 + margin) & (Xp[:, 1] < y1 - margin)
    Xt = np.nan_to_num(cam.ray_plane(uv, TABLE_Z))
    far = (np.hypot(Xt[:, 0] - (x0 + x1) / 2, Xt[:, 1] - (y0 + y1) / 2) > 0.15) & (np.abs(Xt[:, 0]) < 0.5) & (Xt[:, 1] > -0.1) & (Xt[:, 1] < 0.2)
    return inner.reshape(cam.h, cam.w), far.reshape(cam.h, cam.w)


def test_plane_homography_maps_plane_points():
    head = cameras.head_camera()
    K = cameras.intrinsics(13.0, 20.955)
    wrist = _look_at(np.array([-0.3, -0.3, 1.0]), np.array([-0.1, 0.0, TABLE_Z]), K)
    rng = np.random.default_rng(3)
    for z in (TABLE_Z, TABLE_Z + 0.07):
        X = np.c_[rng.uniform(-0.2, 0.0, 20), rng.uniform(-0.1, 0.1, 20), np.full(20, z)]
        a, _ = head.project(X)
        b, _ = wrist.project(X)
        H = plane_homography(head, wrist, z)
        p = (H @ np.c_[a - 0.5, np.ones(20)].T).T  # OpenCV index convention: pixel centre at integer coordinates
        assert np.abs(p[:, :2] / p[:, 2:] - (b - 0.5)).max() < 1e-6


@pytest.mark.parametrize("lr", [False, True])
def test_sweep_recovers_plate_height_and_table(scene, lr):
    head, wrist, ih, iw = scene
    sw = plane_sweep(View(ih, head), [View(iw, wrist)], heights=HEIGHTS, scale=0.5, agg=5, lr=lr)
    inner, far = _masks(scaled_camera(head, 0.5))
    plate_h = sw.height[inner & np.isfinite(sw.height)]
    table_h = sw.height[far & np.isfinite(sw.height)]
    assert plate_h.size > 50 and table_h.size > 500
    assert abs(float(np.median(plate_h)) - PLATE[4]) < 0.003
    assert abs(float(np.median(table_h))) < 0.002
    assert (sw.proud[inner] > 0).mean() > 0.6
    assert (sw.proud[far & (sw.known > 0)] > 0).mean() < 0.03
    assert (sw.on_table[far & (sw.known > 0)] > 0).mean() > 0.8


def _box(sx, sy, z0, z1):
    V = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (z0, z1)], float)
    F = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                  [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int64)
    return V, F


class _Asset:
    def __init__(self, V, F):
        self.Vs, self.Fs = V, F


def _record(pos):
    return {"Geometry": {"vertical_coin_stand": [{"label": "vertical_coin_stand", "category": "vertical_coin_stand",
                                                  "category_idx": 0, "default_pos": list(pos), "default_ori": [1.0, 0, 0, 0],
                                                  "_perception": {"slot": False}}]},
            "Rigid": {"coin": [{"label": "coin0", "category": "coin", "category_idx": 0, "default_pos": list(pos),
                                "default_ori": [1.0, 0, 0, 0], "_perception": {"slot": True, "support_slot": "vertical_coin_stand/0",
                                                                                "stacked_on": "vertical_coin_stand"}}]}}


@pytest.mark.parametrize("where,mismatch", [("plate", False), ("bare_table", True)])
def test_check_record_flags_a_phantom_and_keeps_a_right_object(scene, monkeypatch, where, mismatch):
    pytest.importorskip("trimesh")
    from dojo.perception import render, stereo

    head, wrist, ih, iw = scene
    x0, x1, y0, y1, hp = PLATE
    V, F = _box(x1 - x0 - 0.004, y1 - y0 - 0.004, -0.004, 0.0)  # a thin slab whose top is the plate's
    Vc, Fc = _box(0.002, 0.02, -0.002, 0.0)  # the coin: thin, sitting on the slab
    monkeypatch.setattr(render, "load_asset", lambda key: _Asset(Vc, Fc) if "coin/" in key and "stand" not in key else _Asset(V, F))
    monkeypatch.setattr(render, "_GL", False)  # CPU ray cast
    pos = ((x0 + x1) / 2, (y0 + y1) / 2, TABLE_Z + hp) if where == "plate" else (0.0, 0.10, TABLE_Z + hp)
    rec = _record(pos)
    zeros = {c.name: np.zeros((c.h, c.w), np.uint8) for c in (head, wrist)}
    summ = stereo.check_record(rec, {"cam_head": ih, "cam_left_wrist": iw}, {}, cams={"cam_head": head, "cam_left_wrist": wrist},
                               robot_masks=zeros, sweep_kw={"heights": HEIGHTS, "scale": 0.5})
    st = rec["Geometry"]["vertical_coin_stand"][0]["_perception"]["stereo"]
    assert summ["objects"] == 2
    assert bool(st.get("mismatch")) is mismatch, st
    if not mismatch:
        assert abs(st["dh_mm"]) < 3.0


class _Ext:
    def __init__(self, e):
        self.extents = np.asarray(e, float)


@pytest.mark.parametrize("coin_thickness,dropped", [(0.002, True), (0.02, False)])
def test_reject_slot_bases_drops_a_mismatched_base_and_its_thin_children(monkeypatch, coin_thickness, dropped):
    from dojo.perception import render

    monkeypatch.setattr(render, "load_asset", lambda key: _Ext([0.025, coin_thickness, 0.025] if "Rigid/coin" in key else [0.03, 0.03, 0.03]))
    rec = _record((0.0, 0.0, TABLE_Z))
    rec["Geometry"]["piggy_bank"] = [{"label": "piggy_bank", "category": "piggy_bank", "category_idx": 0,
                                      "default_pos": [0.3, 0, TABLE_Z], "default_ori": [1.0, 0, 0, 0],
                                      "_perception": {"stereo": {"mismatch": True, "dh_mm": 50.0}}}]
    assert reject_slot_bases(rec) == []  # no stereo verdict on the stand: nothing moves (the piggy is no slot base)
    rec["Geometry"]["vertical_coin_stand"][0]["_perception"]["stereo"] = {"mismatch": True, "dh_mm": -8.0}
    if not dropped:  # a base whose child has evidence of its own (a kong tile, a bread slice) is never dropped
        assert reject_slot_bases(rec) == [] and "_not_found" not in rec
        return
    assert reject_slot_bases(rec) == ["coin0", "vertical_coin_stand"]
    assert "Rigid" not in rec and list(rec["Geometry"]) == ["piggy_bank"]
    nf = {x["label"]: x for x in rec["_not_found"]}
    assert set(nf) == {"coin0", "vertical_coin_stand"} and nf["coin0"]["reason"] == "stereo height mismatch"
    assert nf["vertical_coin_stand"]["candidates"] == ["Geometry/vertical_coin_stand/00000"]
