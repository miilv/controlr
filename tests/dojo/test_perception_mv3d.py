"""mv3d: learned multi-view depth put into OUR cameras (no torch, no model: the geometry around it)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dojo.perception import mv3d
from dojo.perception.cameras import TABLE_Z, Camera, head_camera, intrinsics


def _side_camera() -> Camera:
    """A second view of the table from the right, looking at the object region (OpenCV axes)."""
    C = np.array([0.45, -0.35, 1.15])
    target = np.array([0.0, -0.05, TABLE_Z])
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = C
    return Camera("side", intrinsics(13.0, 20.955), T)


def test_umeyama_and_align_sim3_recover_a_similarity():
    rng = np.random.default_rng(0)
    cams = [head_camera(), _side_camera(), Camera("c3", head_camera().K, np.eye(4))]
    cams[2].T_wc[:3, :3] = Rotation.from_euler("xyz", [100, 10, -20], degrees=True).as_matrix()
    cams[2].T_wc[:3, 3] = [-0.4, -0.3, 1.1]
    s, R = 0.37, Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    t = rng.normal(size=3)
    # the model's world: X_model = (R^T (X_ours - t)) / s
    pred_T_cw = []
    for c in cams:
        T_wc_m = np.eye(4)
        T_wc_m[:3, :3] = R.T @ c.T_wc[:3, :3]
        T_wc_m[:3, 3] = R.T @ (c.T_wc[:3, 3] - t) / s
        pred_T_cw.append(np.linalg.inv(T_wc_m))
    s2, R2, t2, res = mv3d.align_sim3(np.array(pred_T_cw), [c.T_wc for c in cams])
    assert s2 == pytest.approx(s, rel=1e-6)
    assert np.allclose(R2, R, atol=1e-6) and np.allclose(t2, t, atol=1e-6)
    assert res["pos_rms_mm"] < 1e-3 and res["rot_max_deg"] < 1e-3
    assert res["umeyama_scale"] == pytest.approx(s, rel=1e-4)


def test_table_depth_unprojects_onto_the_table():
    cam = mv3d.model_camera(head_camera())
    assert (cam.w, cam.h) == (mv3d.MODEL_W, mv3d.MODEL_H)
    td = mv3d.table_depth(cam)
    P = mv3d.unproject(np.nan_to_num(td), cam)
    ok = np.isfinite(td)
    assert ok.mean() > 0.5
    assert np.abs(P[ok][:, 2] - TABLE_Z).max() < 1e-9


def _scene_depth(cam: Camera, box=((-0.05, 0.05), (-0.1, 0.0)), h=0.04):
    """Camera-z depth of the table plus one box top at height h (the top face only: enough for a scale test)."""
    td = mv3d.table_depth(cam)
    top = mv3d.table_depth(cam, TABLE_Z + h)
    P = mv3d.unproject(np.nan_to_num(top), cam)
    inb = (P[..., 0] > box[0][0]) & (P[..., 0] < box[0][1]) & (P[..., 1] > box[1][0]) & (P[..., 1] < box[1][1])
    return np.where(inb, top, td), inb


def test_depth_scale_table_recovers_the_scale_with_an_object_present():
    cam = mv3d.model_camera(head_camera())
    d, inb = _scene_depth(cam)
    assert inb.sum() > 200
    s, share = mv3d.depth_scale_table(d / 1.3, cam)
    assert s == pytest.approx(1.3, rel=2e-3)
    assert share > 0.8


def test_depth_correct_table_removes_a_smooth_warp_and_keeps_the_object_height():
    cam = mv3d.model_camera(head_camera())
    d, inb = _scene_depth(cam, h=0.04)
    vv, uu = np.mgrid[0:cam.h, 0:cam.w]
    warp = 0.8 * (1 + 0.05 * (uu / cam.w - 0.5) - 0.04 * (vv / cam.h - 0.5) ** 2)
    pred = d * warp
    field, share = mv3d.depth_correct_table(pred, cam)
    P = mv3d.unproject(pred * field, cam)
    tab = np.isfinite(d) & ~inb & (mv3d.table_region_mask(cam) > 0)
    assert np.median(np.abs(P[tab][:, 2] - TABLE_Z)) < 2e-4
    assert np.median(P[inb][:, 2] - TABLE_Z) == pytest.approx(0.04, abs=1.5e-3)
    s_only, _ = mv3d.depth_scale_table(pred, cam)  # a single scale leaves the warp
    P1 = mv3d.unproject(pred * s_only, cam)
    assert np.median(np.abs(P1[tab][:, 2] - TABLE_Z)) > 3 * np.median(np.abs(P[tab][:, 2] - TABLE_Z))


def test_triangulate_recovers_points_and_reports_reprojection_error():
    a, b = head_camera(), _side_camera()
    rng = np.random.default_rng(1)
    X = np.c_[rng.uniform(-0.2, 0.2, 50), rng.uniform(-0.25, 0.1, 50), TABLE_Z + rng.uniform(0, 0.1, 50)]
    ua, _ = a.project(X)
    ub, _ = b.project(X)
    Y, err, ang = mv3d.triangulate(a, b, ua, ub)
    assert np.abs(Y - X).max() < 1e-6 and err.max() < 1e-4 and ang.min() > 5
    _, err2, _ = mv3d.triangulate(a, b, ua, ub + [5.0, 0])  # a wrong match breaks the epipolar constraint
    assert np.median(err2) > 1.0


def test_height_in_and_fuse_topdown_see_a_raised_box():
    cam = mv3d.model_camera(head_camera())
    d, inb = _scene_depth(cam, h=0.04)
    P = mv3d.unproject(np.nan_to_num(d), cam)
    P[~np.isfinite(d)] = np.nan
    hm = mv3d.height_in(cam, [P])
    assert np.nanmedian(hm[inb]) == pytest.approx(0.04, abs=1e-3)
    assert np.nanmedian(np.abs(hm[np.isfinite(d) & ~inb])) < 1e-3
    td = mv3d.fuse_topdown([P], cell=0.005)
    ix, iy = td.cell_of(np.array([[0.0, -0.05]]))
    assert td.height[iy[0], ix[0]] == pytest.approx(0.04, abs=1e-3)
    ix, iy = td.cell_of(np.array([[0.3, -0.05]]))
    assert abs(td.height[iy[0], ix[0]]) < 1e-3


# --------------------------------------------------------------------------------------------
# Runtime: server / client (a fake model), the estimator flags, the height check on learned heights, the proposer
# --------------------------------------------------------------------------------------------

def _plate_depth(cam: Camera, plate=(-0.06, 0.06, -0.12, 0.0, 0.04)) -> np.ndarray:
    """Camera-z depth of the table with a plate (x0, x1, y0, y1, height) on it."""
    x0, x1, y0, y1, hp = plate
    td = mv3d.table_depth(cam)
    top = mv3d.table_depth(cam, TABLE_Z + hp)
    P = mv3d.unproject(np.nan_to_num(top), cam)
    on = (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    return np.where(on, top, td)


class _FakeModel:
    """Returns each view's true depth at the requested camera (what a perfect, table-fitted model would)."""

    def handle(self, req):
        cams = [Camera(f"v{i}", np.array(K), np.array(T), int(w), int(h)) for i, (K, T, w, h) in enumerate(req["cams"])]
        return {"depths": [np.nan_to_num(_plate_depth(c)).astype(np.float16) for c in cams], "timings": {"infer_s": 0.0}}

    def info(self):
        return {"model": "fake"}


@pytest.fixture()
def fake_server(tmp_path, monkeypatch):
    import threading
    import time as _t

    sock = str(tmp_path / "mv3d.sock")
    th = threading.Thread(target=mv3d.serve, kwargs={"sock": sock, "M": _FakeModel(), "max_hours": 0.05}, daemon=True)
    th.start()
    for _ in range(100):
        import os

        if os.path.exists(sock):
            break
        _t.sleep(0.02)
    monkeypatch.setenv(mv3d.ENV_ADDR, sock)
    mv3d._CLIENT.clear()
    yield sock
    mv3d._CLIENT.clear()


def test_client_is_none_without_server(monkeypatch, tmp_path):
    monkeypatch.delenv(mv3d.ENV_ADDR, raising=False)
    assert mv3d.client() is None
    monkeypatch.setenv(mv3d.ENV_ADDR, str(tmp_path / "nobody.sock"))
    with pytest.warns(UserWarning, match="unreachable"):
        assert mv3d.client() is None
    mv3d._CLIENT.clear()


def test_server_roundtrip_gives_the_head_height_map(fake_server):
    head, side = head_camera(), _side_camera()
    imgs = {"cam_head": np.zeros((480, 640, 3), np.uint8), "cam_left_wrist": np.zeros((480, 640, 3), np.uint8)}
    r = mv3d.learned_head_height(imgs, {"cam_head": head, "cam_left_wrist": side})
    assert r is not None
    hm, info = r
    assert hm.shape == (480, 640) and info["views"] == ["cam_head", "cam_left_wrist"]
    uv, _ = head.project(np.array([[0.0, -0.06, TABLE_Z + 0.04]]))
    u, v = int(uv[0, 0]), int(uv[0, 1])
    assert hm[v, u] == pytest.approx(0.04, abs=2e-3)
    uv, _ = head.project(np.array([[0.3, -0.06, TABLE_Z]]))
    assert abs(hm[int(uv[0, 1]), int(uv[0, 0])]) < 2e-3


def test_estimator_flags_default_off():
    from dojo.perception.inverse_layout import EstimatorConfig

    c = EstimatorConfig()
    assert c.height_source == "sweep" and c.mv3d_propose is False and c.stereo_check is False


def _box(sx, sy, z0, z1):
    V = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (z0, z1)], float)
    F = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                  [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int64)
    return V, F


class _Asset:
    def __init__(self, V, F):
        self.Vs, self.Fs, self.V, self.F = V, F, V, F


def _true_heights(cam):
    d = _plate_depth(cam)
    P = mv3d.unproject(np.nan_to_num(d), cam)
    return np.where(np.isfinite(d), P[..., 2] - TABLE_Z, np.nan).astype(np.float32)


@pytest.mark.parametrize("where,mismatch", [("plate", False), ("bare_table", True)])
def test_stereo_check_on_learned_heights(monkeypatch, where, mismatch):
    pytest.importorskip("trimesh")
    from dojo.perception import render
    from dojo.perception.inverse_layout import EstimatorConfig, stereo_check

    V, F = _box(0.10, 0.10, -0.006, 0.0)  # a slab whose top is the plate's top (origin at the top)
    monkeypatch.setattr(render, "load_asset", lambda key: _Asset(V, F))
    monkeypatch.setattr(render, "_GL", False)
    head = head_camera()
    hm = _true_heights(head)
    pos = (0.0, -0.06, TABLE_Z + 0.04) if where == "plate" else (0.3, 0.10, TABLE_Z + 0.04)
    rec = {"Geometry": {"box": [{"label": "box", "category": "box", "category_idx": 0, "default_pos": list(pos),
                                 "default_ori": [1.0, 0, 0, 0]}]}}
    stereo_check(rec, {}, {}, {"cam_head": head}, {}, EstimatorConfig(stereo_check=True, height_source="mv3d"), heights=hm)
    st = rec["Geometry"]["box"][0]["_perception"]["stereo"]
    assert rec["_perception"]["stereo_check"]["source"] == "mv3d"
    assert bool(st.get("mismatch")) is mismatch, st


def test_propose_unexplained_places_a_missing_label_at_a_raised_blob(monkeypatch):
    pytest.importorskip("trimesh")
    from types import SimpleNamespace

    from dojo.perception import inverse_layout, render

    head = head_camera()
    hm = _true_heights(head)  # the plate is raised; the record has nothing there
    V, F = _box(0.12, 0.12, -0.04, 0.0)
    monkeypatch.setattr(render, "load_asset", lambda key: _Asset(V, F))
    monkeypatch.setattr(render, "_GL", False)
    monkeypatch.setattr(inverse_layout, "label_groups", lambda task: [{"labels": ["a", "b"], "candidates": ["Rigid/box/00000"],
                                                                        "type": "Rigid"}])

    class _SM:
        R0 = np.eye(3)
        asset = SimpleNamespace(V=V)
        yaw_period = 2 * np.pi

        def pose(self, x, y, yaw, z_off=0.0):
            T = np.eye(4)
            T[:3, 3] = (x, y, TABLE_Z + 0.04)
            return T
    monkeypatch.setattr(inverse_layout, "shape_models", lambda key: [_SM()])
    monkeypatch.setattr(mv3d, "_others_masks", lambda record, views, skip=None: {n: np.zeros((v.cam.h, v.cam.w), np.uint8)
                                                                                 for n, v in views.items()})
    fits = []

    def fake_fit(keys, x0, y0, views, others, n_yaw=8, z_off=0.0):
        fits.append((x0, y0))
        return 0.2, keys[0], _SM(), x0, y0, 0.0
    monkeypatch.setattr(mv3d, "_fit_at", fake_fit)
    view = SimpleNamespace(cam=head, fg=np.zeros((480, 640), np.uint8), known=np.ones((480, 640), np.uint8))
    # "a" sits outside the measured region (no height there: not a suspect); "b" is missing
    rec = {"Rigid": {"box": [{"label": "a", "category": "box", "category_idx": 0, "default_pos": [0.0, 0.45, TABLE_Z + 0.04],
                              "default_ori": [1.0, 0, 0, 0]}]}}
    reg = mv3d.table_region_mask(head) > 0
    log = mv3d.propose_unexplained(rec, {"cam_head": view}, hm, "t", valid=reg)
    assert log and log[0]["accepted"]["label"] == "b"
    placed = [r for r in rec["Rigid"]["box"] if r["label"] == "b"]
    assert len(placed) == 1 and abs(placed[0]["default_pos"][1] - (-0.06)) < 0.03
    assert view.fg.sum() > 0  # the raised, unexplained pixels joined the head foreground
    # nothing raised and unexplained -> nothing proposed
    rec2 = {"Rigid": {"box": [{"label": "a", "category": "box", "category_idx": 0, "default_pos": [0.0, -0.06, TABLE_Z + 0.04],
                               "default_ori": [1.0, 0, 0, 0]}]}}
    assert mv3d.propose_unexplained(rec2, {"cam_head": view}, hm, "t", valid=reg) == []
