"""dojo.perception.sam_masks: wire format, foreground combination, the client/server protocol (a fake model, no torch),
and the estimator's SAM flags (default off; a no-op without a server). No GPU, no weights."""
import threading

import numpy as np
import pytest

from dojo.perception import sam_masks


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.random((48, 64)) > 0.7
    data, shape = sam_masks.pack(m)
    assert len(data) == 48 * 64 // 8
    assert (sam_masks.unpack(data, shape) == m).all()


def test_client_is_none_without_server(monkeypatch, tmp_path):
    monkeypatch.delenv(sam_masks.ENV_ADDR, raising=False)
    assert sam_masks.client() is None
    monkeypatch.setenv(sam_masks.ENV_ADDR, str(tmp_path / "nobody.sock"))
    with pytest.warns(UserWarning, match="unreachable"):
        assert sam_masks.client() is None
    sam_masks._CLIENT.clear()


def test_combine_fg_modes():
    pytest.importorskip("cv2")
    fg = np.zeros((40, 40), np.uint8)
    fg[5:10, 5:10] = 1      # an object the background model sees
    fg[20:24, 30:38] = 1    # a cast shadow / clutter far from any SAM mask
    fg[14:16, 5:10] = 1     # background-model foreground just next to the SAM mask
    sam = np.zeros_like(fg)
    sam[5:14, 5:10] = 1     # SAM's mask: the object, larger than the background model's piece of it
    known = np.ones_like(fg)
    known[0:2] = 0
    u = sam_masks.combine_fg(fg, known, [sam], "union")
    assert u.sum() == (fg | sam).sum()
    g = sam_masks.combine_fg(fg, known, [sam], "gate", dilate_px=3)
    assert g[20:24, 30:38].sum() == 0          # far foreground dropped
    assert g[14:16, 5:10].all()                # near foreground kept
    assert g[5:14, 5:10].all()                 # the mask itself
    fg_t = fg.copy()
    fg_t[8:18, 5:10] = 1                       # the object's piece plus its shadow below it: one component
    t = sam_masks.combine_fg(fg_t, known, [sam], "trim", dilate_px=1)
    assert t[20:24, 30:38].all()               # a component SAM did not touch stays (an object outside the vocabulary)
    assert t[14, 5:10].all() and t[15:18, 5:10].sum() == 0  # the touching component is cut to the mask's neighbourhood
    t = sam_masks.combine_fg(fg_t, known, [sam], "trim", dilate_px=1, trim_cover=0.9)
    assert t[15:18, 5:10].all()                # ... unless the mask covers too little of it (a merged blob)
    r = sam_masks.combine_fg(fg, known, [sam], "replace")
    assert (r == sam).all()
    sam2 = sam.copy()
    sam2[0:2, 0:5] = 1                          # unknown pixels never become foreground
    assert sam_masks.combine_fg(fg, known, [sam2], "replace")[0:2].sum() == 0
    with pytest.raises(ValueError):
        sam_masks.combine_fg(fg, known, [sam], "bogus")


class _FakeModels:
    """Answers every request with a box around each point (points) / one box per prompt (text)."""

    def __init__(self):
        self.images = 0

    def info(self):
        return {"timing": {}, "peak_vram_gb": 0.0}

    def handle(self, req, img):
        h, w = img.shape[:2]
        out = []
        if req["op"] == "text":
            for i, p in enumerate(req["prompts"]):
                m = np.zeros((h, w), bool)
                m[2 + 4 * i: 5 + 4 * i, 2:6] = True
                out.append((*sam_masks.pack(m), 0.9 - 0.1 * i, p))
        else:
            for i, (u, v) in enumerate(req["uv"]):
                for k in range(3 if req.get("multimask", True) else 1):
                    m = np.zeros((h, w), bool)
                    r = 1 + k
                    m[max(0, int(v) - r): int(v) + r + 1, max(0, int(u) - r): int(u) + r + 1] = True
                    out.append((*sam_masks.pack(m), 0.5 + 0.1 * k, i))
        return {"masks": out}


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    sock = str(tmp_path / "sam.sock")
    ready = threading.Event()
    t = threading.Thread(target=sam_masks.serve, args=(sock, ["fake"], 0.01), kwargs={"M": _FakeModels(), "ready": ready},
                         daemon=True)
    t.start()
    assert ready.wait(5)
    monkeypatch.setenv(sam_masks.ENV_ADDR, sock)
    sam_masks._CLIENT.clear()
    yield sock
    sam_masks._CLIENT.clear()


def test_client_server_roundtrip(fake_server):
    c = sam_masks.client()
    assert c is not None
    img = np.zeros((24, 32, 3), np.uint8)
    ms = c.text(img, ["block", "bowl"])
    assert [m.prompt for m in ms] == ["block", "bowl"] and ms[0].mask.shape == (24, 32)
    assert ms[0].mask[2:5, 2:6].all() and ms[0].score == pytest.approx(0.9)
    pts = c.points(img, np.array([[10.0, 12.0], [20.0, 5.0]]))
    assert len(pts) == 6 and {m.point for m in pts} == {0, 1}
    assert pts[0].mask[12, 10] == 1
    # a second request on the same image does not resend it (the server keeps it by key)
    assert len(c.points(img, np.array([[3.0, 3.0]]), multimask=False)) == 1
    assert c.info()["models"] == ["fake"]


def test_estimator_flags_default_off():
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig

    cfg = EstimatorConfig()
    assert cfg.sam_fg == "off" and cfg.sam_propose == "off"
    assert cfg.series_count == "min"  # a bug fix, not an experiment axis: journal 2026-10-27-dojo-sam


def test_sam_fg_hook(fake_server, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il
    from dojo.perception.cameras import head_camera
    from dojo.perception.segment import ViewMasks

    monkeypatch.setattr(sam_masks, "vocabulary", lambda groups, instruction="", max_names=2: ["block"])
    cam = head_camera()
    fg = np.zeros((cam.h, cam.w), np.uint8)
    fg[100:110, 100:110] = 1
    head = ViewMasks(cam, np.zeros((cam.h, cam.w, 3), np.uint8), fg, np.ones_like(fg), np.zeros(fg.shape, np.float32),
                     np.zeros_like(fg))
    ms, info = il._sam_text(head.img, [], "", il.EstimatorConfig())
    assert info["n_masks"] == 1 and info["prompts"] == ["block"]
    v, _ = il._sam_fg(head, ms, il.EstimatorConfig(sam_fg="union"))
    assert v.fg[2:5, 2:6].all() and v.fg[100:110, 100:110].all()
    v, _ = il._sam_fg(head, ms, il.EstimatorConfig(sam_fg="gate"))
    assert v.fg[100:110, 100:110].sum() == 0 and v.fg[2:5, 2:6].all()
    # no server: no masks, the view unchanged
    monkeypatch.delenv(sam_masks.ENV_ADDR)
    sam_masks._CLIENT.clear()
    ms, info = il._sam_text(head.img, [], "", il.EstimatorConfig())
    assert ms is None and "error" in info
    v, info = il._sam_fg(head, ms, il.EstimatorConfig(sam_fg="gate"))
    assert v is head and info.get("kept_fg")


def test_series_count(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    tm = {"t": {"labels": {"rigid2": {"select_mode": {"mode": "hierarchical", "select_instance_nums": [3, 5],
                                                      "label_prefix": ["cube"]}},
                           "broom": {"select_mode": {"mode": "hierarchical"}}}}}
    monkeypatch.setattr(il, "task_map", lambda: tm)
    monkeypatch.setattr(sam_masks, "vocabulary", lambda groups, instruction="", max_names=2:
                        ["small cube"] if groups[0]["labels"][0] == "rigid2" else ["broom"])
    groups = [{"labels": ["rigid2"]}, {"labels": ["broom"]}]
    assert il.instance_range("t", groups[0]) == (3, 5, 1) and il.instance_range("t", groups[1]) is None

    def box(r, c):
        m = np.zeros((20, 20), np.uint8)
        m[r:r + 3, c:c + 3] = 1
        return sam_masks.SamMask(m, 0.9, "small cube")

    ms = [box(0, 0), box(0, 0), box(5, 5), box(10, 10), box(15, 15), sam_masks.SamMask(box(5, 0).mask, 0.9, "broom")]
    out, info = il._series_count("t", groups, ms, "", il.EstimatorConfig(series_count="sam"))
    assert out[0]["labels"] == ["rigid2"] * 4 and out[1]["labels"] == ["broom"]  # 4 distinct cube masks
    assert info["rigid2"] == {"range": [3, 5], "sam": 4, "n": 4}
    out, _ = il._series_count("t", groups, ms[:2], "", il.EstimatorConfig(series_count="sam"))
    assert len(out[0]["labels"]) == 3  # clamped to the config's minimum
    out, _ = il._series_count("t", groups, None, "", il.EstimatorConfig(series_count="min"))
    assert len(out[0]["labels"]) == 3 and groups[0]["labels"] == ["rigid2"]  # the input is not modified
    # whole category sets (classify_objects: 3 categories x 1-3 items): 4 distinct masks -> 6 objects
    tm["t"]["labels"]["rigid2"]["select_mode"].update(select_category_nums=[3, 3], select_instance_nums=[1, 3])
    assert il.instance_range("t", groups[0]) == (3, 9, 3)
    out, info = il._series_count("t", groups, ms, "", il.EstimatorConfig(series_count="sam"))
    assert info["rigid2"] == {"range": [3, 9], "sam": 4, "n": 6}


def test_vocabulary_from_groups(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    names = {"Rigid/t/00000": ["T-block", "L-shaped connector", "red piece"], "Geometry/pen_holder/00000": []}
    monkeypatch.setattr(il, "asset_info", lambda k: {"names": names.get(k, [])})
    groups = [{"candidates": ["Rigid/t/00000"]}, {"candidates": ["Geometry/pen_holder/00000"]}]
    v = sam_masks.vocabulary(groups, "Push the T onto the pad.")
    assert v == ["t-block", "l-shaped connector", "pen holder"]  # one-letter category dropped, two names max
    gp = [{"candidates": [f"Rigid/cat{i}/00000" for i in range(20)]}]
    assert sam_masks.vocabulary(gp, "Pick up the red mug by 10 cm.") == ["red mug"]  # open vocabulary: the phrase


def test_score_record_scores_series_labels(monkeypatch):
    """eval_g1 scores a numbered series (sweep_blocks' cube_k from config label rigid2) instead of dropping it."""
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import eval_g1
    from dojo.perception import inverse_layout as il

    tm = {"t": {"labels": {"rigid2": {"select_mode": {"mode": "hierarchical", "select_instance_nums": [3, 5],
                                                      "label_prefix": ["cube"]}},
                           "broom": {"select_mode": {"mode": "hierarchical"}}}}}
    monkeypatch.setattr(il, "task_map", lambda: tm)
    monkeypatch.setattr(il, "label_groups", lambda task: [{"labels": ["rigid2"]}, {"labels": ["broom"]}])
    monkeypatch.setattr(eval_g1, "footprint_axes", lambda key, q: np.array([1.0, 0.0]))
    monkeypatch.setattr(eval_g1, "gt_world_symmetry", lambda key, q: 4)
    q = [1.0, 0.0, 0.0, 0.0]

    def o(label, x, cat="small_cube"):
        return {"label": label, "type": "Rigid", "category": cat, "category_idx": 1, "pos": [x, 0.0, 0.8], "quat": q}

    gt = {"objects": [o("cube_0", 0.1), o("cube_1", 0.2), o("cube_2", 0.3), o("broom", -0.2, "broom"), o("camera_stand", 0.0, "x")]}
    assert eval_g1.series_labels("t") == {"cube": "rigid2"}
    rec = {"Rigid": {"small_cube": [{"label": "cube_0", "category_idx": 1, "default_pos": [0.2, 0.0, 0.8], "default_ori": q}],
                     "broom": [{"label": "broom", "category_idx": 1, "default_pos": [-0.2, 0.0, 0.8], "default_ori": q}]}}
    rows, n_gt, n_pred = eval_g1.score_record(rec, gt, "t", {})
    assert n_gt == 4 and n_pred == 2            # three cubes + the broom; the unlabelled fixture stays out
    cubes = [r for r in rows if r["label"] == "rigid2"]
    assert sorted(r["series_label"] for r in cubes) == ["cube_0", "cube_1", "cube_2"]
    assert sum(r["found"] for r in cubes) == 1  # one prediction (named as the layout names it) for three cubes


def test_series_labels_out(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    tm = {"t": {"labels": {"rigid2": {"select_mode": {"label_prefix": ["cube"]}},
                           "rigid0": {"select_mode": {"label_prefix": ["cat0", "cat1"]}}}}}
    monkeypatch.setattr(il, "task_map", lambda: tm)

    def r(lab, x):
        return {"label": lab, "default_pos": [x, 0.0, 0.8]}

    rec = {"Rigid": {"small_cube": [r("rigid2", 0.3), r("rigid2", 0.1)], "toy": [r("rigid0", 0.2), r("rigid0", 0.0)]}}
    series = {"rigid2": {"n": 2}, "rigid0": {"n": 2}}
    il._series_labels_out(rec, "t", series)
    assert [e["label"] for e in rec["Rigid"]["small_cube"]] == ["cube_1", "cube_0"]  # numbered by x
    assert series["rigid2"]["labels"] == ["cube_0", "cube_1"]
    assert [e["label"] for e in rec["Rigid"]["toy"]] == ["rigid0", "rigid0"]  # per-category prefixes kept
