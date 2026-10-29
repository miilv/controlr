"""Twin-trained segmenter (stream S): variant layout draws, the training id slice, the arm-pose sampler, Isaac
instance-id mapping, labels (polygons), matching, the power throttle, and the detector client/server protocol (a fake
model, no torch). No GPU, no Isaac, no weights."""
import random
import threading

import numpy as np
import pytest

from dojo.perception import seg_twin
from dojo.simlab import seg_render


# ------------------------------------------------------------------ layout draws and ids

def test_variant_subseeds_stay_in_their_block_and_never_collide():
    from dojo.simlab.gen_layouts import VARIANT_BASE, variant_subseed

    assert variant_subseed(100030, 0, 7) == 100030 * 1000 + 7          # the plain draw = the stored records
    assert variant_subseed(100030, 1, 0) == 100030 * 1000 + VARIANT_BASE
    seen = {}
    for seed in (100030, 100031):
        for v in range(0, 23):
            for att in range(40):
                s = variant_subseed(seed, v, att)
                assert seed * 1000 <= s < (seed + 1) * 1000
                assert s not in seen, (seed, v, att, seen.get(s))
                seen[s] = (seed, v, att)
    with pytest.raises(ValueError):
        variant_subseed(100030, 23, 39)       # would leave the id's 1000-block
    with pytest.raises(ValueError):
        variant_subseed(100030, 1, 40)


@pytest.mark.parametrize("spec", ["100010", "100049-100050", "100100", "100000", "100024", "100029"])
def test_render_refuses_ids_outside_the_training_slice(spec):
    with pytest.raises(SystemExit):
        seg_render.parse_ids(spec)


def test_render_ids_parse():
    assert seg_render.parse_ids("100030-100032,100049") == [100030, 100031, 100032, 100049]


def test_variant_zero_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="variant 0"):
        seg_render.main(["--task", "stack_blocks", "--ids", "100030", "--variants", "0", "--out", str(tmp_path)])


def test_extra_objects_are_training_only(tmp_path):
    from dojo.simlab.gen_layouts import extra_groups, generate

    pool = tmp_path / "pool.txt"
    pool.write_text("Rigid/bell/00000\nRigid/hoe/00000\n")
    with pytest.raises(SystemExit, match="training draws only"):
        seg_render.main(["--task", "general_pickup", "--ids", "100000", "--variants", "0", "--test",
                         "--out", str(tmp_path / "x_test"), "--extra-pool", str(pool)])
    with pytest.raises(ValueError, match="never in a measured record"):
        generate("general_pickup", 100030, variant=0, extra=["Rigid/bell/00000"])
    g = {"Rigid": [{"common": {"xlim": [-0.3, 0.3], "ylim": [-0.2, 0.0], "relative_plane": "Table"},
                    "category": [{"name": "cube"}], "select_mode": {"nums": 1, "label": ["target"]}}]}
    out = extra_groups(g, ["Rigid/bell/00000", "Rigid/hoe/00003"])
    assert len(g["Rigid"]) == 1 and len(out["Rigid"]) == 3                      # the input is not modified
    assert out["Rigid"][2]["category"] == [{"name": "hoe", "index": [3]}]
    assert out["Rigid"][2]["select_mode"]["label"] == ["seg_extra1"] and out["Rigid"][2]["common"]["xlim"] == [-0.3, 0.3]


def test_extra_assets_cycle_the_pool_evenly():
    pool = [f"Rigid/a/{i:05d}" for i in range(7)]
    seen = [x for v in (1, 2) for s in seg_render.TRAIN_IDS for x in seg_render.extra_assets(pool, s, v, 3)]
    counts = [seen.count(p) for p in pool]
    assert max(counts) - min(counts) <= 1 and len(seen) == 120
    assert seg_render.extra_assets([], 100030, 1, 3) == [] and len(seg_render.extra_assets(pool[:2], 100030, 1, 3)) == 2


# ------------------------------------------------------------------ arm poses

def test_sample_moves_is_seeded_and_covers_the_kinds():
    objs = [{"pos": [-0.2, -0.1, 0.8], "label": "a", "inst_name": "a_0"},
            {"pos": [0.25, -0.05, 0.8], "label": "b", "inst_name": "b_1"}]
    a = seg_render.sample_moves(random.Random(5), objs, 7, 0.765)
    b = seg_render.sample_moves(random.Random(5), objs, 7, 0.765)
    assert a == b
    assert {m["kind"] for m in a} == {"hover", "look", "both"}
    for m in a:
        assert m["grip"] in (0.0, 0.5, 1.0)
        if m["kind"] == "hover":
            assert 0.06 <= m["height"] <= 0.16 and m["target_inst"] in ("a_0", "b_1")
        if m["kind"] == "both":
            assert m["tcp"]["left"][0] < m["tcp"]["right"][0]
            assert 0.765 + 0.12 <= m["tcp"]["left"][2] <= 0.765 + 0.30
    # no task objects: only both-arm moves
    assert {m["kind"] for m in seg_render.sample_moves(random.Random(1), [], 5, 0.765)} == {"both"}


# ------------------------------------------------------------------ Isaac instance ids -> our codes

def test_map_instance_ids_by_prim_prefix():
    objects = [{"prim_path": "/World/envs/env_0/rigid/block/block_0_1"},
               {"prim_path": "/World/envs/env_0/rigid/block/block_0_10"},
               {"prim_path": None}]
    robot = {"left": {"link_prims": {"link1": "/World/envs/env_0/robot0/link1"}},
             "right": {"link_prims": {"link1": "/World/envs/env_0/robot1/link1"}}}
    id2l = {"0": "BACKGROUND", "1": "UNLABELLED",
            "5": "/World/envs/env_0/rigid/block/block_0_1/geometry/mesh",
            "6": "/World/envs/env_0/rigid/block/block_0_10/geometry/mesh",      # not block_0_1's child
            "7": "/World/envs/env_0/robot1/link6/visuals/mesh_0",
            "8": "/World/envs/env_0/Table/mesh"}
    m = seg_render.map_instance_ids(id2l, objects, robot)
    assert m == {0: 0, 1: 0, 5: 1, 6: 2, 7: seg_render.ROBOT_CODE, 8: seg_render.OTHER_CODE}
    data = np.array([[0, 5, 6], [7, 8, 99]], np.uint32)
    assert seg_render.relabel(data, m).tolist() == [[0, 1, 2], [seg_render.ROBOT_CODE, seg_render.OTHER_CODE, 0]]
    assert seg_render.ROBOT_CODE == seg_twin.ROBOT_CODE and seg_render.OTHER_CODE == seg_twin.OTHER_CODE


# ------------------------------------------------------------------ labels and metrics

def test_instances_drop_small_and_special_codes():
    c = np.zeros((20, 20), np.uint16)
    c[0:5, 0:5] = 1                    # 25 px
    c[10:12, 10:12] = 2                # 4 px: below min_px
    c[15:20, 15:20] = seg_twin.ROBOT_CODE
    inst = seg_twin.instances(c, min_px=10)
    assert list(inst) == [0] and inst[0].sum() == 25


def test_mask_polygon_joins_fragments():
    cv2 = pytest.importorskip("cv2")
    m = np.zeros((60, 80), np.uint8)
    m[10:30, 10:30] = 1
    m[35:50, 50:70] = 1                # an occluded object seen in two pieces
    poly = seg_twin.mask_polygon(m)
    assert poly is not None and poly.shape[1] == 2
    back = np.zeros_like(m)
    cv2.fillPoly(back, [np.round(poly).astype(np.int32)], 1)
    assert seg_twin.iou(back, m) > 0.85
    line = seg_twin.yolo_line(3, poly, 80, 60).split()
    assert line[0] == "3" and len(line) == 1 + 2 * len(poly)
    xy = np.array(line[1:], float)
    assert xy.min() >= 0 and xy.max() <= 1
    assert seg_twin.mask_polygon(np.zeros((5, 5), np.uint8)) is None


def test_match_greedy_is_one_to_one():
    a = np.zeros((10, 10), np.uint8)
    a[:5] = 1
    b = np.zeros((10, 10), np.uint8)
    b[5:] = 1
    preds = [seg_twin.Det(a.copy(), "Rigid/x/00000", 0.9), seg_twin.Det(a.copy(), "Rigid/y/00000", 0.8)]
    m = seg_twin.match_greedy({0: a.astype(bool), 1: b.astype(bool)}, preds)
    assert m[0][0] in (0, 1) and m[0][1] == 1.0
    assert 1 not in m                   # the second prediction overlaps object 0 only
    assert seg_twin.category_of("Rigid/bottle/00041") == "bottle"
    assert seg_twin.asset_key({"type": "Rigid", "category": "bottle", "category_idx": 41}) == "Rigid/bottle/00041"


# ------------------------------------------------------------------ power throttle

def test_throttle_backs_off_above_target_and_stalls_above_max(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "seg_train", Path(__file__).resolve().parents[2] / "scripts/perception/seg/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    power = [350.0]
    slept = []
    thr = mod.Throttle(270, 420, tmp_path / "p.csv", sleep0=0.1, power_fn=lambda: power[0], sleep_fn=slept.append)
    t = 1000.0
    for _ in range(10):
        thr.step(now=t)
        t += 1
    high = thr.sleep
    assert high > 0.5
    power[0] = 150.0
    for _ in range(60):
        thr.step(now=t)
        t += 1
    assert thr.sleep < high / 4
    power[0] = 500.0                    # over the hard cap: stalls (1-s sleeps) until it gives up after 30 s
    slept.clear()
    thr.step(now=t)
    assert slept.count(1.0) == 30
    assert (tmp_path / "p.csv").read_text().startswith("t,power_w")


# ------------------------------------------------------------------ detector service (fake model)

class _FakeSeg:
    def handle(self, req, img):
        from dojo.perception.sam_masks import pack

        m = np.zeros(img.shape[:2], bool)
        m[2:6, 3:9] = True
        return {"dets": [(*pack(m), "Rigid/block/00003", 0.87, [3.0, 2.0, 9.0, 6.0])]}

    def info(self):
        return {"fake": True}


def test_client_server_roundtrip(tmp_path, monkeypatch):
    from dojo.perception import sam_masks

    sock = str(tmp_path / "seg.sock")
    ready = threading.Event()
    threading.Thread(target=sam_masks.serve, args=(sock, ["seg"], 0.01), kwargs={"M": _FakeSeg(), "ready": ready},
                     daemon=True).start()
    assert ready.wait(5)
    monkeypatch.setenv(seg_twin.ENV_ADDR, sock)
    seg_twin._CLIENT.clear()
    c = seg_twin.client()
    assert c is not None
    img = np.zeros((10, 12, 3), np.uint8)
    dets = c.detect(img)
    assert len(dets) == 1 and dets[0].key == "Rigid/block/00003" and dets[0].mask.sum() == 24
    assert dets[0].box == (3.0, 2.0, 9.0, 6.0)
    seg_twin._CLIENT.clear()


def test_client_none_only_without_an_address_and_loud_when_set_but_down(monkeypatch, tmp_path):
    from dojo.perception.sam_masks import SideServerDown

    monkeypatch.delenv(seg_twin.ENV_ADDR, raising=False)
    assert seg_twin.client() is None
    monkeypatch.setenv(seg_twin.ENV_ADDR, str(tmp_path / "nobody.sock"))
    with pytest.raises(SideServerDown, match="unreachable"):
        seg_twin.client()
    seg_twin._CLIENT.clear()


# ------------------------------------------------------------------ estimator hook (seg_propose)

def test_seg_propose_default_add_and_noop_without_server(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    assert il.EstimatorConfig().seg_propose == "add"     # "off" until 2026-10-28; a no-op without a server
    monkeypatch.delenv(seg_twin.ENV_ADDR, raising=False)
    info = il._seg_propose([], {"cam_head": None}, np.zeros((4, 4, 3), np.uint8), il.EstimatorConfig(seg_propose="on"),
                           [], [], {})
    assert info["error"] == "no segmenter server" and info["n_dets"] == 0


def test_seg_groups_by_asset_then_category():
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    pending = [(0, {"candidates": ["Rigid/block/00001", "Rigid/block/00002"]}, {}),
               (1, {"candidates": ["Rigid/bottle/00041"]}, {}),
               (2, {"candidates": ["Rigid/block/00002"]}, {})]
    assert il._seg_groups("Rigid/block/00002", pending) == [0, 2]
    assert il._seg_groups("Rigid/bottle/00050", pending) == [1]       # an unseen bottle: by category
    assert il._seg_groups("Rigid/shoe/00001", pending) == []
    m = np.zeros((40, 50), np.uint8)
    m[10:20, 5:25] = 1
    known = np.ones_like(m)
    known[10:12] = 0
    b = il._seg_blob(m, known)
    assert b.area == 160 and b.bbox == (5, 12, 20, 8) and np.allclose(b.centroid, [15.0, 16.0])
    assert il._seg_blob(np.zeros_like(m), known) is None


def test_repeat_factor_sampling():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "seg_export", Path(__file__).resolve().parents[2] / "scripts/perception/seg/export_yolo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    imgs = [{"common"}] * 98 + [{"common", "rare"}, set()]
    r = mod.repeat_factors(imgs, 0.1)
    assert r[0] == 1.0 and r[-1] == 1.0                       # frequent class / empty image: once
    assert abs(r[98] - (0.1 / 0.01) ** 0.5) < 1e-9            # the rare class (1 % of images): sqrt(t / f)
    assert mod.repeat_factors(imgs, 0.0) == [1.0] * 100
    f = {"img": "/d/render_v3reh/build_tower/100030_v1/frame_2_cam_head.png", "task": "build_tower",
         "layout": "100030_v1", "frame": 2, "cam": "cam_head"}
    assert mod.frame_stem(f) == "render_v3reh__build_tower__100030_v1__f2__cam_head"   # roots never collide
    fr = [{"objs": [("Rigid/watch/00001", [], 9, False), ("Rigid/watch/00005", [], 9, False)]},
          {"objs": [("Rigid/watch/00001", [], 9, False), ("Rigid/car/00012", [], 9, False)]},
          {"objs": []}]
    kept, out_of = mod.narrow_frames(fr, {"Rigid/watch/00001", "Rigid/watch/00005"})
    assert kept == [fr[0], fr[2]] and out_of == {"Rigid/car/00012": 1}   # a look-alike is never unlabelled background


def test_seg_model_reads_its_training_size(monkeypatch):
    """A model trained at 960 px infers at 960 unless told otherwise (no torch: a stub YOLO)."""
    import sys
    import types

    class FakeYOLO:
        def __init__(self, w):
            self.names = {0: "Rigid/block/00001"}
            self.ckpt = {"train_args": {"imgsz": 960}}

    fake_ul = types.ModuleType("ultralytics")
    fake_ul.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ul)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert seg_twin._SegModel("w.pt").imgsz == 960
    assert seg_twin._SegModel("w.pt", imgsz=640).imgsz == 640


def test_seg_dedup_keeps_one_detection_per_object():
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    a = np.zeros((20, 20), np.uint8)
    a[2:8, 2:8] = 1
    b = a.copy()
    b[2:8, 8] = 1                       # the same block, a slightly larger mask, another class
    c = np.zeros_like(a)
    c[12:18, 12:18] = 1
    dets = [seg_twin.Det(b, "Rigid/block/00014", 0.70), seg_twin.Det(a, "Rigid/block/00015", 0.86),
            seg_twin.Det(c, "Rigid/block/00005", 0.91)]
    kept, n = il._seg_dedup(dets, 0.7)
    assert n == 1 and [d.key for d in kept] == ["Rigid/block/00005", "Rigid/block/00015"]
    assert kept[1].alts == [("Rigid/block/00014", 0.7)]
    assert il._seg_dedup(dets, 0.0)[1] == 0


def test_held_candidates_skip_covered_and_other_categories():
    objs = [{"inst_name": "a", "pos": [0.0, 0.0, 0.78], "type": "Rigid", "category": "mahjong", "clutter": False},
            {"inst_name": "b", "pos": [0.005, 0.0, 0.80], "type": "Rigid", "category": "mahjong", "clutter": False},
            {"inst_name": "c", "pos": [0.2, 0.0, 0.78], "type": "Rigid", "category": "mahjong", "clutter": False},
            {"inst_name": "d", "pos": [0.3, 0.0, 0.78], "type": "Rigid", "category": "block", "clutter": False},
            {"inst_name": "e", "pos": [0.4, 0.0, 0.78], "type": "Geometry", "category": "mahjong", "clutter": False}]
    got = [o["inst_name"] for o in seg_render.held_candidates(objs, "mahjong")]
    assert got == ["b", "c"]               # "a" is under "b"; "d" is another category; "e" is a fixture
    assert [o["inst_name"] for o in seg_render.held_candidates(objs, None)] == ["b", "c", "d"]


def test_test_mode_takes_only_held_out_ids_and_a_test_dir(tmp_path):
    assert seg_render.parse_ids("100000-100002,100024", test=True) == [100000, 100001, 100002, 100024]
    for spec in ("100030", "100010", "100025", "100100"):
        with pytest.raises(SystemExit):
            seg_render.parse_ids(spec, test=True)
    with pytest.raises(SystemExit, match="test"):
        seg_render.main(["--task", "make_kong", "--ids", "100000", "--variants", "0", "--test", "--out", str(tmp_path / "render")])
    with pytest.raises(SystemExit, match="test"):
        seg_render.main(["--task", "make_kong", "--ids", "100000", "--variants", "1", "--test", "--out", str(tmp_path / "render_test")])


def test_rot_between_maps_a_onto_b():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = seg_render._rot_between(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    a = np.array([0.0, 0.0, 1.0])
    R = seg_render._rot_between(a, -a)            # the antiparallel case
    assert np.allclose(R @ a, -a) and np.isclose(np.linalg.det(R), 1.0)
