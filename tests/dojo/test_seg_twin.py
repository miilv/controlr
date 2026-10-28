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


def test_client_is_none_without_server(monkeypatch, tmp_path):
    monkeypatch.delenv(seg_twin.ENV_ADDR, raising=False)
    assert seg_twin.client() is None
    monkeypatch.setenv(seg_twin.ENV_ADDR, str(tmp_path / "nobody.sock"))
    with pytest.warns(UserWarning, match="unreachable"):
        assert seg_twin.client() is None
    seg_twin._CLIENT.clear()


# ------------------------------------------------------------------ estimator hook (seg_propose)

def test_seg_propose_default_off_and_noop_without_server(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    assert il.EstimatorConfig().seg_propose == "off"
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
