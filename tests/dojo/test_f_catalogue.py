"""f-catalogue/1 extraction (dojo/skills/fcatalogue.py, scripts/dojo/fcat/check.py): CPU only, synthetic assets.

A made-up asset tree (RoboDojo layout: Assets/Object/RoboDojo/<Type>/<cat>/<idx>/metadata.json) and catalogue
(index.json + geom.npz) are written per test: a tray, a narrowing slotted stand, a box with a rotated slot frame, a
thin disc, a plug, a lever box (Articulation; fake USD reading), a rack with holes, a garment, a pen with a checker tag
and a hanger. Forbidden files sit next to them (a task config, the catalogue's tasks.json / info.json) and must never
be opened.
"""

from __future__ import annotations

import builtins
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dojo.skills import fcatalogue as fc

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.dojo.fcat import check as ck  # noqa: E402
from scripts.dojo.fcat import functarget_ref as FT  # noqa: E402

# ============================================================================================ synthetic meshes


def prism(poly_yz, x0, x1):
    """A convex polygon in (y, z), counter-clockwise, extruded along x: (V, F)."""
    P = np.asarray(poly_yz, float)
    n = len(P)
    V = np.vstack([np.c_[np.full(n, x0), P], np.c_[np.full(n, x1), P]])
    F = []
    for i in range(1, n - 1):
        F += [[0, i + 1, i], [n, n + i, n + i + 1]]
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    return V, np.asarray(F)


def box(x0, x1, y0, y1, z0, z1):
    return prism([(y0, z0), (y1, z0), (y1, z1), (y0, z1)], x0, x1)


def merge(*parts):
    Vs, Fs, k = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + k)
        k += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def tray_mesh(rim=0.023):
    """A 200 x 160 mm tray: a 5 mm floor, walls ``rim`` above it on three sides (open at the front, -y)."""
    t = 0.004
    return merge(box(-0.1, 0.1, -0.08, 0.08, 0.0, 0.005),
                 box(-0.1, -0.1 + t, -0.08, 0.08, 0.005, 0.005 + rim), box(0.1 - t, 0.1, -0.08, 0.08, 0.005, 0.005 + rim),
                 box(-0.1, 0.1, 0.08 - t, 0.08, 0.005, 0.005 + rim))


def stand_mesh():
    """COIN2's stand in miniature: a 2.6 mm slot 15 mm deep between two walls whose outer face narrows to the top
    (half-width 2.45 mm at the top, + 0.13 mm per mm of depth: 3.1 mm at 5 mm), on a base 11.1 mm wide (bbox
    half-width 5.55 mm). 32 mm long along x; top at z = 0.0085."""
    top, d, hi = 0.0085, 0.015, 0.0013

    def out(depth):
        return 0.00245 + 0.13 * depth
    wall_p = [(hi, top - d), (out(d), top - d), (out(0.0), top), (hi, top)]
    wall_n = [(-y, z) for y, z in wall_p][::-1]
    return merge(box(-0.016, 0.016, -0.00555, 0.00555, -0.0086, top - d), prism(wall_p, -0.016, 0.016),
                 prism(wall_n, -0.016, 0.016))


def rack_mesh():
    """A plate 6 mm thick with two 20 mm square holes (centres x = -/+0.03) on posts, over a floor 34 mm below."""
    plate = [box(-0.06, 0.06, -0.03, -0.01, 0.034, 0.04), box(-0.06, 0.06, 0.01, 0.03, 0.034, 0.04),
             box(-0.06, -0.04, -0.01, 0.01, 0.034, 0.04), box(-0.02, 0.02, -0.01, 0.01, 0.034, 0.04),
             box(0.04, 0.06, -0.01, 0.01, 0.034, 0.04)]
    return merge(box(-0.06, 0.06, -0.03, 0.03, -0.006, 0.0), *plate,
                 box(-0.06, -0.055, -0.03, 0.03, 0.0, 0.034), box(0.055, 0.06, -0.03, 0.03, 0.0, 0.034))


def bbox_md(V) -> dict:
    mn, mx = V.min(0), V.max(0)
    C = [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    return {"vertices": [list(map(float, c)) for c in C], "extents": list(map(float, mx - mn))}


UP = {"up": {"orientation": [1.0, 0.0, 0.0, 0.0]}}
Q_Y_UP = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]       # +90 deg about x: frame y = asset z (up)
Q_HINGE = [0.9871, 0.1604, 0.0, 0.0]                                    # an authored hinge orientation (18.5 deg)
CORNERS = [[-0.04, -0.001, 0.005, 1, 0, 0, 0], [-0.04, 0.001, 0.005, 1, 0, 0, 0],
           [0.04, -0.001, 0.005, 1, 0, 0, 0], [0.04, 0.001, 0.005, 1, 0, 0, 0]]


def synthetic_assets() -> dict:
    """{key: (metadata, mesh | None)}."""
    tray, stand, rack = tray_mesh(), stand_mesh(), rack_mesh()
    slotbox = merge(box(-0.05, 0.05, -0.05, -0.0015, 0.0, 0.03), box(-0.05, 0.05, 0.0015, 0.05, 0.0, 0.03),
                    box(-0.05, 0.05, -0.0015, 0.0015, 0.0, 0.005))       # a 3 mm slot along x, 25 mm deep
    disc = box(-0.014, 0.014, -0.014, 0.014, 0.0, 0.00194)
    lever = box(-0.05, 0.05, -0.04, 0.04, 0.0, 0.06)
    pen = box(-0.07, 0.07, -0.005, 0.005, 0.0, 0.01)
    region = [[0.04, 0.04, 0.005, 1, 0, 0, 0], [-0.04, 0.04, 0.005, 1, 0, 0, 0], [-0.04, -0.04, 0.005, 1, 0, 0, 0],
              [0.04, -0.04, 0.005, 1, 0, 0, 0]]
    return {
        "Geometry/tray/00000": ({"geometry": {"aligned_bbox": bbox_md(tray[0])}, "active": {"place": UP},
                                 "passive": {}}, tray),
        "Geometry/coin_stand/00000": ({"geometry": {"aligned_bbox": bbox_md(stand[0])}, "active": {"place": UP},
                                       "passive": {"support": {"coin_stand/0": {
                                           "center": [[0.0, 0.0, 0.0068, 0.0, 0.0, 0.7071, 0.7071]], "radius": [0.03]}}}},
                                      stand),
        "Geometry/slotbox/00000": ({"geometry": {"aligned_bbox": bbox_md(slotbox[0])}, "active": {"place": UP},
                                    "passive": {"functional": {
                                        "disc_slot": {"frame": [[0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0]]},
                                        "side_slot": {"frame": [[0.0, 0.0, 0.03] + Q_Y_UP]},
                                        "bottom": {"frame": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]},
                                        "tray_bottom": {"frame": region},
                                        "toast_slot1": {"frame": CORNERS}}}}, slotbox),
        "Rigid/disc/00000": ({"geometry": {"aligned_bbox": bbox_md(disc[0])}, "active": {"place": UP, "functional": {
            "center": {"frame": [[0.0, 0.0, 0.0, 0.5, -0.5, 0.5, 0.5]]},
            "disc_insert_center_same_side": {"frame": [[0.0, 0.0, 0.0, 0.5, -0.5, 0.5, 0.5]]}}}, "passive": {}}, disc),
        "Rigid/plug/00000": ({"geometry": {"aligned_bbox": bbox_md(disc[0])}, "active": {"place": UP, "functional": {
            "insert": {"frame": [[0.0, -0.014, 0.0, 0.7071067811865476, -0.7071067811865475, 0.0, 0.0]]}}},
            "passive": {}}, disc),
        "Geometry/rack/00000": ({"geometry": {"aligned_bbox": bbox_md(rack[0])}, "active": {"place": UP},
                                 "passive": {"support": {f"tube_slot/{i}": {"center": [[x, 0.0, 0.038, 1, 0, 0, 0]],
                                                                            "radius": [0.012]}
                                                         for i, x in enumerate((-0.03, 0.03))}}}, rack),
        "Articulation/lever_box/00000": ({"geometry": {"aligned_bbox": bbox_md(lever[0])}, "active": {"place": UP},
                                          "passive": {"functional": {
                                              "press_botton": {"base_link": "link_1", "parent_joint": "joint_1",
                                                               "frame": [[0.0, 0.0, 0.06, 1.0, 0.0, 0.0, 0.0]]},
                                              "lid_hinge": {"base_link": "lid", "parent_joint": "joint_2",
                                                            "frame": [[0.05, 0.0, 0.06] + Q_HINGE]},
                                              "GetParentLink": {"base_link": "lid", "parent_joint": "joint_2",
                                                                "frame": [[0.05, 0.0, 0.06, 1, 0, 0, 0]]}}}}, lever),
        "Garment/shirt/00000": ({"geometry": {"aligned_bbox": bbox_md(pen[0])}, "active": {"place": UP},
                                 "passive": {"functional": {"left_sleeve": {"id": [1, 2], "qpos": [0.0, 0.0, 0.0]}}}},
                                None),
        "Rigid/pen/00000": ({"geometry": {"aligned_bbox": bbox_md(pen[0])}, "active": {"place": UP},
                             "passive": {"functional": {"check": {"frame": [[0.07, 0.0, 0.005, 1, 0, 0, 0]]}}}}, pen),
        "Geometry/hanger/00000": ({"geometry": {"aligned_bbox": bbox_md(pen[0])}, "active": {"place": UP},
                                   "passive": {"functional": {"handle_support_point": {"frame": [
                                       [0.05, 0.0, 0.01, 0.9817933219112157, 0.0, -0.18995229151168458, 0.0]]}}}}, None),
    }


LEVER_USD = {"root_in_asset": [0.0, -0.077112, 0.003522, 1.0, 0.0, 0.0, 0.0], "root_prim": "/root/body",
             "joints": {"joint_1": {"type": "prismatic", "axis": [0.0, 0.0, -1.0], "origin": [0.0, 0.0, 0.0],
                                    "lower": 0.0, "upper": 0.02},
                        "joint_2": {"type": "revolute", "axis": [1.0, 0.0, 0.0], "origin": [0.0, 0.04, 0.06],
                                    "lower": -math.radians(110), "upper": 0.0}}}


def fake_usd(path: Path) -> dict:
    return LEVER_USD


def write_tree(tmp_path):
    assets = tmp_path / "rd" / "Assets" / "Object" / "RoboDojo"
    cat = tmp_path / "perc" / "catalog"
    index = {}
    for key, (md, mesh) in synthetic_assets().items():
        d = assets / key
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps(md))
        if key.startswith("Articulation/"):
            (d / "object.usd").write_text("#usda 1.0\n")
        if mesh is not None:
            (cat / key).mkdir(parents=True)
            np.savez_compressed(cat / key / "geom.npz", vertices=mesh[0].astype(np.float32),
                                faces=mesh[1].astype(np.int32))
            (cat / key / "info.json").write_text(json.dumps({"attributes": {"digit": 3}}))  # never to be read
            index[key] = {"extents": [1, 1, 1]}
    (cat / "index.json").write_text(json.dumps(index))
    (cat / "tasks.json").write_text("{}")                                                   # never to be read
    (tmp_path / "rd" / "task" / "config").mkdir(parents=True)
    (tmp_path / "rd" / "task" / "config" / "x.json").write_text("{}")                      # never to be read
    (assets / "Geometry" / "empty" / "00000").mkdir(parents=True)                          # a key without metadata
    return assets, cat


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return write_tree(tmp_path_factory.mktemp("fcat"))


@pytest.fixture(scope="module")
def doc(tree):
    return fc.build(tree[0], tree[1], usd_reader=fake_usd)


@pytest.fixture(scope="module")
def mds(tree, doc):
    return {k: json.loads((tree[0] / k / "metadata.json").read_text()) for k in doc["assets"]}


# ============================================================================================ schema, sources
def test_build_emits_the_spec_schema(tree, doc):
    assert fc.validate(doc) == []
    b = doc["build"]
    assert b["catalogue_hash"] == fc.sha256_file(tree[1] / "index.json")
    assert b["sources"]["n_assets"] == len(synthetic_assets()) + 1          # the empty dir is listed, not dropped
    assert b["sources"]["skipped"] == {"Geometry/empty/00000": "no metadata.json"}
    assert b["units"] == {"length": "m", "angle_revolute": "rad", "travel_prismatic": "m"}
    assert fc.conv_valid({t: tuple(v) for t, v in b["conv"].items()}) is None and set(b["conv"]) == set(fc.TYPES)
    assert len(b["script"]["sha256"]) == 64 and len(b["sources"]["listing_sha256"]) == 64
    for key, e in doc["assets"].items():
        assert set(e) >= {"root_to_bbox", "bbox", "frames", "joints_by_tag", "tag_types", "derived", "source"}
        assert all(k.startswith(("assets/", "catalog/")) for k in e["source"]["files"])
        assert f"assets/{key}/metadata.json" in e["source"]["files"]
    assert "catalog/Geometry/tray/00000/geom.npz" in doc["assets"]["Geometry/tray/00000"]["source"]["files"]
    assert "assets/Articulation/lever_box/00000/object.usd" in \
        doc["assets"]["Articulation/lever_box/00000"]["source"]["files"]


def test_build_is_deterministic_and_parallel_equal(tree, doc):
    import copy
    d2 = fc.build(tree[0], tree[1], usd_reader=fake_usd, workers=2)
    d1 = copy.deepcopy(doc)
    for d in (d1, d2):
        d["build"].pop("created_utc", None)
        d["build"].pop("wall_s", None)
    assert json.dumps(d2, sort_keys=True, default=float) == json.dumps(d1, sort_keys=True, default=float)


def test_validate_refuses_what_the_adapter_refuses(doc):
    import copy
    d = copy.deepcopy(doc)
    d["build"]["conv"]["slot"] = [2, 1, 2]
    assert any("slot" in x for x in fc.validate(d))
    d = copy.deepcopy(doc)
    del d["assets"]["Rigid/pen/00000"]["tag_types"]["check"]
    assert any("untyped" in x for x in fc.validate(d))
    d = copy.deepcopy(doc)
    d["build"]["units"]["length"] = "mm"
    assert fc.validate(d)
    assert fc.validate({"schema": "f-catalogue/0"}) == ["schema 'f-catalogue/0'"]


def test_index_must_name_the_meshes_read(tmp_path):
    assets, cat = write_tree(tmp_path)
    idx = json.loads((cat / "index.json").read_text())
    idx.pop("Geometry/tray/00000")
    (cat / "index.json").write_text(json.dumps(idx))
    with pytest.raises(ValueError, match="index.json keys differ"):
        fc.build(assets, cat, usd_reader=fake_usd, derive=False)
    d = fc.build(assets, cat, usd_reader=fake_usd, allow_index_mismatch=True, derive=False)
    assert d["build"]["sources"]["index_mismatch"]["mesh_without_index"] == ["Geometry/tray/00000"]


# ============================================================================================ legality
def test_roots_are_checked_themselves(tmp_path):
    for bad in ("task", "config", "Eval_Layout", "Traj", "coord", "dc19", "rh-misha", "scored_rows", "sealed",
                "dojo-official-rh-x", "controlr-dc19"):
        p = tmp_path / bad / "x"
        p.mkdir(parents=True)
        with pytest.raises(PermissionError):
            fc.check_root(p)
    ok = tmp_path / "Assets" / "Object" / "RoboDojo"
    ok.mkdir(parents=True)
    assert fc.check_root(ok) == ok.resolve()


def test_reader_refuses_everything_but_the_asset_files(tree):
    rd = fc.reader(tree[0], tree[1])
    for p in (tree[1] / "tasks.json", tree[1] / "Geometry/tray/00000/info.json",
              tree[0].parents[2] / "task" / "config" / "x.json"):
        with pytest.raises(PermissionError):
            fc.read_bytes(rd, p)
    fc.read_bytes(rd, tree[0] / "Geometry/tray/00000/metadata.json")
    fc.read_bytes(rd, tree[1] / "index.json")


def test_extractor_opens_nothing_outside_the_asset_files(tree, monkeypatch):
    """Every file the whole build opens (``open`` / ``Path.read_bytes``), recorded: only metadata.json, object.usd,
    index.json and geom.npz, under the two roots."""
    opened = []
    real_open, real_rb = builtins.open, Path.read_bytes

    def spy_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def spy_rb(self):
        opened.append(str(self))
        return real_rb(self)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "read_bytes", spy_rb)
    fc.build(tree[0], tree[1], usd_reader=fake_usd, derive=False)
    root = str(tree[0].parents[4])
    mine = [p for p in opened if p.startswith(root)]
    assert mine
    assert {Path(p).name for p in mine} <= {"metadata.json", "object.usd", "index.json", "geom.npz"}, \
        sorted({Path(p).name for p in mine})


def test_no_task_names_in_the_extraction_code():
    """The extraction's own code names no RoboDojo task (the hard rule; dojo/legality has the full audit)."""
    tasks = ["deposit_coin", "make_toast", "organize_table", "play_Xylophone", "hang_mugs", "store_laptop",
             "fill_egg_holder", "make_kong", "sweep_blocks", "insert_key", "build_tower", "press_by_number",
             "general_pickup", "fasten_screws"]
    for f in ("dojo/skills/fcatalogue.py", "scripts/dojo/fcat/check.py", "scripts/dojo/f_catalogue_extract.py"):
        s = (REPO / f).read_text()
        assert not [t for t in tasks if t in s], f


# ============================================================================================ tag -> type
def test_tag_types_by_grammar_and_structure(doc):
    T = {k: e["tag_types"] for k, e in doc["assets"].items()}
    assert T["Geometry/rack/00000"]["tube_slot/0"] == "hole"                 # round support entry: never a slot
    assert T["Geometry/coin_stand/00000"]["coin_stand/0"] == "support"       # support group, no feature word
    assert T["Geometry/slotbox/00000"]["disc_slot"] == "slot"                 # <kind>_slot: the kind left out
    assert T["Geometry/slotbox/00000"]["toast_slot1"] == "slot"
    assert T["Geometry/slotbox/00000"]["bottom"] == "base"
    assert T["Geometry/slotbox/00000"]["tray_bottom"] == "support"            # a bottom outlining a region
    assert T["Rigid/disc/00000"]["center"] == "slot"                          # active, mates with disc_slot
    assert T["Rigid/disc/00000"]["disc_insert_center_same_side"] is None      # Codex list
    assert T["Rigid/plug/00000"]["insert"] == "slot"
    assert T["Rigid/pen/00000"]["check"] is None and "Codex" in doc["assets"]["Rigid/pen/00000"]["tag_rules"]["check"]
    assert T["Garment/shirt/00000"]["left_sleeve"] is None                    # no frame
    assert T["Geometry/hanger/00000"]["handle_support_point"] == "hook"
    lb = T["Articulation/lever_box/00000"]
    assert lb["press_botton"] == "lever" and lb["lid_hinge"] == "hinge" and lb["GetParentLink"] is None
    assert T["Geometry/tray/00000"]["@top"] == "support" and T["Geometry/tray/00000"]["@base"] == "base"


def test_tag_words():
    assert fc.tag_words("test_tube_slot/3") == ["test", "tube", "slot"]
    assert fc.tag_words("toast_slot1") == ["toast", "slot"]
    assert fc.tag_words("GetParentLink") == ["get", "parent", "link"]
    assert fc.codex_flag("checkerboard/0") is None and fc.codex_flag("checkpoint")


def test_lever_without_a_usd_joint_is_null(tree):
    d = fc.build(tree[0], tree[1], usd_reader=lambda p: {"joints": {}}, derive=False,
                 only=["Articulation/lever_box/00000"])
    r = d["assets"]["Articulation/lever_box/00000"]
    assert r["tag_types"]["press_botton"] is None and "no USD joint" in r["tag_rules"]["press_botton"]


def test_joints_and_root_frame(doc):
    e = doc["assets"]["Articulation/lever_box/00000"]
    j = e["joints_by_tag"]["lid_hinge"]
    assert j["type"] == "revolute" and math.isclose(j["lower"], -math.radians(110)) and j["upper"] == 0.0
    assert np.allclose(j["origin"], [0.0, 0.04 + 0.077112, 0.06 - 0.003522])      # root frame
    f = e["frames"]["functional"]["lid_hinge"]["frame"][0]
    assert np.allclose(f[:3], [0.05, 0.077112, 0.06 - 0.003522]) and f[3:] == Q_HINGE  # quaternion kept bit-exact
    assert np.allclose(e["root_to_bbox"], [0.0, 0.077112, 0.03 - 0.003522])
    assert e["root_in_asset"] == LEVER_USD["root_in_asset"]


# ============================================================================================ derived geometry
def test_tray_resting_surface_23mm_below_its_rim():
    """ORG2: a drawer's tray rests 23 mm under the rim the camera sees. heightmap and slice find the floor; the hull
    (a bbox / silhouette top) misses it by the rim."""
    V, F = tray_mesh(0.023)
    hm = fc.resting_surface_heightmap(V, F)
    sl = fc.resting_surface_slice(V, F)
    hu = fc.resting_surface_hull(V, F)
    assert abs(hm["top"] - hm["z_max"] - 0.023) < 0.0005 and abs(hm["z_max"] - 0.005) < 0.0005
    assert abs(hm["rim_height"] - 0.023) < 0.0005
    assert abs(sl["z_max"] - 0.005) < 0.0005
    assert abs(hu["z_max"] - 0.028) < 1e-9                                    # the rim: 23 mm off
    assert hm["area"] > 0.02                                                  # the floor, not a wall top


def test_stand_lip_and_slot_width():
    """COIN2: the stand's outer half-width is 3.1 mm at 5 mm depth (vs its 5.55 mm bbox half-width); its slot is
    2.6 mm wide."""
    V, F = stand_mesh()
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.0068]
    o = fc.opening_fit(V, F, T, into=[0, 0, -1.0], long=[1.0, 0, 0])
    lip = {round(d, 4): w for d, w in o["lip"]}
    assert abs(lip[0.005] - 0.0031) < 1e-4
    assert abs((V[:, 1].max() - V[:, 1].min()) / 2 - 0.00555) < 1e-9
    assert abs(o["width"] - 0.0026) < 1e-4
    assert abs(o["depth"] - 0.015) < 1e-4 and o["length"] > 0.03


def test_stand_entry_measures_its_conv_and_lip(doc):
    e = doc["assets"]["Geometry/coin_stand/00000"]
    assert e["tag_conv"]["coin_stand/0"] == [1, 1, 0]                         # its frame's y is up, x along the slot
    d = e["derived"]["coin_stand/0"]
    assert abs(d["extent"]["width"] - 0.0026) < 1e-4 and d["method"] == "opening_fit" and d["params"]
    assert abs({round(a, 4): b for a, b in d["lip"]}[0.005] - 0.0031) < 1e-4


def test_base_band_depths():
    """A 15 mm pedestal 6 mm tall under a 25 mm body (square, half-sides): a 4 mm band reads the pedestal, 8 / 12 mm
    reach the body (the comparison the band depth is chosen by)."""
    V, F = merge(box(-0.015, 0.015, -0.015, 0.015, 0.0, 0.006), box(-0.025, 0.025, -0.025, 0.025, 0.006, 0.05))
    r = {b: fc.base_band(V, F, b)["radius"] for b in (0.004, 0.008, 0.012)}
    assert abs(r[0.004] - 0.015 * math.sqrt(2)) < 1e-3
    assert abs(r[0.008] - 0.025 * math.sqrt(2)) < 1e-3 and abs(r[0.012] - r[0.008]) < 1e-9


def test_rack_holes_and_slot_corners(doc):
    e = doc["assets"]["Geometry/rack/00000"]
    for tag in ("tube_slot/0", "tube_slot/1"):
        d = e["derived"][tag]
        assert abs(d["extent"]["width"] - 0.02) < 1e-3 and abs(d["extent"]["depth"] - 0.04) < 1e-3
        assert d["width_from"].startswith("mouth")                            # no walls at mid-depth: the plate
    s = doc["assets"]["Geometry/slotbox/00000"]
    assert abs(s["derived"]["disc_slot"]["extent"]["width"] - 0.003) < 1e-4
    c = s["derived"]["toast_slot1"]["corners"]
    assert abs(c["length"] - 0.08) < 1e-9 and abs(c["width"] - 0.002) < 1e-9


def test_tray_top_derived(doc):
    t = doc["assets"]["Geometry/tray/00000"]["derived"]["@top"]
    assert abs(t["below_top"] - 0.023) < 0.0005 and t["method"] == "heightmap" and t["params"]["flat_tol"] == 0.004
    assert abs(t["surface"]["rim_height"] - 0.023) < 0.0005 and len(t["frame"]) == 7


# ============================================================================================ round trip, readers, CONV
def test_round_trip_every_tag_within_1e9(doc, mds):
    for mode in ("typed", "all"):
        n = 0
        for k, e in doc["assets"].items():
            r = ck.round_trip(k, e, mds[k], mode=mode, p1=True)
            assert not r["fail"], (k, r["fail"])
            assert r["file_err"] <= 1e-9
            n += len(r["tags"])
            if mode == "all":
                assert not r["skipped"]
        assert n > 0


def test_verbatim_prototype_drops_a_hinge_orientation(doc, mds):
    """as_frames writes a lever / hinge at the identity orientation: a hinge with an authored rotation fails; patch
    p1 fixes it. A lever at the identity passes either way."""
    k = "Articulation/lever_box/00000"
    r = ck.round_trip(k, doc["assets"][k], mds[k], mode="typed", p1=False)
    assert [f for f in r["fail"] if "lid_hinge" in f] and not [f for f in r["fail"] if "press_botton" in f]


def test_readers_identical_on_round_tripped_frames(doc, mds):
    for i, (k, e) in enumerate(doc["assets"].items()):
        out = ck.reader_check(k, e, mds[k], seed=i)
        assert all(v <= 1e-9 for v in out.values()), (k, out)
    A = doc["assets"]
    for ko in ("Rigid/disc/00000", "Rigid/plug/00000"):
        for kc in ("Geometry/slotbox/00000", "Geometry/rack/00000", "Geometry/coin_stand/00000"):
            r = ck.pair_check((ko, A[ko], mds[ko]), (kc, A[kc], mds[kc]))
            assert r["slot_targets"] <= 1e-9 and r["insert_compose"] <= 1e-9, (ko, kc, r)
    r = ck.pair_check(("Rigid/disc/00000", A["Rigid/disc/00000"], mds["Rigid/disc/00000"]),
                      ("Geometry/slotbox/00000", A["Geometry/slotbox/00000"], mds["Geometry/slotbox/00000"]))
    assert r["n_slot_targets"] == 1                                           # disc 'center' on 'disc_slot'


def test_conv_pinned_by_the_semantic_axis_and_the_slot_reader(doc, mds):
    rows = [r for k, e in doc["assets"].items() for r in ck.conv_check(k, e, mds[k])]
    assert rows and all(r["axis_deg"] <= 25 for r in rows)
    slots = [r for r in rows if r["type"] == "slot"]
    assert slots and all(r["proto_axis_deg"] > 90 for r in slots if not r["override"])   # prototype: z out, not into
    assert all(r["reader_z_out_deg"] < 1e-6 and r["reader_x_along_deg"] < 1e-6 for r in slots)
    # a tag_conv override keeps the raw frame (exact round trip), which is NOT the reader's frame: the adapter must
    # hand perception-style readers the type-row frame
    side = next(r for r in slots if r["tag"] == "side_slot")
    assert side["override"] and abs(side["raw_z_out_deg"] - 90.0) < 1e-6
    assert doc["assets"]["Geometry/slotbox/00000"]["tag_conv"]["side_slot"][:2] == [1, -1]     # its y is up: into = -y
    coin = "Rigid/disc/00000"
    for a in np.linspace(0, math.pi, 5):
        assert ck.slot_reader_check(coin, mds[coin], [0, 0, -1.0], [math.cos(a), math.sin(a), 0.0], [0, 0.1, 0.2]) == 0.0
    # the prototype's slot row would build z INTO the opening; the pinned row builds it out (the reader's frame)
    t = FT.FuncTarget("s", "c", "slot", np.zeros(3), np.array([0, 0, -1.0]),
                      [{"source": "perception", "step": 0, "query": "q", "build_hash": "b"}], across=np.array([1.0, 0, 0]))
    with ck.conv_for({"slot": (2, 1.0, 0)}):
        assert t.frame()[2, 2] == -1.0
    with ck.conv_for({"slot": fc.CONV["slot"]}):
        assert t.frame()[2, 2] == 1.0


def test_vendored_prototype_is_verbatim():
    """scripts/dojo/fcat/functarget_ref.py is the partner's prototype byte for byte (20261107T2309Z-misha-functarget.py)."""
    assert fc.sha256_file(REPO / "scripts/dojo/fcat/functarget_ref.py") == \
        "679865c88ecd01e8138de9de4d8087a91f7e09ce65b7ad31662f8f051683671d"


def test_file_json_round_trips(doc):
    back = json.loads(json.dumps(doc, default=float))
    assert fc.validate(back) == [] and back["assets"].keys() == doc["assets"].keys()


def test_extraction_code_has_no_legality_findings_of_its_own():
    """The shared engine's per-function scan (task names, task-data paths, result-check provenance, task symbols,
    unlisted data-file names) over every function of the extraction code finds nothing but unresolved library calls
    (numpy, builtins). The data names the code may use are declared the way the partner's hybrid profile declares
    them (``data_files``: name -> reason): the intrinsic asset files it reads, the catalogue index it hashes, and its
    own outputs. The denied path components live in dojo/legality/roots.py, which, like the rest of the legality
    tooling, is outside the indexed code by design."""
    import ast
    from dojo.legality import audit as A
    from dojo.legality.profiles import controlr_inference
    base = controlr_inference(REPO)
    files = {"dojo.skills.fcatalogue": REPO / "dojo/skills/fcatalogue.py",
             "scripts.dojo.fcat.check": REPO / "scripts/dojo/fcat/check.py",
             "scripts.dojo.f_catalogue_extract": REPO / "scripts/dojo/f_catalogue_extract.py"}
    data_files = {**{n: "intrinsic asset file (the extraction's input)" for n in ("metadata.json", "geom.npz")},
                  "index.json": "the perception catalogue's index: hashed into catalogue_hash, keys checked vs meshes",
                  **{n: "this tool's own output" for n in ("check.json", "compare.json", ".summary.json",
                                                            "f-catalogue--.json")}}
    prof = A.Profile(name="fcat", index=list(base.index) + [files["scripts.dojo.f_catalogue_extract"],
                                                             files["scripts.dojo.fcat.check"]], roots=[],
                     root_dir=REPO, scan_globals=True, report_unresolved=True, data_files=data_files)
    idx = A.Index(prof)
    rx = A.task_re()
    bad = []
    for mod, path in files.items():
        keys = [(mod, "<module>")]
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                keys.append((mod, node.name))
            elif isinstance(node, ast.ClassDef):
                keys += [(mod, f"{node.name}.{f.name}") for f in node.body if isinstance(f, ast.FunctionDef)]
        for key in keys:
            assert key in idx.funcs, key
            bad += [f"{key}: {w}" for w in A.scan_function(idx, key, rx) if not w.startswith("unresolved")]
    assert not bad, bad
