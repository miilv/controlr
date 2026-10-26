"""held_look verdict rules (jaw opening, absence cap) and refine's support test (pure python + numpy)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from dojo.perception import j1_adapter as J  # noqa: E402
from dojo.perception.refine import _under  # noqa: E402


def test_opening_wider_than_command_is_held_even_when_the_colour_test_sees_pads():
    # 5.1 stack 100004: a dark block lifted 77 mm, colour test "pads" (not held at 1.0 before)
    held, conf = J.held_from_opening(0.55, 0.298, held_px=False, dark_like=True)
    assert held and conf >= 0.7
    held, conf = J.held_from_opening(0.65, 0.344, held_px=False, dark_like=True)  # 100008
    assert held and conf == 1.0


def test_empty_closed_jaw_is_not_held_and_commands_below_the_floor_clip():
    held, conf = J.held_from_opening(0.15, 0.0, held_px=False, dark_like=True)
    assert not held and conf == pytest.approx(0.675)
    assert J.held_from_opening(0.15, 0.1, False, True) == J.held_from_opening(0.15, 0.0, False, True)


def test_disagreement_caps_the_confidence():
    held, conf = J.held_from_opening(J.HELD_CLOSED_FLOOR + 0.15, 0.0, held_px=False, dark_like=False)
    assert held and conf <= 0.6  # opening says held (gap 0.15), colour says pads, near the threshold
    held, conf = J.held_from_opening(J.HELD_CLOSED_FLOOR, 0.0, held_px=True, dark_like=True)
    assert not held and conf <= 0.4  # jaws at the command, but an object-coloured dark region


def test_colour_fallback_absence_is_capped():
    held, conf, ev = J.held_from_colour(0.05, dark_like=True)
    assert not held and conf <= J.HELD_ABSENCE_CAP and ev == "absent"
    held, conf, ev = J.held_from_colour(0.95, dark_like=False)
    assert held and conf == 1.0 and ev == "colour"


def _box(cx, cy, cz, s=0.03):
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s / 2
    return SimpleNamespace(Vs=c), [cx, cy, cz, 1.0, 0.0, 0.0, 0.0]


def test_under_is_a_support_below_with_overlapping_footprint():
    a, p_low = _box(0, 0, 0.78)
    _, p_top = _box(0.005, 0, 0.81)
    assert _under(a, p_low, a, p_top)          # the block of a stack below the target
    assert not _under(a, p_top, a, p_low)      # the one above is not a support
    _, p_side = _box(0.05, 0, 0.78)
    assert not _under(a, p_side, a, p_top)     # no footprint overlap
    _, p_same = _box(0.02, 0, 0.78)
    assert not _under(a, p_same, a, p_low)     # same level


def test_rests_on_needs_contact_and_overlap():
    from dojo.perception.refine import _rests_on

    a, p_low = _box(0, 0, 0.78)
    _, p_top = _box(0.002, 0, 0.81)
    assert _rests_on(a, p_top, a, p_low)          # the next block of a stack covers the lower one
    _, p_hover = _box(0, 0, 0.86)
    assert not _rests_on(a, p_hover, a, p_low)    # a block hovering in the jaws above does not
    _, p_edge = _box(0.027, 0, 0.81)
    assert not _rests_on(a, p_edge, a, p_low)     # 10 % footprint overlap: not covering
    assert not _rests_on(a, p_low, a, p_top)


def test_support_z_is_the_highest_known_top_under_the_footprint(monkeypatch):
    from dojo.perception import render, refine

    a, p_low = _box(0, 0, 0.78)          # table-resting block: top 0.795
    _, p_mid = _box(0.001, 0, 0.81)      # on it: top 0.825
    _, p_far = _box(0.2, 0, 0.78)
    monkeypatch.setattr(render, "load_asset", lambda key: a)
    _, p_float = _box(0.002, 0, 0.896)   # a carried-height belief after a release (5.1 stack 100007 s142)
    assert refine.support_z(a, p_float, [("b", p_low), ("b", p_mid), ("b", p_far)]) == pytest.approx(0.825)
    assert refine.support_z(a, p_float, [("b", p_far)]) == pytest.approx(refine.TABLE_Z)
    _, p_under = _box(0.002, 0, 0.80)    # an "other" above the target's bottom is not its support
    assert refine.support_z(a, p_under, [("b", p_mid)]) == pytest.approx(refine.TABLE_Z)


def _fake_ref(key, M=None):
    return np.asarray({"red": [0.8, 0.1, 0.1], "green": [0.1, 0.7, 0.2], "blue": [0.1, 0.2, 0.8]}[key], float)


def test_identity_rank_pick_and_twins():
    from dojo.perception import identity

    cands = [{"label": "block_0", "key": "red"}, {"label": "block_1", "key": "green"}, {"label": "block_2", "key": "blue"}]
    rk = identity.rank([0.15, 0.65, 0.2], cands, ref=_fake_ref)
    assert rk[0]["label"] == "block_1"
    p = identity.pick(rk)
    assert p["label"] == "block_1" and p["key"] == "green" and p["margin"] > 0.15
    twins = cands + [{"label": "block_3", "key": "green"}]
    p2 = identity.pick(identity.rank([0.15, 0.65, 0.2], twins, ref=_fake_ref))
    assert p2["label"] is None and p2["key"] == "green"  # same key: identity by appearance impossible
    assert p2["margin"] == p["margin"]                     # margin among distinct keys only
    assert identity.rank(None, cands, ref=_fake_ref) == [] and identity.pick([]) is None


def test_identity_asked_verdict():
    from dojo.perception import identity

    asked = {"label": "block_0", "key": "red"}
    assert identity.asked_verdict(None, asked, False) is False
    assert identity.asked_verdict({"key": "red", "conf": 0.3, "margin": 0.01}, asked, True) is True
    assert identity.asked_verdict({"key": "green", "conf": 0.9, "margin": 0.3}, asked, True) is False
    assert identity.asked_verdict({"key": "green", "conf": 0.6, "margin": 0.3}, asked, True) is None
    assert identity.asked_verdict(None, asked, None) is None


def test_catalog_root_treats_unreadable_dirs_as_absent_and_takes_a_list(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from dojo.perception import render

    good = tmp_path / "cat"
    good.mkdir()
    (good / "index.json").write_text("{}")
    real_exists = Path.exists

    def exists(self, *a, **k):  # non-root probing /root/... (compute3: a colon-list DOJO_CATALOG fell through to it)
        if str(self).startswith("/root"):
            raise PermissionError(13, "Permission denied", str(self))
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setenv("DOJO_CATALOG", os.pathsep.join([str(tmp_path / "missing"), "/root/dojo-data/x", str(good)]))
    assert render.catalog_root() == good
    monkeypatch.setenv("DOJO_CATALOG", "")
    monkeypatch.setattr(render, "CATALOG_FALLBACKS", ("/root/dojo-data/perception/catalog", str(tmp_path / "none")))
    with pytest.raises(FileNotFoundError, match="not found in"):
        render.catalog_root()
    assert render.path_exists("/root/anything") is False


def test_look_alike_changes_the_size_not_the_texture(monkeypatch):
    from dojo.perception import refine

    idx = {f"Rigid/block/{i:05d}": {"extents": [s, s, s], "mean_color": c}
           for i, s, c in ((5, 0.03, [0.53, 0.34, 0.2]), (6, 0.035, [0.53, 0.34, 0.2]), (7, 0.04, [0.53, 0.34, 0.2]),
                           (14, 0.03, [0.82, 0.65, 0.45]), (15, 0.035, [0.82, 0.65, 0.45]), (16, 0.04, [0.82, 0.65, 0.45]))}
    monkeypatch.setattr(refine, "_catalog_index", lambda: idx)
    # 5.1 stack 100006: the 40 mm class representative is texture #1 (00007); the block is texture #4
    assert refine.look_alike("Rigid/block/00015", "Rigid/block/00007") == "Rigid/block/00016"
    assert refine.look_alike("Rigid/block/00015", "Rigid/block/00005") == "Rigid/block/00014"
    assert refine.look_alike("Rigid/block/00015", "Rigid/block/00099") == "Rigid/block/00099"  # unknown: unchanged


def test_refine_identity_changes_identity_only_on_a_good_fit(monkeypatch):
    from dojo.perception import refine

    fits = {"Rigid/block/00002": {"residual": 0.30, "iou": 0.66, "ok": True, "pose": [0, 0, 0.78, 1, 0, 0, 0]},
            "Rigid/block/00003": {"residual": 0.20, "iou": 0.67, "ok": True, "pose": [0, 0, 0.79, 0.7, 0.7, 0, 0]}}
    monkeypatch.setattr(refine, "refine", lambda k, *a, **kw: dict(fits[k]))
    monkeypatch.setattr(refine, "look_alike", lambda cur, rep: rep)
    # 5.1 build_tower s0L9: pillar -> prism at iou 0.67 (footprints alike from above): rejected, the pillar's own fit
    r = refine.refine_identity(["Rigid/block/00002", "Rigid/block/00003"], None, {}, [0, 0, 0.78, 1, 0, 0, 0])
    assert r["obj_id"] == "Rigid/block/00002" and r["identity_ok"] is False and r["identity_rival"] == "Rigid/block/00003"
    assert r["pose"] == fits["Rigid/block/00002"]["pose"]
    fits["Rigid/block/00003"]["iou"] = 0.95
    r = refine.refine_identity(["Rigid/block/00002", "Rigid/block/00003"], None, {}, [0, 0, 0.78, 1, 0, 0, 0])
    assert r["obj_id"] == "Rigid/block/00003" and r["identity_ok"] is True


def test_image_edges_see_a_white_block_on_light_wood():
    import cv2

    from dojo.perception.refine import image_edges

    img = np.zeros((120, 160, 3), np.uint8)
    img[:] = (236, 222, 180)                      # light wood (RGB)
    img[40:80, 60:100] = (228, 228, 226)          # a white pillar: about the same grey level, no yellow
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    assert abs(int(grey[60, 80]) - int(grey[10, 10])) < 15
    assert cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120).sum() == 0   # grey-only Canny: nothing
    E = image_edges(img)
    assert E[38:43, 60:100].any() and E[40:80, 58:63].any()                  # chroma edges: the outline


def test_elevated_hypothesis_needs_two_views(monkeypatch):
    from types import SimpleNamespace

    from dojo.perception import inverse_layout as IL

    pil = "Rigid/block/00002"
    on_pillar = IL.Hypothesis(0, pil, "up", 0.19, -0.09, 0.0, 0.3, {"z_off": 0.0378, "stacked_on": pil})
    on_board = IL.Hypothesis(0, pil, "up", 0.0, -0.3, 0.0, 0.3, {"z_off": 0.02, "stacked_on": "Rigid/block/00004"})
    table = IL.Hypothesis(0, pil, "up", 0.2, -0.1, 0.0, 0.3, {})
    ident, every, off = (SimpleNamespace(elevated_multiview=m) for m in ("identical", "all", "off"))
    assert IL._needs_multiview(on_pillar, pil, ident) and not IL._needs_multiview(on_board, "Rigid/block/00004", ident)
    assert IL._needs_multiview(on_board, "Rigid/block/00004", every) and not IL._needs_multiview(table, None, every)
    assert not IL._needs_multiview(on_pillar, pil, off)
    # the tower scan phantom: one view (the scan head view) supports it, frame 0 refutes it
    monkeypatch.setattr(IL, "elevated_support", lambda h, o, v: {"support": 1, "refute": 1, "views": {}})
    assert not IL._multiview_ok(on_pillar, [], {}, ident, pil)
    monkeypatch.setattr(IL, "elevated_support", lambda h, o, v: {"support": 2, "refute": 0, "views": {}})
    assert IL._multiview_ok(on_pillar, [], {}, ident, pil)
    assert IL._multiview_ok(on_board, [], {}, ident, "Rigid/block/00004")  # not identical: not checked


def test_support_slot_pose_reproduces_the_layout_generator(monkeypatch):
    from types import SimpleNamespace

    from dojo.perception import inverse_layout as IL

    sup = {"bread_shelf/0": {"center": [[0.035, 0.0, 0.035, 0.7071067811865476, 0.0, 0.7071067811865475, 0.0]],
                             "radius": [0.05]}}
    infos = {"Geometry/bread_shelf/00000": {"support": sup},
             "Rigid/bread/00000": {"place": {"up": {"center": [0, 0, 0, 1, 0, 0, 0]}}}}
    monkeypatch.setattr(IL, "load_asset", lambda k: SimpleNamespace(info=infos[k]))
    monkeypatch.setattr(IL, "_settled", lambda b, s_, c: None)  # the generator's (drop) pose; the settled table below
    pl = {"xlim": [-0.02, -0.02], "ylim": [-0.007, -0.007], "rotate_rand": False,
          "relative_plane": "bread_shelf/bread_shelf/0", "place_tag": "up"}
    assert IL.support_slot(pl) == ("bread_shelf", "bread_shelf/0")
    assert IL.support_slot({**pl, "xlim": [-0.05, 0.05]}) is None and IL.support_slot({**pl, "relative_plane": "Table"}) is None
    # make_toast 100000 (our dev layout): shelf at (-0.14, -0.02, 0.803), bread_0 at z 0.858 = 0.803 + 0.035 + 0.02
    pos, q = IL.support_slot_pose("Geometry/bread_shelf/00000", [-0.14, -0.02, 0.803], [1, 0, 0, 0], "Rigid/bread/00000", pl)
    assert np.allclose(pos, [-0.14 + 0.035, -0.02 - 0.007, 0.858], atol=1e-6)
    assert np.allclose(q, [0.7071067811865476, 0.0, 0.7071067811865475, 0.0], atol=1e-6)  # the slice stands up
    assert IL.support_slot_pose("Rigid/bread/00000", [0, 0, 0], [1, 0, 0, 0], "Rigid/bread/00000", pl) is None
    # with the settled table: where physics puts the slice (16-17 mm lower in the slot)
    table = {"Geometry/bread_shelf/00000|bread_shelf/0|Rigid/bread/00000": {"pos": [0.0358, -0.0023, 0.0382],
                                                                          "quat": [0.7071, 0, 0.7071, 0]}}
    monkeypatch.setattr(IL, "_settled", lambda b, s_, c: table.get(f"{b}|{s_}|{c}"))
    pos, _ = IL.support_slot_pose("Geometry/bread_shelf/00000", [-0.14, -0.02, 0.803], [1, 0, 0, 0], "Rigid/bread/00000", pl)
    assert np.allclose(pos, [-0.1042, -0.0223, 0.8412], atol=1e-6)


def test_slot_settle_table_is_keyed_by_asset_pairs_only():
    import json as _json
    import re
    from pathlib import Path as _P

    import dojo.perception as P
    d = _json.loads((_P(P.__file__).parent / "slot_settle.json").read_text())
    pat = re.compile(r"^(Rigid|Geometry|Articulation)/[a-z_]+/\d{5}\|[^|]+\|(Rigid|Geometry|Articulation)/[a-z_]+/\d{5}$")
    keys = [k for k in d if not k.startswith("_")]
    assert keys and all(pat.match(k) for k in keys)  # (base asset, support frame, object asset): no task, no layout
