"""make_kong reads from the images (stream S; ``RuntimeConfig.kong_draw_read`` / ``Skills.kong_draw_read``, default off):
the tile identity a view's read gives the belief (merge + persistence), the read-only refine kind, the drawn tile's
standing orientation, the skill-side read helper, and G3's GT answer to a read. No GPU, no Isaac."""
from types import SimpleNamespace

import numpy as np
import pytest


def _pol():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    pol = RuntimePolicy(RuntimeConfig())
    pol.report = {"events": []}
    pol.bridge = SimpleNamespace(steps_used=70)
    return pol


def test_a_tile_read_sets_a_placeholder_confirms_and_drops_on_disagreement():
    pol = _pol()
    o = SimpleNamespace(category="mahjong", idx=3, obj_type="Rigid", identity_known=False,
                        asset=SimpleNamespace(key="Rigid/mahjong/00003"), digit=None, digit_conf=None, digit_source=None)
    prov = SimpleNamespace(belief=SimpleNamespace(objects={"mahjong5_0": o}), assets=None)
    pol._merge_glyph(prov, "mahjong5_0", {"read": True, "name": "Rigid/mahjong/00017", "margin": 0.7, "view": "cam_head"})
    assert o.identity_known and o.asset.key == "Rigid/mahjong/00017" and o.idx == 17
    assert pol._glyph_reads["mahjong5_0"]["asset"] == "Rigid/mahjong/00017"            # kept by later records
    ev = pol.report["events"][-1]
    assert ev["kind"] == "glyph_identity" and ev["why"] == "read" and ev["view"] == "cam_head"
    pol._merge_glyph(prov, "mahjong5_0", {"read": True, "name": "Rigid/mahjong/00017", "margin": 0.9})
    assert o.identity_known and pol.report["events"][-1]["why"] == "confirmed"
    pol._merge_glyph(prov, "mahjong5_0", {"read": True, "name": "Rigid/mahjong/00002", "margin": 0.9})
    assert not o.identity_known and pol.report["events"][-1]["why"] == "disagree"
    assert "mahjong5_0" not in pol._glyph_reads
    n = len(pol.report["events"])
    pol._merge_glyph(prov, "mahjong5_0", {"read": False, "name": "Rigid/mahjong/00002", "why": "low_margin"})
    pol._merge_glyph(prov, "mahjong5_0", {"read": True, "name": "Rigid/block/00002"})       # another category: ignored
    assert len(pol.report["events"]) == n and not o.identity_known


def test_a_persisted_tile_read_survives_a_later_record_without_one():
    from dojo.perception.read_persist import persist_reads
    pol = _pol()
    o = SimpleNamespace(category="mahjong", idx=3, obj_type="Rigid", identity_known=False,
                        asset=SimpleNamespace(key="Rigid/mahjong/00003"))
    prov = SimpleNamespace(belief=SimpleNamespace(objects={"mahjong5_0": o}), assets=None)
    pol._merge_glyph(prov, "mahjong5_0", {"read": True, "name": "Rigid/mahjong/00017", "margin": 0.7})
    later = {"Rigid": {"mahjong": [{"label": "mahjong5_0", "category_idx": 3, "_perception": {"identity": "unknown"}}]}}
    rec, _, evs = persist_reads(later, pol._glyph_reads)
    r = rec["Rigid"]["mahjong"][0]
    assert r["category_idx"] == 17 and r["_perception"]["identity"] == "read" and evs[0]["why"] == "carried"


def _tile(face_body=(0.0, 0.0, 1.0)):
    from dojo.skills.geometry import R_to_quat
    from dojo.skills.world import AssetInfo
    # a mahjong tile: 30 x 40 x 22 mm, its glyph on local +z (the "up" place pose = identity)
    info = AssetInfo(key="Rigid/mahjong/00001", bbox_min=np.array([-0.015, -0.02, -0.011]),
                     bbox_max=np.array([0.015, 0.02, 0.011]), place={"up": R_to_quat(np.eye(3)).tolist()})
    return SimpleNamespace(asset=info)


def test_facing_R_stands_the_tile_on_its_end_with_the_glyph_to_the_robot():
    from dojo.skills.library import facing_R
    R = facing_R(_tile())
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1])                  # the longest axis (local y) up: the checker's stand
    assert np.allclose(R @ [0, 0, 1], [0, -1, 0])                 # the glyph toward the robot / head camera
    assert np.allclose(facing_R(_tile(), toward=(1.0, 0.0, 0.3)) @ [0, 0, 1], [1, 0, 0])   # horizontal part only


def test_the_stand_shows_the_face_to_the_head_camera_best_first():
    from dojo.skills.library import HEAD_CAM_XYZ, facings_to_head
    xy, z = (0.325, -0.15), 0.80
    f = facings_to_head(xy, z)
    d = np.asarray(HEAD_CAM_XYZ) - [xy[0], xy[1], z]
    d /= np.linalg.norm(d)
    cos = [float(np.dot(t, d)) for t in f]
    assert f and cos == sorted(cos, reverse=True) and min(cos) > 0.3
    assert f[0][0] < 0 and f[0][1] < 0                    # at the right end of the row: toward the robot and the middle
    assert all(t[0] < 0.5 for t in f)                     # never the side away from the camera (+x there)


class _FakeSkills:
    """The parts of ``Skills`` ``kong_read`` uses: a refine hook that answers per (arm, call) from ``answers``."""

    def __init__(self, answers, look_ok=True):
        self.no_refine = {"m"}
        self.answers = list(answers)
        self.calls = []
        self.ran = []
        self.last_refine = None
        obj = SimpleNamespace(center_w=np.array([0.0, 0.05, 0.78]), top_z=0.79)
        self.world = SimpleNamespace(obj=lambda label: obj)
        self.kin = SimpleNamespace(base=lambda a: np.eye(4) if a == "left" else np.diag([1.0, 1, 1, 1]) + 0)
        self.ex = SimpleNamespace(q={"left": np.zeros(6), "right": np.zeros(6)}, g={"left": 1.0, "right": 1.0},
                                  run=lambda plans, name="": self.ran.append(name))
        self.refine = object()
        self.look_ok = look_ok

    def _refine(self, label, arm, kind):
        assert label not in self.no_refine                  # a read lifts the skill's no-refine for the label
        self.calls.append((label, arm, kind))
        if kind != "read":                   # a pose refine at a look: no glyph
            self.last_refine = {"label": label, "for": kind}
            return
        self.last_refine = {"label": label, "for": kind, "glyph": self.answers.pop(0)}

    def log(self, m):
        pass


def test_kong_read_reads_once_and_looks_only_when_unread(monkeypatch):
    from dojo.skills import library as L
    from dojo.skills import regrasp
    monkeypatch.setattr(regrasp, "_look_move", lambda sk, arm, xy, z: "move")
    monkeypatch.setattr(L, "ArmPlan", lambda arm, q, g: SimpleNamespace(move=lambda mv: SimpleNamespace(hold=lambda n: "p")))
    sk = _FakeSkills([{"read": True, "name": "Rigid/mahjong/00004"}])
    g = L.kong_read(sk, "m", look="down")
    assert g["name"] == "Rigid/mahjong/00004" and sk.calls == [("m", "left", "read")] and not sk.ran
    assert "m" in sk.no_refine                                     # restored
    sk = _FakeSkills([{"read": False, "why": "low_margin"}, {"read": True, "name": "Rigid/mahjong/00009"}])
    g = L.kong_read(sk, "m", look="down")
    assert g["name"] == "Rigid/mahjong/00009" and sk.ran == ["look at m"]
    assert [c[2] for c in sk.calls] == ["read", "read"]                    # place (v2): no pose refine at the look
    sk = _FakeSkills([{"read": False, "why": "low_margin"}, {"read": True, "name": "Rigid/mahjong/00009"}])
    L.kong_read(sk, "m", look="down", v3=True)
    assert [c[2] for c in sk.calls] == ["read", "look", "read"]            # place3: the pose refined, then read
    monkeypatch.setattr(regrasp, "_look_move", lambda sk, arm, xy, z: None)  # out of reach straight down
    monkeypatch.setattr(L, "_front_look_move", lambda sk, arm, o, looks=None: "oblique" if looks == L.OBLIQUE_LOOKS else None)
    sk = _FakeSkills([{"read": False, "why": "back"}])
    assert L.kong_read(sk, "m", look="down")["why"] == "back" and not sk.ran   # place (v2): no oblique look
    sk = _FakeSkills([{"read": False, "why": "back"}, {"read": True, "name": "Rigid/mahjong/00011"}])
    assert L.kong_read(sk, "m", look="down", v3=True)["name"] == "Rigid/mahjong/00011" and sk.ran == ["look at m"]
    sk = _FakeSkills([{"read": False, "why": "back"}])
    assert L.kong_read(sk, "m", look=None)["why"] == "back" and not sk.ran


def test_the_flag_is_validated_and_off_by_default():
    from dojo.runtime.policy import RuntimeConfig
    assert RuntimeConfig().kong_draw_read == "off"
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["kong_draw_read"][0] == "off" and SKILL_ATTRS["kong_draw_first"][0] == "near"


def test_g3_answers_a_read_only_when_the_glyph_faces_a_camera():
    pytest.importorskip("scipy")
    from dojo.skills import g3

    def env_with(quat):
        row = {"label": "t", "type": "Rigid", "category": "mahjong", "category_idx": 12,
               "pos": [0.3, -0.15, 0.79], "quat": quat}
        return SimpleNamespace(gt_objects=lambda: [row], proprio=lambda: {})
    # standing on its end (local y up), glyph (local +z) toward -y: the head camera (robot side, above) reads it
    c = np.cos(np.pi / 4)
    up_face_robot = [c, c, 0.0, 0.0]                       # +90 deg about x: local y -> +z, local z -> -y
    away = [c, -c, 0.0, 0.0]                               # -90 deg about x: local z -> +y (the back to the robot)
    r = g3.glyph_gt_read(env_with(up_face_robot), None, "t", None)
    assert r["read"] and r["name"] == "Rigid/mahjong/00012" and r["view"] == "cam_head"
    assert g3.glyph_gt_read(env_with(away), None, "t", None) == {"read": False, "name": None, "margin": None,
                                                                 "why": "back"}
    assert g3.glyph_gt_read(env_with(away), None, "nope", None)["why"] == "not visible"


def test_a_second_view_skips_the_first_read_and_looks(monkeypatch):
    from dojo.skills import library as L
    monkeypatch.setattr(L, "_front_look_move", lambda sk, arm, o, looks=None: "move")
    monkeypatch.setattr(L, "ArmPlan", lambda arm, q, g: SimpleNamespace(move=lambda mv: SimpleNamespace(hold=lambda n: "p")))
    sk = _FakeSkills([{"read": True, "name": "Rigid/mahjong/00007", "view": "cam_left_wrist"}])
    g = L.kong_read(sk, "m", look="front", skip_first=True)
    assert g["name"] == "Rigid/mahjong/00007" and len(sk.calls) == 1 and sk.ran == ["look at m"]


def test_place_completes_only_the_read_pair_place3_any_read_hand_glyph():
    """``kong_draw_read`` "place" is the measured v2 (b131264e): the draw completes the one glyph read on exactly two
    unmatched hand tiles; "place3" (#315, not adopted) takes any read hand glyph."""
    from dojo.skills.library import draw_glyphs

    def world(reads):
        objs = {lab: SimpleNamespace(asset=SimpleNamespace(key=k), identity_known=k is not None) for lab, k in reads.items()}
        return SimpleNamespace(obj=lambda lab: objs[lab])
    A, B, C, P = ("Rigid/mahjong/%05d" % i for i in (1, 2, 3, 7))
    clean = {"a0": A, "a1": A, "a2": A, "b0": B, "b1": B, "b2": B, "p0": P, "p1": P, "m0": C}
    hand = list(clean)
    assert draw_glyphs(world(clean), hand, ["m0"])[0] == {P}
    assert draw_glyphs(world(clean), hand, ["m0"], v3=True)[0] == {A, B, P}
    one_misread = dict(clean, a2=None)                     # a triple with one tile unread: its glyph twice as well
    assert draw_glyphs(world(one_misread), hand, ["m0"])[0] is None
    assert draw_glyphs(world(one_misread), hand, ["m0"], v3=True)[0] == {A, B, P}
    assert draw_glyphs(world({k: None for k in clean}), hand, [], v3=True)[0] is None


def test_the_flag_takes_place3():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert "place3" in SKILL_ATTRS["kong_draw_read"]
    RuntimePolicy(RuntimeConfig(kong_draw_read="place3"))       # validated, not refused


def test_insert_key_switches_are_validated_and_off_by_default():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    cfg = RuntimeConfig()
    assert (cfg.key_handover, cfg.key_lean) == ("v1", "off")
    assert SKILL_ATTRS["key_handover"] == ("v1", "v2") and SKILL_ATTRS["key_lean"] == ("off", "on")
    RuntimePolicy(RuntimeConfig(key_handover="v2", key_lean="on"))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(key_lean="maybe"))


def test_fill_holder_switch_is_validated_and_moves_by_default():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().fill_holder == "move" and SKILL_ATTRS["fill_holder"] == ("move", "stay")
    RuntimePolicy(RuntimeConfig(fill_holder="stay"))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(fill_holder="carry"))
