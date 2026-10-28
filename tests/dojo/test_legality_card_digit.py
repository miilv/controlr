"""Legality, card digit (robot-harness ILIA-POLICY-LEGALITY-AUDIT finding 6, G1F "For Ilia" G2-2 / G2-4).

* an explicitly observed digit (the agentic perceiver's ``value``) reaches the planner, also when the pose did not move;
  conflicts with B1's read follow ONE logged rule (``dojo.perception.digits.resolve``);
* the keep and unmentioned-object copies keep B1's measured digit (as ``value``);
* a card whose glyph B1 could not read (no texture score, a tie) gets no digit, and the planner reports the press count
  as unknown instead of pressing a guess.

Fakes only (FakeChat, synthetic records, monkeypatched B1 scorers): no LLM, no GPU, no catalogue. The tests that run
``record.merge_into_record`` or B1's ``_slot_identity`` need scipy / cv2 (the perception extra) and skip without them,
like their neighbours in test_agentic_perceiver.py; the rule, the world, the planner and the J1 copies run everywhere.
"""
import json
import math

import numpy as np
import pytest

from dojo.perception import digits
from dojo.perception.agentic import j1
from dojo.perception.agentic.llm import FakeChat

PRESS = "Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm."
FLANGE = {"left": [-0.3, -0.2, 0.9, 1, 0, 0, 0], "right": [0.3, -0.2, 0.9, 1, 0, 0, 0]}
CARD_Q = [0.0, 0.0, 0.42, -0.91]


def _calib():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    T_h = np.eye(4)
    T_h[:3, :3] = [[1, 0, 0], [0, -c, s], [0, -s, -c]]
    T_h[:3, 3] = [0, -0.41, 1.308]
    return {"cam_head": {"K": [[288.1325, 0, 320], [0, 288.1325, 240], [0, 0, 1]], "T_world_cam": T_h.tolist()}}


def _frames():
    return {"cam_head": np.full((480, 640, 3), 90, np.uint8)}


def _card_entry(label, idx, x, digit_fields):
    return {"category": "number_card", "category_idx": idx, "label": label, "default_pos": [x, -0.05, 0.805],
            "default_ori": CARD_Q, "scale": [1, 1, 1],
            "_perception": {"confidence": 0.95, "residual": 0.05, "place_tag": "zq:0.0400:0,0,0.42,-0.91", "slot": True,
                            **digit_fields}}


def _press_record(card0=None, card1=None):
    """A press_by_number-like B1 record: two red buttons, a blue one, two cards (num0 over button0, num1 over
    button1). ``card*``: the card's ``_perception`` digit fields (B1 read 5 and 3 by default)."""
    read5 = {"digit": 5, "digit_conf": 0.86, "digit_source": "texture"}
    read3 = {"digit": 3, "digit_conf": 1.0, "digit_source": "texture"}
    btn = lambda lab, idx, x: {"category": "SpringButton", "category_idx": idx, "label": lab,  # noqa: E731
                               "default_pos": [x, -0.17, 0.765], "default_ori": [1, 0, 0, 0], "scale": [1, 1, 1],
                               "_perception": {"confidence": 0.9, "residual": 0.1}}
    return {"task": None, "instruction": PRESS,
            "Articulation": {"SpringButton": [btn("button0", 2, -0.15), btn("button1", 2, 0.0), btn("button2", 3, 0.15)]},
            "Geometry": {"number_card": [_card_entry("num0", 5, -0.15, read5 if card0 is None else card0),
                                         _card_entry("num1", 3, 0.0, read3 if card1 is None else card1)]}}


def _plan(record):
    from dojo.runtime.estimate import objects_from_record
    from dojo.skills.planner import plan
    from dojo.skills.world import WorldState

    objs = objects_from_record(record, None, meshes=False)
    return plan(PRESS, WorldState("", PRESS, objs, {}, {})), objs


# ------------------------------------------------------------------------------------------- the rule (CI: numpy only)

def test_digit_map_agrees_with_the_catalogue():
    from dojo.perception.catalog import CATEGORY_ATTRIBUTES

    assert set(digits.DIGIT_OF) == {c for c, f in CATEGORY_ATTRIBUTES.items() if "digit" in f(1)}
    for i in range(1, 10):
        assert digits.digit_of("number_card", i) == CATEGORY_ATTRIBUTES["number_card"](i)["digit"]
        assert digits.idx_for_digit("number_card", i) == i
    assert digits.idx_for_digit("number_card", 8, [1, 5, 3]) is None and digits.digit_of("block", 3) is None
    assert [digits.as_digit(v) for v in (8, "8", " 8 ", True, 12, "12", 8.0, "eight", None)] == \
        [8, 8, 8, None, None, None, None, None, None]


def test_glyph_evidence_reads_only_a_scored_texture_with_a_margin():
    k = "Geometry/number_card/00005"
    flat = digits.glyph_evidence(k, None, 0.0)            # G2-2's probe: no glyph contrast, every candidate 0.25
    assert flat["read"] is False and flat["why"] == "no_texture" and flat["confidence"] == 0.0
    tied = digits.glyph_evidence(k, 0.12, 0.0)
    assert tied["read"] is False and tied["why"] == "tied"
    seed0 = digits.glyph_evidence(k, 0.12, 0.043)         # the seed-0 L9 practice frame: runner-up 0.043 behind
    assert seed0["read"] is True and seed0["confidence"] == pytest.approx(0.86)
    assert digits.glyph_evidence(k, 0.1, 0.2)["confidence"] == 1.0
    assert digits.glyph_evidence(k, 0.12, 0.003, min_margin=0.01)["why"] == "tied"   # the margin is a config axis

    assert digits.record_fields("number_card", 5, seed0, k, 9) == {"digit": 5, "digit_conf": 0.86, "digit_source": "texture"}
    assert digits.record_fields("number_card", 5, flat, k, 9) == {"digit": None, "digit_source": "unread",
                                                                  "digit_unread": "no_texture"}
    # evidence about another key (the identity changed after the read) is no read; one candidate: the config fixes it
    assert digits.record_fields("number_card", 8, seed0, "Geometry/number_card/00008", 9)["digit_unread"] == "not_read"
    assert digits.record_fields("number_card", 4, None, "Geometry/number_card/00004", 1)["digit_source"] == "config"
    assert digits.record_fields("block", 3, None, "Rigid/block/00003", 9) == {}
    # a perception record entry without digit fields (a label found by the runtime's search) is unread
    assert digits.from_record("number_card", {"confidence": 0.6, "found_by_search": True}) == (None, None, "unread")
    assert digits.from_record("number_card", {"digit": 5, "digit_conf": 0.86, "digit_source": "texture"}) == (5, 0.86, "texture")
    assert digits.from_record("block", {}) == (None, None, None)


B1_READ5 = {"digit": 5, "conf": 0.86, "source": "texture", "unread": None}
B1_UNREAD = {"digit": None, "conf": None, "source": "unread", "unread": "no_texture"}


@pytest.mark.parametrize("b1, value, conf, want_digit, want_source, want_event", [
    (B1_READ5, None, None, 5, "texture", None),                    # no agent reading: B1's read stands
    (B1_READ5, 5, None, 5, "agent+texture", None),                 # agreement
    (B1_READ5, 8, None, 8, "agent", "agent_over_b1"),              # the audit's case: an accepted correction wins, logged
    (B1_READ5, 8, 0.3, 5, "texture", "agent_low_conf"),            # an uncertain agent reading does not override
    (B1_READ5, "eight", 0.9, 5, "texture", "agent_invalid"),       # not a digit
    (B1_UNREAD, 7, 0.9, 7, "agent", "agent_reads"),                # the agent reads what B1 could not
    (B1_UNREAD, 7, 0.3, None, "unread", "agent_low_conf"),         # nobody read it with confidence: unknown
    (B1_UNREAD, None, None, None, "unread", None),                 # nobody read it: unknown, never the placeholder
    (None, 4, None, 4, "agent", "agent_reads"),                    # no B1 object: the agent's own card
])
def test_digit_rule(b1, value, conf, want_digit, want_source, want_event):
    r = digits.resolve(b1, value, conf)
    assert (r["digit"], r["source"]) == (want_digit, want_source)
    assert (r["event"] or {}).get("rule") == want_event
    if want_event == "agent_over_b1":
        assert r["event"]["b1"] == 5 and r["event"]["agent"] == 8 and r["event"]["b1_conf"] == 0.86
    if want_digit is None:
        assert r["unread"] in ("no_texture", "agent_rejected")
    # a digit no candidate shows is rejected like any invalid reading
    assert digits.resolve(B1_READ5, 9, 0.9, valid=[1, 5, 3])["event"]["rule"] == "agent_invalid"


# ------------------------------------------------------------------------------------------- world and planner (CI)

def test_observed_digit_reaches_the_planner_and_unknown_is_reported():
    p, objs = _plan(_press_record())
    assert objs["num0"].digit == 5 and objs["num0"].digit_source == "texture" and objs["num0"].digit_conf == 0.86
    assert p.skill == "press_sequence" and p.args["presses"] == [("button0", 5), ("button2", 1), ("button1", 3), ("button2", 1)]

    # the observed digit, not the asset index, is the count (an agent correction recorded against B1's index 5)
    rec = _press_record(card0={"digit": 8, "digit_conf": 0.8, "digit_source": "agent"})
    p, objs = _plan(rec)
    assert objs["num0"].idx == 5 and objs["num0"].digit == 8
    assert p.args["presses"][0] == ("button0", 8)

    # G2-2: a card B1 could not read -> no digit -> the planner reports the count unknown, no press_sequence
    rec = _press_record(card1={"digit": None, "digit_source": "unread", "digit_unread": "tied"})
    p, objs = _plan(rec)
    assert objs["num1"].digit is None and objs["num1"].digit_source == "unread"
    assert p.skill == "none" and p.args["unknown_digits"] == ["button1: num1 digit unknown"]
    assert "press count unknown" in p.why

    # a card entry without any digit read (the runtime's search fills a label with a candidate's index) is unknown too
    rec = _press_record()
    rec["Geometry"]["number_card"][0]["_perception"] = {"confidence": 0.6, "found_by_search": True}
    assert _plan(rec)[0].skill == "none"


def test_the_agent_scene_shows_the_observed_digit_or_none():
    from dojo.agent.scene import shown_digit
    from dojo.runtime.estimate import objects_from_record

    objs = objects_from_record(_press_record(card0={"digit": 8, "digit_conf": 0.8, "digit_source": "agent"},
                                             card1={"digit": None, "digit_source": "unread", "digit_unread": "no_texture"}),
                               None, meshes=False)
    assert shown_digit(objs["num0"]) == 8 and shown_digit(objs["num1"]) is None
    assert shown_digit(objs["button0"]) is None


def test_bridge_server_records_carry_the_value_and_no_value_is_unread():
    from dojo.bridge.perception import record_from_perceived

    per = [{"label": "num0", "asset": "Geometry/number_card/00005", "pos": [-0.15, -0.05, 0.805], "quat": CARD_Q,
            "confidence": 0.9, "value": 8},
           {"label": "num1", "asset": "Geometry/number_card/00003", "pos": [0.0, -0.05, 0.805], "quat": CARD_Q,
            "confidence": 0.9}]
    rec = record_from_perceived(per)
    c0, c1 = rec["Geometry"]["number_card"]
    assert c0["category_idx"] == 8 and c0["_perception"]["digit"] == 8 and c0["_perception"]["digit_source"] == "j1"
    assert c1["_perception"]["digit"] is None and c1["_perception"]["digit_unread"] == "no_value"


# ------------------------------------------------------------------------------------------- J1 copies (CI, FakeChat)

def _b1_cards(unread1=False):
    """B1's J1 output for two cards, as j1_adapter exports them (attributes.digit = the schema's value)."""
    c0 = {"label": "num0", "category": "number_card", "asset": "Geometry/number_card/00005", "pos": [-0.15, -0.05, 0.805],
          "quat": CARD_Q, "size": [0.06, 0.004, 0.08], "confidence": 0.95, "source": "head", "attributes": {"digit": 5},
          "digit_source": "texture", "digit_conf": 0.86, "flags": {"slot": True},
          "alternative": {"asset": "Geometry/number_card/00008", "margin": 0.043}}
    c1 = {"label": "num1", "category": "number_card", "asset": "Geometry/number_card/00003", "pos": [0.0, -0.05, 0.805],
          "quat": CARD_Q, "size": [0.06, 0.004, 0.08], "confidence": 0.95, "source": "head", "flags": {"slot": True}}
    if unread1:
        c1["digit_unread"] = "tied"
    else:
        c1.update(attributes={"digit": 3}, digit_source="texture", digit_conf=1.0)
    return {"objects": [c0, c1], "regions": [], "needs_look": []}


def _perceive(tmp_path, answer, b1):
    fc = FakeChat([json.dumps({"answer": answer})])
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0, "right": 1.0}, "Read the cards.", None, b1=b1,
                           chat=fc, model="fake", refine=False, log_dir=str(tmp_path))
    clean, rep = j1._validate(out)
    assert rep["ok"], rep
    return out, {o["label"]: o for o in clean["objects"]}


def test_keep_and_unmentioned_copies_keep_the_b1_digit(tmp_path):
    """G2-4: a normal keep answer and an unmentioned card deliver B1's digits through the real schema (before: [None,
    None], the press driver stopped before motion)."""
    out, by = _perceive(tmp_path, {"objects": [{"keep": 0}]}, _b1_cards())
    assert "b1_unmentioned_kept=1" in out["notes"]
    assert (by["num0"]["value"], by["num1"]["value"]) == (5, 3)
    raw = {o["label"]: o for o in out["objects"]}
    assert raw["num0"]["digit_source"] == "texture" and raw["num1"]["confidence"] == pytest.approx(0.475)
    # the hint shows the agent B1's digit (and that a card is unread)
    hint = j1.hint_text(_b1_cards(unread1=True), _calib(), "Hint")
    assert "value=5" in hint and "value=unread (tied)" in hint


def test_an_explicit_agent_correction_survives_the_copies(tmp_path):
    """Finding 6 (J1 half): {'keep': 0, 'value': 8} on B1's 5 -> 8, the asset follows, the conflict is logged; the
    agent's own point next to B1's card takes B1's digit when it gives none."""
    out, by = _perceive(tmp_path, {"objects": [{"keep": 0, "value": 8},
                                               {"label": "num1", "pos": [0.001, -0.05, 0.805], "confidence": 0.9}]},
                        _b1_cards())
    assert by["num0"]["value"] == 8 and by["num0"]["asset"] == "Geometry/number_card/00008"
    assert by["num1"]["value"] == 3
    assert "digit num0: rule=agent_over_b1 agent=8 agent_conf=0.8 b1=5 b1_conf=0.86" in out["notes"]
    # an uncertain correction does not override B1's read (logged); a card nobody read stays without a value
    out, by = _perceive(tmp_path, {"objects": [{"keep": 0, "value": 8, "confidence": 0.3}, {"keep": 1}]},
                        _b1_cards(unread1=True))
    assert by["num0"]["value"] == 5 and "value" not in by["num1"]
    assert "rule=agent_low_conf" in out["notes"]
    assert {o["label"]: o for o in out["objects"]}["num1"]["digit_unread"] == "tied"


def test_record_merge_refuses_a_digit_no_candidate_shows():
    from dojo.perception.agentic.record import apply_digit

    r = _card_entry("num0", 5, -0.15, {"digit": 5, "digit_conf": 0.86, "digit_source": "texture"})
    ev = apply_digit(r, {"value": 9, "digit_source": "agent", "digit_conf": 0.8}, cand_idxs=[1, 2, 3, 4, 5, 6, 7, 8])
    assert ev["action"] == "identity_rejected" and r["category_idx"] == 5 and r["_perception"]["digit"] == 5
    ev = apply_digit(r, {"value": 8, "digit_source": "agent", "digit_conf": 0.8}, cand_idxs=[1, 2, 3, 4, 5, 6, 7, 8])
    assert ev["digit_to"] == 8 and r["category_idx"] == 8 and r["_perception"]["digit_source"] == "agent"


def test_b1_j1_export_omits_an_unread_digit():
    """G2-2, J1 half: j1_adapter exports attributes.digit only from a read; an unread card keeps its other attributes
    and says why; a later call that cannot read the card keeps the earlier call's read."""
    from dojo.perception.j1_adapter import _digit_export, carry_prior_digit

    o = {}
    _digit_export(o, {"digit": 5}, {"digit": None, "digit_source": "unread", "digit_unread": "no_texture"})
    assert "attributes" not in o and o["digit_unread"] == "no_texture"
    o = {}
    _digit_export(o, {"digit": 5}, {"digit": 5, "digit_conf": 0.86, "digit_source": "texture"})
    assert o == {"attributes": {"digit": 5}, "digit_source": "texture", "digit_conf": 0.86}
    o = {}
    _digit_export(o, {"joints": 1}, {})               # other categories: the catalogue's attributes as before
    assert o == {"attributes": {"joints": 1}}

    prior = [{"label": "num0", "category": "number_card", "asset": "Geometry/number_card/00005", "attributes": {"digit": 5},
              "pos": [-0.15, -0.05, 0.805], "quat": CARD_Q, "confidence": 0.86,
              "alternative": {"asset": "Geometry/number_card/00006", "margin": 0.053}}]

    def cur():
        return [{"label": "num0", "category": "number_card", "asset": "Geometry/number_card/00001", "digit_unread": "tied",
                 "alternative": {"asset": "Geometry/number_card/00002", "margin": 0.0}}]
    c = cur()
    assert carry_prior_digit(c, prior) == 1
    assert c[0]["attributes"]["digit"] == 5 and c[0]["asset"].endswith("00005") and "digit_unread" not in c[0]
    # a prior entry without its own confidence (merge_prior's prior_unusable) re-identifies nothing
    c = cur()
    assert carry_prior_digit(c, [{k: v for k, v in prior[0].items() if k != "confidence"}]) == 0
    assert c[0]["digit_unread"] == "tied" and "attributes" not in c[0]


def test_a_tracked_agent_digit_survives_the_next_b1_gated_call():
    """AGENTIC_GATE=budget: after the agent's call, B1-gated calls return the tracker's scene, which pairs each tracked
    object with B1's re-read and takes the re-read's fields: the agent's 8 came back as B1's 5."""
    from dojo.perception.tracker import Tracker

    tr = Tracker()
    agent = dict(_b1_cards()["objects"][0], value=8, digit_source="agent", digit_conf=0.8, asset="Geometry/number_card/00008")
    agent.pop("attributes")
    tr.update([agent])
    objs, _, ev = tr.update([_b1_cards()["objects"][0]])          # B1 reads 5 at the same place
    assert objs[0]["value"] == 8 and "attributes" not in objs[0] and objs[0]["asset"].endswith("00008")
    assert {"event": "digit", "label": "num0", "value": 8} in ev
    objs, _, ev = tr.update([_b1_cards()["objects"][0]])          # and again on the call after
    assert objs[0]["value"] == 8 and objs[0]["digit_source"] == "agent"


# ------------------------------------------------------------------------------------------- record path (scipy)

def _agentic(tmp_path, record, answer):
    pytest.importorskip("scipy")
    import unittest.mock as um

    from dojo.perception.agentic import record as R

    st = {"left_ee_pose": FLANGE["left"], "right_ee_pose": FLANGE["right"], "left_ee_joint_state": [1.0],
          "right_ee_joint_state": [1.0]}
    with um.patch.object(R, "_calib_from_state", lambda s: _calib()):
        return R.agentic_estimate(_frames(), st, "press_by_number", PRESS, bgs={}, b1_record=record,
                                  chat=FakeChat([json.dumps({"answer": answer})]), model="fake", refine=False,
                                  log_dir=str(tmp_path))


def _keep_all(**num0):
    """Keep every hint object (record_objects' order: Geometry first, so num0 = 0, num1 = 1, the buttons 2-4)."""
    return {"objects": [dict({"keep": 0}, **num0)] + [{"keep": i} for i in range(1, 5)]}


def test_audit_same_pose_card_correction_reaches_the_planner(tmp_path):
    """The audit's probe: B1 record at index 5, the agent answers {'keep': 0, 'value': 8} for that card (same pose) ->
    the planner presses 8 (before: merge_log=[] and card_digit 5). Also on a record without digit fields."""
    new = _agentic(tmp_path, _press_record(), _keep_all(value=8))
    ag = new["_perception"]["agentic"]
    assert ag["result"] == "agent"
    ident = [e for e in ag["merge"] if e["action"] == "identity"]
    assert ident == [{"label": "num0", "action": "identity", "digit_from": 5, "digit_to": 8, "idx_from": 5, "idx_to": 8,
                      "source": "agent", "was": "texture"}]
    assert any("rule=agent_over_b1" in p for p in ag["problems"])
    card = next(r for r in new["Geometry"]["number_card"] if r["label"] == "num0")
    assert card["default_pos"] == [-0.15, -0.05, 0.805] and card["category_idx"] == 8
    p, objs = _plan(new)
    assert objs["num0"].digit == 8 and objs["num0"].digit_source == "agent"
    assert p.args["presses"] == [("button0", 8), ("button2", 1), ("button1", 3), ("button2", 1)]

    legacy = _press_record()
    for r in legacy["Geometry"]["number_card"]:
        r["_perception"] = {"confidence": 0.95, "residual": 0.05, "slot": True}
    assert _plan(_agentic(tmp_path, legacy, _keep_all(value=8)))[0].skill == "none"   # num1 nobody read
    both = {"objects": [{"keep": 0, "value": 8}, {"keep": 1, "value": "3"}] + [{"keep": i} for i in range(2, 5)]}
    assert _plan(_agentic(tmp_path, legacy, both))[0].args["presses"][::2] == [("button0", 8), ("button1", 3)]


def test_conflicting_and_uncertain_readings_follow_the_rule(tmp_path):
    # an uncertain agent reading does not override B1's read: the planner presses 5, the rejection is logged
    new = _agentic(tmp_path, _press_record(), _keep_all(value=8, confidence=0.3))
    assert not [e for e in new["_perception"]["agentic"]["merge"] if e["action"] == "identity"]
    assert any("rule=agent_low_conf" in p for p in new["_perception"]["agentic"]["problems"])
    assert _plan(new)[0].args["presses"][0] == ("button0", 5)
    # B1 could not read num0 (tie) and the agent kept it without a value: unknown, the planner reports it
    unread = {"digit": None, "digit_source": "unread", "digit_unread": "tied"}
    new = _agentic(tmp_path, _press_record(card0=unread), _keep_all())
    p, objs = _plan(new)
    assert objs["num0"].digit is None and p.skill == "none" and p.args["unknown_digits"] == ["button0: num0 digit unknown"]
    # ... and an agent that reads it supplies the count
    new = _agentic(tmp_path, _press_record(card0=unread), _keep_all(value=7))
    assert _plan(new)[0].args["presses"][0] == ("button0", 7)
    assert next(r for r in new["Geometry"]["number_card"] if r["label"] == "num0")["_perception"]["digit_source"] == "agent"


# ------------------------------------------------------------------------------------------- B1's identity read (cv2)

class _SM:
    tag = "zq:0.0400:0,0,0.42,-0.91"

    class asset:
        extents = np.array([0.06, 0.004, 0.08])

    def pose(self, x, y, yaw, z_off=0.0):
        return np.eye(4)


@pytest.mark.parametrize("tex, want_key, want_why", [
    ({1: None, 5: None, 8: None}, "00001", "no_texture"),   # G2-2's probe: flat glyphs, every candidate on the 0.25 penalty
    ({1: 0.30, 5: 0.10, 8: 0.10}, "00005", "tied"),         # two candidates explain the glyph equally
    ({1: 0.30, 5: 0.10, 8: 0.25}, "00005", "texture"),      # a read: margin 0.4 x 0.15 = 0.06
])
def test_b1_slot_identity_without_glyph_evidence_exports_no_digit(monkeypatch, tex, want_key, want_why):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as il

    monkeypatch.setattr(il, "_slot_shape", lambda k, placement, tag: _SM())
    monkeypatch.setattr(il, "shape_by_tag", lambda k, tag: _SM())
    monkeypatch.setattr(il, "texture_cost", lambda sm, T, views, k, **kw: tex[int(k[-5:])])
    monkeypatch.setattr(il, "observed_color", lambda sm, T, views, **kw: (None, 0))
    cands = [f"Geometry/number_card/{i:05d}" for i in (1, 5, 8)]
    h = il.Hypothesis(0, cands[0], _SM.tag, -0.15, -0.05, 0.0, 0.1, {"sil_loss": 0.1, "geom_group": cands})
    out = il._slot_identity(h, {"candidates": cands}, {}, None, {}, il.EstimatorConfig())
    assert out.key.endswith(want_key) and out.per_view["glyph"]["why"] == want_why
    f = digits.record_fields("number_card", int(out.key[-5:]), out.per_view["glyph"], out.key, len(cands))
    if want_why == "texture":
        assert f == {"digit": 5, "digit_conf": 1.0, "digit_source": "texture"}
    else:  # no digit leaves B1, and the planner reports the count unknown
        assert f["digit"] is None and f["digit_unread"] == want_why
        rec = _press_record(card0=f)
        assert _plan(rec)[0].skill == "none"
