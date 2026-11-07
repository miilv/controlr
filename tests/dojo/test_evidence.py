"""dojo.perception.evidence: the episode's persistent per-instance evidence (candidates from unexplained head regions,
identity votes apart from pose writes) and the runtime's default-off ``instance_evidence: log``. CPU only."""

import pytest

from dojo.perception.evidence import CAND_MIN_PX, Ledger, candidate_regions


def _rec(unexplained, rigid=(), basket=(0.0, 0.05)):
    return {"Geometry": {"basket": [{"category_idx": 2, "default_pos": [basket[0], basket[1], 0.80],
                                     "default_ori": [1, 0, 0, 0]}]},
            "Rigid": {"car": [{"label": "rigid0", "category_idx": i, "default_pos": [x, y, 0.78],
                               "_perception": {"confidence": c}} for i, (x, y, c) in enumerate(rigid)]},
            "_perception": {"unexplained": [{"xy": list(xy), "area_px": a} for xy, a in unexplained]}}


HALF = {"Geometry/basket/00002": (0.126, 0.084)}


def test_candidates_are_object_sized_off_fixtures_and_off_believed_objects():
    rec = _rec([((0.30, -0.15), 900),            # an unseen object on the table: kept
                ((0.20, -0.20), CAND_MIN_PX - 1),  # a fragment: dropped
                ((0.05, 0.06), 800),             # inside the basket's footprint: the basket's
                ((-0.10, -0.12), 500),           # 20 mm from a believed object: its unexplained part
                ((-0.20, -0.08), 3500)],         # a merged region (shadow + objects): kept, marked
               rigid=[(-0.10, -0.10, 0.5)])
    c = candidate_regions(rec, HALF.get)
    assert [(x["xy"], x["merged"]) for x in c] == [([0.3, -0.15], False), ([-0.2, -0.08], True)]
    assert len(candidate_regions(rec, lambda k: None)) == 3          # no fixture geometry: nothing is filtered by it


def test_ledger_keeps_candidates_across_reads_until_resolved():
    led = Ledger()
    led.update(_rec([((0.30, -0.15), 900)]), 0, HALF.get)
    led.update(_rec([((0.31, -0.16), 950)]), 40, HALF.get)          # sighted again 14 mm away: the same one
    assert len(led.candidates) == 1 and led.candidates[0].sightings == 2 and led.candidates[0].area_px == 950
    led.update(_rec([]), 80, HALF.get)                              # not seen in a later read: still kept
    assert len(led.open_candidates()) == 1
    led.resolve(led.candidates[0], True, 90, key="Rigid/car/00011")
    assert led.open_candidates() == [] and led.events[-1]["kind"] == "candidate_confirmed"
    assert led.summary()["candidates"][0]["status"] == "confirmed"


def test_identity_votes_stay_apart_from_the_key_until_they_outweigh_it():
    led = Ledger(switch_margin=0.5)
    led.update(_rec([], rigid=[(0.1, -0.1, 0.6)]), 0, HALF.get)
    assert led.identity("rigid0") == "Rigid/car/00000"
    led.vote("rigid0", "Rigid/watch/00005", 0.9, 30, "refine")      # one stronger read: a vote, not a switch
    assert led.identity("rigid0") == "Rigid/car/00000" and "rigid0" in led.summary()["contested"]
    led.vote("rigid0", "Rigid/watch/00005", 0.3, 60, "refine")      # 1.2 vs 0.6: outweighs by the margin
    assert led.identity("rigid0") == "Rigid/watch/00005"
    assert any(e["kind"] == "identity_switch" for e in led.events)


def test_runtime_switch_defaults_off_and_logs_only():
    from types import SimpleNamespace

    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().instance_evidence == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(instance_evidence="write"))
    pol = RuntimePolicy(RuntimeConfig(instance_evidence="log"))
    pol.report = {"events": []}
    pol.bridge = SimpleNamespace(steps_used=12)
    pol._half_extent = HALF.get
    pol._ledger = Ledger()
    rec = _rec([((0.30, -0.15), 900)])
    assert pol._persist_reads(rec, "initial") == rec                 # the record goes on unchanged
    assert [e["kind"] for e in pol.report["events"]] == ["candidate"]
    assert pol.report["events"][0]["stage"] == "initial" and pol.report["evidence"]["candidates"][0]["first_step"] == 12
    off = RuntimePolicy(RuntimeConfig())
    off.report, off.bridge, off._ledger = {"events": []}, SimpleNamespace(steps_used=0), None
    off._persist_reads(rec, "initial")
    assert off.report["events"] == [] and "evidence" not in off.report
