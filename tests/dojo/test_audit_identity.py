"""Offline identities remain fixed when a ring is transported past another instance."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import audit_identity as ai
from dojo.perception.grasp_metric import Footprints


def _frame(a=0., b=.2):
    return {"objects": [{"label": label, "type": "Rigid", "category": "block", "category_idx": 5,
                         "pos": [x, 0, .8], "quat": [1, 0, 0, 0]} for label, x in (("physical0", a), ("physical1", b))]}


def _write(step, x, label="belief0"):
    return {"step": step, "label": label, "before": [x, 0, .8], "q_before": [1, 0, 0, 0]}


def _tracker(monkeypatch, report, frames):
    monkeypatch.setattr(ai, "Groups", lambda task: SimpleNamespace(of=lambda label: "blocks"))
    fp = Footprints(loader=lambda key: np.array([[-.01, -.01, -.01], [.01, .01, .01]]))
    return ai.AuditIdentity(report, frames, "task", fp, back=0)


def test_consumed_pick_identity_survives_transport_and_a_misplaced_retry(monkeypatch):
    report = {"events": [{"kind": "consume", "skill": "pick", "step": s, "label": "belief0",
                           "key": "Rigid/block/00005", "pose7_used": [x, 0, .8, 1, 0, 0, 0]}
                          for s, x in ((1, 0), (9, .2))]}
    frames = {"1": _frame(), "5": _frame(.3, .001), "9": _frame(.3, .2)}
    tr = _tracker(monkeypatch, report, frames)
    for s in (5, 9):
        a = tr.resolve(_write(s, .001), frames[str(s)], np.array([.001, 0, .8]))
        assert a["gt_label"] == "physical0" and not a["identity_fallback"]
        assert a["identity_source"] == "consumed_pick" and a["identity_step"] == 1


def test_initial_tight_association_and_two_distinct_instances(monkeypatch):
    rec = {"Rigid": {"block": [{"label": f"belief{i}", "category_idx": 5, "default_pos": [x, 0, .8]}
                               for i, x in enumerate((.003, .198))]}}
    f = _frame()
    tr = _tracker(monkeypatch, {"records": {"initial": rec}}, {"0": f})
    for i, x in enumerate((.003, .198)):
        a = tr.resolve(_write(0, x, f"belief{i}"), f, np.array([x, 0, .8]))
        assert a["gt_label"] == f"physical{i}" and a["identity_source"] == "initial_tight"


def test_distant_fallback_is_reported_not_persisted_and_has_no_future_leak(monkeypatch):
    report = {"events": [{"kind": "consume", "skill": "pick", "step": 10, "label": "belief0",
                           "pose7_used": [0, 0, .8, 1, 0, 0, 0], "key": "Rigid/block/00005"}]}
    frames = {"0": _frame(), "1": _frame(1., .2)}
    tr = _tracker(monkeypatch, report, frames)
    assert not tr.keys  # future consumed-pick asset/identity evidence is not available yet
    a = tr.resolve(_write(0, 1.), frames['0'], np.array([1., 0, .8]))
    assert a["gt_label"] == "physical1" and a["identity_fallback"] and not tr.bindings
    b = tr.resolve(_write(1, 1.), frames['1'], np.array([1., 0, .8]))
    assert b["gt_label"] == "physical0" and not b["identity_fallback"]


def test_missing_persistent_object_never_silently_switches_to_neighbour(monkeypatch):
    frames = {"0": _frame()}
    tr = _tracker(monkeypatch, {}, frames)
    tr.resolve(_write(0, 0.), frames['0'], np.array([0, 0, .8]))
    f = {"objects": frames['0']['objects'][1:]}
    with pytest.raises(ValueError, match="refusing nearest substitution"):
        tr.resolve(_write(1, .2), f, np.array([.2, 0, .8]))


def test_later_perception_record_is_not_backdated_to_initial_frame(monkeypatch):
    def rec(x):
        return {'Rigid': {'block': [{'label': 'belief0', 'category_idx': 5, 'default_pos': [x, 0, .8]}]}}
    f = _frame()
    tr = _tracker(monkeypatch, {'records': {'initial': rec(0.), 'agentic2': rec(.2)}}, {'0': f})
    a = tr.resolve(_write(0, 0.), f, np.array([0, 0, .8]))
    assert a['gt_label'] == 'physical0'


def test_first_consumed_pick_finalizes_provisional_label_but_retries_do_not_rebind(monkeypatch):
    # Initial detector labels can be renamed before use. First physical consumption is authoritative;
    # no hindsight is applied to the earlier write, and subsequent misplaced retries cannot move the anchor.
    initial = {'Rigid': {'block': [{'label': 'belief0', 'category_idx': 5, 'default_pos': [.2, 0, .8]}]}}
    events = [{'kind': 'consume', 'skill': 'pick', 'label': 'belief0', 'step': s,
               'pose7_used': [x, 0, .8, 1, 0, 0, 0], 'key': 'Rigid/block/00005'} for s, x in ((1, 0.), (2, .2))]
    frames = {str(s): _frame() for s in range(3)}
    tr = _tracker(monkeypatch, {'records': {'initial': initial}, 'events': events}, frames)
    assert tr.resolve(_write(0, .2), frames['0'], np.array([.2, 0, .8]))['gt_label'] == 'physical1'
    for s in (1, 2):
        a = tr.resolve(_write(s, .2), frames[str(s)], np.array([.2, 0, .8]))
        assert a['gt_label'] == 'physical0' and a['identity_source'] == 'consumed_pick'
        assert a['identity_prior_gt_label'] == 'physical1'


def test_saved_grasp_close_snapshot_finalizes_when_consume_logging_is_absent(monkeypatch):
    report = {'events': [{'kind': 'refine', 'for': 'pick', 'step': 0, 'arm': 'left', 'label': 'belief0'}],
              'dev_belief_probe': {'snaps': [{'step': 1, 'objects': {'belief0': {
                  'pos': [0, 0, .8], 'quat': [1, 0, 0, 0], 'key': 'Rigid/block/00005'}}}]}}
    monkeypatch.setattr(ai, 'Groups', lambda task: SimpleNamespace(of=lambda label: 'blocks'))
    fp = Footprints(loader=lambda key: np.array([[-.01, -.01, -.01], [.01, .01, .01]]))
    frames = {'1': _frame(), '5': _frame(.3, .001)}
    actions = [{'left_ee_joint_state': [g]} for g in (1., 0., 1.)]
    tr = ai.AuditIdentity(report, frames, 'task', fp, back=0, actions=actions)
    a = tr.resolve(_write(5, .001), frames['5'], np.array([.001, 0, .8]))
    assert a['gt_label'] == 'physical0' and a['identity_source'] == 'grasp_close'
