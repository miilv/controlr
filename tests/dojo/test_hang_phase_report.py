"""The diagnostic must not mistake a release's delayed trace name for its current action."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "hang_phase_report", Path(__file__).resolve().parents[2] / "scripts/skills/hang_phase_report.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
annotate_segments = _module.annotate_segments


def test_segment_boundaries_use_trajectory_lengths():
    frames = [{'step': k, 'seg': 'previous'} for k in range(1, 7)]
    annotate_segments(frames, [{'name': 'pick', 'steps': 2}, {'name': 'carry', 'steps': 1},
                               {'name': 'hang', 'steps': 2}, {'name': 'retreat', 'steps': 1}])
    assert [f['current_segment'] for f in frames] == ['pick', 'pick', 'carry', 'hang', 'hang', 'retreat']
    assert all(f['seg'] == 'previous' for f in frames)


def test_loose_retry_assignments_do_not_duplicate_one_release():
    logs = ['[hang] mug0 -> hook 4: left (wall)', '[hang] mug0 loose; retry',
            '[hang] mug1 -> hook 5: right (wall)', '[hang] mug0 -> hook 0: right (handle)']
    assert _module.hang_assignments(logs) == [('mug0', 0, 'right'), ('mug1', 5, 'right')]
