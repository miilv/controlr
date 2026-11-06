"""The diagnostic must not mistake a release's delayed trace name for its current action."""
from scripts.skills.hang_phase_report import annotate_segments


def test_segment_boundaries_use_trajectory_lengths():
    frames = [{'step': k, 'seg': 'previous'} for k in range(1, 7)]
    annotate_segments(frames, [{'name': 'pick', 'steps': 2}, {'name': 'carry', 'steps': 1},
                               {'name': 'hang', 'steps': 2}, {'name': 'retreat', 'steps': 1}])
    assert [f['current_segment'] for f in frames] == ['pick', 'pick', 'carry', 'hang', 'hang', 'retreat']
    assert all(f['seg'] == 'previous' for f in frames)
