"""Read-only report contracts: partial credit, thresholds, coverage and sample SD."""
import math
from pathlib import Path
import runpy

import pytest

F = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/skills/pour_g3_report.py'))


def row(seed, score, success=False, **kw):
    return {'seed': seed, 'score': score, 'success': success, 'stable': True, 'steps': 123, **kw}


def test_partial_win_is_not_a_full_success_win():
    pairs = [(row(1, 0), row(1, .2))]
    assert F['scored_pairs'](pairs)['wins'] == 1
    assert F['scored_pairs'](pairs, True)['wins'] == 0
    assert F['scored_pairs']([(row(1, .2), row(1, .3))])['wins'] == 1


def test_sample_sd_and_missing_pairs_cannot_pass():
    assert math.isclose(F['moments']([0, 100])['sd'], math.sqrt(5000))
    r = F['paired_summary']([row(100080, 0)], [row(100080, 1, True)], [100080, 100081], 'pour_by_language')
    assert r['n'] == 1 and not r['complete'] and r['missing'] == [100081]
    assert F['decision']({'pour_by_language': r})['passes'] is None
    assert r['clean_successes']['B'] == 0  # absent diagnostics are not evidence of clean capture
    assert r['success_flips'] == {'B_only': 1}


def test_duplicate_and_infrastructure_rows():
    with pytest.raises(AssertionError, match='duplicate'):
        F['paired_summary']([row(1, 0), row(1, 0)], [], [1], 'pour_by_language')
    r = F['paired_summary']([row(1, 0, infra_failure=['uncooked mesh'])], [row(1, 1, True)], [1], 'pour_by_language')
    assert r['n'] == 0 and r['excluded'] == [1] and not r['complete']


def test_official_success_overrides_an_earlier_conservative_skill_failure():
    r = row(1, 1, True, result={'ok': False, 'stage': 'pour 3'})
    assert F['failure_stage'](r, 'pour_by_language') == 'success'
