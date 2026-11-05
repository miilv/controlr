"""The scoring-slice census must never expose individual diagnostics or open other slices."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'imit_infra_census', Path(__file__).resolve().parents[2] / 'scripts/dojo/imit_infra_census.py')
CENSUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENSUS)


def write(root, suffix, row):
    directory = root / ('c4b_imitatesg_' + suffix)
    directory.mkdir()
    (directory / 'imitate_episodes.jsonl').write_text(json.dumps(row) + '\n')
    return directory


def test_only_aggregate_fields_escape_and_archived_attempts_stay_separate(tmp_path):
    write(tmp_path, 'o2owaitlook_100005', dict(category='infra_failure', success=False,
          steps_used=0, why='SECRET DIAGNOSTICS', score=0.25, actions=['PRIVATE']))
    write(tmp_path, 'o2owaitlook_100005__retry', dict(category='perception_gross', success=False,
          steps_used=900))
    write(tmp_path, 'o2owaitlook_100006.codebug', dict(category='error', success=False, steps_used=400))
    rows = CENSUS.census(tmp_path)
    active = next(r for r in rows if r['archive'] == 'active')
    assert active['attempts'] == 2 and active['infra_rate'] == 0.5
    assert active['categories'] == {'infra_failure': 1, 'perception_gross': 1}
    assert active['rc'] == {'None': 2} and active['zero_steps'] == 1
    assert len(rows) == 2
    text = json.dumps(rows)
    assert all(s not in text for s in ('SECRET', 'PRIVATE', '100005', 'score', 'why', 'actions'))


def test_forbidden_slices_families_and_symlinks_are_not_opened(tmp_path):
    for suffix in ('o2owaitlook_100100', 'o2owaitlook_100010', 'o2owaitlook_100030',
                   'dev_100000', 'unknown_100005'):
        directory = tmp_path / ('c4b_imitatesg_' + suffix)
        directory.mkdir()
        (directory / 'bad_episodes.jsonl').write_text('not json; must never open')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'bad_episodes.jsonl').write_text('not json')
    (tmp_path / 'c4b_imitatesg_o2owaitlook_100005').symlink_to(outside, target_is_directory=True)
    directory = tmp_path / 'c4b_imitatesg_o2o_100006'
    directory.mkdir()
    (directory / 'bad_episodes.jsonl').symlink_to(outside / 'bad_episodes.jsonl')
    assert CENSUS.census(tmp_path) == []
