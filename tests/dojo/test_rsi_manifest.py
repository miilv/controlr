"""Input freeze rejects off-slice ids and byte changes without a GPU."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('rsi_manifest', Path(__file__).parents[2] / 'baselines/rsi/manifest.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('ids', [[], [100000, 100000], [100010], [100030], [100100], [0]])
def test_refuse_ids(ids):
    with pytest.raises(ValueError):
        m.validate_ids(ids)


def test_hash_and_order(tmp_path):
    m.validate_ids([100000, 100009, 100020, 100029])
    directory = tmp_path / 'stack_blocks'
    directory.mkdir()
    record = directory / '100000.json'
    record.write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='rsi-yardstick/1', salt=m.SALT,
        tasks=dict(stack_blocks=[dict(id=100000, position=0, file='stack_blocks/100000.json',
                                      sha256=m.digest(record))]))))
    (tmp_path / 'manifest.sha256').write_text(m.digest(manifest))
    m.verify(tmp_path)
    record.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='layout hash'):
        m.verify(tmp_path)


def test_summary_does_not_impute_random():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / 'baselines/rsi'))
    try:
        import summarize
        rows = [dict(task='stack_blocks', layout_id=100000, success=True,
                     official_score=100, steps=140, wall_s=10)]
        result = summarize.per_cell(rows)
        assert len(result) == 54
        assert result['stack_blocks']['official_score'] == 100
        assert result['stack_blocks_random']['official_score'] is None
        with pytest.raises(ValueError, match='duplicate'):
            summarize.per_cell(rows * 2)
    finally:
        sys.path.pop(0)
