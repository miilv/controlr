"""The research board must exclude sealed rows and replay its recovery checkpoint."""
import hashlib
import pytest
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

BOARD = Path(__file__).resolve().parents[2] / 'scripts/dojo/board'


def test_state_rejects_sealed_and_unknown_ids(tmp_path):
    spec = importlib.util.spec_from_file_location('board_state', BOARD / 'state.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'rows.json'
    path.write_text(json.dumps([{'layout_id': 100099}, {'layout_id': 100100},
                                {'layout_id': 100120}, {'layout_id': None},
                                {'seed': 100020}]))
    assert module.load_rows(path) == [{'layout_id': 100099}, {'seed': 100020}]


def test_fetch_rejects_directory_before_open(tmp_path):
    # Execute only the remote read-only program with poisoned report/episode contents.
    import ast
    tree = ast.parse((BOARD / 'fetch_rows.py').read_text())
    remote = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in n.targets))
    for name in ['c4b_sealed_100100', 'c4b_unknown', 'c4b_dc19_100005']:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'x_episodes.jsonl').write_text(json.dumps(
            {'layout_id': 100005, 'success': True, 'score': 1.0}) + '\n')
        with gzip.open(folder / 'x_report.json.gz', 'wt') as report:
            json.dump({'config': {'must_never_read': True}}, report)
    (tmp_path / 'c4b_alias_100005').symlink_to(tmp_path / 'c4b_sealed_100100', target_is_directory=True)
    result = subprocess.run([sys.executable, '-c', remote, str(tmp_path / '*/*_episodes.jsonl'), '{}', '[]'],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'rows': [], 'cfgs': {}, 'norow': []}


def test_checkpoint_replays_strict_board(tmp_path):
    if subprocess.run(['git', 'cat-file', '-e', '69890e2^{commit}'],
                      capture_output=True).returncode:
        pytest.skip('Replay requires repository history (shallow CI checkout)')
    manifest = json.loads((BOARD / 'snapshot/manifest.json').read_text())
    for source in (BOARD / 'snapshot').glob('*.json'):
        shutil.copyfile(source, tmp_path / source.name)
    subprocess.run([sys.executable, str(BOARD / 'cells.py'), '--state-dir', str(tmp_path),
                    '--main-ref', manifest['repo_main']],
                   capture_output=True, text=True, check=True)
    ns = json.loads((tmp_path / 'northstar.json').read_text())
    rows = json.loads((tmp_path / 'cells.json').read_text())
    assert len(rows) == 54
    assert round(ns['headline_strict'], 2) == 51.53
    assert abs(sum(r['w'] for r in rows) - 1) < 1e-10
    assert ns['dc19_tranche'] is None


def test_fetch_filters_payload_ids_and_preserves_allowed_row(tmp_path):
    import ast
    tree = ast.parse((BOARD / 'fetch_rows.py').read_text())
    remote = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in n.targets))
    folder = tmp_path / 'c4b_allowed_100005'
    folder.mkdir()
    (folder / 'x_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'task': 'make_kong', 'layout_id': 100005, 'success': True, 'score': 1.0},
        {'layout_id': 100100, 'success': True}, {'layout_id': None, 'success': True}]))
    result = subprocess.run([sys.executable, '-c', remote, str(tmp_path / '*/*_episodes.jsonl'), '{}', '[]'],
                            capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)['rows']
    assert [(r['task'], r['layout_id']) for r in rows] == [('make_kong', 100005)]


def test_snapshot_manifest_and_no_sealed_rows():
    manifest = json.loads((BOARD / 'snapshot/manifest.json').read_text())
    for name, expected in manifest['inputs'].items():
        payload = (BOARD / 'snapshot' / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        for row in json.loads(payload):
            assert row.get('layout_id', row.get('seed')) < 100100
