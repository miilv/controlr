"""Git-backed ratchet comparisons must retain pins after a subtree import."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location(
    'subtree_scan', Path(__file__).resolve().parents[2] / 'scripts/legality/scan.py')
scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan)


def test_nested_base_comparison(tmp_path, capsys):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    (tmp_path / 'README').write_text('parent')
    git('add', 'README')
    git('commit', '-qm', 'parent')
    parent = git('rev-parse', 'HEAD')
    root = tmp_path / 'controlr'
    folder = root / 'scripts/legality'
    folder.mkdir(parents=True)
    (folder / 'scan.py').write_text('# scanner')
    (folder / 'allowlist.json').write_text('{"old": 1}')
    (folder / 'transition-review.json').write_text('{}')
    git('add', 'controlr')
    git('commit', '-qm', 'import')
    base = git('rev-parse', 'HEAD')
    assert scan.base_pins(parent, root) == (None, None, None)
    assert 'bootstrap' in capsys.readouterr().err
    previous, reviewed, gains = scan.base_pins(base, root)
    assert previous == {'old': 1} and reviewed == {}
    assert 'allowlist grew relative to base' in scan.gate([], {'old': 2}, previous)
    (folder / 'allowlist.json').unlink()
    git('add', '-u')
    git('commit', '-qm', 'broken pins')
    with pytest.raises(SystemExit, match='allowlist.json is missing'):
        scan.base_pins('HEAD', root)
    with pytest.raises(subprocess.CalledProcessError):
        scan.base_pins('invalid-base', root)
