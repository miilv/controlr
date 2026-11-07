"""Transitive checks must catch leaks below wrappers and dynamic boundaries."""
from pathlib import Path
import sys

# Scripts are intentionally not installed as controller packages.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.legality.scan import Scanner, declared_entries, gate


def test_transitive_import_alias_attribute_and_registry(tmp_path):
    pkg = tmp_path / 'dojo'
    pkg.mkdir()
    (pkg / 'entry.py').write_text('from . import hidden as h\nREG = {"x": h.worker}\ndef run():\n    return REG["x"]()\n')
    (pkg / 'hidden.py').write_text('def worker():\n    return deeper()\ndef deeper():\n    return "make_kong", "tasks.json", "ported_plan"\n')
    findings = Scanner(tmp_path, ['make_kong']).scan(['dojo.entry:run'])
    assert {'task_name', 'task_config', 'task_switch'} <= {f.category for f in findings}
    assert any(f.symbol == 'dojo.hidden.deeper' and f.entry == 'dojo.entry:run' for f in findings)


def test_unresolved_dynamic_calls_are_visible(tmp_path):
    (tmp_path / 'dojo').mkdir()
    (tmp_path / 'dojo/x.py').write_text('def run(callback):\n    return callback()\n')
    assert any(f.category == 'unresolved' and f.evidence == 'callback'
               for f in Scanner(tmp_path, ['make_kong']).scan(['dojo.x:run']))


def test_allowlist_only_shrinks_and_general_cannot_be_exempted(tmp_path):
    from scripts.legality.scan import Finding
    f = Finding('dojo/general/x.py', 1, 'x', 'task_name', 'make_kong', 'x')
    assert gate([f], [f.key])
    assert gate([], ['new'], ['old'])
    assert not gate([], ['old'], ['old', 'retired'])


def test_repository_legality_baseline():
    import json
    findings = Scanner(ROOT).scan(declared_entries(ROOT))
    allowed = json.loads((ROOT / 'scripts/legality/allowlist.json').read_text())
    problems = gate(findings, allowed)
    assert not problems, '\n'.join(problems[:30])


def test_general_entry_cannot_inherit_a_legacy_dependency_exemption(tmp_path):
    (tmp_path / 'dojo/general').mkdir(parents=True)
    (tmp_path / 'dojo/general/entry.py').write_text('from dojo.old import helper\ndef run():\n    return helper()\n')
    (tmp_path / 'dojo/old.py').write_text('def helper():\n    return "make_kong"\n')
    findings = Scanner(tmp_path, ['make_kong']).scan(['dojo.general.entry:run'])
    assert gate(findings, [f.key for f in findings])


def test_long_strings_and_prompt_resources_are_scanned(tmp_path):
    (tmp_path / 'dojo').mkdir()
    (tmp_path / 'dojo/manual.md').write_text('A very long instruction: ' + 'ordinary text ' * 20 + 'make_kong checker 0.035\n')
    found = Scanner(tmp_path, ['make_kong']).scan(['file:dojo/manual.md'])
    assert {'task_name', 'checker', 'checker_tolerance'} <= {f.category for f in found}


def test_exemption_occurrence_budgets_cannot_grow():
    from scripts.legality.scan import Finding
    a = Finding('dojo/old.py', 1, 'old', 'task_name', 'make_kong', 'old')
    b = Finding('dojo/old.py', 2, 'old', 'task_name', 'make_kong', 'old')
    assert gate([a, b], {a.key: 1})
    assert gate([], {a.key: 2}, {a.key: 1})
    assert not gate([a], {a.key: 1}, {a.key: 2})
