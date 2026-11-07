"""Our integration/proposed additions to the shared partner legality engine."""
import json
from pathlib import Path
import sys

from dojo.legality import audit as A
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.legality.scan import gate, inventory


def profile(tmp_path, files, root='pkg.entry:run'):
    paths = []
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        paths.append(path)
    return A.Profile('fixture', index=paths, roots=[root], root_dir=tmp_path,
                     scan_globals=True, report_unresolved=True)


def test_imported_global_dispatch_table_reaches_the_hidden_policy(tmp_path):
    p = profile(tmp_path, {
        'pkg/entry.py': 'from . import hidden as h\nBANK = {"x": h.worker}\ndef run():\n    return BANK["x"]()\n',
        'pkg/hidden.py': 'RULE = "make_kong"\ndef worker():\n    return RULE\n'})
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any(v.func == 'pkg.hidden:<module>' and 'task name' in v.why for v in found)
    assert any('pkg.entry:run' in v.path for v in found)


def test_unresolved_callbacks_are_explicit(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run(callback):\n    return callback()\n'})
    assert any('unresolved call callback' in v.why for v in A.audit_callgraph(profile=p, names=[]))


def test_policy_switch_checker_tolerance_and_index_rule_are_visible(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'CATEGORY_ATTRIBUTES = {"number_card": lambda idx: int(idx)}\ndef run(cfg):\n    x = cfg.watch_relocate_task\n    return 0.035 # checker threshold\n'})
    whys = [v.why for v in A.audit_callgraph(profile=p, names=['make_kong'])]
    assert any('task switch' in why for why in whys)
    assert any('checker-shaped tolerance' in why for why in whys)
    assert any('category-index identity' in why for why in whys)


def test_reflective_pin_cannot_exempt_a_second_dynamic_read(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run(obj, name):\n    x = getattr(obj, name)\n    return getattr(obj, name + "suffix")\n'})
    p.reflective_ok = {('pkg.entry', 'run'): 'x = getattr(obj, name)'}
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any('non-literal getattr (line 3)' in v.why for v in found)
    p.reflective_ok[('pkg.entry', 'run')] = 'x = getattr(obj, name) # changed'
    assert any('pinned reflective line changed' in v.why for v in A.audit_callgraph(profile=p, names=['make_kong']))


def test_prompt_resources_and_long_strings_are_included(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run():\n    return 1\n'})
    resource = tmp_path / 'manual.md'
    resource.write_text('ordinary text ' * 20 + 'make_kong checker rules\n')
    p.resources = [resource]
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any(v.func == 'manual.md:<resource>' and 'task name' in v.why for v in found)


def test_intrinsic_catalogue_files_are_not_task_data(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run(root):\n    return root / "metadata.json", root / "description.json", root / "object.usd"\n'})
    assert not [v for v in A.audit_callgraph(profile=p, names=['make_kong']) if 'task data' in v.why]


def row(file='dojo/old.py', path=None):
    return {'key': 'old', 'file': file, 'line': 1, 'category': 'task_name', 'why': 'test',
            'path': path or ['dojo.old:run']}


def test_allowlist_budgets_only_shrink_and_general_dependencies_cannot_inherit_them():
    assert not gate([row()], {'old': 1}, {'old': 2})
    assert gate([row(), row()], {'old': 1})
    assert gate([], {'old': 2}, {'old': 1})
    assert gate([row(file='dojo/general/x.py')], {'old': 1})
    assert gate([row(path=['dojo.general.x:run', 'dojo.old:run'])], {'old': 1})


def test_repository_shared_profile_matches_its_frozen_baseline():
    report = inventory(ROOT)
    allowed = json.loads((ROOT / 'scripts/legality/allowlist.json').read_text())
    problems = gate(report['findings'], allowed)
    assert not problems, '\n'.join(problems[:30])


def test_reexported_registry_roots_are_followed(tmp_path):
    p = profile(tmp_path, {
        'pkg/entry.py': 'from .hidden import run\n',
        'pkg/hidden.py': 'def run():\n    return "make_kong"\n'})
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any(v.func == 'pkg.hidden:run' and 'task name' in v.why for v in found)
    assert not any('not in the index' in v.why for v in found)


def test_legacy_first_call_path_cannot_hide_clean_root_reachability():
    r = row()
    r['clean_reachable'] = True
    assert gate([r], {'old': 1})
    r.pop('clean_reachable')
    r['root'] = 'dojo.general.alias:run'
    assert gate([r], {'old': 1})


def test_evaluator_termination_and_historical_log_sources_are_inventoried(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run(env):\n    return env.episode_over(), "logs/answer.json", "Traj/reference.json"\n'})
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any('termination feedback surface' in v.why for v in found)
    assert sum('task data path' in v.why for v in found) == 2


def test_carrier_pin_is_one_line_and_duplicate_lines_fail_closed(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run(obj):\n    x = obj.instruction\n    return obj.instruction.lower()\n'})
    p.carriers = {('pkg.entry', 'run'): 'x = obj.instruction'}
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert not any(v.why == '.instruction read (line 2)' for v in found)
    assert any(v.why == '.instruction read (line 3)' for v in found)
    (tmp_path / 'pkg/entry.py').write_text('def run(obj):\n    x = obj.instruction\n    x = obj.instruction\n')
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any('carrier line is ambiguous' in v.why for v in found)
    assert sum('.instruction read' in v.why for v in found) == 2


def test_path_loaded_helpers_are_explicit_roots_even_without_run(tmp_path):
    from dojo.legality.profiles import controlr_inference
    path = tmp_path / 'dojo/skills/ported/skills/hidden/helper.py'
    path.parent.mkdir(parents=True)
    path.write_text('def acceptance(x):\n    return "make_kong", "task/RoboDojo/tasks/check.py"\n')
    p = controlr_inference(tmp_path)
    target = 'dojo.skills.ported.skills.hidden.helper:acceptance'
    assert target in p.roots
    found = A.audit_callgraph([target], p, names=['make_kong'])
    assert any(v.root == target and 'task data path' in v.why for v in found)


def test_unknown_dynamic_loader_is_not_silently_certified(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'import importlib.util\ndef run(path):\n    return importlib.util.spec_from_file_location("external", path)\n'})
    found = A.audit_callgraph(profile=p, names=['make_kong'])
    assert any(v.why.startswith('dynamic code loader') for v in found)


def test_reviewed_vendored_checker_predicate_is_detected(tmp_path):
    from dojo.legality.profiles import controlr_inference
    path = tmp_path / 'dojo/skills/ported/skills/pour_p1/skill_robodojo.py'
    path.parent.mkdir(parents=True)
    path.write_text('def _in_target(bounds, pos):\n    return pos[2] > bounds[0][2]\ndef run(robot):\n    verified = _in_target(robot.bounds, robot.pos)\n    return verified\n')
    p = controlr_inference(tmp_path)
    root = 'dojo.skills.ported.skills.pour_p1.skill_robodojo:run'
    found = A.audit_callgraph([root], p, names=['make_kong'])
    assert any(v.why.startswith('audited checker-shaped acceptance predicate') for v in found)
    assert any(v.why.startswith('audited checker predicate drives verified outcome') for v in found)


def test_multiline_exemptions_pin_the_offending_line_and_each_match(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'def run():\n    """generic opener\n    make_kong\n    stack_blocks\n    task/RoboDojo/config/hidden.yml\n    """\n'})
    found = A.audit_callgraph(profile=p, names=['make_kong', 'stack_blocks'])
    assert any(v.why == "task name in a string: 'make_kong' (line 3)" for v in found)
    assert any(v.why == "task name in a string: 'stack_blocks' (line 4)" for v in found)
    assert any(v.why.startswith('task data path:') and '(line 5)' in v.why for v in found)


def test_path_loaded_class_constants_are_module_code_too(tmp_path):
    p = profile(tmp_path, {'pkg/entry.py': 'class Bank:\n    candidates = "make_kong"\ndef run():\n    return Bank.candidates\n'})
    assert any('task name' in v.why for v in A.audit_callgraph(profile=p, names=['make_kong']))
