"""The converged legality engine (Ilia's PR #436, 304c5511, with our A1/A2 in it): the detection gains it brings to our
two profiles, each planted here (each of the first seven fails on the engine before the port), plus Ilia's standalone
planted tests for the opt-in inventory switches (``scan_globals``, ``report_unresolved``) that ride along unused by
``controlr-glib`` / ``controlr-hybrid-agent``. Pure AST: nothing here runs a simulator or imports a skill."""

from pathlib import Path

import pytest

from dojo.legality import audit as AU


def _pkg(tmp_path, files: dict) -> list[Path]:
    out = []
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        out.append(p)
    return out


def _whys(p, names=("stack_blocks",)):
    return [(v.func, v.why) for v in AU.audit_callgraph(profile=p, names=list(names))]


# ---------------------------------------------------------------------------------------------- gains on our profiles
def test_a_relative_import_inside_a_package_init_is_followed(tmp_path):
    """``from .impl import helper`` in ``pkg/__init__.py`` resolves against the package itself (it used to drop one
    level and miss the helper), at module level and inside a function."""
    idx = _pkg(tmp_path, {
        "ww/__init__.py": "from .impl import helper\n\n\ndef entry(robot):\n    return helper(robot)\n\n\n"
                          "def lazy(robot):\n    from .impl import other\n    return other(robot)\n",
        "ww/impl.py": "def helper(robot):\n    return robot.instruction\n\n\ndef other(robot):\n    return 'stack_blocks'\n"})
    for root, reached, why in (("ww:entry", "ww.impl:helper", ".instruction read"),
                               ("ww:lazy", "ww.impl:other", "task name in a string")):
        p = AU.Profile(name="t", index=idx, roots=[root], root_dir=tmp_path)
        assert any(f == reached and why in w for f, w in _whys(p)), root


def test_a_root_given_by_its_reexported_name_is_audited(tmp_path):
    """A root named through a package re-export (``ww:helper`` for ``ww.impl:helper``) is resolved, not reported as
    'not in the index'."""
    idx = _pkg(tmp_path, {"ww/__init__.py": "from .impl import helper\n",
                          "ww/impl.py": "def helper(robot):\n    return robot.instruction\n"})
    p = AU.Profile(name="t", index=idx, roots=["ww:helper"], root_dir=tmp_path)
    whys = _whys(p)
    assert not any("not in the index" in w for _, w in whys), whys
    assert any(f == "ww.impl:helper" and ".instruction read" in w for f, w in whys), whys


def test_a_class_body_comes_with_a_class_reached_through_a_module(tmp_path):
    """A1 follow-up: ``pkg.mod.Bank()`` and ``getattr(M, "Bank")()`` bring the class body (class constants) of a
    field-only class too, one with no method that would bring it along."""
    idx = _pkg(tmp_path, {
        "zz/__init__.py": "", "zz/sub/__init__.py": "",
        "zz/sub/mod.py": "class Bank:\n    names = 'stack_blocks'\n",
        "zz/entry.py": "import zz.sub.mod\nimport zz.sub.mod as M\n\n\ndef a():\n    return zz.sub.mod.Bank()\n\n\n"
                       "def b():\n    return getattr(M, 'Bank')()\n"})
    for root in ("zz.entry:a", "zz.entry:b"):
        p = AU.Profile(name="t", index=idx, roots=[root], root_dir=tmp_path)
        assert ("zz.sub.mod", "Bank.<body>") in AU.reachable(profile=p), root
        assert any(f == "zz.sub.mod:Bank.<body>" and "task name" in w for f, w in _whys(p)), root


def test_a_pin_is_the_whole_line_and_a_duplicated_pinned_line_fails_closed(tmp_path):
    """Pins match a whole (stripped) line: a fragment that sits inside a longer line exempts nothing, and a pinned
    line that appears twice is ambiguous, so neither copy is exempt (Ilia's carrier test, plus the dispatch case)."""
    idx = _pkg(tmp_path, {"pp/__init__.py": "", "pp/e.py":
                          "def run(obj):\n    x = obj.instruction\n    return obj.instruction.lower()\n"})
    p = AU.Profile(name="t", index=idx, roots=["pp.e:run"], root_dir=tmp_path,
                   carriers={("pp.e", "run"): "x = obj.instruction"})
    whys = [w for _, w in _whys(p)]
    assert ".instruction read (line 2)" not in whys and ".instruction read (line 3)" in whys, whys
    idx[1].write_text("def run(obj):\n    x = obj.instruction\n    x = obj.instruction\n")
    whys = [w for _, w in _whys(p)]
    assert any("carrier line is ambiguous" in w for w in whys), whys
    assert sum(".instruction read" in w for w in whys) == 2, whys
    idx[1].write_text("class S:\n    def go(self, fn, other):\n"
                      "        f = getattr(self, fn) if fn in self.ok else getattr(self, other)\n        return f\n")
    p = AU.Profile(name="t", index=idx, roots=["pp.e:S.go"], root_dir=tmp_path,
                   dispatch={("pp.e", "S.go"): ("getattr(self, fn)", [])})
    whys = [w for _, w in _whys(p)]
    assert "pinned dispatch line changed: re-audit it" in whys and "non-literal getattr (line 3)" in whys, whys


def test_every_task_name_in_one_string_is_reported(tmp_path):
    idx = _pkg(tmp_path, {"tt/__init__.py": "", "tt/e.py": "def run():\n    return 'first stack_blocks then make_kong'\n"})
    p = AU.Profile(name="t", index=idx, roots=["tt.e:run"], root_dir=tmp_path)
    whys = {w for _, w in _whys(p, names=("stack_blocks", "make_kong"))}
    assert {"task name in a string: 'stack_blocks'", "task name in a string: 'make_kong'"} <= whys, whys


def test_task_accessor_names_are_task_symbols(tmp_path):
    idx = _pkg(tmp_path, {"aa/__init__.py": "", "aa/e.py":
                          "def run(cfg, label_candidates, instance_range):\n"
                          "    return label_candidates(cfg), instance_range, cfg.quota_lists\n"})
    p = AU.Profile(name="t", index=idx, roots=["aa.e:run"], root_dir=tmp_path)
    whys = {w for _, w in _whys(p)}
    assert {"task symbol label_candidates (line 2)", "task symbol instance_range (line 2)"} <= whys, whys


def test_run_and_log_directories_are_task_data_paths_but_an_attribute_name_is_not(tmp_path):
    """Development logs and run directories hold the checker's outcomes: ``runs`` / ``logs`` / ``Traj`` path parts
    are task data paths. The literal attribute name of a ``getattr`` (``getattr(policy, "log", None)``) is an
    identifier, not a path, and stays clean."""
    idx = _pkg(tmp_path, {"dd/__init__.py": "", "dd/e.py":
                          "from pathlib import Path\n\n\ndef run(root, policy):\n"
                          "    a = Path(root) / 'runs'\n    b = 'logs/episode.txt'\n"
                          "    return a, b, getattr(policy, 'log', None)\n"})
    p = AU.Profile(name="t", index=idx, roots=["dd.e:run"], root_dir=tmp_path)
    whys = {w for _, w in _whys(p)}
    assert {"task data path: 'runs'", "task data path: 'logs/episode.txt'"} <= whys, whys
    assert "task data path: 'log'" not in whys, whys


# ---------------------------------------------------------------- Ilia's planted tests for the opt-in inventory switches
def _inventory(tmp_path, text, names=("make_kong",)):
    idx = _pkg(tmp_path, {"pkg/__init__.py": "", "pkg/entry.py": text})
    p = AU.Profile(name="t", index=idx, roots=["pkg.entry:run"], root_dir=tmp_path, scan_globals=True,
                   report_unresolved=True)
    return [v.why for v in AU.audit_callgraph(profile=p, names=list(names))]


def test_unknown_dynamic_loader_is_not_silently_certified(tmp_path):
    whys = _inventory(tmp_path, 'import importlib.util\ndef run(path):\n'
                                '    return importlib.util.spec_from_file_location("external", path)\n')
    assert any(w.startswith("dynamic code loader") for w in whys), whys


def test_multiline_exemptions_pin_the_offending_line_and_each_match(tmp_path):
    whys = _inventory(tmp_path, 'def run():\n    """generic opener\n    make_kong\n    stack_blocks\n'
                                '    task/RoboDojo/config/hidden.yml\n    """\n', names=("make_kong", "stack_blocks"))
    assert "task name in a string: 'make_kong' (line 3)" in whys, whys
    assert "task name in a string: 'stack_blocks' (line 4)" in whys, whys
    assert any(w.startswith("task data path:") and "(line 5)" in w for w in whys), whys


def test_path_loaded_class_constants_are_module_code_too(tmp_path):
    whys = _inventory(tmp_path, 'class Bank:\n    candidates = "make_kong"\ndef run():\n    return Bank.candidates\n')
    assert any("task name" in w for w in whys), whys


def test_known_pure_leaves_and_field_only_records_are_resolved(tmp_path):
    whys = _inventory(tmp_path, 'from dataclasses import dataclass, field\nimport numpy as np\n@dataclass\n'
                                'class Mask:\n    value: list = field(default_factory=list)\n'
                                'def run(x):\n    return Mask(list(x)), float(1), np.ascontiguousarray(x)\n')
    assert not any(w.startswith("unresolved") for w in whys), whys


def test_leaf_resolution_cannot_hide_callbacks_or_file_reads(tmp_path):
    whys = _inventory(tmp_path, 'import numpy as np\ndef run(float, np, x):\n'
                                '    return float(x), np.ascontiguousarray(x), np.load(x)\n')
    for call in ("float", "np.ascontiguousarray", "np.load"):
        assert any(f"unresolved call {call}" in w for w in whys), (call, whys)


@pytest.mark.parametrize('precise', [False, True])
def test_attribute_identifier_exemption_preserves_real_path_reads(tmp_path, precise):
    idx = _pkg(tmp_path, {'fixture.py': '''def run(policy):
    a = getattr(policy, "log", None)
    setattr(policy, "logs", 0)
    b = hasattr(policy, "runs")
    delattr(policy, "Traj")
    open("logs/x.json")
    open("Traj/episode")
    c = getattr(policy, "logger", "logs/fallback.json")
    setattr(policy, "log", "logs/value.json")
    return getattr(policy, "stack_blocks"), a, b, c
'''})
    p = AU.Profile(name='attribute-paths', index=idx, roots=['fixture:run'], root_dir=tmp_path,
                   scan_globals=precise)
    vs = AU.audit_callgraph(profile=p, names=['stack_blocks'])
    paths = [v.why for v in vs if v.why.startswith('task data path:')]
    for line, path in enumerate(('logs/x.json', 'Traj/episode', 'logs/fallback.json', 'logs/value.json'), 6):
        assert any((f'(line {line})' in why if precise else repr(path) in why) for why in paths), (path, paths)
    for name in ('log', 'logs', 'runs', 'Traj'):
        assert not any(why.startswith(f'task data path: {name!r}') for why in paths), (name, paths)
    assert any('task name in a string:' in v.why and 'stack_blocks' in v.why for v in vs)


def test_portable_profile_registration_preserves_our_cli_default(tmp_path, monkeypatch, capsys):
    assert {'controlr-inference', 'controlr-glib', 'controlr-hybrid-agent'} <= set(AU.PROFILES)
    monkeypatch.setitem(AU.PROFILES, 'controlr-inference',
                        lambda: AU.Profile(name='controlr-inference', root_dir=tmp_path))
    assert AU.main(['--json']) == 0
    import json
    assert json.loads(capsys.readouterr().out)['profile'] == 'controlr-inference'
