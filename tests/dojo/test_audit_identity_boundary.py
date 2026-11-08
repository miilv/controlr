"""The offline v2 scorer must stay outside controller import closures and index."""
import ast
from collections import deque
from pathlib import Path

import pytest


V2 = 'scripts.perception.audit_identity_v2'


def _import_graph(root):
    files = {'.'.join(p.relative_to(root).with_suffix('').parts).removesuffix('.__init__'): p
             for base in ('dojo', 'controlr', 'scripts') for p in (root / base).rglob('*.py')}
    graph = {}
    for module, path in files.items():
        package = module if path.name == '__init__.py' else module.rpartition('.')[0]
        imports = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package.split('.')[:len(package.split('.')) - node.level + 1] if node.level else []
                target = '.'.join(prefix + ([node.module] if node.module else []))
                imports.add(target)
                imports.update(f'{target}.{a.name}' for a in node.names)
            elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and ((isinstance(node.func, ast.Name) and node.func.id == '__import__')
                       or (isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'))):
                imports.add(node.args[0].value)
        # Importing a submodule also executes any local parent package initializers.
        graph[module] = {'.'.join(name.split('.')[:i]) for name in imports
                         for i in range(1, len(name.split('.')) + 1)} & files.keys()
    return graph


def _controller_path(graph):
    todo = deque((module, [module]) for module in graph
                 if any(module == base or module.startswith(base + '.')
                        for base in ('dojo.runtime', 'dojo.skills', 'dojo.agent', 'dojo.bridge')))
    seen = set()
    while todo:
        module, path = todo.popleft()
        if module == V2:
            return path
        if module not in seen:
            seen.add(module)
            todo.extend((other, path + [other]) for other in graph.get(module, ()))
    return None


def test_v2_outside_inference_index_and_controller_import_closure():
    from dojo.legality.profiles import controlr_inference
    root = Path(__file__).resolve().parents[2]
    assert root / 'scripts/perception/audit_identity_v2.py' not in controlr_inference(root).index
    assert _controller_path(_import_graph(root)) is None


@pytest.mark.parametrize('edge', ['from . import audit_identity_v2',
                                  'from scripts.perception.audit_identity_v2 import AuditIdentityV2',
                                  'import scripts.perception.audit_identity_v2',
                                  'importlib.import_module("scripts.perception.audit_identity_v2")'])
def test_boundary_detects_transitive_import(tmp_path, edge):
    for name, text in {'dojo/runtime/policy.py': 'from scripts.perception import helper',
                       'scripts/perception/helper.py': edge,
                       'scripts/perception/audit_identity_v2.py': ''}.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    assert _controller_path(_import_graph(tmp_path)) == [
        'dojo.runtime.policy', 'scripts.perception.helper', V2]
