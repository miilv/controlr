"""Our inference roots for the legality tool shared with the partner team.

No task-scoped policies are exempted. An instruction read is permitted to a planner,
but still inventoried; a skill boundary must consume explicit arguments instead.
"""
import ast
from pathlib import Path
import re

from .audit import Profile

ROOT = Path(__file__).resolve().parents[2]
GENERAL = ('dojo/general/', 'dojo/skills/general/', 'dojo/perception/general/', 'dojo/runtime/general/')


def controlr_inference(root=ROOT):
    root = Path(root).resolve()
    index = sorted(p for base in ('dojo', 'controlr') for p in (root / base).rglob('*.py')
                   if '/legality/' not in str(p))
    roots = ['dojo.skills.planner:plan', 'dojo.skills.planner:solve',
             'dojo.runtime.supervisor:RuleSupervisor.decide',
             'dojo.runtime.policy:RuntimePolicy.__init__', 'dojo.runtime.policy:RuntimePolicy.reset',
             'dojo.runtime.policy:RuntimePolicy.next', 'dojo.runtime.policy:RuntimePolicy._setup',
             'dojo.agent.policy:AgentPolicy.__init__',
             'dojo.perception.inverse_layout:estimate', 'dojo.perception.j1_adapter:perceive',
             'dojo.skills.ported.harness_robot:HarnessRobot.cmd_obs']
    registry = root / 'skills/registry.yaml'
    if registry.exists():
        roots += re.findall(r'^\s+entry:\s*[\'\"]?(dojo\.[\w.]+:[\w]+)', registry.read_text(), re.M)
    # Scan ALL dispatch-table implementations even when absent from the chooser.
    # Their references are globals, including imported ported and route-only tables.
    roots += ['dojo.skills.library:<module>', 'dojo.skills.ported:<module>']
    for path in (root / 'dojo/skills/ported/skills').rglob('skill_robodojo.py'):
        if any(isinstance(n, ast.FunctionDef) and n.name == 'run' for n in ast.parse(path.read_text()).body):
            roots.append('.'.join(path.relative_to(root).with_suffix('').parts) + ':run')
    # The clean paths cannot inherit a legacy exemption, even through a helper.
    for base in GENERAL:
        for path in (root / base).rglob('*.py'):
            module = '.'.join(path.relative_to(root).with_suffix('').parts).removesuffix('.__init__')
            roots.append(module + ':<module>')
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    roots.append(module + ':' + node.name)
                elif isinstance(node, ast.ClassDef):
                    roots += [module + ':' + node.name + '.' + method.name for method in node.body
                              if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))]
    resources = [p for pattern in ('dojo/agent/prompts/*.md', 'dojo/perception/agentic/prompts/*.md',
                                  'dojo/perception/agentic/vendor/PERCEIVER-PROTOCOL.md')
                 for p in root.glob(pattern)]
    return Profile(name='controlr-inference', index=index, roots=sorted(set(roots)), root_dir=root,
                   scan_globals=True, report_unresolved=True, resources=resources)
