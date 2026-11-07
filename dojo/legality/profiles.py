"""Our inference roots for the legality tool shared with the partner team.

No task-scoped policies are exempted. An instruction read is permitted to a planner,
but still inventoried; a skill boundary must consume explicit arguments instead.
"""
import ast
from pathlib import Path
import re

from .audit import Profile

ROOT = Path(__file__).resolve().parents[2]
# Transitional additions are inventoried explicitly, not certified as clean roots.
REVIEW_MODULES = ('dojo/perception/general_masks.py', 'dojo/perception/general_prompts.py',
                  'dojo/perception/general_filters.py', 'dojo/perception/base_refit.py',
                  'dojo/skills/coin_contact.py', 'dojo/skills/push_plan.py', 'dojo/skills/torso.py')
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
    # Path-loaded code has no reliable ordinary import edge. Explicitly root
    # every module and callable in the vendored subtree, not just its run().
    dynamic = sorted((root / 'dojo/skills/ported/skills').rglob('*.py'))
    dynamic += [root / p for p in REVIEW_MODULES if (root / p).exists()]
    dynamic += [root / p for p in ('scripts/skills/articulation_links.py',
                                   'controlr/robot/so101/protocol.py', 'controlr/robot/so101/arm_id.py')
                if (root / p).exists()]
    for path in dynamic:
        if path not in index:
            index.append(path)
        module = '.'.join(path.relative_to(root).with_suffix('').parts).removesuffix('.__init__')
        roots.append(module + ':<module>')
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                roots.append(module + ':' + node.name)
            elif isinstance(node, ast.ClassDef):
                roots += [module + ':' + node.name + '.' + f.name for f in node.body
                          if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # cmd_* is selected reflectively. Root all commands so nested cmd_skill and
    # its file-loader cannot disappear behind getattr('cmd_' + cmd).
    adapter = root / 'dojo/skills/ported/harness_robot.py'
    if adapter.exists():
        for cls in ast.parse(adapter.read_text()).body:
            if isinstance(cls, ast.ClassDef):
                roots += ['dojo.skills.ported.harness_robot:' + cls.name + '.' + f.name
                          for f in cls.body if isinstance(f, ast.FunctionDef) and f.name.startswith('cmd_')]
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
