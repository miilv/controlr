#!/usr/bin/env python3
"""Conservative, non-importing inference reachability inventory (stdlib only).

Not a proof of legality: receiver types are approximated by matching member names;
unresolved calls are explicit findings. Both sides can supply --entry module:symbol.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTRIES = (
    'dojo.skills.library', 'dojo.skills.planner:plan', 'dojo.skills.planner:solve',
    'dojo.runtime.supervisor:RuleSupervisor', 'dojo.runtime.policy:RuntimePolicy',
    'dojo.agent.policy:AgentPolicy', 'dojo.perception.inverse_layout:estimate',
    'dojo.perception.j1_adapter:infer_task',
)
SCHEMA_NAMES = {'task_map', 'label_groups', 'label_candidates', 'instance_range',
                'fixed_slot_labels', 'scene_keys', 'public_constants', 'task_candidates',
                'load_sets', '_placement_prior', 'quota_lists'}
SWITCH = re.compile(r'\b(?:watch_relocate_task|key_scope_task|sam_task_routes|sam_task_banks|ported_plan|ported_routes|ported_\w+|new_routes)\b')
CHECKER = re.compile(r'checker|scorer|reward_manager|success_flag', re.I)
TERMINATION = re.compile(r'episode_over|env_end|success_latch|termination_feedback')
GENERAL = ('dojo/general/', 'dojo/skills/general/', 'dojo/perception/general/', 'dojo/runtime/general/')


@dataclass(frozen=True, order=True)
class Finding:
    file: str
    line: int
    entry: str
    category: str
    evidence: str
    symbol: str

    @property
    def key(self):
        # Stable through line moves; source changes must be reviewed. All reachable roots are reported; the exemption identifies source evidence.
        return hashlib.sha256(json.dumps([self.file, self.category,
                                         self.evidence, self.symbol]).encode()).hexdigest()


class Scanner:
    def __init__(self, root: Path, task_names=None):
        self.root = root
        self.nodes, self.files, self.module_of = {}, {}, {}
        self.classified = {}
        self.aliases = defaultdict(dict)
        self.members = defaultdict(set)
        self.sources = {}
        for base in ('dojo', 'controlr'):
            for p in sorted((root / base).rglob('*.py')):
                module = '.'.join(p.relative_to(root).with_suffix('').parts)
                if module.endswith('.__init__'):
                    module = module[:-9]
                source = p.read_text()
                tree = ast.parse(source, filename=str(p))
                self.sources[module] = source.splitlines()
                self._index(module, p, tree, '')
        if task_names is None:
            # This is audit tooling, never loaded by inference. Freeze the full
            # canonical table and all observed vocabulary keys in terms.json.
            terms = json.loads((ROOT / 'scripts/legality/terms.json').read_text())
            task_names = terms['tasks']
        self.task_rx = re.compile(r'(?<![\w])(?:' + '|'.join(
            re.escape(t).replace('_', '[_ ]') for t in sorted(task_names, key=len, reverse=True)) + r')(?![\w])', re.I)
        self.prefixes = {k.rsplit('.', i)[0] for k in self.nodes for i in range(1, k.count('.') + 1)}
        self.descendants = defaultdict(set)
        for name in self.nodes:
            for i in range(1, name.count('.') + 1):
                self.descendants[name.rsplit('.', i)[0]].add(name)
        self.edges = defaultdict(set)
        self.unresolved = defaultdict(list)
        self._graph()

    def _index(self, module, path, node, qual):
        key = module + ('.' + qual if qual else '')
        self.nodes[key], self.files[key], self.module_of[key] = node, path, module
        if qual:
            self.members[qual.split('.')[-1]].add(key)
        for child in getattr(node, 'body', []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._index(module, path, child, '.'.join(filter(None, (qual, child.name))))
        # Module aliases plus local imports: conservative union, never execute them.
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for a in n.names:
                    self.aliases[key][a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
            elif isinstance(n, ast.ImportFrom):
                package = module.split('.') if path.name == '__init__.py' else module.split('.')[:-1]
                prefix = '.'.join(package[:len(package) - n.level + 1]) if n.level else ''
                target = '.'.join(filter(None, (prefix, n.module)))
                for a in n.names:
                    self.aliases[key][a.asname or a.name] = target + '.' + a.name

    def _body_nodes(self, node):
        # Do not attribute sibling function bodies to a wrapper or module.
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield from self._body_nodes(child)

    def _resolve(self, expr, key):
        if isinstance(expr, ast.Name):
            name = expr.id
            mod = self.module_of[key]
            aliases = {**self.aliases[mod], **self.aliases[key]}
            if name in aliases:
                return {aliases[name]}
            if name in ('self', 'cls'):
                parent = key
                while parent != mod:
                    parent = parent.rpartition('.')[0]
                    if isinstance(self.nodes.get(parent), ast.ClassDef):
                        return {parent}
            candidates = [key.rsplit('.', 1)[0] + '.' + name, mod + '.' + name]
        elif isinstance(expr, ast.Attribute):
            bases = self._resolve(expr.value, key)
            candidates = [p + '.' + expr.attr for p in bases]
            # Unknown receivers: conservative local member-name expansion.
            # Known external imports stay opaque; never connect np.foo to a
            # repository method named foo. Calls to those are still reported.
            if not bases:
                candidates += list(self.members[expr.attr])
            else:
                return set(candidates)
        else:
            return set()
        return {c for c in candidates if c in self.nodes or c in self.prefixes}

    def _graph(self):
        for key, node in self.nodes.items():
            mod = self.module_of[key]
            if key != mod:
                self.edges[key].add(mod)  # imports execute global tables
            if isinstance(node, ast.ClassDef) or str(self.files[key].relative_to(self.root)).startswith(GENERAL):
                self.edges[key].update(self.descendants[key])
            for n in self._body_nodes(node):
                if isinstance(n, (ast.Name, ast.Attribute)):
                    self.edges[key].update(t for t in self._resolve(n, key) if t in self.nodes)
                if isinstance(n, ast.Call):
                    targets = self._resolve(n.func, key)
                    if not any(t in self.nodes for t in targets):
                        self.unresolved[key].append((n.lineno, ast.unparse(n.func)))
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for target in self.aliases[key].values():
                        while target and target not in self.nodes:
                            target = target.rpartition('.')[0]
                        if target:
                            self.edges[key].add(target)

    def categories(self, path, text):
        cats = []
        if self.task_rx.search(text):
            cats.append('task_name')
        if 'tasks.json' in text or re.search(r'\b(?:' + '|'.join(SCHEMA_NAMES) + r')\b', text) or 'task/RoboDojo/config' in text:
            cats.append('task_config')
        if CHECKER.search(text):
            cats.append('checker')
        if '/ported/' in '/' + path and re.search(r'\binstruction\b', text):
            cats.append('instruction_boundary')
        if CHECKER.search(text) and re.search(r'\d', text):
            cats.append('checker_tolerance')
        if re.match(r'def (?:tower_from_boards|declare_kong|copy_demonstration|play_board_game|row_by_size|align_in_row|sweep_into|pack_objects)\b', text):
            cats.append('benchmark_composite')
        if SWITCH.search(text):
            cats.append('task_switch')
        if TERMINATION.search(text):
            cats.append('termination')
        return cats

    def scan(self, entries):
        findings = set()
        for entry in entries:
            if entry.startswith('file:'):
                path = entry[5:]
                for line, text in enumerate((self.root / path).read_text().splitlines(), 1):
                    for category in self.categories(path, text):
                        findings.add(Finding(path, line, entry, category, text.strip(), '<resource>'))
                continue
            start = entry.replace(':', '.')
            if start not in self.nodes:
                findings.add(Finding('<entry>', 0, entry, 'unresolved', start, start))
                continue
            todo, seen = [start], set()
            while todo:
                key = todo.pop()
                if key in seen:
                    continue
                seen.add(key)
                todo.extend(self.edges[key] - seen)
                if key in self.classified:
                    findings.update(Finding(*row[:2], entry, *row[2:]) for row in self.classified[key])
                    continue
                local = set()
                path = str(self.files[key].relative_to(self.root))
                node = self.nodes[key]
                lines = set(range(getattr(node, 'lineno', 1),
                                  getattr(node, 'end_lineno', len(self.sources[self.module_of[key]])) + 1))
                def exclude_definitions(parent):
                    for child in ast.iter_child_nodes(parent):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            lines.difference_update(range(child.lineno, child.end_lineno + 1))
                        else:
                            exclude_definitions(child)
                exclude_definitions(node)
                for line in sorted(lines):
                    text = self.sources[self.module_of[key]][line - 1].strip()
                    cats = self.categories(path, text)
                    for cat in cats:
                        local.add(Finding(path, line, entry, cat, text, key))
                for line, call in self.unresolved[key]:
                    local.add(Finding(path, line, entry, 'unresolved', call, key))
                self.classified[key] = [(f.file, f.line, f.category, f.evidence, f.symbol) for f in local]
                findings.update(local)
        grouped = {}
        roots = defaultdict(set)
        for f in findings:
            ident = (f.file, f.line, f.category, f.evidence, f.symbol)
            grouped[ident] = f
            roots[ident].add(f.entry)
        return sorted(Finding(f.file, f.line, ';'.join(sorted(roots[k])), f.category, f.evidence, f.symbol)
                      for k, f in grouped.items())


def declared_entries(root):
    entries = list(DEFAULT_ENTRIES)
    # Read registry as data only, including candidates (no cherry-picking).
    path = root / 'skills/registry.yaml'
    if path.exists():
        entries += re.findall(r'^\s+entry:\s*[\'\"]?(dojo\.[\w.]+:[\w]+)', path.read_text(), re.M)
    for pattern in ('dojo/agent/prompts/*.md', 'dojo/perception/agentic/prompts/*.md',
                    'dojo/perception/agentic/vendor/PERCEIVER-PROTOCOL.md'):
        entries.extend('file:' + str(p.relative_to(root)) for p in root.glob(pattern))
    # Future general-system files are automatically roots, even before registration.
    for base in GENERAL:
        for p in (root / base).rglob('*.py'):
            entries.append('.'.join(p.relative_to(root).with_suffix('').parts).removesuffix('.__init__'))
    return sorted(set(entries))


def gate(findings, allowlist, previous=None):
    allowed = set(allowlist)
    problems = []
    if previous is not None and allowed - set(previous):
        problems.append('allowlist grew relative to base')
    for f in findings:
        if f.key not in allowed or f.file.startswith(GENERAL) or any(
                e.replace('.', '/').startswith(GENERAL) for e in f.entry.split(';')):
            problems.append(f'{f.file}:{f.line} {f.entry} {f.category}: {f.evidence}')
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--entry', action='append')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--base-ref')
    a = ap.parse_args(argv)
    findings = Scanner(a.root).scan(a.entry or declared_entries(a.root))
    report = {'entries': a.entry or declared_entries(a.root), 'counts': dict(Counter(f.category for f in findings)),
              'findings': [dict(asdict(f), key=f.key) for f in findings]}
    text = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end='')
    if a.check:
        relative = 'scripts/legality/allowlist.json'
        allowed = json.loads((a.root / relative).read_text())
        previous = None
        if a.base_ref:
            raw = subprocess.run(['git', 'show', f'{a.base_ref}:{relative}'], cwd=a.root, capture_output=True, text=True)
            # Bootstrap PR only: the first frozen baseline has no predecessor.
            if raw.returncode == 0:
                previous = json.loads(raw.stdout)
            else:
                subprocess.run(['git', 'cat-file', '-e', a.base_ref + '^{commit}'],
                               cwd=a.root, check=True, capture_output=True)
                existed = subprocess.run(['git', 'cat-file', '-e',
                                          a.base_ref + ':scripts/legality/scan.py'],
                                         cwd=a.root, capture_output=True)
                if existed.returncode == 0:
                    raise SystemExit('base scanner exists but its allowlist is missing')
        problems = gate(findings, allowed, previous)
        for problem in problems:
            print(problem)
        return bool(problems)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
