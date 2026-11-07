#!/usr/bin/env python3
"""Report and debt-ratchet CLI over the legality engine shared with the partner team."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dojo.legality.audit import Index, _walk, task_re
from dojo.legality.profiles import GENERAL, controlr_inference


def category(why):
    for prefix, cat in [('task name', 'task_name'), ('task data', 'task_data'),
                        ('imports task', 'checker_module'), ('task symbol', 'task_vocabulary'),
                        ('task switch', 'task_switch'), ('checker-shaped', 'checker_tolerance'),
                        ('checker provenance', 'checker_provenance'), ('benchmark composite', 'benchmark_composite'), ('category-index', 'identity_source'),
                        ('checker / env', 'checker_state'), ('unresolved', 'unresolved')]:
        if why.startswith(prefix):
            return cat
    if 'instruction' in why or '.task read' in why or "'task'" in why:
        return 'instruction_or_identity'
    return 'reflection'


def inventory(root=ROOT):
    profile = controlr_inference(root)
    idx = Index(profile)
    violations = _walk(idx, profile.roots, task_re())
    clean_roots = [r for r in profile.roots if r.replace('.', '/').startswith(GENERAL)]
    clean = {(v.func, v.why) for v in _walk(idx, clean_roots, task_re())} if clean_roots else set()
    files = {}
    for p in profile.index:
        module = str(p.relative_to(profile.root_dir).with_suffix('')).replace('/', '.').removesuffix('.__init__')
        files[module] = str(p.relative_to(profile.root_dir))
    rows = []
    for v in violations:
        module, qual = v.func.split(':', 1)
        source = idx.mods.get(module)
        match = re.search(r'\(line (\d+)\)', v.why)
        node = idx.funcs.get((module, qual))
        line = int(match[1]) if match else getattr(node, 'lineno', 0)
        evidence = source[1][line - 1].strip() if source and 0 < line <= len(source[1]) else v.why
        file = files.get(module, module)
        if source is None and (profile.root_dir / file).is_file():
            evidence = (profile.root_dir / file).read_text().splitlines()[line - 1].strip()
        row = {**asdict(v), 'file': file, 'line': line, 'category': category(v.why), 'evidence': evidence, 'clean_reachable': (v.func, v.why) in clean}
        # Exact source evidence and reason pin the debt, independent of line-number moves.
        # Editing an exemption requires retiring it, never regenerating the baseline.
        row['key'] = hashlib.sha256(json.dumps([file, qual, re.sub(r' \(line \d+\)', '', v.why), evidence]).encode()).hexdigest()
        rows.append(row)
    return {'profile': profile.name, 'roots': profile.roots,
            'counts': dict(Counter(r['category'] for r in rows)), 'findings': rows}


def gate(rows, allowlist, previous=None):
    counts = Counter(r['key'] for r in rows)
    issues = []
    if previous is not None and any(n > previous.get(k, 0) for k, n in allowlist.items()):
        issues.append('allowlist grew relative to base')
    for r in rows:
        clean_path = r.get('clean_reachable', False) or r.get('root', '').replace('.', '/').startswith(GENERAL) or r['file'].startswith(GENERAL) or any(p.replace('.', '/').startswith(GENERAL) for p in r['path'])
        if counts[r['key']] > allowlist.get(r['key'], 0) or clean_path:
            issues.append(f"{r['file']}:{r['line']} {r['category']}: {r['why']}")
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--candidate', type=Path, help='attach exact audited candidate flags (no code execution)')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--base-ref')
    args = ap.parse_args(argv)
    report = inventory(args.root)
    if args.candidate:
        text = args.candidate.read_text()
        cfg_line = next(line[len('--config '):] for line in text.splitlines() if line.startswith('--config '))
        report['candidate'] = {'sha256': hashlib.sha256(text.encode()).hexdigest(),
                               'runtime_overrides': json.loads(cfg_line.replace(chr(92) + '"', '"')),
                               'flags': text,
                               'scope': 'conservative superset; no path-feasibility claim'}
    output = json.dumps(report, indent=1) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    if args.check:
        relative = 'scripts/legality/allowlist.json'
        allowed = json.loads((ROOT / relative).read_text())
        previous = None
        if args.base_ref:
            subprocess.run(['git', 'cat-file', '-e', args.base_ref + '^{commit}'], cwd=ROOT, check=True, capture_output=True)
            old = subprocess.run(['git', 'show', f'{args.base_ref}:{relative}'], cwd=ROOT, capture_output=True, text=True)
            if old.returncode == 0:
                previous = json.loads(old.stdout)
            elif subprocess.run(['git', 'cat-file', '-e', args.base_ref + ':scripts/legality/scan.py'], cwd=ROOT, capture_output=True).returncode == 0:
                raise SystemExit('base scanner exists but its allowlist is missing')
        problems = gate(report['findings'], allowed, previous)
        print('\n'.join(problems[:100]))
        return bool(problems)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
