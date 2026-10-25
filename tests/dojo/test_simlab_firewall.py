"""Asset firewall: no dojo/ code reads Assets/Eval_Layout or the eval-paired Assets/Traj demos (pure + subprocess)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from dojo.simlab import paths

REPO = Path(__file__).resolve().parents[2]


def test_violation_rules(tmp_path):
    a = tmp_path / "RoboDojo" / "Assets"
    assert paths.eval_asset_violation(a / "Eval_Layout" / "RoboDojo" / "push_T" / "0" / "push_T_3.json")
    assert paths.eval_asset_violation(a / "Eval_Layout")
    assert paths.eval_asset_violation(a / "Traj" / "RoboDojo" / "imitate_sorting_sequence" / "0" / "12.pkl")
    assert paths.eval_asset_violation(a / "Traj" / "RoboDojo" / "imitate_sorting_sequence")       # listing it
    assert paths.eval_asset_violation(a / "Traj")
    assert paths.eval_asset_violation(a / "Traj" / "RoboDojo" / "some_new_task" / "0.pkl")        # unknown: closed
    assert paths.eval_asset_violation(a / "Traj" / "RoboDojo" / "make_kong" / "2.pkl") is None    # generic demos
    assert paths.eval_asset_violation(a / "Traj" / "RoboDojo" / "play_tic_tac_toe" / "1_4.pkl") is None
    assert paths.eval_asset_violation(a / "Object" / "RoboDojo" / "Rigid" / "block" / "00000" / "metadata.json") is None
    assert paths.eval_asset_violation(tmp_path / "dojo-data" / "simlab" / "traj" / "x" / "1.pkl") is None  # ours
    assert paths.eval_asset_violation(3) is None                                                   # an fd


def test_firewall_blocks_reads_in_process(tmp_path):
    """Audit hooks cannot be removed, so this runs in a fresh interpreter."""
    a = tmp_path / "Assets"
    ev = a / "Eval_Layout" / "RoboDojo" / "push_T" / "0"
    imitate = a / "Traj" / "RoboDojo" / "imitate_sorting_sequence" / "0"
    kong = a / "Traj" / "RoboDojo" / "make_kong"
    for d in (ev, imitate, kong):
        d.mkdir(parents=True)
    (ev / "push_T_0.json").write_text("{}")
    (imitate / "0.pkl").write_bytes(b"x")
    (kong / "0.pkl").write_bytes(b"x")
    code = f"""
import glob, os, shutil, sys
from pathlib import Path
from dojo.simlab import paths
paths.install_asset_firewall()
blocked = [lambda: open({str(ev / 'push_T_0.json')!r}).read(),
           lambda: Path({str(ev / 'push_T_0.json')!r}).read_text(),
           lambda: os.listdir({str(ev)!r}),
           lambda: list(os.scandir({str(ev)!r})),
           lambda: glob.glob({str(a / 'Eval_Layout' / '*')!r}),
           lambda: shutil.copyfile({str(imitate / '0.pkl')!r}, {str(tmp_path / 'stolen.pkl')!r}),
           lambda: open({str(imitate / '0.pkl')!r}, 'rb').read()]
for i, f in enumerate(blocked):
    try:
        f()
    except PermissionError:
        continue
    sys.exit(f"not blocked: {{i}}")
assert open({str(kong / '0.pkl')!r}, 'rb').read() == b"x"
print("ok")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=60, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr + r.stdout


def test_eval_paired_demo_is_routed_never_opened(tmp_path, monkeypatch):
    from dojo.simlab import runner as R
    monkeypatch.setattr(paths, "DATA", tmp_path)
    opened = []
    load = R._routed_load_pkl(lambda p: opened.append(p) or {"joint_path": []})
    t = tmp_path / "RoboDojo" / "Assets" / "Traj" / "RoboDojo"
    d = load(t / "imitate_sorting_sequence" / "0" / f"{paths.SEED_BASE + 1}.pkl")
    assert d["_simlab_no_demo"] and not opened                    # the stub, the eval file untouched
    load(t / "make_kong" / "1.pkl")
    assert opened == [t / "make_kong" / "1.pkl"]                    # generic demos load as usual
    with pytest.raises(PermissionError):
        load(t / "some_new_task" / "0" / "3.pkl")
    with pytest.raises(RuntimeError):                               # eval seeds are refused, not stubbed
        load(t / "imitate_sorting_sequence" / "0" / "3.pkl")


def _string_constants(tree: ast.AST):
    """(node, parent) for every str constant that is not a docstring."""
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and id(child) not in docs:
                yield child, parent


def test_no_dojo_code_builds_eval_asset_paths():
    """Outside the firewall itself (paths.py), "Eval_Layout" / "Traj" may appear in dojo/ code only as a
    membership guard (``"Eval_Layout" in p.parts``), never as a path component to open or list."""
    bad = []
    for f in sorted((REPO / "dojo").rglob("*.py")):
        rel = f.relative_to(REPO).as_posix()
        if rel == "dojo/simlab/paths.py":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for c, parent in _string_constants(tree):
            v = c.value
            hit = "Eval_Layout" in v or v == "Traj" or "Assets/Traj" in v or "/Traj/" in v
            if not hit:
                continue
            guard = isinstance(parent, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in parent.ops)
            if not guard:
                bad.append(f"{rel}:{c.lineno}: {v!r}")
    assert not bad, "eval asset paths built outside the firewall:\n" + "\n".join(bad)
