"""Sealed-block records (robot-harness D-C19): layouts neither side has opened, run once through the official path.

The block is a directory ``DIR/<task>/<id>.settled.json`` (``<id>.json`` for a task the manifest marks
``"settled": false``: make_toast and the conveyors run raw) plus a manifest (JSON)::

    {"block": "D-C19 tranche A", "tasks": {"stack_blocks": {"settled": true, "records": {"100100": "<md5>", ...}}, ...}}

Rules, enforced here and again in RoboDojo's process (``dev_eval.py`` with ``SIMLAB_SEALED_MANIFEST``):
- only ids the manifest lists for the task run, and only ids >= ``paths.RESERVED_MIN_ID`` (the joint block's range);
- each record's bytes are md5-verified against the manifest before staging; a mismatch, a missing file or an unlisted
  id refuses the whole call;
- staging writes the verified bytes straight into the run's layout dir (read-only) and nowhere else;
- nothing here prints, logs or returns a record's content: callers get (id, md5, pass / fail) only.

    python -m dojo.simlab.sealed stage --dir DIR --manifest FILE --task T --ids "100100 100101" --dest RUN/layouts
    python -m dojo.simlab.sealed settled --manifest FILE --task T          # -> "1" / "0"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dojo.simlab import paths

MD5 = re.compile(r"^[0-9a-f]{32}$")


class SealedError(SystemExit):
    """A sealed-block rule refused the call (never carries a record's content)."""


def load_manifest(path: str | os.PathLike) -> dict:
    m = json.loads(Path(path).read_text())
    tasks = m.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise SealedError(f"sealed manifest {path}: no 'tasks'")
    for t, e in tasks.items():
        recs = e.get("records") if isinstance(e, dict) else None
        if not isinstance(recs, dict) or not recs or not isinstance(e.get("settled"), bool):
            raise SealedError(f"sealed manifest {path}: task {t!r} needs 'settled' (bool) and 'records' {{id: md5}}")
        for i, h in recs.items():
            if not str(i).isdigit() or not MD5.match(str(h)):
                raise SealedError(f"sealed manifest {path}: task {t!r}: bad entry id {i!r} / md5")
            if int(i) < paths.RESERVED_MIN_ID:
                raise SealedError(f"sealed manifest {path}: task {t!r}: id {i} is not a joint-block id "
                                  f"(>= {paths.RESERVED_MIN_ID})")
    return m


def task_entry(manifest: dict, task: str) -> dict:
    e = manifest["tasks"].get(task)
    if e is None:
        raise SealedError(f"sealed block: task {task!r} is not in the manifest")
    return e


def record_name(layout_id: int | str, settled: bool) -> str:
    return f"{int(layout_id)}{'.settled' if settled else ''}.json"


def verify(block_dir: str | os.PathLike, manifest: dict, task: str, ids) -> list[tuple[int, str, bytes]]:
    """(id, md5, bytes) of each requested record after its md5 check; refuses on any unlisted id, missing file or
    mismatch (all ids are checked first, nothing is staged on a partial pass)."""
    e = task_entry(manifest, task)
    out, bad = [], []
    for i in ids:
        i = int(i)
        want = e["records"].get(str(i))
        if want is None:
            bad.append(f"{i} unlisted")
            continue
        p = Path(block_dir) / task / record_name(i, e["settled"])
        if not p.is_file():
            bad.append(f"{i} missing")
            continue
        data = p.read_bytes()
        got = hashlib.md5(data).hexdigest()
        if got != want:
            bad.append(f"{i} md5 {got} != {want}")
            continue
        out.append((i, got, data))
    if bad:
        raise SealedError(f"sealed block {task}: refused ({'; '.join(bad)})")
    return out


def stage(block_dir, manifest: dict, task: str, ids, dest: str | os.PathLike) -> list[tuple[int, str]]:
    """Verify, then write each record's verified bytes to ``dest/<task>/`` (mode 0400). Returns (id, md5)."""
    e = task_entry(manifest, task)
    recs = verify(block_dir, manifest, task, ids)
    d = Path(dest) / task
    d.mkdir(parents=True, exist_ok=True)
    for i, h, data in recs:
        p = d / record_name(i, e["settled"])
        if p.exists():
            p.chmod(0o600)
            p.unlink()
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    return [(i, h) for i, h, _ in recs]


def check_staged(layout_dir: str | os.PathLike, manifest: dict, task: str, files) -> None:
    """RoboDojo's side (dev_eval.py): every staged record of ``task`` is listed and still matches its md5."""
    e = task_entry(manifest, task)
    bad = []
    for p in files:
        p = Path(p)
        i = p.name.split(".")[0]
        want = e["records"].get(i)
        if want is None or p.name != record_name(i, e["settled"]):
            bad.append(f"{p.name} unlisted")
        elif hashlib.md5(p.read_bytes()).hexdigest() != want:
            bad.append(f"{i} md5 mismatch")
    if bad or not list(files):
        raise SealedError(f"sealed block {task}: staged records refused in {layout_dir} ({'; '.join(bad) or 'none'})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sealed-block staging (prints id + md5 + pass only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--dir", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--ids", required=True, help="space / comma separated layout ids")
    s.add_argument("--dest", required=True)
    q = sub.add_parser("settled")
    q.add_argument("--manifest", required=True)
    q.add_argument("--task", required=True)
    a = ap.parse_args(argv)
    m = load_manifest(a.manifest)
    if a.cmd == "settled":
        print("1" if task_entry(m, a.task)["settled"] else "0")
        return 0
    ids = [x for x in re.split(r"[ ,]+", a.ids.strip()) if x]
    for i, h in stage(a.dir, m, a.task, ids, a.dest):
        print(f"[sealed] {a.task} {i} md5 {h} pass", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
