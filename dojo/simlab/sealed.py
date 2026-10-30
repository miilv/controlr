"""Sealed-block records (robot-harness D-C19): layouts neither side has opened, run once through the official path.

The block is a directory ``DIR/<task>/<id>.settled.json`` (``<id>.json`` for a task the manifest marks
``"settled": false``: make_toast and the conveyors run raw) plus a manifest, schema ``dc19-sealed/1`` (agreed with
Mikhail's side, bus 30 Oct 0008Z / 0035Z)::

    {"schema": "dc19-sealed/1", "block": "D-C19",
     "generator": {"repo": "controlr", "commit": "<sha>", "GENERATOR": "simlab/v15", "settler": "<sha>",
                   "robodojo": "f57e163", "isaacsim": "5.1.0"},
     "seed_salt_sha256": "<64 hex>", "created_utc": "<ts>",
     "tasks": {"<task>": {"settled": true, "records": {"<id>": "<md5>", ...},
                          "tranche_a": [<ids>], "tranche_b": [<ids>], "unstable_count": <int>,
                          "shortfall": <int, optional>}}}

Rules (``load_manifest``; Mikhail's side enforces the same on its own path):
- every id in [100100, 100119]; ``tranche_a`` / ``tranche_b`` ascending, disjoint, every A id below every B id, together
  exactly the ``records``; 5 ids each unless ``shortfall`` (the cell's missing stable ids, never padded) says otherwise:
  A takes the first min(5, n) of the n = 10 - shortfall stable ids, B the rest;
- tranche B runs only once ``tranche_b.preregistered`` exists beside the manifest;
- position k of a tranche is its k-th id (``position_id``; ids may skip, the unstable ones are not in the block).

Enforced here and again in RoboDojo's process (``dev_eval.py`` with ``SIMLAB_SEALED_MANIFEST``): only listed ids run;
each record's bytes are md5-verified before staging; a mismatch, a missing file or an unlisted id refuses the whole call;
staging writes the verified bytes into the run's layout dir (read-only) and nowhere else; nothing prints, logs or
returns a record's content (id, md5, pass / fail only). Run once (``claim``): one O_EXCL marker per (task, id) in
``$COORD_DIR/scored/<block>/<task>-<id>.json`` before the server starts; a second claim refuses; the one disclosed
infrastructure re-run (before any verdict was read) needs ``DC19_INFRA_RERUN=<reason>`` and makes ``<marker>.rerun``.

    python -m dojo.simlab.sealed verify --dir DIR --manifest FILE        # task id md5 pass / fail, then its own md5
    python -m dojo.simlab.sealed stage --dir DIR --manifest FILE --task T --ids "100100 100101" --dest RUN/layouts
    python -m dojo.simlab.sealed claim --coord-dir C --manifest FILE --task T --ids "..." --run-id ID
    python -m dojo.simlab.sealed settled --manifest FILE --task T          # -> "1" / "0"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SCHEMA = "dc19-sealed/1"
BLOCK_IDS = (100100, 100119)
TRANCHE = 5
PREREGISTERED = "tranche_b.preregistered"     # beside the manifest: tranche B may run
SALT_FILE = ".salt"                           # DIR/.salt (0400, never printed): the only non-record file DIR may hold
GENERATOR_KEYS = ("repo", "commit", "GENERATOR", "settler", "robodojo", "isaacsim")
MD5 = re.compile(r"^[0-9a-f]{32}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SealedError(SystemExit):
    """A sealed-block rule refused the call (never carries a record's content)."""


def _ascending(xs) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:]))


def check_task(task: str, e: dict) -> None:
    recs, a, b = e.get("records"), e.get("tranche_a"), e.get("tranche_b")
    if not isinstance(recs, dict) or not isinstance(e.get("settled"), bool) or not isinstance(a, list) \
            or not isinstance(b, list) or not isinstance(e.get("unstable_count"), int) or e["unstable_count"] < 0:
        raise SealedError(f"task {task!r}: needs settled (bool), records {{id: md5}}, tranche_a / tranche_b (lists), "
                          "unstable_count (int >= 0)")
    for i, h in recs.items():
        if not str(i).isdigit() or not MD5.match(str(h)):
            raise SealedError(f"task {task!r}: bad entry id {i!r} / md5")
        if not BLOCK_IDS[0] <= int(i) <= BLOCK_IDS[1]:
            raise SealedError(f"task {task!r}: id {i} outside the block [{BLOCK_IDS[0]}, {BLOCK_IDS[1]}]")
    if not all(isinstance(x, int) for x in a + b):
        raise SealedError(f"task {task!r}: tranche ids must be integers")
    if not _ascending(a) or not _ascending(b) or set(a) & set(b) or (a and b and max(a) >= min(b)):
        raise SealedError(f"task {task!r}: tranches must be ascending, disjoint, every A id below every B id")
    if sorted(a + b) != sorted(int(i) for i in recs):
        raise SealedError(f"task {task!r}: tranche_a + tranche_b must equal the records exactly")
    short = e.get("shortfall", 0)
    if not isinstance(short, int) or not 0 <= short <= 2 * TRANCHE:
        raise SealedError(f"task {task!r}: shortfall must be an integer in [0, {2 * TRANCHE}]")
    n = 2 * TRANCHE - short
    if len(a) != min(TRANCHE, n) or len(b) != n - min(TRANCHE, n):
        raise SealedError(f"task {task!r}: tranche sizes {len(a)} / {len(b)} do not match shortfall {short} "
                          f"(A = the first min({TRANCHE}, n) stable ids, B the rest; never padded)")


def load_manifest(path: str | os.PathLike) -> dict:
    """The manifest after every ``dc19-sealed/1`` rule (refuses anything else); ``_path`` keeps where it lives."""
    m = json.loads(Path(path).read_text())
    if m.get("schema") != SCHEMA:
        raise SealedError(f"sealed manifest {path}: schema must be {SCHEMA!r}")
    for k in ("block", "created_utc"):
        if not isinstance(m.get(k), str) or not m[k]:
            raise SealedError(f"sealed manifest {path}: {k!r} required")
    g = m.get("generator")
    if not isinstance(g, dict) or any(not isinstance(g.get(k), str) or not g[k] for k in GENERATOR_KEYS):
        raise SealedError(f"sealed manifest {path}: generator needs {list(GENERATOR_KEYS)}")
    if not SHA256.match(str(m.get("seed_salt_sha256", ""))):
        raise SealedError(f"sealed manifest {path}: seed_salt_sha256 must be 64 hex")
    tasks = m.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise SealedError(f"sealed manifest {path}: no 'tasks'")
    for t, e in tasks.items():
        if not isinstance(e, dict):
            raise SealedError(f"sealed manifest {path}: task {t!r} is not an object")
        check_task(t, e)
    m["_path"] = str(Path(path).resolve())
    return m


def task_entry(manifest: dict, task: str) -> dict:
    e = manifest["tasks"].get(task)
    if e is None:
        raise SealedError(f"sealed block: task {task!r} is not in the manifest")
    return e


def record_name(layout_id: int | str, settled: bool) -> str:
    return f"{int(layout_id)}{'.settled' if settled else ''}.json"


def position_id(manifest: dict, task: str, tranche: str, k: int) -> int:
    """The k-th (0-based) id of a task's tranche ("a" / "b"): the agreed position -> id mapping."""
    ids = task_entry(manifest, task)[f"tranche_{tranche}"]
    if not 0 <= k < len(ids):
        raise SealedError(f"sealed block {task}: tranche {tranche} has {len(ids)} ids, no position {k}")
    return ids[k]


def b_preregistered(manifest: dict) -> bool:
    return (Path(manifest["_path"]).parent / PREREGISTERED).exists()


def check_runnable(manifest: dict, task: str, ids) -> None:
    """Requested ids must be the task's tranche A ids, or tranche B ids once it is pre-registered."""
    e = task_entry(manifest, task)
    bad = []
    for i in (int(x) for x in ids):
        if i in e["tranche_a"]:
            continue
        if i in e["tranche_b"]:
            if not b_preregistered(manifest):
                bad.append(f"{i} tranche B not pre-registered")
            continue
        bad.append(f"{i} unlisted")
    if bad:
        raise SealedError(f"sealed block {task}: refused ({'; '.join(bad)})")


def verify_records(block_dir: str | os.PathLike, manifest: dict, task: str, ids) -> list[tuple[int, str, bytes]]:
    """(id, md5, bytes) of each requested record after its md5 check; refuses on any unlisted or not-runnable id,
    missing or empty file or mismatch (all ids are checked first, nothing is staged on a partial pass)."""
    check_runnable(manifest, task, ids)
    e = task_entry(manifest, task)
    out, bad = [], []
    for i in (int(x) for x in ids):
        p = Path(block_dir) / task / record_name(i, e["settled"])
        if not p.is_file() or p.stat().st_size == 0:
            bad.append(f"{i} missing or empty")
            continue
        data = p.read_bytes()
        got = hashlib.md5(data).hexdigest()
        if got != e["records"][str(i)]:
            bad.append(f"{i} md5 {got} != {e['records'][str(i)]}")
            continue
        out.append((i, got, data))
    if bad:
        raise SealedError(f"sealed block {task}: refused ({'; '.join(bad)})")
    return out


def stage(block_dir, manifest: dict, task: str, ids, dest: str | os.PathLike) -> list[tuple[int, str]]:
    """Verify, then write each record's verified bytes to ``dest/<task>/`` (mode 0400). Returns (id, md5)."""
    e = task_entry(manifest, task)
    recs = verify_records(block_dir, manifest, task, ids)
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
    """RoboDojo's side (dev_eval.py): every staged record of ``task`` is listed, runnable and still matches its md5."""
    e = task_entry(manifest, task)
    files = list(files)
    bad = []
    for p in files:
        p = Path(p)
        i = p.name.split(".")[0]
        want = e["records"].get(i)
        if want is None or p.name != record_name(i, e["settled"]):
            bad.append(f"{p.name} unlisted")
        elif hashlib.md5(p.read_bytes()).hexdigest() != want:
            bad.append(f"{i} md5 mismatch")
    if not bad and files:
        try:
            check_runnable(manifest, task, [Path(p).name.split(".")[0] for p in files])
        except SealedError as err:
            bad.append(str(err))
    if bad or not files:
        raise SealedError(f"sealed block {task}: staged records refused in {layout_dir} ({'; '.join(bad) or 'none'})")


def verify_dir(block_dir: str | os.PathLike, manifest: dict) -> tuple[list[tuple[str, int, str, bool]], list[str]]:
    """The hash-only check of a whole block: (task, id, md5, pass) per listed record, and the problems that are not a
    record (an unlisted file or task folder, an empty record). Never opens a record beyond hashing it."""
    root = Path(block_dir)
    rows, problems = [], []
    for task, e in sorted(manifest["tasks"].items()):
        for i in sorted(int(x) for x in e["records"]):
            p = root / task / record_name(i, e["settled"])
            if not p.is_file():
                rows.append((task, i, "-", False))
                continue
            got = hashlib.md5(p.read_bytes()).hexdigest()
            if p.stat().st_size == 0:
                problems.append(f"empty {task}/{p.name}")
            rows.append((task, i, got, got == e["records"][str(i)] and p.stat().st_size > 0))
    listed = {(t, record_name(i, e["settled"])) for t, e in manifest["tasks"].items() for i in e["records"]}
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_dir():
            if p.name not in manifest["tasks"]:
                problems.append(f"unlisted task folder {p.name}")
                continue
            for f in sorted(p.iterdir()):
                if (p.name, f.name) not in listed:
                    problems.append(f"unlisted file {p.name}/{f.name}")
        elif p.name != SALT_FILE:
            problems.append(f"unlisted file {p.name}")
    if not root.is_dir():
        problems.append(f"no block directory {root}")
    return rows, problems


def self_md5() -> str:
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()


def claim(coord_dir: str | os.PathLike, manifest: dict, task: str, ids, run_id: str) -> list[str]:
    """Run once: an O_EXCL marker ``scored/<block>/<task>-<id>.json`` per id, all ids checked before any is made. A
    cell with a marker refuses, unless ``DC19_INFRA_RERUN=<reason>`` (once per cell: ``<marker>.rerun``)."""
    d = Path(coord_dir) / "scored" / manifest["block"]
    d.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in ids]
    check_runnable(manifest, task, ids)
    rerun = os.environ.get("DC19_INFRA_RERUN", "").strip()
    paths = [d / f"{task}-{i}.json" for i in ids]
    done = [p for p in paths if p.exists()]
    if done and not rerun:
        raise SealedError(f"sealed block {task}: already claimed {[p.name for p in done]} (run once; an infrastructure "
                          "re-run before any verdict needs DC19_INFRA_RERUN=<reason>)")
    again = [p for p in done if Path(str(p) + ".rerun").exists()]
    if again:
        raise SealedError(f"sealed block {task}: the one re-run is spent for {[p.name for p in again]}")
    made = []
    rec = {"run_id": run_id, "task": task, "block": manifest["block"], "manifest": manifest["_path"],
           "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for p, i in zip(paths, ids):
        target = Path(str(p) + ".rerun") if p in done else p
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({**rec, "id": i, **({"infra_rerun_reason": rerun} if p in done else {})}) + "\n")
        made.append(str(target))
    return made


def _ids(spec: str) -> list[str]:
    return [x for x in re.split(r"[ ,]+", spec.strip()) if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sealed-block tools (print id + md5 + pass / fail only)")
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
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--manifest", required=True)
    c = sub.add_parser("claim")
    c.add_argument("--coord-dir", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--task", required=True)
    c.add_argument("--ids", required=True)
    c.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    m = load_manifest(a.manifest)
    if a.cmd == "settled":
        print("1" if task_entry(m, a.task)["settled"] else "0")
        return 0
    if a.cmd == "verify":
        rows, problems = verify_dir(a.dir, m)
        for task, i, h, ok in rows:
            print(f"{task} {i} {h} {'pass' if ok else 'fail'}", flush=True)
        for p in problems:
            print(f"FAIL {p}", flush=True)
        n_ok = sum(1 for r in rows if r[3])
        print(f"verify: {n_ok}/{len(rows)} pass, {len(problems)} other problems; verifier md5 {self_md5()}", flush=True)
        return 0 if n_ok == len(rows) and not problems else 1
    if a.cmd == "claim":
        for p in claim(a.coord_dir, m, a.task, _ids(a.ids), a.run_id):
            print(f"[sealed] claimed {Path(p).name}", flush=True)
        return 0
    for i, h in stage(a.dir, m, a.task, _ids(a.ids), a.dest):
        print(f"[sealed] {a.task} {i} md5 {h} pass", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
