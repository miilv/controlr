"""D-C19's unattended block generation (agreed with Mikhail's side, bus 30 Oct 0008Z / 0035Z). It NEVER runs until the
lead says so, after both freeze manifests are committed:

    SIMLAB_GPU=0 scripts/simlab/rdpy.sh -u scripts/dojo/dc19_generate.py --out DIR --manifest FILE --lock LOCK \\
        --v6-checkout ~/dojo-j5-gen-v6 --go

1. Salt: 32 bytes from /dev/urandom, written as hex to ``DIR/.salt`` (0400) and never printed; only sha256(salt) is
   published (the manifest's ``seed_salt_sha256``).
2. Generation: every board task, ids 100100-100119, salted (``gen_layouts.salted_seed``: the random streams start from
   H(id || salt), the reset seed too). fill_egg_holder runs on its measurement set's generator (``--v6-checkout``, the
   b009d17c simlab/v6 tree) with the same salted streams. Raw records go to a private work dir beside DIR, never DIR.
3. Presettle, physics-only, per task (``dojo.simlab.presettle --reserved-slice``, then ``--keep-raw-unmoved``), under
   ``--lock``. An id is stable iff it has a settled record afterwards; the settled:false tasks (make_toast and the two
   conveyors) run their RAW record, stable by the same check (the agreed rule).
4. Selection: per task the first 10 stable ids ascending; tranche A = the first 5, tranche B = the next 5; a short cell
   says so (``shortfall``), never padded; ``unstable_count`` = the unstable ids among 100100-100119.
5. DIR/<task>/<id>[.settled].json (0400) for the selected ids only, the manifest (schema dc19-sealed/1) at ``--manifest``
   (outside DIR), the work dir deleted.
6. Prints only sha256(manifest) and per cell "task n_stable n_unstable". Nothing else (no listing, no record content, no
   presettle output: its log stays in the work dir and is deleted with it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDS = list(range(100100, 100120))
RAW_TASKS = ("make_toast", "match_and_pick_from_conveyor", "pick_from_conveyor_by_image")   # settled: false
V6_TASKS = ("fill_egg_holder",)
TRANCHE = 5


def tranches(stable: list[int]) -> dict:
    """The agreed selection from a cell's stable ids: A = first 5, B = next 5, shortfall explicit."""
    s = sorted(stable)[:2 * TRANCHE]
    return {"tranche_a": s[:TRANCHE], "tranche_b": s[TRANCHE:], "shortfall": 2 * TRANCHE - len(s)}


def draw_salt(out: Path) -> str:
    """32 random bytes (hex) into out/.salt, mode 0400; returned, never printed."""
    salt = os.urandom(32).hex()
    fd = os.open(out / ".salt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as f:
        f.write(salt)
    return salt


def generate_raw(task: str, layout_id: int, salt: str, v6_checkout: str | None) -> dict | None:
    """One salted raw record (None if the generator finds no valid layout)."""
    from dojo.simlab import gen_layouts as G
    if task in V6_TASKS:
        base = G.salted_seed(layout_id, salt)       # the same salted streams through the v6 generator's own formula
        code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from dojo.simlab import gen_layouts as G6\n"
                "try:\n    print(json.dumps(G6.generate(sys.argv[2], int(sys.argv[3]))))\n"
                "except G6.LayoutError:\n    print('null')")
        r = subprocess.run([sys.executable, "-c", code, v6_checkout, task, str(base)], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": f"{v6_checkout}:{os.environ.get('PYTHONPATH', '')}"})
        if r.returncode != 0:
            raise SystemExit(f"v6 generator failed for {task} (rc {r.returncode})")
        lay = json.loads(r.stdout.strip().splitlines()[-1])
        if lay is not None:
            lay["_simlab"].update({"id": layout_id, "seed": G.env_seed(base), "seed_salt": G.salt_fingerprint(salt)})
        return lay
    try:
        return G.generate(task, layout_id, seed_salt=salt)
    except G.LayoutError:
        return None


def presettle(task: str, work: Path, lock: str, log: Path) -> None:
    """Pass 1, then --keep-raw-unmoved, physics-only, on the work dir (output only to the private log)."""
    env = {**os.environ, "DOJO_DATA": str(work), "DOJO_GPU_LOCK": lock, "SIMLAB_GPU": "1"}
    ids = [str(i) for i in IDS]
    for extra in ([], ["--keep-raw-unmoved"]):
        with open(log, "a") as f:
            subprocess.run(["bash", str(ROOT / "scripts/simlab/rdpy.sh"), "-m", "dojo.simlab.presettle", "--task", task,
                            "--ids", *ids, "--reserved-slice", *extra], env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=3600, check=False)


def provenance() -> dict:
    from dojo.simlab import gen_layouts as G
    from dojo.simlab import presettle as P
    dep = ROOT / ".deployed.json"
    commit = json.loads(dep.read_text()).get("head") if dep.exists() else None
    if commit is None:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        commit = r.stdout.strip() or None
    rd = os.environ.get("RD_HOME", "")
    rd_commit = (Path(rd) / "robodojo_commit.txt").read_text().strip()[:7] if (Path(rd) / "robodojo_commit.txt").exists() else "unknown"
    try:
        import importlib.metadata as md
        isaac = md.version("isaacsim")
    except Exception:  # noqa: BLE001
        isaac = "unknown"
    return {"repo": "controlr", "commit": commit or "unknown", "GENERATOR": G.GENERATOR,
            "settler": f"{P.SETTLER} @ {commit or 'unknown'}", "robodojo": rd_commit, "isaacsim": isaac}


def run(out: Path, manifest: Path, lock: str, tasks: list[str], v6_checkout: str | None,
        gen=generate_raw, settle=presettle, prov=provenance) -> dict:
    from dojo.simlab import sealed
    out.mkdir(parents=True)
    work = out.parent / f".{out.name}.work-{int(time.time())}"
    try:
        salt = draw_salt(out)
        man = {"schema": sealed.SCHEMA, "block": "D-C19", "generator": prov(),
               "seed_salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": {}}
        counts = {}
        for task in tasks:
            raw_dir = work / "simlab" / "layouts" / task
            raw_dir.mkdir(parents=True, exist_ok=True)
            for i in IDS:
                lay = gen(task, i, salt, v6_checkout)
                if lay is not None:
                    (raw_dir / f"{i}.json").write_text(json.dumps(lay, indent=1))
            settle(task, work, lock, work / "presettle.log")
            stable = [i for i in IDS if (raw_dir / f"{i}.settled.json").is_file()]
            raw = task in RAW_TASKS
            sel = tranches(stable)
            d = out / task
            d.mkdir()
            recs = {}
            for i in sel["tranche_a"] + sel["tranche_b"]:
                src = raw_dir / sealed.record_name(i, not raw)
                data = src.read_bytes()
                fd = os.open(d / sealed.record_name(i, not raw), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                recs[str(i)] = hashlib.md5(data).hexdigest()
            entry = {"settled": not raw, "records": recs, **sel, "unstable_count": len(IDS) - len(stable)}
            if task in V6_TASKS:
                entry["generator"] = {"GENERATOR": "simlab/v6", "checkout": str(v6_checkout)}
            sealed.check_task(task, entry)
            man["tasks"][task] = entry
            counts[task] = (len(stable), len(IDS) - len(stable))
        manifest.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        sealed.load_manifest(manifest)              # the loader's rules, once more, on the written file
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "counts": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the sealed block directory (must not exist)")
    ap.add_argument("--manifest", required=True, help="the manifest file (outside --out)")
    ap.add_argument("--lock", required=True, help="the GPU lock presettle takes (agreed per D-C19; no default)")
    ap.add_argument("--v6-checkout", required=True, help="fill_egg_holder's simlab/v6 generator tree (b009d17c)")
    ap.add_argument("--tasks", default=None, help="comma list (default: every task with dev layouts)")
    ap.add_argument("--go", action="store_true", help="required: the lead's go, after both freezes")
    a = ap.parse_args(argv)
    if not a.go:
        raise SystemExit("dc19_generate: not without --go (the lead's go, after both freeze manifests are committed)")
    out, manifest = Path(a.out).expanduser().resolve(), Path(a.manifest).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: the block is generated once")
    if out in manifest.parents:
        raise SystemExit("the manifest lives outside the block directory")
    from dojo.simlab import paths
    tasks = a.tasks.split(",") if a.tasks else sorted(p.name for p in paths.LAYOUTS.iterdir() if p.is_dir())
    res = run(out, manifest, a.lock, tasks, os.path.expanduser(a.v6_checkout))
    print(f"manifest sha256 {res['sha256']}")
    for t, (ns, nu) in sorted(res["counts"].items()):
        print(f"{t} n_stable {ns} n_unstable {nu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
