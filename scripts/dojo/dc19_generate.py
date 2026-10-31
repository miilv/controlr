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
3b. imitate_sorting_sequence (``sealed.DEMO_TASKS``): its support-arm demo, planned and validated blind by
   ``dojo.simlab.demo_gen`` under ``--lock`` on the stable ids, ascending, in rounds until 10 validate or the stable ids
   run out (agreed 30 Oct). An id is eligible iff it is stable AND its demo validated; ``demo_dropped`` counts the stable
   ids tried whose demo did not.
4. Selection: per task the first 10 eligible (stable) ids ascending; tranche A = the first 5, tranche B = the next 5; a
   short cell says so (``shortfall``), never padded; ``unstable_count`` = the unstable ids among 100100-100119.
5. DIR/<task>/<id>[.settled].json (0400) for the selected ids only (plus ``<id>.demo.pkl``, 0400, for imitate; its md5
   in the manifest's ``demos``), the manifest (schema dc19-sealed/1) at ``--manifest`` (outside DIR), the work dir deleted.
6. Prints only sha256(manifest) and per cell "task n_stable n_unstable" (plus "n_demo_dropped" for imitate). Nothing
   else (no listing, no record content, no presettle / demo_gen output: their log stays in the work dir and is deleted
   with it).

Options for a later, separately agreed generation (all off by default: the D-C19 block of 30 Oct is exactly this
script at 127b65c4 with none of them):
- ``--raw-check``: the settled:false tasks are judged by ``presettle --raw-check`` (``raw_index.json`` "raw ok": the
  raw record as stored is RoboDojo-stable, runs 250 steps, is at rest with the belt subtracted, nothing fell), instead
  of "a settled record exists", which they never get (D-C19: 0 / 20 each; journal 2026-10-31-dojo-dc19-supp-gen).
- ``--demo-speed S``: demo_gen's ``--speed`` (near / carry joint speeds; the dev demos used 1.5, at 1.0 about a third
  of the orders plan past demo_gen's 900-env-step cap).
- ``--block NAME``: the manifest's block name (default "D-C19"), e.g. a supplementary block's own name.
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


def presettle(task: str, work: Path, lock: str, log: Path, raw_check: bool = False) -> None:
    """Pass 1, then --keep-raw-unmoved, physics-only, on the work dir (output only to the private log); with
    ``raw_check`` one ``--raw-check`` run instead (a settled:false task: raw_index.json, no record written)."""
    env = {**os.environ, "DOJO_DATA": str(work), "DOJO_GPU_LOCK": lock, "SIMLAB_GPU": "1"}
    ids = [str(i) for i in IDS]
    for extra in ([["--raw-check"]] if raw_check else [[], ["--keep-raw-unmoved"]]):
        with open(log, "a") as f:
            subprocess.run(["bash", str(ROOT / "scripts/simlab/rdpy.sh"), "-m", "dojo.simlab.presettle", "--task", task,
                            "--ids", *ids, "--reserved-slice", *extra], env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=3600, check=False)


def demo_cmd(task: str, ids: list[int], speed: float | None = None) -> list[str]:
    """The demo_gen command (``speed``: its ``--speed``; None = demo_gen's default, as the 30 Oct block ran)."""
    return (["bash", str(ROOT / "scripts/simlab/rdpy.sh"), "-m", "dojo.simlab.demo_gen", "--task", task,
             "--ids", *[str(i) for i in ids]] + ([] if speed is None else ["--speed", f"{float(speed):g}"]))


def make_demos(task: str, work: Path, lock: str, ids: list[int], log: Path, speed: float | None = None) -> None:
    """demo_gen on ``ids`` (physics-only, under ``lock``, output only to the private log); a validated demo is
    ``work/simlab/traj/<task>/<id>.pkl`` afterwards (demo_gen deletes the ones that fail validation)."""
    env = {**os.environ, "DOJO_DATA": str(work), "DOJO_GPU_LOCK": lock, "SIMLAB_GPU": "1"}
    with open(log, "a") as f:
        subprocess.run(demo_cmd(task, ids, speed), env=env, stdout=f, stderr=subprocess.STDOUT,
                       timeout=4 * 3600, check=False)


def demo_ids(task: str, work: Path, lock: str, stable: list[int], demos=make_demos) -> tuple[list[int], int]:
    """(stable ids with a validated demo, ascending; how many tried ones failed), in rounds over the stable ids
    ascending until 2 x TRANCHE validate or none are left."""
    traj = work / "simlab" / "traj" / task
    ok: list[int] = []
    tried: list[int] = []
    left = sorted(stable)
    while left and len(ok) < 2 * TRANCHE:
        batch, left = left[:2 * TRANCHE - len(ok)], left[2 * TRANCHE - len(ok):]
        demos(task, work, lock, batch, work / "demo_gen.log")
        tried += batch
        ok += [i for i in batch if (traj / f"{i}.pkl").is_file()]
    return sorted(ok), len(tried) - len(ok)


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


def _seal(src: Path, dst: Path) -> str:
    """Copy ``src`` into the block as ``dst`` (0400, O_EXCL); its md5."""
    data = src.read_bytes()
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return hashlib.md5(data).hexdigest()


def run(out: Path, manifest: Path, lock: str, tasks: list[str], v6_checkout: str | None,
        gen=generate_raw, settle=presettle, prov=provenance, demos=make_demos, raw_check: bool = False,
        demo_speed: float | None = None, block: str = "D-C19") -> dict:
    """The block (see the module doc). ``raw_check`` / ``demo_speed`` / ``block``: the off-by-default options; when
    one is set, the manifest's ``generator.options`` records them (the 30 Oct block has no such key)."""
    import functools
    from dojo.simlab import presettle as P
    from dojo.simlab import sealed
    opts = {k: v for k, v in (("raw_check", P.RAW_CHECK if raw_check else None), ("demo_speed", demo_speed))
            if v is not None}
    demos = functools.partial(demos, speed=demo_speed) if demo_speed is not None else demos
    out.mkdir(parents=True)
    work = out.parent / f".{out.name}.work-{int(time.time())}"
    try:
        salt = draw_salt(out)
        man = {"schema": sealed.SCHEMA, "block": block, "generator": {**prov(), **({"options": opts} if opts else {})},
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
            raw = task in RAW_TASKS
            if raw and raw_check:                   # a settled:false task runs its raw record: judge THAT record
                settle(task, work, lock, work / "presettle.log", raw_check=True)
                ri = raw_dir / "raw_index.json"
                index = json.loads(ri.read_text()) if ri.is_file() else {}
                stable = [i for i in IDS if (index.get(str(i)) or {}).get("status") == P.RAW_OK
                          and (raw_dir / f"{i}.json").is_file()]
            else:
                settle(task, work, lock, work / "presettle.log")
                stable = [i for i in IDS if (raw_dir / f"{i}.settled.json").is_file()]
            eligible, dropped = (demo_ids(task, work, lock, stable, demos) if task in sealed.DEMO_TASKS
                                 else (stable, None))
            sel = tranches(eligible)
            d = out / task
            d.mkdir()
            recs, dems = {}, {}
            for i in sel["tranche_a"] + sel["tranche_b"]:
                recs[str(i)] = _seal(raw_dir / sealed.record_name(i, not raw), d / sealed.record_name(i, not raw))
                if dropped is not None:
                    dems[str(i)] = _seal(work / "simlab" / "traj" / task / f"{i}.pkl", d / sealed.demo_name(i))
            entry = {"settled": not raw, "records": recs, **sel, "unstable_count": len(IDS) - len(stable)}
            if dropped is not None:
                entry.update({"demos": dems, "demo_dropped": dropped})
            if task in V6_TASKS:
                entry["generator"] = {"GENERATOR": "simlab/v6", "checkout": str(v6_checkout)}
            sealed.check_task(task, entry)
            man["tasks"][task] = entry
            counts[task] = (len(stable), len(IDS) - len(stable)) + (() if dropped is None else (dropped,))
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
    ap.add_argument("--raw-check", action="store_true",
                    help="settled:false tasks judged by presettle --raw-check ('raw ok'), not 'a settled record "
                         "exists' (off: the 30 Oct D-C19 block)")
    ap.add_argument("--demo-speed", type=float, default=None,
                    help="demo_gen --speed for the imitate demos (default: demo_gen's own, as the 30 Oct block)")
    ap.add_argument("--block", default="D-C19", help="the manifest's block name (a supplementary block's own name)")
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
    res = run(out, manifest, a.lock, tasks, os.path.expanduser(a.v6_checkout), raw_check=a.raw_check,
              demo_speed=a.demo_speed, block=a.block)
    print(f"manifest sha256 {res['sha256']}")
    for t, (ns, nu, *nd) in sorted(res["counts"].items()):
        print(f"{t} n_stable {ns} n_unstable {nu}" + (f" n_demo_dropped {nd[0]}" if nd else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
