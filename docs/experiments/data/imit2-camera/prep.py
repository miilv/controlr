"""IMIT2 camera A/B: fresh-salt layout + demo preparation on compute3 (physics only, light locks, no model calls).

usage (compute3): python3 prep.py ~/controlr-imit2-cam ~/controlr-imit2-cam-data
Every step is idempotent (a .done stamp); every attempt, log and index is kept.
"""
import fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

H = Path.home()
REPO = Path(sys.argv[1]).resolve()
ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REPO))
from controlr.keyguard import child_env                      # noqa: E402
from dojo.simlab.paths import is_our_measurement_id         # noqa: E402

D = json.loads((ROOT / "design.json").read_text())
DATA = ROOT / D["block"]
GPU_MAX, DISK_MIN = 24576, 10 * 1024 ** 3
LOCKS = ["dojo-gpu-light.lock", "dojo-gpu-light2.lock", "dojo-gpu-light3.lock", "dojo-gpu-light4.lock"]


def note(msg):
    with (ROOT / "notes.log").open("a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")


def used_mib():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       text=True).split()[0])


def free_disk():
    return int(subprocess.check_output(["df", "--output=avail", "-B1", str(ROOT)], text=True).splitlines()[-1])


def run(tag, args, gpu=False):
    done = ROOT / (tag + ".done")
    if done.exists():
        return
    env = child_env({"DOJO_DATA": str(DATA), "SIMLAB_GPU": "0", "SIMLAB_MIN_FREE_MB": "0", "RD_HOME": str(H / "dojo-robodojo"),
                     "DOJO_CATALOG": os.environ.get("DOJO_CATALOG", "")})
    lock = None
    if gpu:
        deadline = time.monotonic() + 4 * 3600
        while time.monotonic() < deadline and lock is None:
            if used_mib() <= GPU_MAX and free_disk() >= DISK_MIN:
                for name in LOCKS:
                    f = (H / name).open("a+")
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        f.close()
                        continue
                    lock = f
                    break
            if lock is None:
                time.sleep(20)
        if lock is None:
            raise RuntimeError("no admitted free light lock")
        lock.seek(0); lock.truncate(); lock.write(f"pid={os.getpid()} stream=IMIT2 prep={tag}\n"); lock.flush()
        note(f"START {tag} lock={lock.name} used_MiB={used_mib()} free_GB={free_disk() // 1024 ** 3}")
    cmd = ["bash", str(REPO / "scripts/simlab/rdpy.sh")] + args
    if gpu:
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=8G", "-p", "CPUQuota=600%"] + cmd
    try:
        with (ROOT / (tag + ".log")).open("a") as out:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        note(f"END {tag} rc={rc}")
        if rc:
            raise RuntimeError(f"{tag}: rc={rc}")
        done.touch()
    finally:
        if lock:
            lock.close()


assert (REPO / ".deployed_commit").read_text().strip() == D["code"]
assert hashlib.sha256(D["salt"].encode()).hexdigest() == D["salt_sha256"]
assert free_disk() >= DISK_MIN
note("FROZEN before generation: design_sha256=" + hashlib.sha256((ROOT / "design.json").read_bytes()).hexdigest())
selection, attempts = {}, {}
for cell in D["cells"]:
    task, ids, n_want = cell["task"], [], cell["pairs"]
    tried = []
    for start, n in cell["batches"]:
        if len(ids) >= n_want:
            break
        batch = list(range(start, start + n))
        assert all(is_our_measurement_id(i) for i in batch), batch
        tag = f"{task}_{start}_{batch[-1]}"
        run("gen_" + tag, ["-m", "dojo.simlab.gen_layouts", "--task", task, "--start", str(start), "--n", str(n),
                           "--seed-salt", D["salt"]])
        run("settle_" + tag, ["-m", "dojo.simlab.presettle", "--task", task, "--ids", *map(str, batch)], gpu=True)
        index = json.loads((DATA / "simlab/layouts" / task / "settled_index.json").read_text())
        for lid in batch:
            rec = index.get(str(lid), {"status": "missing"})
            tried.append({"id": lid, **rec})
            if rec.get("status") == "ok" and len(ids) < n_want:
                ids.append(lid)
    selection[task], attempts[task] = ids, tried
    (ROOT / "selection.json").write_text(json.dumps({"selected": selection, "attempts": attempts}, indent=2) + "\n")
    if task == "imitate_sorting_sequence":
        demo_ok = {}
        for speed in D["demo_speeds"]:
            todo = [i for i in ids if not demo_ok.get(i)]
            if not todo:
                break
            run(f"demo{int(speed * 100)}_{task}", ["-m", "dojo.simlab.demo_gen", "--task", task, "--speed", str(speed),
                                                   "--ids", *map(str, todo)], gpu=True)
            index = json.loads((DATA / "simlab/traj" / task / "demo_index.json").read_text())
            for i in todo:
                demo_ok[i] = index.get(str(i), {}).get("status") == "ok"
                attempts.setdefault("demos", []).append({"id": i, "speed": speed,
                                                         "status": index.get(str(i), {}).get("status")})
        selection["imitate_excluded"] = [i for i in ids if not demo_ok.get(i)]
        selection[task] = [i for i in ids if demo_ok.get(i)]
        (ROOT / "selection.json").write_text(json.dumps({"selected": selection, "attempts": attempts}, indent=2) + "\n")
(ROOT / "prep.done").touch()
note("PREP COMPLETE " + json.dumps({k: len(v) for k, v in selection.items()}) + "; no model calls")
