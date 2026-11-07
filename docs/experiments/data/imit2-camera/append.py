"""IMIT2 camera A/B: append the staged rows at the END of compute6's queue2.txt (never reorder), under wave-start.lock
then queue2.txt.lock; verify every row first; write dispatch.json (receipt)."""
import collections, datetime, fcntl, hashlib, json, os, re, statistics
from pathlib import Path

H = Path.home(); R = H / "dojo-data/imit2-cam"; S = H / "dojo-integ-c6-state"; Q = S / "queue2.txt"
D = json.loads((R / "design-cam.json").read_text()); M = json.loads((R / "manifest.json").read_text())
new = (R / "queue-imit2.txt").read_bytes(); newrows = new.decode().splitlines(keepends=True)
assert not (R / "dispatch.json").exists() and len(newrows) == M["queue"]["rows"] == len(M["schedule"])
assert hashlib.sha256(new).hexdigest() == M["queue"]["sha256"]
assert (R / "prereg-commit.txt").exists(), "preregistration must be on main first"
fh = {f["path"]: f["sha256"] for f in M["files"]}
for line, m in zip(newrows, M["schedule"]):
    t = line.split()
    assert t[0] == m["name"] and t[0].startswith("c6b_") and "_tie" not in t[0]
    assert t[t.index("--repo") + 1] == D["deployment"] and t[t.index("--layouts") + 1] == m["layout"]
    assert t[t.index("--task") + 1] == m["task"]
    assert [t[i + 1] for i, z in enumerate(t[:-1]) if z == "--setenv" and t[i + 1].startswith("AGENTIC_MODEL=")] == \
        ["AGENTIC_MODEL=" + D["model"]]
    cfg = json.loads(t[t.index("--config") + 1])
    assert all(cfg.get(k) == v for k, v in D["overrides"][m["arm"]].items())
    if m["arm"] == "A":
        assert cfg["imitate_wait"] == "origin" and cfg["imitate_pairing"] == "one_to_one" and "imitate_keep_sides" not in cfg
    assert hashlib.sha256(Path(m["layout"]).read_bytes()).hexdigest() == fh[m["layout"]]
for p, h in fh.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
with (S / "wave-start.lock").open("a") as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with (S / "queue2.txt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        starts, dones, samples = {}, {}, collections.defaultdict(list)
        for p in S.glob("lane2_heavy*.log"):
            for l in p.read_text().splitlines():
                m = re.match(r"^(\S+) \[([^]]+)\] start (\S+)", l)
                if m:
                    starts[m[3]] = {"time": m[1], "lane": m[2]}
                m = re.match(r"^(\S+) \[([^]]+)\] done (\S+) rc=(\d+) lock (\d+) s", l)
                if m:
                    dones[m[3]] = m[1]
                    if int(m[4]) == 0:
                        samples[m[3].split("_")[1]].append(int(m[5]))
        names = {m["name"] for m in M["schedule"]}
        assert not names.intersection(starts)
        before = Q.read_bytes(); assert not before or before.endswith(b"\n")
        existing = before.decode().splitlines(keepends=True)
        assert not names.intersection(l.split()[0] for l in existing)
        after = before + new
        (R / "queue-before.txt").write_bytes(before)
        tmp = S / "queue2.imit2.tmp"; tmp.write_bytes(after)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, Q); (R / "queue-after.txt").write_bytes(after)
        now = datetime.datetime.now(datetime.timezone.utc)
        means = {k: statistics.mean(v[-20:]) for k, v in samples.items() if v}
        fallback = {"imitatesg": 750, "stackblockssg": 160, "classifysg": 300, "poursg": 225, "matchpicksg": 420}
        est = lambda n: means.get(n.split("_")[1], fallback.get(n.split("_")[1], 600))
        wait = sum(est(l.split()[0]) for l in existing); active = []
        for p in S.glob("job.heavy*"):
            n = p.read_text().strip()
            if n in starts and (n not in dones or starts[n]["time"] > dones[n]):
                el = (now - datetime.datetime.fromisoformat(starts[n]["time"].replace("Z", "+00:00"))).total_seconds()
                rem = max(0, est(n) - el); wait += rem
                active.append({"name": n, "elapsed_s": round(el), "estimated_remaining_s": round(rem)})
        block = sum(est(m["name"]) for m in M["schedule"])
        receipt = {"inserted_utc": now.isoformat(), "prereg_commit": (R / "prereg-commit.txt").read_text().strip(),
                   "positions_1_based": [len(existing) + 1, len(existing) + len(newrows)], "new_rows": len(newrows),
                   "existing_rows": len(existing), "queue_rows_after": len(existing) + len(newrows),
                   "queue_sha256_before": hashlib.sha256(before).hexdigest(),
                   "queue_sha256_after": hashlib.sha256(after).hexdigest(),
                   "manifest_sha256": hashlib.sha256((R / "manifest.json").read_bytes()).hexdigest(),
                   "new_rows_sha256": hashlib.sha256(new).hexdigest(), "appended_at_end": True,
                   "rows_ahead": [l.split()[0] for l in existing], "active_jobs": active,
                   "route_checked_rows": len(newrows), "model": D["model"], "deployment": D["code"],
                   "estimated_start_utc": (now + datetime.timedelta(seconds=wait / 2)).isoformat(),
                   "estimated_finish_utc": (now + datetime.timedelta(seconds=(wait + block) / 2)).isoformat(),
                   "estimated_block_lane_s": round(block),
                   "timing_basis": {"recent_mean_seconds": means, "fallback_s": fallback, "lanes": 2,
                                    "note": "rough; admission/startup/time caps/retries/power drains may delay"}}
        (R / "dispatch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (R / "notes.log").open("a") as f:
            f.write(now.isoformat() + " DISPATCH " + json.dumps(receipt) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "rows_ahead"}, indent=2))
