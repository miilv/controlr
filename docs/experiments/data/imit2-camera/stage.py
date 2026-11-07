"""IMIT2 camera A/B: stage design-cam.json, the queue rows and manifest.json (inputs + hashes). No queue write here.
usage (compute6): python3 stage.py <full sha>"""
import hashlib, json, sys
from pathlib import Path

H = Path.home(); R = H / "dojo-data/imit2-cam"; SHA = sys.argv[1]; REPO = H / f"dojo-integ-c6-{SHA[:8]}"
assert (REPO / ".deployed_commit").read_text().strip() == SHA
sys.path.insert(0, str(REPO))
from dojo.simlab.paths import is_our_measurement_id  # noqa: E402

BLOCK = R / "imit2-cam-1"
sel = json.loads((R / "selection.json").read_text())["selected"]
CELLS = [("imitate_sorting_sequence", "imitatesg", sel["imitate_sorting_sequence"]),
         ("stack_blocks", "stackblockssg", sel["stack_blocks"]),
         ("classify_objects", "classifysg", sel["classify_objects"])]
for _, _, ids in CELLS:
    assert ids and all(is_our_measurement_id(i) for i in ids)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


cand = [l for l in (R / "candidate.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
assert len(cand) == 3 and cand[2].startswith("--config ")
base = json.loads(cand[2][len("--config "):])
assert base["imitate_pairing"] == "one_to_one" and base["imitate_wait"] == "origin"
MODEL = "claude/claude-sonnet-5-5"
OVER = {"A": {},
        "C": {"imitate_wait": "home", "imitate_keep_sides": True},
        "B": {"imitate_wait": "home", "imitate_keep_sides": True, "imitate_order": "persist_elim",
              "imitate_pairing": "cues", "imitate_carry": "reach", "imitate_retry": "refind"}}
for k in ("imitate_keep_sides", "imitate_order", "imitate_carry", "imitate_retry"):
    assert k not in base, k
ROT = [("A", "C", "B"), ("C", "B", "A"), ("B", "A", "C")]
rows, sched, files = [], [], []
k = 0
for task, short, ids in CELLS:
    for lid in ids:
        lay = BLOCK / "simlab/layouts" / task / f"{lid}.settled.json"
        files.append({"path": str(lay), "sha256": sha(lay)})
        if task == "imitate_sorting_sequence":
            demo = BLOCK / "simlab/traj" / task / f"{lid}.pkl"
            files.append({"path": str(demo), "sha256": sha(demo)})
        for arm in ROT[k % 3]:
            cfg = {**base, **OVER[arm]}
            name = f"c6b_{short}_imit2{arm}r1_{lid}"
            row = " ".join([name, "--repo", str(REPO), cand[0], cand[1], "--config", json.dumps(cfg, separators=(",", ":")),
                            "--setenv", "AGENTIC_MODEL=" + MODEL, "--setenv", "DOJO_DATA=" + str(BLOCK), "--task", task,
                            "--renderer", "isaac51", "--episode-max-minutes", "25", "--max-minutes", "25",
                            "--layouts", str(lay)])
            rows.append(row + "\n")
            sched.append({"name": name, "arm": arm, "task": task, "id": lid, "ordinal": k, "layout": str(lay)})
        k += 1
q = "".join(rows).encode()
(R / "queue-imit2.txt").write_bytes(q)
design = {"stream": "IMIT2-CAM", "code": SHA, "deployment": str(REPO), "model": MODEL,
          "baseline_candidate": "scripts/dojo/queues/2026-11-06-candidate.txt",
          "baseline_candidate_sha256": sha(REPO / "scripts/dojo/queues/2026-11-06-candidate.txt"),
          "candidate_robotics_sha256": sha(R / "candidate.txt"), "overrides": OVER, "cells": {t: ids for t, _, ids in CELLS},
          "salt": json.loads((R / "design.json").read_text())["salt"], "prep_design_sha256": sha(R / "design.json"),
          "order": "per id the three arms back to back, rotated by ordinal: ACB, CBA, BAC", "repeats_per_arm": 1,
          "comparisons": ["C vs A", "B vs C"]}
(R / "design-cam.json").write_text(json.dumps(design, indent=2) + "\n")
man = {"design_cam_sha256": sha(R / "design-cam.json"), "prep_design_sha256": sha(R / "design.json"),
       "selection_sha256": sha(R / "selection.json"),
       "queue": {"path": str(R / "queue-imit2.txt"), "sha256": hashlib.sha256(q).hexdigest(), "rows": len(rows)},
       "schedule": sched, "files": files, "parity": json.loads((R / "parity.json").read_text()).get("diff"),
       "deployed_json_sha256": sha(REPO / ".deployed.json")}
(R / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "queue_sha256": man["queue"]["sha256"], "manifest_sha256": sha(R / "manifest.json")}))
