"""CLASSIFY3 camera A/B: queue rows for compute4 from the frozen design, the stability selection and the frozen
candidate; every row validated before the queue write (no queue is touched here).

    <deploy>/.venv/bin/python make_rows.py DESIGN SELECTION CANDIDATE DEPLOY DATA_ROOT OUT_DIR

Writes OUT_DIR/queue-classify3.txt (one lane2.sh queue line per row) and OUT_DIR/rows-check.json:
- per row, the resolved RuntimeConfig (``dojo.runtime.eval.build_config``);
- the A/B difference per pair (must be exactly ``sort_contain``);
- the route (``AGENTIC_MODEL``), the layout file's existence and id slice.
"""
import hashlib
import json
import sys
from pathlib import Path

design, selection, cand, deploy, data, out = (Path(x) for x in sys.argv[1:7])
D = json.loads(design.read_text())
S = json.loads(selection.read_text())["selected"]
sys.path.insert(0, str(deploy))
from dojo.runtime.eval import build_config          # noqa: E402
from dojo.simlab.paths import is_our_measurement_id  # noqa: E402

lines = [ln.strip() for ln in cand.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
toks = " ".join(lines).split(" ")
cfg_i = toks.index("--config") + 1
base_cfg = json.loads(toks[cfg_i])
head = " ".join(toks[:cfg_i - 1])                     # preset / env / side-server tokens, before --config
MODEL = D["model"]
SHORT = {"classify_objects": "classifysg", "classify_objects_by_language": "classifylangsg", "stack_blocks": "stackblockssg"}
KIT = data / "meta" / "kit-env.sh"
rows, check = [], {"rows": [], "pairs": []}
for cell in D["cells"]:
    task = cell["task"]
    for k, lid in enumerate(S[task]):
        assert is_our_measurement_id(lid), lid
        lay = data / D["block"] / "simlab/layouts" / task / f"{lid}.settled.json"
        assert lay.exists(), lay
        order = ("A", "B") if k % 2 == 0 else ("B", "A")
        cfgs = {}
        for arm in order:
            c = dict(base_cfg, **D["overrides"][arm])
            name = f"c4b_{SHORT[task]}_cls3{arm}r1_{lid}"
            line = " ".join([name, "--repo", str(deploy), head, "--config", json.dumps(c, separators=(",", ":")),
                             "--setenv", f"AGENTIC_MODEL={MODEL}", "--setenv", f"DOJO_DATA={data / D['block']}",
                             "--task", task, "--renderer", "isaac51", "--episode-max-minutes", "25", "--max-minutes", "25",
                             "--layouts", str(lay), "--setenv", f"BASH_ENV={KIT}",
                             "--setenv", "DOJO_BG_FOLDS=/home/haptics/dojo-data/bg_folds_51_ilia",
                             "--setenv", "INTEG_WATCHDOG_S=600", "--setenv", "INTEG_WATCHDOG_HARD_S=600"])
            rc = build_config("submit_v3", c)
            cfgs[arm] = rc.__dict__
            rows.append(line)
            assert f"AGENTIC_MODEL={MODEL}" in line and line.count("AGENTIC_MODEL=") == 1
            check["rows"].append({"name": name, "task": task, "id": lid, "arm": arm,
                                  "layout_sha256": hashlib.sha256(lay.read_bytes()).hexdigest(),
                                  "model": MODEL, "sort_contain": rc.sort_contain, "sort_park": rc.sort_park,
                                  "instance_evidence": rc.instance_evidence, "ported_classify": rc.ported_classify})
        diff = sorted(f for f in cfgs["A"] if cfgs["A"][f] != cfgs["B"][f])
        assert diff == ["sort_contain"], (task, lid, diff)
        check["pairs"].append({"task": task, "id": lid, "order": "".join(order), "diff": diff})
out.mkdir(parents=True, exist_ok=True)
q = out / "queue-classify3.txt"
q.write_text("\n".join(rows) + "\n")
check["n_rows"] = len(rows)
check["queue_sha256"] = hashlib.sha256(q.read_bytes()).hexdigest()
check["deploy"] = str(deploy)
check["deploy_commit"] = (deploy / ".deployed_commit").read_text().strip()
(out / "rows-check.json").write_text(json.dumps(check, indent=1) + "\n")
print(f"{len(rows)} rows, every pair differs only in sort_contain; queue sha256 {check['queue_sha256']}")
