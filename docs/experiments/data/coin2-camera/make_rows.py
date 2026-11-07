"""COIN2-CAM camera A/B rows for compute4 (lane2.sh queue lines) from the frozen design, the stability selection and
the frozen candidate; every row checked before any queue write (this script touches no queue).

    <deploy>/.venv/bin/python make_rows.py DESIGN SELECTION CANDIDATE DEPLOY DATA_ROOT OUT_DIR
Writes OUT_DIR/queue-coin2cam.txt and OUT_DIR/rows-check.json: per row the resolved RuntimeConfig fields of the
lever, the route, the layout hash; per pair the A / B difference, which must be exactly the five disc fields.
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
head = " ".join(toks[:cfg_i - 1])
MODEL = D["model"]
LEVER = sorted(D["overrides"]["B"])
SHORT = {"deposit_coin": "depositcoin", "stack_blocks": "stackblockssg", "insert_key": "insertkeysg",
         "play_tic_tac_toe": "tttsg"}
KIT = data / "meta" / "kit-env.sh"
rows, check = [], {"rows": [], "pairs": []}
for cell in D["cells"]:
    task = cell["task"]
    for k, lid in enumerate(S[task]):
        assert is_our_measurement_id(lid) and lid < 100100, lid
        lay = data / D["block"] / "simlab/layouts" / task / f"{lid}.settled.json"
        assert lay.exists(), lay
        order = ("A", "B") if k % 2 == 0 else ("B", "A")
        cfgs = {}
        for arm in order:
            c = dict(base_cfg, **D["overrides"][arm])
            name = f"c4b_{SHORT[task]}_coin2cam{arm}r1_{lid}"
            line = " ".join([name, "--repo", str(deploy), head, "--config", json.dumps(c, separators=(",", ":")),
                             "--setenv", f"AGENTIC_MODEL={MODEL}", "--setenv", f"DOJO_DATA={data / D['block']}",
                             "--task", task, "--renderer", "isaac51", "--episode-max-minutes", "25", "--max-minutes", "25",
                             "--layouts", str(lay), "--setenv", f"BASH_ENV={KIT}",
                             "--setenv", "DOJO_BG_FOLDS=/home/haptics/dojo-data/bg_folds_51_ilia",
                             "--setenv", "INTEG_WATCHDOG_S=600", "--setenv", "INTEG_WATCHDOG_HARD_S=600"])
            rc = build_config("submit_v3", c)
            cfgs[arm] = rc.__dict__
            rows.append(line)
            assert line.count("AGENTIC_MODEL=") == 1 and f"AGENTIC_MODEL={MODEL}" in line
            assert "DOJO_GRIP_SOURCE=v2" in line and rc.grip_check == "on"
            check["rows"].append({"name": name, "task": task, "role": cell["role"], "id": lid, "arm": arm,
                                  "layout_sha256": hashlib.sha256(lay.read_bytes()).hexdigest(), "model": MODEL,
                                  **{f: getattr(rc, f) for f in LEVER + ["disc_reserve"]}})
        diff = sorted(f for f in cfgs["A"] if cfgs["A"][f] != cfgs["B"][f])
        assert diff == LEVER, (task, lid, diff)
        check["pairs"].append({"task": task, "role": cell["role"], "id": lid, "order": "".join(order), "diff": diff})
out.mkdir(parents=True, exist_ok=True)
block = "".join(r + "\n" for r in rows)
(out / "queue-coin2cam.txt").write_text(block)
check["n_rows"] = len(rows)
check["block_sha256"] = hashlib.sha256(block.encode()).hexdigest()
(out / "rows-check.json").write_text(json.dumps(check, indent=1) + "\n")
print(len(rows), check["block_sha256"])
