"""COIN2-CAM: stability-only selection from the frozen design (no outcome exists yet).

    python3 select.py DESIGN DATA_ROOT OUT   -> OUT = selection.json
Per cell, ids in pool order whose settled record is ok: "all stable" keeps them all; "first N stable by id" the first N.
"""
import json
import sys
from pathlib import Path

design, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
D = json.loads(design.read_text())
sel, drops = {}, {}
for c in D["cells"]:
    idx = json.loads((root / D["block"] / "simlab/layouts" / c["task"] / "settled_index.json").read_text())
    pool = [s + i for s, n in c["pool"] for i in range(n)]
    ok = [i for i in pool if idx.get(str(i), {}).get("status", "missing") == "ok"]
    drops[c["task"]] = {str(i): idx.get(str(i), {}).get("status", "missing") for i in pool if i not in ok}
    sel[c["task"]] = ok if c["select"] == "all stable" else ok[:c["pairs"]]
    assert len(sel[c["task"]]) >= (18 if c["role"] == "gain" else c["pairs"]), (c["task"], len(sel[c["task"]]))
out.write_text(json.dumps({"selected": sel, "dropped": drops}, indent=1) + "\n")
print(json.dumps({k: len(v) for k, v in sel.items()}), json.dumps(drops))
