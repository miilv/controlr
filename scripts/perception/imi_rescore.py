"""Re-score cached frame-0 records (rec_<id>.json of val_imitate0) with val_imitate0.score. usage: imi_rescore.py REC_DIR"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from val_imitate0 import ROOT, score  # noqa: E402
from dojo.perception import d0  # noqa: E402
d = Path(sys.argv[1])
rows = []
for f in sorted(d.glob("rec_*.json")):
    lid = f.stem[4:]
    r, ph = score(json.loads(f.read_text()), d0.load_gt(ROOT / lid, "W0000"))
    for x in r:
        x["ep"] = lid
    rows += r
    print(lid, "found", sum(x["found"] for x in r), "cat", sum(x["cat_ok"] for x in r), "phantoms", ph)
for side in ("own", "demo"):
    rs = [r for r in rows if r["side"] == side]
    print(side, "found", sum(r["found"] for r in rs), "cat", sum(r["cat_ok"] for r in rs), "key", sum(r["key_ok"] for r in rs))
per = {}
for r in rows:
    per.setdefault(r["ep"], []).append(r)
print("layouts all right", sum(all(r["cat_ok"] for r in rs) for rs in per.values()), "of", len(per))
