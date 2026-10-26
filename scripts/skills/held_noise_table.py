"""Build ``dojo/skills/data/held_noise.json``: perception-held's dev evaluation of ``held_pose`` (PR #59, the
``held_eval`` jsonl rows on our own dev layouts >= 100000) reduced to what the G3 test provider
(``dojo.skills.heldsim``) resamples: per object category and view kind, the read's ok / why, its error (leading point
and rotation, TCP frame) and its reported sigma.

    python scripts/skills/held_noise_table.py OUT.json EVAL.jsonl [EVAL.jsonl ...]

View kinds (``held_eval``'s frame kinds, views "all" = every camera that sees the object):
- hover: the place hover as the skills reach it (``hover``) and its grasp variants (``grasp``);
- look: the other arm's wrist at a ranked observer look, the holding arm at the hover (``look_ranked``);
- show: the holding arm at a ranked show pose + the ranked observer look (``show_ranked``).
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

KINDS = {"hover": "hover", "grasp": "hover", "look_ranked": "look", "show_ranked": "show"}
# the merged estimator's calibrated sigma (``dojo.perception.held.SIGMA_SCALE``; the eval rows hold the raw sigma) and its
# ok bars (``OK_MAX_LEAD_SIGMA_MM`` / ``OK_MAX_AXIS_SIGMA_DEG``): rows re-decided as ``held_eval.rescale`` does
SIGMA_SCALE = {"rigid": {"pos": 1.21, "rot": 1.31}, "spin": {"pos": 1.62, "rot": 1.61}}
OK_LAT, OK_AXIS = 1.0, 2.0
SPIN = {"test_tube", "pen", "oil_pen", "coin"}     # continuous symmetry (``held.spin_axis``), by category here
CAP = 40          # rows per (category, kind), a seeded random subsample (the file stays small)


def _r(v, n=2):
    return [round(float(x), n) for x in v] if isinstance(v, (list, tuple)) else round(float(v), n)


def reduce_row(r: dict) -> dict:
    """ok, why (re-decided on the calibrated sigma); err = the fit's error (leading point mm, rotation deg as a rotation
    vector; TCP frame); sig = the calibrated sigma the read reports (leading point per TCP axis and across the insertion
    axis, mm; insertion axis, deg)."""
    err = _r(list(r["err_fit"]["lead_mm"]) + list(r["err_fit"]["rot_deg"]))
    s = r.get("sigma") or {}
    if not s:
        return {"ok": 0, "why": r.get("why") or "fit_poor", "err": err}
    k = SIGMA_SCALE["spin" if r["key"].split("/")[1] in SPIN else "rigid"]
    lead = [v * k["pos"] for v in (s.get("lead_mm") or [99.0] * 3)]
    lat, ax = s.get("lead_lat_mm", 99.0) * k["pos"], s.get("axis_deg", 99.0) * k["rot"]
    ok = r.get("why", "") in ("", "ambiguous") and lat <= OK_LAT and ax <= OK_AXIS
    return {"ok": int(ok), "why": "" if ok else (r.get("why") or "ambiguous"), "err": err, "sig": _r(lead + [lat, ax])}


def main(argv: list[str]) -> None:
    out_p, files = argv[0], argv[1:]
    groups: dict = defaultdict(list)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            kind = KINDS.get(r.get("kind"))
            if kind is None or r.get("views_set") != "all" or "err_fit" not in r:
                continue
            cat = r["key"].split("/")[1]
            groups[(cat, kind)].append(r)
    rng = random.Random(0)
    table: dict = {}
    for (cat, kind), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: (r["path"], r.get("prior_i", 0)))
        if len(rows) > CAP:
            rows = rng.sample(rows, CAP)
        table.setdefault(cat, {})[kind] = [reduce_row(r) for r in rows]
    stats = {cat: {k: {"n": len(v), "ok": sum(x["ok"] for x in v)} for k, v in kinds.items()}
             for cat, kinds in table.items()}
    doc = {"source": "perception-held held_eval (PR #59), views 'all', dev layouts >= 100000; sigma calibrated with "
                     f"{SIGMA_SCALE}, ok re-decided at {OK_LAT} mm / {OK_AXIS} deg",
           "files": [f.split("/")[-1] for f in files], "kinds": KINDS, "stats": stats, "table": table}
    with open(out_p, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    for cat, kinds in stats.items():
        print(cat, kinds)


if __name__ == "__main__":
    main(sys.argv[1:])
