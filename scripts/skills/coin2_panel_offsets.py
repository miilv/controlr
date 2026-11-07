"""COIN2 panel: per-layout camera-like belief offsets for G3 ``--belief-offset-file`` (dev only).

    python3 scripts/skills/coin2_panel_offsets.py SALTS.json ERRORS.json ROOT OUT_DIR

For each block of SALTS (``{"blocks": [{"block", "salt", ...}]}``) and each stable settled layout under
``ROOT/<block>/simlab/layouts/deposit_coin/`` (``settled_index.json`` status ok): sample index
``int(sha256(f"{salt}|{id}"), 16) % n`` of ERRORS' ``samples`` (``yaw_deg``, ``dn_mm``, ``du_mm``). The coin and its
stand get the same yaw error and the xy shift ``dn`` along the coin's normal + ``du`` along its plane (from the
settled record: dev ground truth, used only to express the camera's error in this layout's frame). Writes
``OUT_DIR/<block>.offsets.json`` ({seed: {label: [dx, dy, dyaw]}}) and prints their sha256.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def qR(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def coin_record(lay: dict) -> dict:
    for grp in lay.values():
        if isinstance(grp, dict):
            for recs in grp.values():
                for r in recs if isinstance(recs, list) else []:
                    if isinstance(r, dict) and r.get("label") == "coin0":
                        return r
    raise KeyError("coin0")


def offsets_for(salt: str, lid: int, lay: dict, samples: list) -> tuple[dict, dict]:
    i = int(hashlib.sha256(f"{salt}|{lid}".encode()).hexdigest(), 16) % len(samples)
    s = samples[i]
    R = qR(coin_record(lay)["default_ori"])
    n = R[:, 2].copy()
    n[2] = 0.0
    n /= np.linalg.norm(n)
    u = np.cross([0.0, 0.0, 1.0], n)
    dxy = (float(s["dn_mm"]) * n + float(s["du_mm"]) * u) / 1000.0
    off = [round(float(dxy[0]), 6), round(float(dxy[1]), 6), float(s["yaw_deg"])]
    return {"coin0": off, "vertical_coin_stand": off}, {"sample": i, **s}


def main(argv=None):
    salts, errors, root, out = (argv or sys.argv[1:])[:4]
    blocks = json.loads(Path(salts).read_text())["blocks"]
    samples = json.loads(Path(errors).read_text())["samples"]
    Path(out).mkdir(parents=True, exist_ok=True)
    for b in blocks:
        d = Path(root) / b["block"] / "simlab" / "layouts" / b["task"]
        idx = json.loads((d / "settled_index.json").read_text())
        res, meta = {}, {}
        for sid, st in sorted(idx.items()):
            if st.get("status", "ok") != "ok":
                continue
            lay = json.loads((d / f"{sid}.settled.json").read_text())
            res[sid], meta[sid] = offsets_for(b["salt"], int(sid), lay, samples)
        p = Path(out) / f"{b['block']}.offsets.json"
        p.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        (Path(out) / f"{b['block']}.offsets-meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        print(p, hashlib.sha256(p.read_bytes()).hexdigest(), len(res), "stable of", len(idx))


if __name__ == "__main__":
    main()
