"""A catalogue-format colour map from a frozen calibration set's own ``color_M.json``, no refit (D-C19 parity).

A frozen set (``<bg_folds>/<set>/color_M.json``: {"name", "M": 4x3, "n"}) is what our perception service and
``identity.load_color_M`` use when ``DOJO_CALIB_SET`` is set. The J1 adapter's ``_color_m`` reads the catalogue format
(``DOJO_COLOR_CALIB``: {"all": 4x3, ...}). This writes {"all": M, "<name>": M} with the set's M unchanged, so both
routes' B1 use one map. Deterministic: sorted keys, compact separators, Python's float repr; the same source gives the
same bytes (sha256) on every box.

    python3 scripts/perception/calib_catalog_from_set.py <bg_folds>/<set>/color_M.json <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def catalog_from_set(src: Path) -> str:
    d = json.loads(Path(src).read_text())
    M = d["M"]
    if len(M) != 4 or any(len(r) != 3 for r in M):
        raise ValueError(f"{src}: M is not 4x3")
    return json.dumps({"all": M, str(d["name"]): M}, sort_keys=True, separators=(",", ":")) + "\n"


def main(argv=None) -> None:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 2:
        raise SystemExit(__doc__)
    text = catalog_from_set(Path(a[0]))
    Path(a[1]).write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), a[1])


if __name__ == "__main__":
    main()
