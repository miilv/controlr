"""The package versions a venv runs the candidate with (D-C19 parity: the official-path venv vs a compute4 lane venv).
Run it with the venv's own python; prints JSON (the python version + PACKAGES -> installed version or null).

    ~/dojo-integ-c4-<sha>/.venv/bin/python scripts/dojo/pkg_versions.py --out lane_pkgs.json     # compute4 lane venv
    python scripts/dojo/candidate_parity.py --candidate C --packages-ref lane_pkgs.json            # fails on a mismatch
"""
from __future__ import annotations

import argparse
import json
import platform
import sys

# what perception / the runtime / the side servers load (lane venvs: uv sync --frozen + venv_extras.sh)
PACKAGES = ("numpy", "pillow", "opencv-python-headless", "opencv-python", "torch", "torchvision", "ultralytics",
            "transformers", "timm", "scipy", "trimesh", "pyrender", "PyOpenGL", "shapely", "networkx")


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return {"python": platform.python_version(), "executable": sys.executable, "packages": out}


def diff(ref: dict, got: dict) -> dict:
    """{name: [ref, got]} for every package (and the python version) that differs; {} = the same set."""
    d = {p: [ref["packages"].get(p), got["packages"].get(p)] for p in sorted(set(ref["packages"]) | set(got["packages"]))
         if ref["packages"].get(p) != got["packages"].get(p)}
    if ref.get("python") != got.get("python"):
        d["python"] = [ref.get("python"), got.get("python")]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    v = versions()
    text = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
