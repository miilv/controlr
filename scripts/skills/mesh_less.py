"""List the assets of our dev layouts that no catalogue has a mesh for (``dojo/skills/data/mesh_less.json``).

G3 refuses to run when an asset lacks a mesh on this box that is NOT in that list (an incomplete catalogue: a sync still
running silently turned meshes into bboxes and changed episodes). Regenerate on a box with the complete catalogue:

    python3 scripts/skills/mesh_less.py [--layouts ~/dojo-data/simlab/layouts] [--out dojo/skills/data/mesh_less.json]

Catalogues: ``$DOJO_CATALOG`` (os.pathsep list) plus the defaults G3 uses. Plain python, no repo imports.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default="~/dojo-data/simlab/layouts")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "dojo", "skills", "data",
                                                  "mesh_less.json"))
    a = ap.parse_args()
    cats = [c for c in os.environ.get("DOJO_CATALOG", "").split(os.pathsep) if c]
    cats += ["~/dojo-data/skills4/catalog", "~/dojo-data/skills3/catalog", "~/dojo-perception-data/catalog",
             "~/dojo-data/skills2/catalog"]
    cats = [os.path.expanduser(c) for c in cats if os.path.isdir(os.path.expanduser(c))]
    root = os.path.expanduser(a.layouts)
    missing: dict[str, set] = {}
    for task in sorted(os.listdir(root)):
        for f in sorted(glob.glob(os.path.join(root, task, "1*.json"))):
            d = json.load(open(f))
            for typ in ("Rigid", "Geometry", "Articulation", "Dynamic", "Garment"):
                for cat, recs in (d.get(typ) or {}).items():
                    for r in recs if isinstance(recs, list) else []:
                        if not isinstance(r, dict) or r.get("clutter"):
                            continue
                        key = "%s/%s/%05d" % (typ, cat, int(r.get("category_idx", 0) or 0))
                        if not any(os.path.exists(os.path.join(c, key, "geom.npz")) for c in cats):
                            missing.setdefault(key, set()).add(task)
    json.dump({"note": "assets (type/category/idx) of non-clutter dev-layout objects with no geom.npz in any catalogue "
                       "(on a box with the complete catalogue); G3 plans on their bboxes by design. Regenerate: "
                       "scripts/skills/mesh_less.py",
               "keys": sorted(missing)}, open(a.out, "w"), indent=0)
    print(f"{len(missing)} assets without a mesh -> {a.out}")


if __name__ == "__main__":
    main()
