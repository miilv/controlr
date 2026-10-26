"""Does perception's catalogue hold every candidate asset of every vocabulary the runtime can choose, and if not, put
them there reproducibly.

Mode B sees a scene only through the candidate sets of ``catalog/tasks.json`` (the runtime picks one per instruction,
``supervisor.VOCABULARIES`` + the catalogue fallback). An asset missing from the catalogue is an object perception can
never find, so a mode-B number for that task measures the catalogue, not the system (2026-10-25: make_toast,
play_Xylophone and hang_mugs had none of their assets on compute3).

    python -m dojo.runtime.catalog_check [--catalog DIR] [--vocabularies table|all]           # report
    python -m dojo.runtime.catalog_check --sync --sources ~/dojo-data/skills4/catalog ... \\
        [--assets ~/dojo-robodojo/RoboDojo/Assets]                                             # fill the gaps

``--sync`` copies a missing asset's ``geom.npz`` + ``info.json`` from the first source catalogue that has it (the
skills catalogues are built with the same ``dojo.perception.catalog`` builder), else builds it from RoboDojo's assets
(``catalog build --keys-json``); then ``catalog annotate`` / ``index`` / ``precompute`` for the added keys.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def vocabulary_names(which: str = "table", tasks: dict | None = None) -> list[str]:
    from .supervisor import VOCABULARIES
    names = [v.name for v in VOCABULARIES]
    if which == "all" and tasks:
        names += [n for n in tasks if n not in names]
    return names


def missing_assets(catalog: str | Path, names: list[str] | None = None) -> dict[str, list[str]]:
    """{vocabulary: [candidate keys without an index entry or geometry]} (only vocabularies with gaps; a vocabulary
    absent from tasks.json maps to ["<not in tasks.json>"])."""
    c = Path(catalog)
    tasks = json.loads((c / "tasks.json").read_text())
    index = json.loads((c / "index.json").read_text()) if (c / "index.json").exists() else {}
    out = {}
    for name in names or vocabulary_names("table", tasks):
        if name not in tasks:
            out[name] = ["<not in tasks.json>"]
            continue
        keys = sorted({k for d in tasks[name]["labels"].values() for k in d.get("candidates") or []})
        miss = [k for k in keys if k not in index or not (c / k / "geom.npz").is_file() or not (c / k / "info.json").is_file()]
        if miss:
            out[name] = miss
    return out


def sync(catalog: str | Path, keys: list[str], sources: list[str | Path], assets: str | Path | None = None,
         run_tools: bool = True, python: str = sys.executable) -> dict:
    """Copy / build the given keys into ``catalog``, then annotate / index / precompute them. Returns what was done."""
    c = Path(catalog)
    copied, to_build = [], []
    for k in keys:
        src = next((Path(s) / k for s in sources if (Path(s) / k / "geom.npz").is_file() and (Path(s) / k / "info.json").is_file()),
                   None)
        if src is None:
            to_build.append(k)
            continue
        (c / k).mkdir(parents=True, exist_ok=True)
        for f in ("geom.npz", "info.json"):
            shutil.copy2(src / f, c / k / f)
        copied.append((k, str(src)))
    done = {"copied": copied, "built": [], "unresolved": []}
    if not run_tools:
        done["unresolved"] = to_build
        return done
    with tempfile.TemporaryDirectory() as td:
        if to_build and assets is not None:
            kj = Path(td) / "build.json"
            kj.write_text(json.dumps(to_build))
            subprocess.run([python, "-m", "dojo.perception.catalog", "build", "--assets", str(assets), "--out", str(c),
                            "--keys-json", str(kj)], check=True)
            done["built"] = [k for k in to_build if (c / k / "geom.npz").is_file()]
        done["unresolved"] = [k for k in to_build if k not in done["built"]]
        added = [k for k, _ in copied] + done["built"]
        if added:
            kj = Path(td) / "added.json"
            kj.write_text(json.dumps(added))
            if assets is not None:
                subprocess.run([python, "-m", "dojo.perception.catalog", "annotate", "--assets", str(assets), "--out", str(c),
                                "--keys-json", str(kj)], check=True)
            subprocess.run([python, "-m", "dojo.perception.catalog", "index", "--out", str(c)], check=True)
            subprocess.run([python, "-m", "dojo.perception.catalog", "precompute", "--out", str(c), "--keys-json", str(kj)],
                           check=True, env=dict(os.environ, DOJO_CATALOG=str(c)))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", default=os.path.expanduser(os.environ.get("DOJO_PERCEPTION_CATALOG",
                                                                          "~/dojo-perception-data/catalog")))
    ap.add_argument("--vocabularies", choices=("table", "all"), default="table",
                    help="table: the runtime's instruction table; all: every candidate set in tasks.json")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--sources", nargs="*", default=[], help="catalogues to copy missing assets from (skills catalogues)")
    ap.add_argument("--assets", default=None, help="RoboDojo Assets dir: build what no source has; annotate")
    a = ap.parse_args(argv)
    tasks = json.loads((Path(a.catalog) / "tasks.json").read_text())
    miss = missing_assets(a.catalog, vocabulary_names(a.vocabularies, tasks))
    for name, keys in miss.items():
        print(f"{name}: {len(keys)} missing {keys[:8]}{' ...' if len(keys) > 8 else ''}")
    if not miss:
        print("every candidate asset is in the catalogue")
        return 0
    if not a.sync:
        return 1
    keys = sorted({k for v in miss.values() for k in v if not k.startswith("<")})
    done = sync(a.catalog, keys, [os.path.expanduser(s) for s in a.sources], a.assets and os.path.expanduser(a.assets))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in done.items()}))
    left = missing_assets(a.catalog, vocabulary_names(a.vocabularies, tasks))
    print("still missing:", left or "none")
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
