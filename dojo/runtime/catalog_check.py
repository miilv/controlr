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

Slot bases (``missing_support``): an object the config pins in a support slot of another object (make_toast's bread,
play_Xylophone's mallet) is placed from the base's ``passive.support`` frame, which only ``catalog annotate`` writes.
A base without it fails the check (exit 1, ERROR on stderr) even with every asset present; ``--sync --assets``
annotates it.

Articulated assets (``missing_links``): the skills read an Articulation's links and joints (a lid's hinge, a lever)
from ``links.json`` next to its geometry (``scripts/skills/articulation_links.py``; ``world.AssetDB``). An
Articulation candidate without it fails the check (fill_egg_holder's lid was invisible to the runtime); ``--sync``
copies it from a source catalogue, else writes it from the asset's USD (``--assets``).
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


# Candidate keys the catalogue cannot have because RoboDojo's Assets do not ship them (checked on compute3 and in
# ``--sync --assets``: no source catalogue has them and ``catalog build`` finds no USD). A known gap is reported
# separately and does not fail the check; any other missing key does.
KNOWN_UNSHIPPED: dict[str, str] = {
    "Rigid/broom_shovel/00001": "sweep_blocks candidate; not in RoboDojo's Assets",
    "Rigid/broom_shovel/00004": "sweep_blocks candidate; not in RoboDojo's Assets",
}


def missing_assets(catalog: str | Path, names: list[str] | None = None,
                   known: dict | None = None) -> dict[str, list[str]]:
    """{vocabulary: [candidate keys without an index entry or geometry]} (only vocabularies with gaps; a vocabulary
    absent from tasks.json maps to ["<not in tasks.json>"]). Keys in ``known`` (default ``KNOWN_UNSHIPPED``; pass
    {} to list them too) are not gaps."""
    known = KNOWN_UNSHIPPED if known is None else known
    c = Path(catalog)
    tasks = json.loads((c / "tasks.json").read_text())
    index = json.loads((c / "index.json").read_text()) if (c / "index.json").exists() else {}
    out = {}
    for name in names or vocabulary_names("table", tasks):
        if name not in tasks:
            out[name] = ["<not in tasks.json>"]
            continue
        keys = sorted({k for d in tasks[name]["labels"].values() for k in d.get("candidates") or []})
        miss = [k for k in keys if k not in known
                and (k not in index or not (c / k / "geom.npz").is_file() or not (c / k / "info.json").is_file())]
        if miss:
            out[name] = miss
    return out


def missing_support(catalog: str | Path, names: list[str] | None = None) -> dict[str, list[str]]:
    """{vocabulary: ["<base key>#<support key>", ...]} for objects the config pins in a support slot of another object
    (make_toast's bread in the shelf, play_Xylophone's mallet on its stand), for asset pairs perception places from the
    slot (``dojo/perception/slot_settle.json``), whose base asset has no such ``passive.support`` frame in the
    catalogue (``catalog annotate`` not run). Without it perception cannot place them:
    the task scores 0 with every asset present (2026-10-26 breadth, mode B)."""
    from dojo.perception.slots import slot_requirements

    c = Path(catalog)
    tasks = json.loads((c / "tasks.json").read_text())
    out = {}
    for name in names or vocabulary_names("table", tasks):
        if name not in tasks:
            continue
        gaps = set()
        for _lab, key, skey, _ck, validated in slot_requirements(tasks[name]["labels"]):
            if not validated:  # no settle data for the pair: perception's generic path, nothing to require
                continue
            info_p = c / key / "info.json"
            sup = (json.loads(info_p.read_text()).get("support") or {}) if info_p.is_file() else {}
            if skey not in sup:
                gaps.add(f"{key}#{skey}")
        if gaps:
            out[name] = sorted(gaps)
    return out


# Articulation categories whose links / joints the skills read from ``links.json`` (holders.py: the egg holder's lid hinge;
# the toaster's lever; the laptop's lid). SpringButton is pressed at perception's annotated press point
# (``catalog annotate``, info.json ``articulation``) and has no links.json anywhere: adding one would change what the
# skills load for press_by_number, so it is not required.
LINKS_CATEGORIES = ("egg_holder", "laptop", "toaster")


def missing_links(catalog: str | Path, names: list[str] | None = None) -> dict[str, list[str]]:
    """{vocabulary: [Articulation candidate keys (``LINKS_CATEGORIES``) present in the catalogue without ``links.json``]}."""
    c = Path(catalog)
    tasks = json.loads((c / "tasks.json").read_text())
    out = {}
    for name in names or vocabulary_names("table", tasks):
        if name not in tasks:
            continue
        keys = sorted({k for d in tasks[name]["labels"].values() for k in d.get("candidates") or []
                       if k.startswith("Articulation/") and k.split("/")[1] in LINKS_CATEGORIES
                       and (c / k / "geom.npz").is_file() and not (c / k / "links.json").is_file()})
        if keys:
            out[name] = keys
    return out


def _robodojo_objects(assets: str | Path) -> Path:
    """``Assets`` or ``Assets/Object/RoboDojo`` -> the per-type asset root."""
    a = Path(assets).expanduser()
    return a / "Object" / "RoboDojo" if (a / "Object" / "RoboDojo").is_dir() else a


def sync_links(catalog: str | Path, keys: list[str], sources: list[str | Path], assets: str | Path | None = None) -> dict:
    """``links.json`` for the given Articulation keys: copied from the first source catalogue that has it, else written
    from the asset's USD (``articulation_links.read_links``; needs ``pxr``)."""
    c = Path(catalog)
    done = {"copied": [], "written": [], "unresolved": []}
    for k in keys:
        src = next((Path(s) / k / "links.json" for s in sources if (Path(s) / k / "links.json").is_file()), None)
        if src is not None:
            shutil.copy2(src, c / k / "links.json")
            done["copied"].append((k, str(src)))
            continue
        usd = None
        if assets is not None:
            d = _robodojo_objects(assets) / k
            usd = next((d / n for n in ("object.usdz", "object.usd") if (d / n).is_file()), None)
        if usd is None:
            done["unresolved"].append(k)
            continue
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "articulation_links", Path(__file__).resolve().parents[2] / "scripts" / "skills" / "articulation_links.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        (c / k / "links.json").write_text(json.dumps(mod.read_links(usd)))
        done["written"].append(k)
    return done


def annotate(catalog: str | Path, keys: list[str], assets: str | Path, python: str = sys.executable) -> None:
    """``catalog annotate`` (support / functional frames, articulations) for already-present keys."""
    with tempfile.TemporaryDirectory() as td:
        kj = Path(td) / "annotate.json"
        kj.write_text(json.dumps(sorted(set(keys))))
        subprocess.run([python, "-m", "dojo.perception.catalog", "annotate", "--assets", str(assets), "--out", str(catalog),
                        "--keys-json", str(kj)], check=True)


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
    names = vocabulary_names(a.vocabularies, tasks)
    unshipped = missing_assets(a.catalog, names, known={})
    unshipped = sorted({k for v in unshipped.values() for k in v if k in KNOWN_UNSHIPPED})
    if unshipped:
        print(f"known unshipped (not a gap): {unshipped}")
    miss = missing_assets(a.catalog, names)
    for name, keys in miss.items():
        print(f"{name}: {len(keys)} missing {keys[:8]}{' ...' if len(keys) > 8 else ''}")
    if not miss:
        print("every candidate asset is in the catalogue")
    rc = 0
    if miss:
        if not a.sync:
            rc = 1
        else:
            keys = sorted({k for v in miss.values() for k in v if not k.startswith("<")})
            done = sync(a.catalog, keys, [os.path.expanduser(s) for s in a.sources], a.assets and os.path.expanduser(a.assets))
            print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in done.items()}))
            left = missing_assets(a.catalog, names)
            print("still missing:", left or "none")
            rc = 0 if not left else 1
    # support frames of slot bases (the asset can be present and still unusable without ``catalog annotate``)
    gaps = missing_support(a.catalog, names)
    if gaps and a.sync and a.assets:
        annotate(a.catalog, [g.split("#")[0] for v in gaps.values() for g in v], os.path.expanduser(a.assets))
        gaps = missing_support(a.catalog, names)
    for name, g in gaps.items():
        print(f"ERROR {name}: slot base without its passive.support frame {g} -- perception cannot place the objects in "
              f"that slot (run --sync --assets <RoboDojo Assets>)", file=sys.stderr)
    if not gaps:
        print("every slot base has its support frames")
    # links / joints of articulated assets (the skills' lid hinges, levers)
    lgaps = missing_links(a.catalog, names)
    if lgaps and a.sync:
        keys = sorted({k for v in lgaps.values() for k in v})
        done = sync_links(a.catalog, keys, [os.path.expanduser(s) for s in a.sources], a.assets)
        print("links.json:", json.dumps({k: len(v) for k, v in done.items()}))
        lgaps = missing_links(a.catalog, names)
    for name, g in lgaps.items():
        print(f"ERROR {name}: Articulation without links.json {g} -- the skills cannot see its joints / links "
              f"(run --sync --sources <catalogue with it> or --assets <RoboDojo Assets>)", file=sys.stderr)
    if not lgaps:
        print("every Articulation candidate has its links.json")
    return 1 if (rc or gaps or lgaps) else 0


if __name__ == "__main__":
    sys.exit(main())
