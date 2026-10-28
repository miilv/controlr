"""A frozen perception calibration set: backgrounds + colour map from an explicit list of D0 layout ids, with a manifest.

The runtime's perception service uses ONE such set for every episode of an evaluation (``DOJO_CALIB_SET``). The set
is chosen once per deployment, never per episode: the old choice, ``fold = crc32(layout id) % 2``, made the layout id
a policy input (legality audit 2026-10-28, robot-harness docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md, P1-1).
To keep dev numbers clean the set is built only from layouts that are never evaluated: the calibration-only slice
``paths.CALIBRATION_SLICE`` (100040-49, D-C10b). ``build`` refuses any other id; the dev drivers refuse to evaluate a
calibration id and any layout the set's manifest lists (``overlap``).

The default set, ``frozen_d0_100040_49``: D0-format frames (frames 0-3: home + 3 scripted arm moves) of ids
100040-49 of the 6 D0 tasks, rendered for this (6.0 proxy on compute3: ``bg_folds``; genuine 5.1 on compute2:
``bg_folds_51``).

Layout of a set directory (``$DOJO_BG_FOLDS/<name>/``)::

    cam_head.png cam_left_wrist.png cam_right_wrist.png cam_head_clean.png cam_head_clean_unseen.png
    color_M.json          {"name": ..., "M": 4x3, "n": rows}      (the colour map; else DOJO_COLOR_CALIB[<name>])
    scene/<Type_cat_idx>/ scene backgrounds (conveyor, basket, ...): copied from ``--scene-from`` when given, and
                          the manifest says so (their own provenance is not re-derived here)
    manifest.json         {"name", "layout_ids", "tasks", "episodes", "d0_root", "built_by", "built_at", "scene", ...}

    python -m dojo.perception.calib_set build --d0 ~/dojo-data/d0cal_51 --ids 100040-100049 \\
        --name frozen_d0_100040_49 --out ~/dojo-data/calib_sets/bg_folds_51 [--scene-from <fold dir>]

DEV-ONLY tool (the colour map is fitted at GT poses); the runtime reads only the PNGs, ``color_M.json`` and the
manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

DEFAULT_SET = "frozen_d0_100040_49"
DEFAULT_IDS = "100040-100049"


def parse_ids(spec: str) -> list[int]:
    """``"100010-100019,100055"`` -> sorted unique ints."""
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out |= set(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def check_buildable(ids) -> None:
    """A frozen set is built only from the calibration-only slice (D-C10b): never from a layout anyone evaluates (our
    measurement slices, Mikhail's slices, reserved ids)."""
    from dojo.simlab import paths
    bad = [i for i in ids if not paths.is_calibration_id(i)]
    if bad or not ids:
        raise ValueError(f"ids {bad[:5]}{'...' if len(bad) > 5 else ''} are outside the calibration-only slice "
                         f"{paths.CALIBRATION_SLICE} (D-C10b): a calibration set is built from nothing else")


def load_manifest(set_dir) -> dict | None:
    p = Path(set_dir) / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def overlap(manifest: dict | None, layout_id) -> bool | None:
    """True when ``layout_id`` is one the set was built from, False when not, None when the set has no manifest
    (a legacy ``fold<k>`` dir: unknown)."""
    if not manifest or manifest.get("layout_ids") is None:
        return None
    try:
        return int(layout_id) in {int(i) for i in manifest["layout_ids"]}
    except (TypeError, ValueError):
        return None


def _commit() -> str | None:
    import os
    if os.environ.get("CALIB_SET_COMMIT"):          # a deploy without .git (the builder's commit, given by the caller)
        return os.environ["CALIB_SET_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def build(d0_root, out_root, name: str, ids, scene_from=None, max_n: int = 40, n_clean: int = 160,
          seed: int = 0, colour: bool = True) -> dict:
    """Write ``<out_root>/<name>/``: backgrounds + colour map from the D0 episodes whose id is in ``ids`` (every task
    of ``d0_root``), and the manifest. Refuses ids in our measurement slices."""
    import cv2

    from . import d0
    from .eval_g1 import backgrounds_from

    ids = sorted({int(i) for i in ids})
    check_buildable(ids)
    d0_root, out = Path(d0_root), Path(out_root) / name
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen set is never rebuilt in place (pick a new name)")
    eps = [e for e in d0.episodes(d0_root) if e.name.isdigit() and int(e.name) in ids]
    if not eps:
        raise FileNotFoundError(f"no D0 episodes with ids {ids[0]}..{ids[-1]} under {d0_root}")
    bgs = backgrounds_from(eps, np.random.default_rng(seed), max_n=max_n, n_clean=n_clean)
    if "cam_head" not in bgs:
        raise RuntimeError("no head frames: no background")
    tmp = out.with_name(out.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for k, v in bgs.items():
        if k.endswith("_unseen"):
            cv2.imwrite(str(tmp / f"{k}.png"), (v * 255).astype(np.uint8))
        else:
            cv2.imwrite(str(tmp / f"{k}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    man: dict = {"name": name, "layout_ids": ids, "tasks": sorted({e.parent.name for e in eps}),
                 "episodes": len(eps), "episode_ids": sorted(f"{e.parent.name}/{e.name}" for e in eps),
                 "d0_root": str(d0_root), "backgrounds": {"max_n": max_n, "n_clean": n_clean, "seed": seed},
                 "built_by": f"dojo.perception.calib_set @ {_commit()}",
                 "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "rule": "never calibrate an evaluation of these layout ids (legality audit 2026-10-28, P1-1)"}
    if colour:
        from .color_calib import collect, fit
        rows = collect(d0_root, episodes=eps, bgs_for=lambda _ep: bgs)
        if len(rows) < 8:
            raise RuntimeError(f"{len(rows)} colour rows: too few for a colour map")
        (tmp / "color_M.json").write_text(json.dumps({"name": name, "M": fit(rows).tolist(), "n": len(rows)}, indent=1))
        man["colour"] = {"rows": len(rows), "file": "color_M.json"}
    if scene_from:
        src = Path(scene_from) / "scene"
        if src.is_dir():
            shutil.copytree(src, tmp / "scene")
            man["scene"] = {"copied_from": str(src), "dirs": sorted(p.name for p in src.iterdir() if p.is_dir()),
                            "provenance": "not re-derived: the scene backgrounds' own source layouts are those of "
                                          "their builder (journal); not covered by layout_ids"}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(out)
    return man


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--d0", required=True, help="D0 root (<root>/<task>/<id>/frame_*)")
    b.add_argument("--ids", default=DEFAULT_IDS)
    b.add_argument("--name", default=DEFAULT_SET)
    b.add_argument("--out", required=True, help="the DOJO_BG_FOLDS root the set goes under")
    b.add_argument("--scene-from", default=None, help="a fold dir whose scene/ backgrounds are copied (recorded)")
    b.add_argument("--n-clean", type=int, default=160)
    b.add_argument("--no-colour", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("set_dir")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        man = build(a.d0, a.out, a.name, parse_ids(a.ids), scene_from=a.scene_from, n_clean=a.n_clean,
                    colour=not a.no_colour)
        print(json.dumps({k: v for k, v in man.items() if k != "episode_ids"}, indent=1))
    else:
        print(json.dumps(load_manifest(a.set_dir), indent=1))


if __name__ == "__main__":
    main()
