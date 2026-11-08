"""Build and check the ``f-catalogue/1`` file (the partner's F adapter input), catalogue-wide, CPU only.

    # one file per catalogue build (pxr for the Articulations' USD: perception's venv)
    python scripts/dojo/f_catalogue_extract.py build --assets ~/dojo-robodojo/RoboDojo/Assets/Object/RoboDojo \\
        --catalog ~/dojo-perception-data/catalog --out ~/dojo-data/fcat/ [--workers 2]
    # the catalogue-wide round trip, the verb readers and CONV, against the raw metadata
    python scripts/dojo/f_catalogue_extract.py check FILE --assets ... --out check.json
    # the derived-geometry method comparison (surface: heightmap / slice / hull; base band 4 / 8 / 12 mm; lip section
    # vs slab; slot width corners vs mesh)
    python scripts/dojo/f_catalogue_extract.py compare --assets ... --catalog ... --out compare.json [--workers 2]

The build writes ``f-catalogue-<catalogue_hash[:12]>-<code sha256[:8]>.json`` into ``--out`` (a directory; the code
hash covers this script and ``dojo/skills/fcatalogue.py``) and
prints its sha256 (the value for ``hybrid_pins``). Reads only what ``dojo.skills.fcatalogue.reader`` allows: RoboDojo's
``metadata.json`` / ``object.usd[z]`` and the catalogue's ``index.json`` / ``geom.npz``, nothing else. No LLM calls, no
simulator.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dojo.skills import fcatalogue as fc  # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def cmd_build(a) -> int:
    doc = fc.build(a.assets, a.catalog, script_path=str(Path(__file__).resolve()), log=log, workers=a.workers,
                   allow_index_mismatch=a.allow_index_mismatch)
    doc["build"]["script"]["path"] = "scripts/dojo/f_catalogue_extract.py"
    bad = fc.validate(doc)
    if bad:
        log(f"[fcat] VALIDATION: {len(bad)} problems, first: {bad[:5]}")
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    name = f"f-catalogue-{(doc['build']['catalogue_hash'] or 'nohash')[:12]}-{doc['build']['script']['code_sha256'][:8]}.json"
    p = out / name
    p.write_text(json.dumps(doc, default=_jsonable, separators=(",", ":")) + "\n")
    sha = fc.sha256_file(p)
    summ = summary(doc)
    summ.update({"file": str(p), "file_sha256": sha, "file_bytes": p.stat().st_size, "validation": bad})
    (out / (name[:-5] + ".summary.json")).write_text(json.dumps(summ, indent=1, default=_jsonable) + "\n")
    print(json.dumps({k: summ[k] for k in ("file", "file_sha256", "file_bytes", "n_assets", "n_entries", "wall_s")}))
    return 1 if bad else 0


def summary(doc: dict) -> dict:
    A = doc["assets"]
    s = {"n_assets": doc["build"]["sources"]["n_assets"], "n_entries": len(A), "wall_s": doc["build"]["wall_s"],
         "skipped": doc["build"]["sources"]["skipped"], "catalogue_hash": doc["build"]["catalogue_hash"],
         "listing_sha256": doc["build"]["sources"]["listing_sha256"]}
    sec = collections.Counter()
    for e in A.values():
        for k in fc.SECTIONS:
            if e["frames"][k]:
                sec[k] += 1
        for k in ("joints_by_tag", "tag_conv", "root_in_asset"):
            if e.get(k):
                sec[k] += 1
        if e.get("derived"):
            sec["derived"] += 1
    s["with_section"] = dict(sec)
    types, nulls, codex = collections.Counter(), collections.defaultdict(set), collections.defaultdict(set)
    for k, e in A.items():
        for tag, t in e["tag_types"].items():
            types[t or "null"] += 1
            if t is None:
                r = e["tag_rules"][tag]
                (codex if r.startswith("null for Codex") else nulls)[r].add(f"{k.rsplit('/', 1)[0]}:{tag}")
    s["types"] = dict(types)
    s["null_for_codex"] = {r: sorted(v) for r, v in codex.items()}
    s["null_other"] = {r: sorted(v)[:20] + ([f"... {len(v)} in all"] if len(v) > 20 else []) for r, v in nulls.items()}
    der, ref = collections.Counter(), collections.Counter()
    for e in A.values():
        for tag, d in (e.get("derived") or {}).items():
            der[tag if tag.startswith("@") else d["type"]] += 1
        for tag, why in (e.get("derived_refused") or {}).items():
            ref[(tag if tag.startswith("@") or tag == "*" else e["tag_types"].get(tag)) + ": " + why] += 1
    s["derived"] = dict(der)
    s["derived_refused"] = dict(ref.most_common())
    s["tag_conv"] = sum(len(e.get("tag_conv") or {}) for e in A.values())
    return s


def raw_metadata(assets_root: str, keys) -> dict:
    rd = fc.reader(assets_root)
    return {k: fc.read_json(rd, rd["assets"] / k / "metadata.json") for k in keys}


def cmd_check(a) -> int:
    from scripts.dojo.fcat import check as ck
    t0 = time.time()
    doc = json.loads(Path(a.file).expanduser().read_text())
    bad = fc.validate(doc)
    A = doc["assets"]
    mds = raw_metadata(a.assets, A)
    out = {"file": str(a.file), "file_sha256": fc.sha256_file(Path(a.file).expanduser()), "validation": bad}
    # 1. the round trip, four ways
    rt = {}
    for mode in ("typed", "all"):
        for p1 in (False, True):
            name = f"{mode}/{'p1' if p1 else 'verbatim'}"
            n_fr = n_tags = 0
            fails, skipped, worst = [], 0, 0.0
            file_worst = 0.0
            for k, e in A.items():
                r = ck.round_trip(k, e, mds[k], mode, p1)
                n_fr += r["n_frames"]
                n_tags += len(r["tags"])
                skipped += len(r["skipped"])
                fails += [f"{k}: {x}" for x in r["fail"]]
                worst = max([worst] + [v for v in r["tags"].values() if math.isfinite(v)])
                file_worst = max(file_worst, r["file_err"])
            rt[name] = {"frames": n_fr, "tags": n_tags, "skipped_null_tags": skipped, "fails": len(fails),
                        "fail_list": fails[:40], "max_err": worst, "file_vs_raw_max_err": file_worst}
            log(f"[check] round trip {name}: {n_fr} frames, {n_tags} tags, {len(fails)} fails, max {worst:.3g}")
    out["round_trip"] = rt
    # 2. single-asset readers, verbatim and p1
    for p1 in (False, True):
        rd = collections.defaultdict(float)
        worst_key = {}
        for i, (k, e) in enumerate(A.items()):
            if not any(e["frames"][s] for s in fc.SECTIONS):
                continue
            for n, v in ck.reader_check(k, e, mds[k], seed=i, p1=p1).items():
                if v > rd[n]:
                    rd[n], worst_key[n] = v, k
        out[f"readers/{'p1' if p1 else 'verbatim'}"] = {"max_diff": dict(rd), "worst_key": worst_key}
        log(f"[check] readers {'p1' if p1 else 'verbatim'}: {dict(rd)}")
    # 3. two-asset readers (slot_targets, insert_into's composition), every occupant x every container
    occ = [k for k, e in A.items() if e["frames"]["active_functional"]]
    con = [k for k, e in A.items() if e["frames"]["functional"] or e["frames"]["support"]]
    pm, n = collections.defaultdict(float), collections.Counter()
    for i, ko in enumerate(occ):
        for kc in con:
            if ko == kc:
                continue
            r = ck.pair_check((ko, A[ko], mds[ko]), (kc, A[kc], mds[kc]), seed=i)
            for kk, v in r.items():
                if kk.startswith("n_"):
                    n[kk] += v
                else:
                    pm[kk] = max(pm[kk], v)
    out["pair_readers"] = {"occupants": len(occ), "containers": len(con), "max_diff": dict(pm), "counts": dict(n)}
    log(f"[check] pair readers: {out['pair_readers']}")
    # 4. CONV: semantic axis per type, pinned vs the prototype's table; the slot reader's convention
    rows = [r for k, e in A.items() for r in ck.conv_check(k, e, mds[k])]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    conv = {}
    for t, rs in sorted(by.items()):
        ax = np.array([r["axis_deg"] for r in rs])
        pr = np.array([r["proto_axis_deg"] for r in rs])
        conv[t] = {"n": len(rs), "pinned_within_25deg": int((ax <= 25).sum()), "pinned_max_deg": float(ax.max()),
                   "proto_within_25deg": int((pr <= 25).sum()), "overrides": int(sum(r["override"] for r in rs)),
                   "type_row_frames": int(sum(not r["override"] for r in rs))}
        if t == "slot":
            conv[t]["reader_z_out_max_deg"] = float(max(r["reader_z_out_deg"] for r in rs))
            conv[t]["reader_x_along_max_deg"] = float(max(r["reader_x_along_deg"] for r in rs))
            conv[t]["raw_frame_not_reader_frame"] = sorted(f"{r['key']}:{r['tag']}" for r in rs
                                                          if r["raw_z_out_deg"] > 1 or r["raw_x_along_deg"] > 1)
    out["conv"] = {"pinned": doc["build"]["conv"], "per_type": conv,
                   "tag_conv": {f"{k}:{tag}": {"conv": tc, "why": (e.get("tag_conv_why") or {}).get(tag)}
                                for k, e in A.items() for tag, tc in (e.get("tag_conv") or {}).items()}}
    coin = next((k for k in A if k.startswith("Rigid/coin/")), None)
    if coin:
        worst = 0.0
        for k, e in A.items():
            for tag, d in (e.get("derived") or {}).items():
                if d.get("type") != "slot" or tag.startswith("@"):
                    continue
                for t in ck.targets_of(k, e, "typed", True):
                    if t.provenance[0]["tag"] == tag:
                        worst = max(worst, ck.slot_reader_check(coin, mds[coin], t.axis, t.across, t.point))
        out["conv"]["slot_targets_from_pose_max_diff"] = worst
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).expanduser().write_text(json.dumps(out, indent=1, default=_jsonable) + "\n")
    ok = not bad and rt["typed/p1"]["fails"] == 0 and rt["all/p1"]["fails"] == 0
    print(json.dumps({"ok": ok, "round_trip": {k: (v["frames"], v["fails"]) for k, v in rt.items()},
                      "readers": out["readers/p1"]["max_diff"], "pairs": out["pair_readers"]["max_diff"]}))
    return 0 if ok else 1


# ============================================================================================ method comparison
def _compare_job(args):
    key, md, mesh, entry = args
    V, F = mesh
    rm = fc.root_map(fc.pose7_T(entry["root_in_asset"]) if entry.get("root_in_asset") else None)
    if not rm["identity"]:
        V = (np.asarray(V, float) - rm["T"][:3, 3]) @ rm["T"][:3, :3]
    out = {"key": key, "n_faces": int(len(F))}
    if fc.upright((md.get("active") or {}).get("place")):
        hm = fc.top_height_map(V, F, fc.SURFACE_PARAMS["cell"])
        for name, f in fc.SURFACE_METHODS.items():
            t0 = time.time()
            kw = {"flat_tol": 0.004, "min_area": 4e-4}
            if name == "heightmap":
                kw.update(cell=0.003, hmap=hm)
            r = f(V, F, **kw)
            out[f"surface/{name}"] = ({"refused": r["refused"]} if "refused" in r else
                                      {"z_max": r["z_max"], "below_top": r["top"] - r["z_max"], "area": r["area"],
                                       "rim": r["rim_height"], "xy": [float(x) for x in r["xy"]]})
            out[f"surface/{name}"]["s"] = round(time.time() - t0, 3)
        for band in (0.004, 0.008, 0.012):
            b = fc.base_band(V, F, band)
            out[f"base/{int(band * 1000)}mm"] = b.get("refused") or {"radius": b["radius"],
                                                                     "radius_about_origin": b["radius_about_origin"]}
    for tag, d in (entry.get("derived") or {}).items():
        if tag.startswith("@") or d.get("method") != "opening_fit":
            continue
        section = "support" if tag in entry["frames"]["support"] else "functional"
        fr = entry["frames"][section][tag]
        fl = [c[0] for c in fr] if section == "support" else fr["frame"]
        a_, s_, c_ = entry["tag_conv"].get(tag) or fc.CONV[d["type"]]
        R = fc.quat_R(fl[0][3:7])
        into = s_ * R[:, a_] * (-1.0 if d["type"] == "support" else 1.0)
        P = np.mean([np.asarray(f[:3], float) for f in fl], axis=0) if d["type"] == "slot" and len(fl) > 1 \
            else np.asarray(fl[0][:3], float)
        T = np.eye(4)
        T[:3, 3] = P
        win = tuple(d["params"]["window_m"])
        long = R[:, d["long_col"]]
        slab = fc.opening_fit(V, F, T, into, long, half_window=win, lip_slab=0.001)
        out[f"opening/{tag}"] = {"type": d["type"], "width_mesh": d["extent"]["width"], "depth": d["extent"]["depth"],
                                 "length_mesh": d["extent"].get("length"), "corners": d.get("corners"),
                                 "lip_section": d.get("lip"), "lip_slab_1mm": slab.get("lip"),
                                 "width_from": d.get("width_from"), "floor_below_frame": d.get("floor_below_frame")}
    return out


def cmd_compare(a) -> int:
    doc = json.loads(Path(a.file).expanduser().read_text())
    A = doc["assets"]
    rd = fc.reader(a.assets, a.catalog)
    mds = {k: fc.read_json(rd, rd["assets"] / k / "metadata.json") for k in A}

    def jobs():
        for k, e in A.items():
            p = rd["catalog"] / k / "geom.npz"
            if p.is_file():
                yield k, mds[k], fc.read_mesh(rd, p), e
    rows = []
    t0 = time.time()
    if a.workers > 1:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            pend = set()
            for j in jobs():
                pend.add(ex.submit(_compare_job, j))
                if len(pend) >= 2 * a.workers:
                    fin, pend = wait(pend, return_when=FIRST_COMPLETED)
                    rows += [f.result() for f in fin]
                    if len(rows) % 50 < len(fin):
                        log(f"[compare] {len(rows)} ({time.time() - t0:.0f} s)")
            rows += [f.result() for f in pend]
    else:
        for j in jobs():
            rows.append(_compare_job(j))
    rows.sort(key=lambda r: r["key"])
    Path(a.out).expanduser().write_text(json.dumps({"file": a.file, "rows": rows, "wall_s": round(time.time() - t0, 1)},
                                                   default=_jsonable) + "\n")
    print(json.dumps({"rows": len(rows), "wall_s": round(time.time() - t0, 1)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--assets", required=True, help="RoboDojo Assets/Object/RoboDojo")
    b.add_argument("--catalog", required=True, help="perception's catalogue root (ONE root)")
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--allow-index-mismatch", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("--assets", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("file")
    m.add_argument("--assets", required=True)
    m.add_argument("--catalog", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
