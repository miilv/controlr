"""Per-link geometry and joints of RoboDojo Articulation assets -> ``links.json`` in a mesh catalogue.

A functional point's metadata names the link it rides on (``base_link``) and its joint
(``parent_joint``), but its frame does not always sit on that link (toaster 00001: the lever's
``toast_botton`` frame is at the toaster's centre; the lever is a 53 mm bar on its side). This
reads the asset's USD and writes, per link, its visual-mesh bbox and points (asset frame, metres,
the frame the layout pose places) and, per joint, its type, axis (asset frame), limits and rest
position. ``world.AssetDB`` loads it as ``AssetInfo.links`` / ``AssetInfo.joints``.

Run in a python with ``pxr`` (perception's venv) on the machine with the assets:

    ~/dojo-perception/.venv/bin/python scripts/skills/articulation_links.py \\
        --assets ~/dojo-robodojo/RoboDojo/Assets/Object/RoboDojo --out ~/dojo-data/skills4/catalog \\
        Articulation/toaster Articulation/laptop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _quat_R(wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_links(usd: Path, max_pts: int = 4000) -> dict:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_w = np.array(cache.GetLocalToWorldTransform(root), dtype=float).T if root.IsA(UsdGeom.Xformable) else np.eye(4)
    root_inv = np.linalg.inv(root_w)

    def to_asset(prim) -> np.ndarray:
        return root_inv @ np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T

    # links: the direct Xform children of the default prim that hold meshes (RoboDojo's articulations:
    # /root/base, /root/link_0, /root/link_1, ...); a mesh belongs to the nearest such ancestor
    links: dict[str, list] = {}
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        p = prim
        while p.GetParent() and p.GetParent() != root:
            p = p.GetParent()
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, float)
        m = to_asset(prim)
        P = (np.c_[pts, np.ones(len(pts))] @ m.T)[:, :3] * mpu
        links.setdefault(p.GetName(), []).append(P)
    out_links = {}
    for name, Ps in links.items():
        P = np.vstack(Ps)
        if len(P) > max_pts:
            P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
        out_links[name] = {"bbox_min": np.round(P.min(0), 5).tolist(), "bbox_max": np.round(P.max(0), 5).tolist(),
                           "points": np.round(P, 4).tolist()}
    joints = {}
    for prim in Usd.PrimRange(root):
        t = prim.GetTypeName()
        if t not in ("PhysicsPrismaticJoint", "PhysicsRevoluteJoint"):
            continue
        b0 = prim.GetRelationship("physics:body0").GetTargets()
        b1 = prim.GetRelationship("physics:body1").GetTargets()
        ax = str(prim.GetAttribute("physics:axis").Get() or "X")
        e = {"X": [1.0, 0, 0], "Y": [0, 1.0, 0], "Z": [0, 0, 1.0]}[ax]
        r0 = prim.GetAttribute("physics:localRot0").Get()
        p0 = prim.GetAttribute("physics:localPos0").Get()
        R0 = _quat_R([r0.GetReal(), *r0.GetImaginary()]) if r0 is not None else np.eye(3)
        Tb0 = to_asset(stage.GetPrimAtPath(b0[0])) if b0 else np.eye(4)
        axis = Tb0[:3, :3] @ R0 @ np.asarray(e)
        origin = Tb0[:3, :3] @ (np.asarray(p0, float) * mpu if p0 is not None else np.zeros(3)) + Tb0[:3, 3] * mpu
        lo, hi = prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()
        kind = "prismatic" if "Prismatic" in t else "revolute"
        pos = prim.GetAttribute(f"state:{'linear' if kind == 'prismatic' else 'angular'}:physics:position").Get()
        stiff = prim.GetAttribute(f"drive:{'linear' if kind == 'prismatic' else 'angular'}:physics:stiffness").Get()
        scale = mpu if kind == "prismatic" else 1.0
        joints[prim.GetName()] = {
            "type": kind, "body0": b0[0].name if b0 else None, "body1": b1[0].name if b1 else None,
            "axis": np.round(axis / (np.linalg.norm(axis) or 1.0), 5).tolist(),
            "origin": np.round(origin, 5).tolist(),
            # revolute limits are in degrees in USD; prismatic in stage units
            "lower": None if lo is None or not np.isfinite(lo) else float(lo) * scale,
            "upper": None if hi is None or not np.isfinite(hi) else float(hi) * scale,
            "position": float(pos or 0.0) * scale, "stiffness": None if stiff is None else float(stiff)}
    return {"links": out_links, "joints": joints, "meters_per_unit": mpu}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="Assets/Object/RoboDojo")
    ap.add_argument("--out", required=True, help="catalogue root (writes <Type>/<cat>/<idx>/links.json)")
    ap.add_argument("cats", nargs="+", help="Type/category, e.g. Articulation/toaster")
    a = ap.parse_args(argv)
    root = Path(a.assets).expanduser()
    for tc in a.cats:
        for d in sorted((root / tc).iterdir()):
            usd = next((d / n for n in ("object.usdz", "object.usd") if (d / n).exists()), None)
            if usd is None:
                continue
            info = read_links(usd)
            od = Path(a.out).expanduser() / tc / d.name
            od.mkdir(parents=True, exist_ok=True)
            (od / "links.json").write_text(json.dumps(info))
            print(f"{tc}/{d.name}: links {sorted(info['links'])} joints "
                  f"{ {k: (v['type'], v['body1'], v['axis'], v['lower'], v['upper']) for k, v in info['joints'].items()} }",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
