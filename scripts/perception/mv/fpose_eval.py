"""DEV-ONLY: FoundationPose (model-based: our catalogue meshes) refining B1's frame-0 poses on learned RGB-D.

RGB = the head image; depth = the learned fused depth in the head camera (VGGT through our cameras, table-fitted:
``mv3d_eval`` predictions); hypotheses = B1's pose and its yaw turns about world z; FoundationPose's refiner (5
iterations) then its scorer picks one. Scored with eval_g1 before / after on the same records (GT read after).

FoundationPose (NVlabs, weights: HF gpue/foundationpose-weights) imports pytorch3d / nvdiffrast / open3d at module level
and needs CUDA builds of them; we run only its two networks and its crop / batch code, with
- pytorch3d.transforms: the few rotation maps it uses (roma),
- nvdiffrast_render: our pyrender per-face-colour renderer (``render.ColorMeshRenderer``) on the crop camera, lit the
  way nvdiffrast_render lights it (0.8 ambient + 0.5 diffuse from the camera, normals from the rendered depth),
- open3d / nvdiffrast: empty stubs (not on this path).

    python scripts/perception/mv/fpose_eval.py --pred ~/dojo-data/pbake-mv/vggt --b1dump DIR --out DIR [--classes gross ok]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

EXT = Path(os.environ.get("DOJO_MV_EXT", os.path.expanduser("~/controlr-mv/ext")))
FP = EXT / "FoundationPose"


def _install_stubs():
    import roma
    import torch

    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    class _Dummy:
        def __init__(self, *a, **k):
            raise RuntimeError("stubbed out (not on the refine / score path)")

    p3 = mod("pytorch3d")
    tr = mod("pytorch3d.transforms")
    tr.so3_exp_map = lambda v: roma.rotvec_to_rotmat(v)
    tr.so3_log_map = lambda R: roma.rotmat_to_rotvec(R)
    tr.matrix_to_axis_angle = lambda R: roma.rotmat_to_rotvec(R)

    def rot6d(d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)
    tr.rotation_6d_to_matrix = rot6d
    for n in ("se3_exp_map", "se3_log_map", "matrix_to_euler_angles", "euler_angles_to_matrix"):
        setattr(tr, n, _Dummy)
    p3.transforms = tr
    rd = mod("pytorch3d.renderer")
    for n in ("FoVPerspectiveCameras", "PerspectiveCameras", "look_at_view_transform", "look_at_rotation", "RasterizationSettings",
              "MeshRenderer", "MeshRasterizer", "BlendParams", "SoftSilhouetteShader", "HardPhongShader", "PointLights", "TexturesVertex"):
        setattr(rd, n, _Dummy)
    mod("pytorch3d.renderer.mesh")
    mod("pytorch3d.renderer.mesh.rasterize_meshes").barycentric_coordinates = _Dummy
    sh = mod("pytorch3d.renderer.mesh.shader")
    sh.SoftDepthShader = sh.HardFlatShader = _Dummy
    mod("pytorch3d.renderer.mesh.textures").Textures = _Dummy
    mod("pytorch3d.structures").Meshes = _Dummy
    nv = mod("nvdiffrast")
    nvt = mod("nvdiffrast.torch")
    nvt.RasterizeCudaContext = nvt.RasterizeGLContext = _Dummy
    nv.torch = nvt
    mod("open3d")


class CropRenderer:
    """nvdiffrast_render's contract on pyrender: (B, h, w, 3) colour in [0, 1], (B, h, w) z-depth, xyz_map in the
    camera frame, for poses ``ob_in_cams`` rendered into the crop windows ``bbox2d`` (l, t, r, b; FoundationPose's
    integer-pixel coordinates)."""

    def __init__(self):
        from dojo.perception.render import ColorMeshRenderer

        self.r = ColorMeshRenderer(160, 160)
        self.key = None

    def __call__(self, K=None, H=None, W=None, ob_in_cams=None, output_size=None, bbox2d=None, use_light=False,
                 extra=None, w_ambient=0.8, w_diffuse=0.5, **_):
        import torch

        from dojo.perception.cameras import Camera

        h, w = (int(output_size[0]), int(output_size[1])) if output_size is not None else (H, W)
        poses = ob_in_cams.detach().cpu().numpy() if hasattr(ob_in_cams, "detach") else np.asarray(ob_in_cams)
        bb = bbox2d.detach().cpu().numpy() if bbox2d is not None else np.tile([0, 0, W, H], (len(poses), 1))
        cols, deps, xyzs = [], [], []
        for T, (l, t, r, b) in zip(poses, bb):
            sx, sy = w / max(r - l, 1e-6), h / max(b - t, 1e-6)
            Kc = np.array([[K[0][0] * sx, 0, (K[0][2] + 0.5 - l) * sx], [0, K[1][1] * sy, (K[1][2] + 0.5 - t) * sy], [0, 0, 1.0]])
            cam = Camera("crop", Kc, np.eye(4), w, h)
            col, d = self.r.render(cam, self.key, T)
            col = col[..., :3].astype(np.float32) / 255.0
            d = d.astype(np.float32)
            vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
            xyz = np.stack([(uu + 0.5 - Kc[0, 2]) / Kc[0, 0] * d, (vv + 0.5 - Kc[1, 2]) / Kc[1, 1] * d, d], -1)
            if use_light:  # nvdiffrast_render: colour * 0.8 + diffuse(light along +z from the camera) * colour * 0.5
                gx = np.gradient(xyz, axis=1)
                gy = np.gradient(xyz, axis=0)
                n = np.cross(gx, gy)
                n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
                n = np.where(n[..., 2:3] > 0, -n, n)  # facing the camera
                diff = np.clip(-n[..., 2:3], 0, 1)
                col = np.clip(col * w_ambient + diff * col * w_diffuse, 0, 1)
            col = col * (d > 0)[..., None]
            cols.append(col)
            deps.append(d)
            xyzs.append(xyz * (d > 0)[..., None])
        dev = "cuda"
        if extra is not None:
            extra["xyz_map"] = torch.as_tensor(np.stack(xyzs), device=dev, dtype=torch.float)
        return (torch.as_tensor(np.stack(cols), device=dev, dtype=torch.float),
                torch.as_tensor(np.stack(deps), device=dev, dtype=torch.float), None)


def load_fp():
    _install_stubs()
    sys.path[:0] = [str(FP), str(FP / "learning" / "models")]
    cwd = os.getcwd()
    os.chdir(FP)
    try:
        import learning.training.predict_pose_refine as ppr
        import learning.training.predict_score as pps
    finally:
        os.chdir(cwd)
    import torch

    torch.set_default_tensor_type("torch.FloatTensor")
    rend = CropRenderer()
    ppr.nvdiffrast_render = rend
    pps.nvdiffrast_render = rend
    refiner = ppr.PoseRefinePredictor()
    scorer = pps.ScorePredictor()
    torch.set_default_tensor_type("torch.FloatTensor")
    return refiner, scorer, rend


def head_depth(z: dict, vs: list[dict], mode: str = "poly") -> np.ndarray:
    """The learned fused z-depth of the head view at full resolution (0 = unknown)."""
    import cv2

    from dojo.perception import mv3d
    from dojo.perception.mv3d_eval import _pred_points

    P, _, _ = _pred_points(z, vs, mode, 0.0)
    _, d = mv3d.splat(P[0]["cam"], [p["points"] for p in P], [p["use"] for p in P])
    d = np.where(P[0]["rob"] == 0, d, np.nan)
    head = vs[0]["cam"]
    return np.nan_to_num(cv2.resize(d.astype(np.float32), (head.w, head.h), interpolation=cv2.INTER_NEAREST), nan=0.0)


def yaw_turns(T_wo: np.ndarray, n: int) -> list[np.ndarray]:
    """The pose and its turns about world z through its own origin."""
    from scipy.spatial.transform import Rotation

    out = []
    for k in range(n):
        Rz = Rotation.from_euler("z", 2 * np.pi * k / n).as_matrix()
        T = T_wo.copy()
        T[:3, :3] = Rz @ T_wo[:3, :3]
        out.append(T)
    return out


def flagged_objects(check_file: str | None, flag_mm: float) -> dict | None:
    """(set, task, episode) -> {(label, rounded pos)} of B1 objects the learned height check flags (|dh| > flag_mm)."""
    if not check_file:
        return None
    out: dict = {}
    for line in open(os.path.expanduser(check_file)):
        r = json.loads(line)
        if "error" in r:
            continue
        s = out.setdefault((r["set"], r["task"], r["episode"]), set())
        for o in r["objects"]:
            if o.get("mv3d_dh_mm") is not None and abs(o["mv3d_dh_mm"]) > flag_mm:
                s.add((o["label"], tuple(np.round(o["pos"], 3))))
    return out


def run_scene(f: Path, b1dump: Path, fp, classes, n_yaw: int = 8, iters: int = 5, flagged: dict | None = None) -> dict:
    import copy

    import torch

    from dojo.perception import d0
    from dojo.perception.cameras import R_to_quat_wxyz, pose_to_T
    from dojo.perception.mv3d_eval import SET_ALIAS, _b1_record, _score, root_dir, scene_views
    from dojo.perception.render import load_asset

    refiner, scorer, rend = fp
    root_name, task, epn = f.name.split("__")[:3]
    ep = root_dir(root_name) / task / epn
    mf = b1dump / {v: k for k, v in SET_ALIAS.items()}.get(root_name, root_name) / task / epn / "meta.json"
    meta = json.loads(mf.read_text())
    z = dict(np.load(f))
    vs = scene_views(ep, "home")
    depth = head_depth(z, vs)
    cam = vs[0]["cam"]
    rgb = vs[0]["img"]
    K = cam.K.astype(np.float32).copy()
    K[0, 2] -= 0.5  # FoundationPose: pixel centres at integer coordinates
    K[1, 2] -= 0.5
    depth = depth.astype(np.float32)
    T_cw = np.linalg.inv(cam.T_wc)
    rec0 = _b1_record(meta)
    gt = d0.load_gt(ep, 0)  # DEV-ONLY: which B1 objects to try is chosen by their class (a dev selection, then scored)
    m = {"set": root_name, "task": task, "episode": epn, "frame": 0}
    c0, _ = _score(rec0, gt, task, m)
    gpos = {i: o.get("pos") for i, o in enumerate(gt["objects"])}
    todo = []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec0.get(typ) or {}).items():
            for j, r in enumerate(lst):
                gi = r.get("_gt_i")
                if gi is None or gi < 0 or gpos.get(gi) is None:
                    continue
                o = gt["objects"][gi]
                k = (o.get("label"), o["category_idx"], tuple(np.round(o["pos"], 4)))
                cls = next((v for kk, v in c0.items() if kk[2] == k[2]), None)
                if flagged is not None and (r.get("label"), tuple(np.round(r["default_pos"], 3))) not in flagged.get((root_name, task, epn), set()):
                    continue
                if cls in classes:
                    todo.append((typ, cat, j, r, cls))
    rec1 = copy.deepcopy(rec0)
    log = []
    for typ, cat, j, r, cls in todo:
        key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
        a = load_asset(key)
        diam = float(np.linalg.norm(a.V.max(0) - a.V.min(0)))
        mesh = types.SimpleNamespace(vertices=a.Vs.astype(np.float32))
        T_wo = pose_to_T(r["default_pos"], r["default_ori"])
        hyps = np.stack([T_cw @ T for T in yaw_turns(T_wo, n_yaw)]).astype(np.float32)
        rend.key = key
        t0 = time.time()
        torch.set_default_tensor_type("torch.FloatTensor")
        xyz = _xyz(depth, K)
        poses, _ = refiner.predict(rgb=rgb, depth=depth, K=K, ob_in_cams=hyps, xyz_map=xyz, mesh=mesh, mesh_tensors={},
                                   glctx=object(), mesh_diameter=diam, iteration=iters)
        torch.set_default_tensor_type("torch.FloatTensor")
        scores, _ = scorer.predict(rgb=rgb, depth=depth, K=K, ob_in_cams=poses.detach().cpu().numpy(), mesh=mesh,
                                   mesh_tensors={}, glctx=object(), mesh_diameter=diam)
        torch.set_default_tensor_type("torch.FloatTensor")
        best = int(torch.argmax(scores).item())
        T_new = cam.T_wc @ poses[best].detach().cpu().numpy()
        e = rec1[typ][cat][j]
        e["default_pos"] = [float(v) for v in T_new[:3, 3]]
        e["default_ori"] = [float(v) for v in R_to_quat_wxyz(T_new[:3, :3])]
        log.append({"key": key, "label": r.get("label"), "b1_class": cls, "best": best, "s": round(time.time() - t0, 2),
                    "moved_mm": round(float(np.linalg.norm(T_new[:3, 3] - T_wo[:3, 3])) * 1000, 1)})
    c1, _ = _score(rec1, gt, task, m)
    changes = [{"label": k[0], "before": c0.get(k), "after": c1.get(k)} for k in c0 if c0.get(k) != c1.get(k)]
    return {**m, "tried": log, "changes": changes}


def _xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    vv, uu = np.mgrid[0:depth.shape[0], 0:depth.shape[1]].astype(np.float32)
    x = (uu - K[0, 2]) / K[0, 0] * depth
    y = (vv - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], -1) * (depth > 0.001)[..., None]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--b1dump", default="~/dojo-data/pbake-mv/c4/sam/b1/dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=["gross"])
    ap.add_argument("--match", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-yaw", type=int, default=8)
    ap.add_argument("--flagged", default="", help="only B1 objects this learned height check (mv3d_eval check frames.jsonl) flags")
    ap.add_argument("--flag-mm", type=float, default=15.0)
    a = ap.parse_args(argv)
    os.environ.setdefault("DOJO_GL", "1")
    fp = load_fp()
    flagged = flagged_objects(a.flagged, a.flag_mm)
    files = sorted(Path(a.pred).expanduser().glob("pred/*__home.npz"))
    if a.match:
        files = [f for f in files if any(m in f.name for m in a.match)]
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from collections import Counter

    tot = Counter()
    with open(out / "frames.jsonl", "w") as fh:
        for f in files:
            try:
                r = run_scene(f, Path(a.b1dump).expanduser(), fp, set(a.classes), n_yaw=a.n_yaw, flagged=flagged)
            except Exception as e:
                import traceback

                r = {"file": f.name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1200:]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r["file"], r["error"], r["tb"], flush=True)
                continue
            for c in r["changes"]:
                tot[(c["before"], c["after"])] += 1
            tot["tried"] += len(r["tried"])
            if r["tried"]:
                print(r["task"], r["episode"], r["changes"], [(t["label"], t["b1_class"], t["moved_mm"], t["s"]) for t in r["tried"]], flush=True)
    print(dict(tot))


if __name__ == "__main__":
    main()
