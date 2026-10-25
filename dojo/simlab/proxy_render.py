"""Isaac Sim 6.0 proxy renderer for RoboDojo states (runs inside Isaac 6.0's python, after SimulationApp).

Shared by ``render_d0_proxy.py`` (offline D0 frames) and ``render_service.py`` (render on demand). It opens
RoboDojo's own USD stage of a layout (exported by the physics-only 5.1 twin), builds the three cameras the
way RoboDojo's CameraManager does, writes recorded object / robot-link poses onto the prims (no physics),
and renders 640x480 RGB with IsaacLab's "quality" settings + DLAA (RoboDojo's DEFAULT_RENDER_CONFIG).
"""

from __future__ import annotations

import math

import numpy as np

RENDERER_TAG = "isaac-6.0 (proxy for 5.1)"
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
QUALITY = {
    "/rtx/translucency/enabled": True, "/rtx/reflections/enabled": True, "/rtx/reflections/denoiser/enabled": True,
    "/rtx/directLighting/sampledLighting/enabled": True, "/rtx/sceneDb/ambientLightIntensity": 1.0,
    "/rtx/shadows/enabled": True, "/rtx/indirectDiffuse/enabled": True, "/rtx/indirectDiffuse/denoiser/enabled": True,
    "/rtx/ambientOcclusion/enabled": True, "/rtx/ambientOcclusion/denoiserMode": 0,
    "/rtx/raytracing/subpixel/mode": 1, "/rtx/raytracing/cached/enabled": True, "/rtx-transient/dlssg/enabled": False,
    "/rtx-transient/dldenoiser/enabled": True, "/rtx/post/dlss/execMode": 2, "/omni/replicator/RTSubframes": 3,
}
APP_CONFIG = {"headless": True, "width": 640, "height": 480, "renderer": "RaytracedLighting", "anti_aliasing": 3,
              "multi_gpu": False, "sync_loads": True}


def euler_xyz_intrinsic_deg_to_quat(e) -> tuple[float, float, float, float]:
    """RoboDojo ``utils.rotations.euler_angles_to_quat(degrees=True)`` (default extrinsic=False, i.e.
    scipy "XYZ" intrinsic): R = Rx(a) Ry(b) Rz(c); wxyz."""
    a, b, c = (math.radians(v) / 2 for v in e)
    qx = (math.cos(a), math.sin(a), 0.0, 0.0)
    qy = (math.cos(b), 0.0, math.sin(b), 0.0)
    qz = (math.cos(c), 0.0, 0.0, math.sin(c))

    def mul(p, q):
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return mul(qx, mul(qy, qz))


class ProxyRenderer:
    def __init__(self, app):
        import carb
        import omni.replicator.core as rep
        import omni.usd
        self.app, self.rep = app, rep
        st = carb.settings.get_settings()
        for k, v in QUALITY.items():
            st.set(k, v)
        try:
            rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
        except Exception as e:  # noqa: BLE001
            print(f"[proxy_render] DLAA not set: {e}", flush=True)
        self.ctx = omni.usd.get_context()
        self.stage = None
        self.cams: dict = {}
        self.rps: dict = {}
        self.anns: dict = {}

    def open(self, stage_path: str, cams: dict, warmup: int = 20) -> None:
        from pxr import Gf, UsdGeom
        self.close_stage()
        self.ctx.open_stage(str(stage_path))
        for _ in range(200):
            self.app.update()
            if self.ctx.get_stage_loading_status()[2] == 0:
                break
        self.stage = self.ctx.get_stage()
        self.cams = cams
        for name in CAMS:
            c = cams[name]
            parent = c["mount"] or "/World/envs/env_0"
            xpath = f"{parent}/SimlabCam_{name}"
            xform = UsdGeom.Xform.Define(self.stage, xpath)
            q = euler_xyz_intrinsic_deg_to_quat(c["ori_euler_deg"])
            xform.MakeMatrixXform().Set(Gf.Matrix4d(Gf.Rotation(Gf.Quatd(*q)), Gf.Vec3d(*map(float, c["pos"]))))
            cam = UsdGeom.Camera.Define(self.stage, f"{xpath}/{name}")
            cam.CreateFocalLengthAttr().Set(float(c["focal_length"]))
            cam.CreateHorizontalApertureAttr().Set(float(c["horizontal_aperture"]))
            cam.CreateVerticalApertureAttr().Set(float(c["vertical_aperture"]))
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(*map(float, c["clipping_range"])))
            self.rps[name] = self.rep.create.render_product(f"{xpath}/{name}", tuple(c["resolution"]))
            self.anns[name] = self.rep.AnnotatorRegistry.get_annotator("rgb")
            self.anns[name].attach([self.rps[name]])
        for _ in range(warmup):          # shader / texture warm-up
            self.app.update()

    def close_stage(self) -> None:
        for name in list(self.rps):
            try:
                self.anns[name].detach([self.rps[name]])
                self.rps[name].destroy()
            except Exception:  # noqa: BLE001
                pass
        self.rps, self.anns = {}, {}

    def _set_world_pose(self, cache, path: str, pos, quat) -> bool:
        from pxr import Gf, UsdGeom
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return False
        cur = cache.GetLocalToWorldTransform(prim)
        scale = [Gf.Vec3d(cur.GetRow3(i)).GetLength() for i in range(3)]
        q = np.asarray(quat, float)
        q = q / np.linalg.norm(q)                 # IsaacLab poses are float32, not exactly unit
        rot = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(*map(float, q))))
        world = Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(*scale)) * Gf.Matrix4d(rot, Gf.Vec3d(*map(float, pos)))
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(world * parent.GetInverse())
        return True

    def render(self, gt: dict, subframes: int = 8) -> tuple[dict, dict, list]:
        """Images {cam: HxWx3 uint8}, camera GT {cam: K, T_world_cam_*}, prim paths not found."""
        from pxr import UsdGeom
        org = np.asarray(gt.get("env_origin", [0, 0, 0]), float)
        cache = UsdGeom.XformCache()
        missing = []
        for o in gt["objects"]:
            if o.get("pos") is None or not o.get("prim_path"):
                continue
            if not self._set_world_pose(cache, o["prim_path"], np.asarray(o["pos"]) + org, o["quat"]):
                missing.append(o["prim_path"])
        for _, r in gt["robot"].items():
            lp = r.get("link_prims", {})
            for ln in sorted(r["links"], key=lambda n: lp.get(n, "").count("/")):
                if ln not in lp:
                    continue
                pose = np.asarray(r["links"][ln], float)
                cache.Clear()
                if not self._set_world_pose(cache, lp[ln], pose[:3] + org, pose[3:]):
                    missing.append(lp[ln])
        self.rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0, pause_timeline=False)
        images, cam_gt = {}, {}
        cache = UsdGeom.XformCache()
        for name in CAMS:
            images[name] = np.ascontiguousarray(np.asarray(self.anns[name].get_data())[..., :3]).astype(np.uint8)
            c = self.cams[name]
            parent = c["mount"] or "/World/envs/env_0"
            m = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(f"{parent}/SimlabCam_{name}/{name}"))
            T = np.array(m, float).T
            T[:3, 3] -= org
            W, H = c["resolution"]
            fx = W * float(c["focal_length"]) / float(c["horizontal_aperture"])
            cam_gt[name] = {"K": [[fx, 0.0, W / 2.0], [0.0, fx, H / 2.0], [0.0, 0.0, 1.0]],
                            "T_world_cam_usd": T.tolist(),
                            "T_world_cam_cv": (T @ np.diag([1.0, -1.0, -1.0, 1.0])).tolist(), "width": W, "height": H,
                            "fy_if_vertical_aperture_honoured": H * float(c["focal_length"]) / float(c["vertical_aperture"])}
        return images, cam_gt, missing
