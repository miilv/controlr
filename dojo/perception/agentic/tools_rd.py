"""RoboDojo tools for the agentic perceiver: a per-call workspace and a restricted command executor.

Allowed commands = the protocol's own scripts (``vendor/robodojo``; same argument syntax as in
docs/PERCEIVER-PROTOCOL.md, file arguments are workspace file names) plus two built-ins over the call's calibration
(``geom.py`` math, the same as the server's verbs):

    locate cam=<cam> px=[u,v] [cam2=<cam> px2=[u,v]] [z=<m>]   two views: triangulation + ray gap; one view: plane hit
                                                               at z (default the table, 0.765)
    project pts=[[x,y,z],...] [img=1]                           pixels in every camera (img=1: marked full frames)

Anything else (shell, paths outside the workspace, other programs) is refused with a message to the model.
Images a command writes into the workspace (zoom/mosaic outputs, project marks) come back to the model.
"""
from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor"
RD_TOOLS = VENDOR / "robodojo"
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SCRIPTS = ("zoom.py", "mosaic.py", "ph.py", "tri2.py", "sil.py", "cube.py", "cubez.py", "tp.py", "tpose.py", "btn.py",
           "rim.py", "line2v.py", "vcheck.py")
TABLE_Z = 0.765
MAX_IMG_SIDE = 1280
W, H = 640, 480


def write_calib(path: Path, calib: dict, flange: dict | None) -> None:
    """calib {cam: {K, T_world_cam}} + flange {arm: [x,y,z,qw,qx,qy,qz]} -> the protocol tools' calib file
    (``obs calib_out=true`` layout: {"calib": {cam: {K, T_world_cam}}, "eef": {arm: {"flange": {pos, quat}}}})."""
    d = {"calib": {c: {"K": np.asarray(v["K"], float).tolist(), "T_world_cam": np.asarray(v["T_world_cam"], float).tolist()}
                   for c, v in calib.items()}, "eef": {}}
    for arm, f in (flange or {}).items():
        f = [float(x) for x in f]
        d["eef"][arm] = {"flange": {"pos": f[:3], "quat": f[3:7]}}
    path.write_text(json.dumps(d))


def _geom():
    sys.path.insert(0, str(RD_TOOLS))
    try:
        import geom  # noqa: F401  (vendored, byte-identical)
    finally:
        sys.path.pop(0)
    return sys.modules["geom"]


def _vec(s: str):
    v = json.loads(s)
    return [float(x) for x in v]


def _png_state(ws: Path) -> dict:
    return {p.name: p.stat().st_mtime_ns for p in ws.glob("*.png")}


def _fit(png: bytes) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    if max(im.size) <= MAX_IMG_SIDE:
        return png
    s = MAX_IMG_SIDE / max(im.size)
    im = im.convert("RGB").resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


class RoboDojoTools:
    """Executor for one perceive call. ``ws`` holds cam_*.png, calib0.json (+ b1.json)."""

    def __init__(self, ws: Path, calib_file: str = "calib0.json", timeout_s: float = 30.0):
        self.ws = Path(ws)
        self.calib_file = calib_file
        self.timeout_s = timeout_s
        self.protected = {f"{c}.png" for c in CAMS}
        self.n = 0

    # ---------------------------------------------------------------- dispatch
    def run(self, cmd: str) -> tuple[str, list[tuple[str, bytes]]]:
        """-> (text for the model, [(image name, png bytes)])."""
        self.n += 1
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            return f"error: cannot parse the command ({e})", []
        while argv and argv[0] in ("python", "python3"):
            argv = argv[1:]
        if len(argv) >= 3 and argv[0].endswith("bin/robot"):  # "bin/robot <id> locate ..." -> "locate ..."
            argv = argv[2:]
        if not argv:
            return "error: empty command", []
        prog = argv[0].rsplit("/", 1)[-1]
        if prog in ("locate", "project"):
            try:
                return getattr(self, "_" + prog)(argv[1:])
            except Exception as e:  # report to the model, never crash the call
                return f"error: {prog}: {type(e).__name__}: {e}", []
        if prog not in SCRIPTS:
            return (f"error: '{prog}' is not an available command. Available: {', '.join(SCRIPTS)}, locate, project", [])
        for a in argv[1:]:
            if a.startswith("/") or ".." in a.split("/"):
                return f"error: argument {a!r}: only workspace file names (no paths)", []
        out_idx = {"zoom.py": 5, "mosaic.py": 2}.get(prog)  # never let a tool overwrite the observation's frames
        if out_idx is not None and len(argv) > out_idx and argv[out_idx].rsplit("/", 1)[-1] in self.protected:
            return f"error: {argv[out_idx]} is an input frame; name the output differently", []
        before = _png_state(self.ws)
        try:
            p = subprocess.run([sys.executable, str(RD_TOOLS / prog)] + [a.rsplit("/", 1)[-1] if a.endswith(".json") or a.endswith(".png") else a
                                                                       for a in argv[1:]],
                               cwd=self.ws, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return f"error: {prog} timed out after {self.timeout_s:.0f} s", []
        out = (p.stdout or "").strip()
        if p.returncode != 0:
            err = (p.stderr or "").strip().splitlines()[-3:]
            out = (out + "\n" if out else "") + "error (exit %d): %s" % (p.returncode, " | ".join(err))
        imgs = []
        after = _png_state(self.ws)
        for name, mt in sorted(after.items()):
            if name in self.protected:
                continue
            if before.get(name) != mt:
                imgs.append((name, _fit((self.ws / name).read_bytes())))
        return out[:4000] or "(no output)", imgs

    # ---------------------------------------------------------------- built-ins
    def _kv(self, args: list[str]) -> dict:
        kv = {}
        for a in args:
            if "=" not in a:
                raise ValueError(f"expected key=value, got {a!r}")
            k, v = a.split("=", 1)
            kv[k.strip()] = v.strip()
        return kv

    def _calib(self, cam: str):
        g = _geom()
        d = g.load(str(self.ws / self.calib_file))
        if cam not in d["calib"]:
            raise ValueError(f"no calibration for {cam!r} in this call (have {sorted(d['calib'])})")
        return g.calib(d, cam)

    def _locate(self, args):
        g = _geom()
        kv = self._kv(args)
        cam, px = kv.get("cam", "cam_head"), _vec(kv["px"])
        K, T = self._calib(cam)
        if "cam2" in kv and "px2" in kv:
            K2, T2 = self._calib(kv["cam2"])
            p, gap = g.tri(K, T, px, K2, T2, _vec(kv["px2"]))
            return json.dumps({"xyz": [round(float(v), 4) for v in p], "ray_gap_mm": round(gap * 1000, 1),
                               "how": f"triangulated {cam} + {kv['cam2']}"}), []
        z = float(kv.get("z", TABLE_Z))
        p = g.plane(K, T, px, z)
        return json.dumps({"xyz": [round(float(v), 4) for v in p], "how": f"{cam} ray at z={z:.4f}"}), []

    def _project(self, args):
        g = _geom()
        kv = self._kv(args)
        pts = json.loads(kv["pts"])
        pts = [[float(v) for v in p] for p in pts]
        d = g.load(str(self.ws / self.calib_file))
        out, imgs = {}, []
        for cam in d["calib"]:
            K, T = g.calib(d, cam)
            px = g.project(K, T, pts)
            out[cam] = [None if q is None or not (0 <= q[0] < W and 0 <= q[1] < H) else [round(q[0], 1), round(q[1], 1)]
                        for q in px]
            if kv.get("img") in ("1", "true", "yes") and any(out[cam]) and (self.ws / f"{cam}.png").exists():
                from PIL import Image, ImageDraw

                im = Image.open(self.ws / f"{cam}.png").convert("RGB")
                dr = ImageDraw.Draw(im)
                for i, q in enumerate(out[cam]):
                    if q is None:
                        continue
                    dr.line([(q[0] - 7, q[1]), (q[0] + 7, q[1])], fill=(255, 0, 0), width=2)
                    dr.line([(q[0], q[1] - 7), (q[0], q[1] + 7)], fill=(255, 0, 0), width=2)
                    dr.text((q[0] + 5, q[1] + 3), str(i), fill=(255, 255, 0))
                b = io.BytesIO()
                im.save(b, format="PNG")
                imgs.append((f"project_{cam}.png", b.getvalue()))
        return json.dumps(out), imgs


def project_points(calib: dict, pts) -> dict:
    """{cam: [[u,v] | None]} for world points (hint tables)."""
    g = _geom()
    out = {}
    for cam, c in calib.items():
        K, T = np.asarray(c["K"], float), np.asarray(c["T_world_cam"], float)
        px = g.project(K, T, pts)
        out[cam] = [None if q is None or not (0 <= q[0] < W and 0 <= q[1] < H) else [round(q[0]), round(q[1])] for q in px]
    return out

