"""Observation -> image parts + a short legend for one user turn.

Why overlays exist at all: in the published LLM-as-policy transcripts
(Robocurve inspect-robots, research/sources/robocurve-gpt6-astra.md §5, §10)
every trial spent calls re-deriving which image direction is +x, and grasp
misalignment from parallax was the dominant failure. Projecting the base-frame
grid, the TCP and the gripper footprint into the image with the calibrated
camera removes that guesswork. Each overlay is a config switch so its value
can be measured.

Pipeline per camera (``ObservationConfig.cameras`` order):
  resize to the long edge (LANCZOS, never upscale) -> overlays in configured
  order (``grid``, ``axes``, ``ee_marker``; ``raw`` = none) -> main image;
  ``diff`` / ``heatmap`` add one derived image each (needs ``prev``); then
  optionally everything is tiled into one labelled image (``tile``).

Everything is deterministic for identical input (fixed colours, fixed font,
no randomness): identical frames must encode to identical JPEG bytes or the
prompt cache breaks.

The STATE line is NOT produced here — it is part of the feedback text
(``controlr.protocol.feedback.format_state``); ``RenderedObservation.text``
only describes the images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from controlr.config import ActionConfig, ObservationConfig
from controlr.protocol.feedback import rotvec_to_matrix
from controlr.protocol.grammar import fmt_num, pos_decimals, pos_factor
from controlr.types import CameraInfo, Observation, RobotSpec

OVERLAYS = ("grid", "axes", "ee_marker")
DERIVED = ("diff", "heatmap")
KNOWN = ("raw",) + OVERLAYS + DERIVED + ("tile",)

_NEAR = 0.02          # m, near clipping plane for projected segments
_AXIS_LEN = 0.05      # m, length of the base-axis arrows drawn at the TCP
_DIFF_GAIN = 4.0      # |I_t - I_{t-1}| amplification
_COLORS = {
    "grid": (255, 255, 255, 110),
    "grid_label": (255, 255, 255, 255),
    "label_bg": (0, 0, 0, 150),
    "x": (255, 60, 60, 255),
    "y": (60, 230, 60, 255),
    "z": (80, 140, 255, 255),
    "tcp": (255, 0, 255, 255),
    "finger": (255, 230, 0, 255),
    "drop": (255, 0, 255, 160),
}

Encoder = Callable[[np.ndarray, int, str], Any]   # (rgb, quality, label) -> ImagePart


def resolve_grid_z(cfg: ObservationConfig, spec: RobotSpec | None) -> float:
    """Grid plane height: the configured value, else the robot's table top, else 0.
    Defaulting to the table keeps the grid (and the TCP drop line) on the surface
    the model sees, whichever backend runs."""
    if cfg.grid_z is not None:
        return float(cfg.grid_z)
    if spec is not None and spec.table_z is not None:
        return float(spec.table_z)
    return 0.0


# Anthropic looks back at most 20 content blocks from a cache breakpoint for an earlier
# cache entry; one turn adds 1 assistant block + feedback text + images + legend text.
MAX_BLOCKS_PER_TURN = 18


def blocks_per_turn(cfg: ObservationConfig) -> int:
    """Upper bound of content blocks one turn adds to the transcript."""
    if cfg.tile or "tile" in cfg.renderers:
        n_img = 1
    else:
        n_img = len(cfg.cameras) * (1 + sum(r in DERIVED for r in cfg.renderers))
    return 1 + 1 + n_img + 1        # assistant reply, feedback text, images, legend


def lookback_warning(cfg: ObservationConfig) -> str | None:
    n = blocks_per_turn(cfg)
    if n > MAX_BLOCKS_PER_TURN:
        return (f"{n} content blocks per turn exceed the ~{MAX_BLOCKS_PER_TURN} that fit Anthropic's "
                f"20-block cache lookback: every turn would miss the previous cache entry; set "
                f"observation.tile=true")
    return None


# Tool-frame axis the gripper fingers move along (0 = x). See ee_marker.
JAW_AXIS = 0


@dataclass
class RenderedObservation:
    images: list            # list[controlr.llm.transcript.ImagePart]
    text: str


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:     # Pillow < 10.1
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _scaled_K(info: CameraInfo, img_w: int, img_h: int) -> np.ndarray:
    K = np.asarray(info.K, dtype=float).copy()
    sx, sy = img_w / info.width, img_h / info.height
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def to_camera(info: CameraInfo, pts_base: np.ndarray) -> np.ndarray:
    """(N,3) base-frame points -> (N,3) OpenCV camera-frame points."""
    P = np.atleast_2d(np.asarray(pts_base, dtype=float))
    T = np.asarray(info.T_cam_base, dtype=float)
    return P @ T[:3, :3].T + T[:3, 3]


def project_points(info: CameraInfo, pts_base: np.ndarray, img_w: int | None = None,
                   img_h: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points to pixel coords of an image of size
    (img_w, img_h) (defaults: the camera's native size; intrinsics are scaled
    for resized images). Returns (uv (N,2), valid (N,)); points at or behind
    the near plane are invalid (their uv is NaN)."""
    w = info.width if img_w is None else img_w
    h = info.height if img_h is None else img_h
    pc = to_camera(info, pts_base)
    valid = pc[:, 2] > _NEAR
    K = _scaled_K(info, w, h)
    uv = np.full((len(pc), 2), np.nan)
    z = pc[valid, 2:3]
    if len(z):
        hom = pc[valid] @ K.T
        uv[valid] = hom[:, :2] / z
    return uv, valid


def _project_segment(info: CameraInfo, p0: np.ndarray, p1: np.ndarray, w: int, h: int
                     ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Project a 3D segment, clipping the part behind the camera."""
    c = to_camera(info, np.stack([p0, p1]))
    z0, z1 = c[0, 2], c[1, 2]
    if z0 <= _NEAR and z1 <= _NEAR:
        return None
    if z0 <= _NEAR or z1 <= _NEAR:
        t = (_NEAR + 1e-6 - z0) / (z1 - z0)
        cut = c[0] + t * (c[1] - c[0])
        if z0 <= _NEAR:
            c[0] = cut
        else:
            c[1] = cut
    K = _scaled_K(info, w, h)
    hom = c @ K.T
    uv = hom[:, :2] / hom[:, 2:3]
    if not np.all(np.isfinite(uv)) or np.any(np.abs(uv) > 1e5):
        return None
    return (float(uv[0, 0]), float(uv[0, 1])), (float(uv[1, 0]), float(uv[1, 1]))


def describe_axes(info: CameraInfo, origin: np.ndarray | None = None, long_edge: int | None = None,
                  per: str = "100 mm") -> str:
    """One sentence: how the base +x/+y/+z axes appear in this camera, with the
    projected length of a 100 mm step, e.g. "+x points right (98 px per 100 mm),
    +y points up (85 px per 100 mm), +z points up (39 px per 100 mm) in the
    448-px image".

    WHY the lengths: on a camera tilted ~25 deg from vertical, +y (away from the
    camera) and +z (up) both go "up" the image and differ only in magnitude —
    direction words alone cannot separate depth from height, the dominant failure
    of the live runs. Generated from the calibration so the manual cannot
    contradict the images. ``long_edge``: the size the image is sent at (pixel
    lengths are scaled to it; default native). ``per``: label of the 0.1 m step
    in the LLM unit."""
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    scale = (long_edge / max(info.width, info.height)) if long_edge else 1.0
    parts = []
    for name, d in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        seg = _project_segment(info, o, o + 0.1 * np.asarray(d, dtype=float), info.width, info.height)
        if seg is None:
            parts.append(f"+{name} is not visible")
            continue
        du, dv = seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]
        n = math.hypot(du, dv)
        if n < 2.0:   # axis along the viewing ray: report depth direction instead
            depth = to_camera(info, np.stack([o, o + 0.1 * np.asarray(d, float)]))[:, 2]
            parts.append(f"+{name} points {'away from' if depth[1] > depth[0] else 'toward'} the camera")
            continue
        parts.append(f"+{name} points {_direction_word(du / n, dv / n)} ({n * scale:.0f} px per {per})")
    size = f"{long_edge}-px" if long_edge else f"{max(info.width, info.height)}-px"
    return ", ".join(parts) + f" in the {size} image"


def axis_directions(info: CameraInfo, origin: np.ndarray | None = None) -> dict[str, str | None]:
    """Direction word per base axis (None when along the viewing ray / invisible)."""
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    out: dict[str, str | None] = {}
    for name, d in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        seg = _project_segment(info, o, o + 0.1 * np.asarray(d, dtype=float), info.width, info.height)
        if seg is None or math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) < 2.0:
            out[name] = None
        else:
            out[name] = _direction_word(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])
    return out


def _direction_word(du: float, dv: float) -> str:
    """Eight-way word for an image direction (any length; v grows downward)."""
    ang = math.degrees(math.atan2(-dv, du)) % 360   # 0 = right, 90 = up (image v grows downward)
    words = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return words[int(((ang + 22.5) % 360) // 45)]


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def resize_long_edge(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    """LANCZOS downscale so max(h, w) == long_edge; never upscale."""
    h, w = rgb.shape[:2]
    s = long_edge / max(h, w)
    if s >= 1.0:
        return np.ascontiguousarray(rgb[..., :3]).astype(np.uint8, copy=True)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).resize(
        size, Image.Resampling.LANCZOS))


def _label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill,
           w: int, h: int) -> None:
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x = min(max(x, 1), w - tw - 3)
    y = min(max(y, 1), h - th - 3)
    draw.rectangle((x - 1, y - 1, x + tw + 2, y + th + 2), fill=_COLORS["label_bg"])
    draw.text((x - l, y - t), text, font=font, fill=fill)


def _arrow(draw: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float], fill, width: int = 2):
    draw.line([a, b], fill=fill, width=width)
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 4:
        return
    ux, uy = dx / n, dy / n
    head = min(8.0, n * 0.4)
    left = (b[0] - head * ux + head * 0.5 * uy, b[1] - head * uy - head * 0.5 * ux)
    right = (b[0] - head * ux - head * 0.5 * uy, b[1] - head * uy + head * 0.5 * ux)
    draw.polygon([b, left, right], fill=fill)


def _heat_color(m: np.ndarray) -> np.ndarray:
    """Jet-like colormap for m in [0,1] -> float RGB in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * m - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * m - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * m - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

class ObservationRenderer:
    """``render(obs, prev, turn)`` -> RenderedObservation (encoded ImageParts).

    ``action_cfg`` (units for labels) and ``spec`` (workspace extent for the
    grid) are optional contract additions; defaults: mm/deg and a 1 m square
    around the base. ``encoder`` defaults to
    ``controlr.llm.transcript.encode_image`` (imported lazily)."""

    def __init__(self, cfg: ObservationConfig, action_cfg: ActionConfig | None = None,
                 spec: RobotSpec | None = None, encoder: Encoder | None = None):
        unknown = [r for r in cfg.renderers if r not in KNOWN]
        if unknown:
            raise ValueError(f"unknown renderers {unknown}; known: {list(KNOWN)}")
        self.cfg = cfg
        self.acfg = action_cfg or ActionConfig()
        self.spec = spec
        self._encoder = encoder
        self.grid_z = resolve_grid_z(cfg, spec)
        self.tile = cfg.tile or "tile" in cfg.renderers
        self.lookback_warning = lookback_warning(cfg)

    # -- public -------------------------------------------------------------

    def render(self, obs: Observation, prev: Observation | None, turn: int) -> RenderedObservation:
        arrays = self.render_arrays(obs, prev, turn)
        enc = self._get_encoder()
        images = [enc(rgb, self.cfg.jpeg_quality, label) for label, rgb in arrays]
        return RenderedObservation(images=images, text=self.legend(arrays))

    def render_arrays(self, obs: Observation, prev: Observation | None, turn: int
                      ) -> list[tuple[str, np.ndarray]]:
        """The pixels that will be sent, as (label, HxWx3 uint8) — split out
        from ``render`` so tests and benchmarks need no encoder."""
        size = self.cfg.first_turn_size if (turn == 0 and self.cfg.first_turn_size) else self.cfg.size
        out: list[tuple[str, np.ndarray]] = []
        for cam in self._cameras(obs):
            base = resize_long_edge(obs.images[cam], size)
            info = obs.cameras.get(cam)
            main = self._overlay(base, info, obs) if info is not None else base.copy()
            out.append((cam, main))
            prev_img = None
            if prev is not None and cam in prev.images:
                p = resize_long_edge(prev.images[cam], size)
                if p.shape == base.shape:
                    prev_img = p
            for r in self.cfg.renderers:
                if r == "diff" and prev_img is not None:
                    out.append((f"{cam} diff", self._diff(base, prev_img)))
                elif r == "heatmap" and prev_img is not None:
                    out.append((f"{cam} heatmap", self._heatmap(base, prev_img)))
        if self.tile and len(out) > 1:
            return [("tile", self._tile(out))]
        return out

    def legend(self, arrays: list[tuple[str, np.ndarray]]) -> str:
        """Short, stable description of the images of this turn."""
        ov = [r for r in self.cfg.renderers if r in OVERLAYS]
        ov_txt = {
            "grid": (f"base-frame grid every {self._len_text(self.cfg.grid_step)} "
                     f"on the plane z={self._len_text(self.grid_z)}"),
            "axes": "base +x/+y/+z arrows (red/green/blue) at the TCP",
            "ee_marker": "TCP (magenta) + finger tips (yellow) + drop line to the table",
        }
        parts = []
        for label, rgb in arrays:
            h, w = rgb.shape[:2]
            if label == "tile":
                parts.append(f"tile {w}x{h} (labelled panels)")
            elif label.endswith(" diff"):
                parts.append(f"{label} {w}x{h} (amplified change since the previous image)")
            elif label.endswith(" heatmap"):
                parts.append(f"{label} {w}x{h} (change since the previous image, hot = moved)")
            else:
                extra = f"; overlays: {', '.join(ov_txt[o] for o in ov)}" if ov else ""
                parts.append(f"{label} {w}x{h}{extra}")
        return "IMAGES: " + " | ".join(parts)

    # -- internals ----------------------------------------------------------

    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            from controlr.llm.transcript import encode_image   # written by the llm owner
            self._encoder = encode_image
        return self._encoder

    def _cameras(self, obs: Observation) -> list[str]:
        cams = list(self.cfg.cameras) if self.cfg.cameras else sorted(obs.images)
        missing = [c for c in cams if c not in obs.images]
        if missing:
            raise KeyError(f"cameras {missing} not in observation (have {sorted(obs.images)})")
        return cams

    def _len_text(self, v_m: float) -> str:
        a = self.acfg
        return f"{fmt_num(v_m / pos_factor(a), pos_decimals(a))} {a.pos_unit}"

    def _num(self, v_m: float) -> str:
        return fmt_num(v_m / pos_factor(self.acfg), pos_decimals(self.acfg))

    def _overlay(self, rgb: np.ndarray, info: CameraInfo, obs: Observation) -> np.ndarray:
        ov = [r for r in self.cfg.renderers if r in OVERLAYS]
        if not ov:
            return rgb.copy()
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = _font(max(9, round(h / 36)))
        for r in ov:
            getattr(self, f"_draw_{r}")(draw, info, obs, w, h, font)
        return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))

    def _draw_grid(self, draw, info, obs, w, h, font) -> None:
        step = self.cfg.grid_step
        z = self.grid_z
        if self.spec is not None:
            lo, hi = self.spec.workspace_lo, self.spec.workspace_hi
        else:
            lo, hi = (-0.5, -0.5, 0.0), (0.5, 0.5, 0.5)
        xs = np.arange(math.ceil(lo[0] / step - 1e-9), math.floor(hi[0] / step + 1e-9) + 1) * step
        ys = np.arange(math.ceil(lo[1] / step - 1e-9), math.floor(hi[1] / step + 1e-9) + 1) * step
        if len(xs) == 0 or len(ys) == 0:
            return
        labels = []
        for i, x in enumerate(xs):
            seg = _project_segment(info, np.array([x, ys[0], z]), np.array([x, ys[-1], z]), w, h)
            if seg:
                draw.line(seg, fill=_COLORS["grid"], width=1)
                if round(x / step) % 2 == 0:
                    labels.append((seg[0], f"x={self._num(x)}"))
        for j, y in enumerate(ys):
            seg = _project_segment(info, np.array([xs[0], y, z]), np.array([xs[-1], y, z]), w, h)
            if seg:
                draw.line(seg, fill=_COLORS["grid"], width=1)
                if round(y / step) % 2 == 0:
                    labels.append((seg[1], f"y={self._num(y)}"))
        for (u, v), text in labels:
            if -20 <= u <= w + 20 and -20 <= v <= h + 20:
                _label(draw, (u + 2, v + 2), text, font, _COLORS["grid_label"], w, h)

    def _draw_axes(self, draw, info, obs, w, h, font) -> None:
        o = np.asarray(obs.state.tcp_pos, dtype=float).reshape(3)
        for name, d in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
            seg = _project_segment(info, o, o + _AXIS_LEN * np.asarray(d, float), w, h)
            if seg is None:
                continue
            _arrow(draw, seg[0], seg[1], _COLORS[name])
            _label(draw, (seg[1][0] + 3, seg[1][1] - 6), f"+{name}", font, _COLORS[name], w, h)

    def _draw_ee_marker(self, draw, info, obs, w, h, font) -> None:
        st = obs.state
        tcp = np.asarray(st.tcp_pos, dtype=float).reshape(3)
        # drop line: TCP straight down to the grid plane (height cue)
        floor = np.array([tcp[0], tcp[1], self.grid_z])
        seg = _project_segment(info, tcp, floor, w, h)
        if seg:
            draw.line(seg, fill=_COLORS["drop"], width=1)
            fu, fv = seg[1]
            draw.line([(fu - 4, fv - 4), (fu + 4, fv + 4)], fill=_COLORS["drop"], width=1)
            draw.line([(fu - 4, fv + 4), (fu + 4, fv - 4)], fill=_COLORS["drop"], width=1)
        # gripper footprint: finger tips along the jaw axis, +-width/2. The pads
        # close along tool x (PHANTOM scene.py: prismatic pads on gripper_housing x,
        # gripper.yaw = 0; the Isaac scripted grasp aligns tool x with the packet's
        # thin axis) — verified against the rendered fingers in the Isaac frames.
        R = rotvec_to_matrix(st.tcp_rotvec)
        half = max(st.gripper_mm, 0.0) * 1e-3 / 2
        jaw = R[:, JAW_AXIS]
        seg = _project_segment(info, tcp - half * jaw, tcp + half * jaw, w, h)
        if seg:
            draw.line(seg, fill=_COLORS["finger"], width=2)
            for (u, v) in seg:
                draw.rectangle((u - 3, v - 3, u + 3, v + 3), outline=_COLORS["finger"], width=2)
        uv, ok = project_points(info, tcp[None], w, h)
        if ok[0]:
            u, v = uv[0]
            draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=_COLORS["tcp"], width=2)
            draw.line([(u - 9, v), (u + 9, v)], fill=_COLORS["tcp"], width=1)
            draw.line([(u, v - 9), (u, v + 9)], fill=_COLORS["tcp"], width=1)
            _label(draw, (u - 80, v - 24), f"TCP z={self._num(tcp[2])}", font, _COLORS["tcp"], w, h)

    @staticmethod
    def _diff(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
        d = np.abs(cur.astype(np.int16) - prev.astype(np.int16)).max(axis=2).astype(np.float32)
        g = np.clip(d * _DIFF_GAIN, 0, 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)

    @staticmethod
    def _heatmap(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
        d = np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean(axis=2).astype(np.uint8)
        blurred = np.asarray(Image.fromarray(d).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
        m = np.clip(blurred * _DIFF_GAIN / 255.0, 0.0, 1.0)     # fixed scale: no change -> no heat
        color = _heat_color(m) * 255.0
        alpha = (0.65 * m)[..., None]
        base = cur.astype(np.float32) * 0.8                     # dim the frame a little
        return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)

    def _tile(self, items: list[tuple[str, np.ndarray]]) -> np.ndarray:
        n = len(items)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        cw = max(a.shape[1] for _, a in items)
        ch = max(a.shape[0] for _, a in items)
        strip = max(12, round(ch / 22))
        font = _font(max(9, strip - 3))
        canvas = Image.new("RGB", (cols * cw, rows * (ch + strip)), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for k, (label, a) in enumerate(items):
            r, c = divmod(k, cols)
            x0, y0 = c * cw, r * (ch + strip)
            draw.text((x0 + 3, y0 + 1), label, font=font, fill=(255, 255, 255))
            canvas.paste(Image.fromarray(a), (x0, y0 + strip))
        return np.asarray(canvas)
