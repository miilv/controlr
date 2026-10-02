"""Cache probe: does a growing image transcript get cached the way we think?

Replays (or synthesises) frames into an append-only transcript exactly like
the turn loop does — system prompt, then per turn one user message with an
image and one short assistant reply — and records per-turn latency and how
much of the prompt was served from cache. Expected on anthropic routes: from
turn 1 on, cache_read ≈ everything except the newest turn.

Also hosts the small frame helpers shared with ``bench.latency`` (loading a
frame directory, synthetic frames, resizing) so both benchmarks feed the
model identical pixels.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from controlr.runlog import to_jsonable, utc_stamp

_IMG_EXT = (".jpg", ".jpeg", ".png")

# Stand-in operating manual for the probe. Long enough (> 4096 tokens with a
# few images; > 1024 on its own) to clear every Claude route's minimum
# cacheable prefix, deterministic so repeated probes hit the same cache.
PROBE_SYSTEM = (
    "You are the low-level controller of a UR3 robot arm with a parallel gripper. "
    "Each user turn shows one camera frame of the workspace. Reply with exactly two lines:\n"
    "MOVE ee_delta <dx> <dy> <dz>   (millimetres, base frame)\n"
    "STATUS OK\n"
    "No other text. This is a latency/caching benchmark; any small plausible motion is fine.\n\n"
    + "\n".join(f"Rule {i}: keep the gripper above the table, move at most 50 mm per line, "
                f"prefer small corrective motions, and never leave the workspace box." for i in range(60))
)
PROBE_REPLY = "MOVE ee_delta 5 0 0\nSTATUS OK"


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def load_frames(directory: str | Path, limit: int | None = None) -> list[np.ndarray]:
    """RGB uint8 frames from a directory of images (sorted by name), or from
    a run dir's ``images/`` subdirectory."""
    d = Path(directory)
    if (d / "images").is_dir():
        d = d / "images"
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no images in {d}")
    return [np.asarray(Image.open(p).convert("RGB")) for p in files]


def synthetic_frame(i: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Deterministic scene-like frame: table gradient + a block that moves
    with ``i`` + a gripper-ish bar. Distinct per turn (no accidental dedup)."""
    y, x = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), np.uint8)
    img[..., 0] = (90 + 60 * x / width).astype(np.uint8)
    img[..., 1] = (80 + 50 * y / height).astype(np.uint8)
    img[..., 2] = 70
    cx = int(width * 0.3 + (i * 17) % int(width * 0.4))
    cy = int(height * 0.6)
    img[cy - 25:cy + 25, cx - 25:cx + 25] = (200, 40, 40)
    gx = int(width * 0.5 + 60 * np.sin(i / 3))
    img[40:int(height * 0.45), gx - 6:gx + 6] = (40, 40, 40)
    return img


def resize_long_edge(rgb: np.ndarray, size: int) -> np.ndarray:
    """LANCZOS to ``size`` long edge, never upscale (same rule as the renderers)."""
    h, w = rgb.shape[:2]
    if max(h, w) <= size:
        return rgb
    s = size / max(h, w)
    im = Image.fromarray(rgb).resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(im)


def frame_source(frames_dir: str | Path | None) -> Callable[[int], np.ndarray]:
    frames = load_frames(frames_dir) if frames_dir else None
    if frames:
        return lambda i: frames[i % len(frames)]
    return synthetic_frame


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

@dataclass
class ProbeTurn:
    turn: int
    n_images: int
    ttft: float | None
    t_end: float | None
    prompt_tokens: int | None
    cache_read: int | None
    cache_write: int | None
    completion_tokens: int | None
    share: float | None
    error: str | None


def run_cache_probe(llm, model: str, *, turns: int = 20, size: int = 448,
                    frames_dir: str | Path | None = None, cache: str = "auto",
                    ttl: str = "5m", jpeg_quality: int = 90, max_tokens: int = 40,
                    extra_body: dict | None = None,
                    out_dir: str | Path | None = None,
                    printer: Callable[[str], None] | None = print) -> list[ProbeTurn]:
    """Grow a transcript for ``turns`` turns; one request per turn. Writes
    ``probe.csv`` to ``out_dir`` when given. Spend: exactly ``turns`` calls."""
    from controlr.llm.caching import apply_cache_markers, cache_style_for
    from controlr.llm.transcript import Transcript, encode_image

    src = frame_source(frames_dir)
    style = cache_style_for(model, cache)
    tr = Transcript(PROBE_SYSTEM)
    rows: list[ProbeTurn] = []
    if printer:
        printer(f"cache probe: model={model} style={style} size={size} turns={turns}")
        printer(f"{'turn':>4} {'imgs':>4} {'ttft':>6} {'end':>6} {'prompt':>7} {'read':>7} {'write':>6} {'share':>6}")
    for i in range(turns):
        img = encode_image(resize_long_edge(src(i), size), jpeg_quality, f"scene t{i}")
        tr.add_user([f"TURN {i}", img])
        msgs = apply_cache_markers(tr.to_messages(), style, ttl)
        res = llm.complete(model, msgs, max_tokens=max_tokens, extra_body=extra_body or None)
        u = res.usage
        pt = getattr(u, "prompt_tokens", None)
        cr = getattr(u, "cache_read_tokens", None)
        row = ProbeTurn(i, tr.n_images(), res.timings.ttft, res.timings.t_end, pt, cr,
                        getattr(u, "cache_write_tokens", None), getattr(u, "completion_tokens", None),
                        (cr / pt) if (pt and cr is not None) else None, res.error)
        rows.append(row)
        if printer:
            printer(_fmt_row(row))
        # Keep the transcript identical across runs: a fixed assistant reply
        # (the model's actual text varies, which is irrelevant to caching).
        tr.add_assistant(PROBE_REPLY if not res.error else "STATUS OK")
    if out_dir:
        write_csv(Path(out_dir) / "probe.csv", [to_jsonable(r) for r in rows])
    return rows


def _fmt(v, nd=2) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _fmt_row(r: ProbeTurn) -> str:
    s = (f"{r.turn:>4} {r.n_images:>4} {_fmt(r.ttft):>6} {_fmt(r.t_end):>6} {_fmt(r.prompt_tokens):>7} "
         f"{_fmt(r.cache_read):>7} {_fmt(r.cache_write):>6} {_fmt(r.share):>6}")
    return s + (f"  ERROR {r.error}" if r.error else "")


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def default_out_dir(root: str | Path, kind: str) -> Path:
    return Path(root) / f"{utc_stamp(time.time())}_{kind}"
