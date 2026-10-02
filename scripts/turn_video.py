"""Turn-by-turn video of a controlr run: per turn, the image the model saw, its reply,
the per-turn LLM latency / cache share, the STATE it was given and the feedback its
command produced; the final frame shows the end state and the goal check.

Standalone (Pillow + the ffmpeg binary; does not import controlr):
    uv run --no-project --with pillow python scripts/turn_video.py RUN_DIR OUT.mp4 "title"
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL_W = 560
WRAP = 62
BG = (18, 20, 24)


def _font(name: str, size: int):
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


MONO, MONO_B, HEAD = _font("DejaVuSansMono.ttf", 13), _font("DejaVuSansMono-Bold.ttf", 13), _font("DejaVuSans-Bold.ttf", 17)


def _sha(x) -> str:
    return x if isinstance(x, str) else (x.get("sha") or x.get("image_sha"))


def _state_line(text: str | None) -> str:
    for ln in reversed((text or "").splitlines()):
        if ln.startswith("STATE:"):
            return ln
    return ""


def _first_user_text(run: Path) -> str:
    for ln in (run / "messages.jsonl").read_text().splitlines():
        m = json.loads(ln)
        if m.get("role") == "user":
            c = m.get("content")
            parts = c if isinstance(c, list) else [c]
            return "\n".join(p if isinstance(p, str) else str(p.get("text", "")) for p in parts
                             if isinstance(p, str) or "text" in p)
    return ""


def _llm_line(t: dict) -> str:
    """Per-turn LLM latency: ``llm.t_end`` (request start -> stream end, incl. thinking) is the
    field that turns.jsonl records for every call; ``timings.llm_wall`` adds retries."""
    llm = t.get("llm") or {}
    end = llm.get("t_end") if llm.get("t_end") is not None else (t.get("timings") or {}).get("llm_wall")
    s = f"LLM {end:.1f} s" if isinstance(end, (int, float)) else "LLM ?"
    if isinstance(llm.get("ttft"), (int, float)):
        s += f" (first token {llm['ttft']:.1f} s)"
    u = t.get("usage") or {}
    if u.get("prompt_tokens"):
        s += f" | cache {100 * (u.get('cache_read_tokens') or 0) / u['prompt_tokens']:.0f}%"
    if u.get("reasoning_tokens"):
        s += f" | think {u['reasoning_tokens']} tok"
    return s


def _card(run: Path, sha: str, head: str, blocks: list[tuple[str, list[str], tuple]]) -> Image.Image:
    im = Image.open(run / "images" / f"{sha}.jpg").convert("RGB")
    H = max(im.height, 420)
    c = Image.new("RGB", (im.width + PANEL_W, H), BG)
    c.paste(im, (0, (H - im.height) // 2))
    d = ImageDraw.Draw(c)
    x, y = im.width + 12, 10
    d.text((x, y), head, font=HEAD, fill=(255, 210, 90))
    y += 26
    for title, lines, color in blocks:
        if title:
            d.text((x, y), title, font=MONO_B, fill=(150, 190, 255))
            y += 16
        for ln in lines:
            for w in textwrap.wrap(ln, WRAP, subsequent_indent="  ") or [""]:
                if y > H - 16:
                    return c
                d.text((x, y), w, font=MONO, fill=color)
                y += 16
        y += 6
    return c


def main() -> None:
    run, out, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    turns = [json.loads(ln) for ln in (run / "turns.jsonl").read_text().splitlines() if ln.strip()]
    setup = json.loads((run / "setup.json").read_text())
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    scene = setup.get("scene") or {}
    sub = []
    if scene.get("packet"):
        sub.append(f"packet yaw offset {scene['packet'].get('yaw_offset_deg', 0):+.0f} deg, "
                   f"start yaw offset {scene['start'].get('yaw_offset_deg', 0):+.0f} deg")
    white, grey, green, red = (230, 230, 230), (170, 170, 170), (120, 230, 140), (255, 120, 110)
    frames: list[Image.Image] = []
    prev_state = _state_line(_first_user_text(run))
    for t in turns:
        imgs = [_sha(i) for i in (t.get("obs_images") or [])]
        if not imgs:
            continue
        fb = [ln for ln in (t.get("feedback") or "").splitlines()[1:] if not ln.startswith("STATE:")]
        if not t.get("feedback"):
            g = t.get("goal") or {}
            fb = [f"GOAL: {'reached' if g.get('success') else 'not reached'} - {g.get('message', '')}"]
        blocks = [(title, sub, grey), ("saw", [prev_state], grey),
                  ("model reply", (t.get("reply") or "(none)").strip().splitlines(), white),
                  ("", [_llm_line(t)], (255, 210, 90)),
                  ("result", fb, red if any(ln.startswith("STOP") for ln in fb) else green)]
        frames.append(_card(run, imgs[0], f"turn {t['turn']}", blocks))
        prev_state = _state_line(t.get("feedback")) or prev_state
    last = turns[-1]
    nxt = [_sha(i) for i in (last.get("next_obs_images") or [])]
    if nxt:
        g = last.get("goal") or {}
        end = [f"outcome: {summary.get('outcome', '?')} after {len(turns)} turns",
               f"goal: {'SUCCESS' if g.get('success') else 'not reached'} - {g.get('message', '')}"]
        frames += [_card(run, nxt[0], "end", [(title, sub, grey), ("", end, green if g.get("success") else red)])] * 2
    tmp = Path(tempfile.mkdtemp())
    W = max(f.width for f in frames)
    H = max(f.height for f in frames)
    for i, f in enumerate(frames):
        c = Image.new("RGB", (W + W % 2, H + H % 2), BG)
        c.paste(f, (0, 0))
        c.save(tmp / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                    "-vf", "fps=10", "-c:v", "libx264", "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p",
                    str(out)], check=True)
    print(out, len(frames), "frames", f"{out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
