"""Episode video with the agent's chat log: RoboDojo's recording on top, the conversation
below. The video freezes while the model thinks (for the call's real wall time) and plays the
env steps its command produced, so the clip runs in real time.

    uv run --no-project --with pillow --with numpy python scripts/robodojo/chat_video.py \
        RUN_DIR HEAD.mp4 WRIST.mp4 OUT.mp4 "title"

RUN_DIR is a controlr run directory (turns.jsonl, messages.jsonl, summary.json); HEAD / WRIST are
RoboDojo's per-camera mp4s of the same episode (``robodojo_result/episode_*_cam_*.mp4``).
RoboDojo appends one frame per ``get_obs``: 2 at the start, then per turn one per env step, one
after an execute and one for the loop's observation — that is how frames are matched to turns
(off by ≤ 2 frames at the end). Needs ffmpeg.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 25
W, VH, PH = 960, 360, 540                     # width, video height, chat panel height
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
COL = {"user": (150, 200, 255), "assistant": (255, 225, 140), "think": (170, 170, 170), "sys": (140, 255, 160)}


def frames(path: str, w: int, h: int) -> list[np.ndarray]:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3))


def user_text(content) -> str:
    if isinstance(content, str):
        return content
    parts, imgs = [], 0
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p["text"])
        elif isinstance(p, dict) and ("image_sha" in p or p.get("type") == "image_url"):
            imgs += 1
        elif isinstance(p, str):
            parts.append(p)
    text = "\n".join(parts).strip()
    return text + (f"\n[{imgs} camera images]" if imgs else "")


class Panel:
    def __init__(self, title: str):
        self.f = ImageFont.truetype(FONT, 15)
        self.fb = ImageFont.truetype(BOLD, 15)
        self.title = title
        self.cw = self.f.getlength("M")
        self.cols = int((W - 24) / self.cw)
        self._cache: dict = {}

    def render(self, msgs: list[tuple[str, str]], status: str) -> np.ndarray:
        key = (tuple(msgs[-8:]), status)
        if key in self._cache:
            return self._cache[key]
        img = Image.new("RGB", (W, PH), (18, 18, 22))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 26], fill=(40, 40, 50))
        d.text((10, 5), self.title, font=self.fb, fill=(255, 255, 255))
        d.text((W - 12 - self.f.getlength(status), 5), status, font=self.f, fill=(255, 200, 120))
        lines: list[tuple[str, str, bool]] = []          # (role, text, is_header)
        for role, text in msgs[-8:]:
            head = {"user": "HARNESS →", "assistant": "MODEL →", "think": "MODEL", "sys": ""}[role]
            if head:
                lines.append((role, head, True))
            for para in text.splitlines() or [""]:
                for ln in textwrap.wrap(para, self.cols) or [""]:
                    lines.append((role, ln, False))
            lines.append((role, "", False))
        lh = 19
        room = (PH - 34) // lh
        y = 32
        for role, ln, head in lines[-room:]:
            d.text((12, y), ln, font=self.fb if head else self.f, fill=COL[role])
            y += lh
        arr = np.asarray(img)
        self._cache[key] = arr
        return arr


def main(run_dir: str, head: str, wrist: str, out: str, title: str) -> None:
    run = Path(run_dir)
    turns = [json.loads(x) for x in (run / "turns.jsonl").read_text().splitlines()]
    msgs_log = [json.loads(x) for x in (run / "messages.jsonl").read_text().splitlines()]
    users = [user_text(m["content"]) for m in msgs_log if m["role"] == "user"]
    summary = json.loads((run / "summary.json").read_text())
    hv, wv = frames(head, W // 2, VH), frames(wrist, W // 2, VH)
    n = min(len(hv), len(wv))
    vid = [np.concatenate([hv[i], wv[i]], axis=1) for i in range(n)]
    panel = Panel(title)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{VH + PH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", out],
                           stdin=subprocess.PIPE)

    def emit(frame: np.ndarray, msgs, status: str, count: int = 1) -> None:
        both = np.concatenate([frame, panel.render(msgs, status)], axis=0).tobytes()
        for _ in range(count):
            enc.stdin.write(both)

    pos = min(2, n)
    chat: list[tuple[str, str]] = [("user", users[0] if users else "")]
    emit(vid[pos - 1], chat, "start", FPS)
    for k, t in enumerate(turns):
        u = t.get("usage") or {}
        think_s = float((t.get("timings") or {}).get("llm_wall") or (t.get("llm") or {}).get("t_end") or 0.0)
        rtok = u.get("reasoning_tokens") or 0
        # thinking: frozen frame, a live counter
        nthink = max(1, int(round(think_s * FPS)))
        for j in range(nthink):
            s = (j + 1) / FPS
            note = f"thinking… {s:4.1f} s" + (f" ({rtok} reasoning tokens, hidden)" if rtok else "")
            emit(vid[pos - 1], chat + [("think", note)], f"turn {k}  thinking {s:4.1f} s")
        reply = (t.get("reply") or t.get("error") or "").strip()
        chat.append(("assistant", reply + f"\n[{think_s:.1f} s{', ' + str(rtok) + ' reasoning tokens' if rtok else ''}]"))
        b = t.get("backend") or {}
        nmove = (b.get("env_steps") or 0) + (1 if b else 0) + 1
        used = b.get("steps_used")
        for j in range(nmove):
            if pos < n:
                pos += 1
            emit(vid[pos - 1], chat, f"turn {k}  executing" + (f"  steps {used}/200" if used is not None else ""))
        if k + 1 < len(users):
            chat.append(("user", users[k + 1]))
    while pos < n:                                        # the episode's last frames
        pos += 1
        emit(vid[pos - 1], chat, "end")
    verdict = "SUCCESS" if summary.get("success") else "FAIL"
    chat.append(("sys", f"episode over: {summary.get('outcome')} -> RoboDojo verdict {verdict}"))
    emit(vid[n - 1], chat, verdict, FPS * 3)
    enc.stdin.close()
    enc.wait()
    print(out, f"{len(turns)} turns")


if __name__ == "__main__":
    main(*sys.argv[1:6])
