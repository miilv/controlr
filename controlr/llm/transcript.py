"""Append-only control transcript.

WHY append-only + images stored as encoded bytes: prompt caching only pays off
if every request's prefix is byte-identical to the previous request's. So
(1) nothing ever rewrites or drops an earlier message, (2) each frame is
JPEG-encoded exactly once and the same bytes (and the same base64 data URL
string) are re-sent on every later turn, and (3) serialisation is a pure
function of the stored records. Cache markers are NOT part of the transcript;
``controlr.llm.caching`` adds them on a copy at request time.

Assistant replies are stored exactly as received, including early-stopped
truncation — the model must see what it actually "said". The single
exception is the serialised form of an EMPTY reply (failed/aborted call):
Anthropic rejects empty assistant content, so ``to_messages`` emits a fixed
placeholder. It is deterministic, so prefix stability is unaffected.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

EMPTY_ASSISTANT_PLACEHOLDER = "(no reply)"


@dataclass(frozen=True)
class ImagePart:
    """One encoded image. ``sha`` = sha256 hex of ``jpeg`` (content address
    for ``runs/.../images/<sha>.jpg``). ``data_url`` is computed once at
    construction so every serialisation reuses the identical string."""
    jpeg: bytes
    sha: str
    width: int
    height: int
    label: str = ""
    data_url: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "data_url",
            "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii"))

    @classmethod
    def from_jpeg(cls, jpeg: bytes, label: str = "") -> "ImagePart":
        """Wrap existing JPEG bytes (e.g. replayed frames) without re-encoding."""
        with Image.open(io.BytesIO(jpeg)) as im:
            w, h = im.size
        return cls(jpeg=jpeg, sha=hashlib.sha256(jpeg).hexdigest(), width=w, height=h, label=label)


def encode_image(rgb: np.ndarray, quality: int = 90, label: str = "") -> ImagePart:
    """HxWx3 uint8 RGB -> ImagePart. Deterministic for identical pixels and
    quality (fixed subsampling, no optimize pass, no metadata), so identical
    frames give identical bytes and shas."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 RGB image, got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(np.ascontiguousarray(arr))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    jpeg = buf.getvalue()
    h, w = arr.shape[:2]
    return ImagePart(jpeg=jpeg, sha=hashlib.sha256(jpeg).hexdigest(), width=w, height=h, label=label)


@dataclass(frozen=True)
class _Msg:
    role: str
    parts: tuple   # tuple[str | ImagePart, ...] for user; (str,) for assistant/system


class Transcript:
    """System message + alternating user/assistant turns. The only mutators
    are ``add_user`` / ``add_assistant``, and they only append."""

    def __init__(self, system_text: str) -> None:
        self.system_text = system_text
        self._msgs: list[_Msg] = []

    # -- append --------------------------------------------------------------
    def add_user(self, parts: list[str | ImagePart]) -> None:
        clean: list[str | ImagePart] = []
        for p in parts:
            if isinstance(p, ImagePart):
                clean.append(p)
            elif isinstance(p, str):
                if p:                       # empty text blocks are rejected by Anthropic
                    clean.append(p)
            else:
                raise TypeError(f"user part must be str or ImagePart, got {type(p).__name__}")
        if not clean:
            raise ValueError("user turn has no content")
        self._msgs.append(_Msg("user", tuple(clean)))

    def add_assistant(self, text: str) -> None:
        self._msgs.append(_Msg("assistant", (text,)))

    # -- read ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of messages including the system message."""
        return 1 + len(self._msgs)

    def images(self) -> list[ImagePart]:
        return [p for m in self._msgs for p in m.parts if isinstance(p, ImagePart)]

    def n_images(self) -> int:
        return len(self.images())

    def to_messages(self) -> list[dict]:
        """OpenAI chat format, images as data URLs, no cache markers. Fresh
        dicts each call (callers may mutate them), identical content."""
        out: list[dict] = [{"role": "system", "content": self.system_text}]
        for m in self._msgs:
            if m.role == "assistant":
                out.append({"role": "assistant",
                            "content": m.parts[0] if m.parts[0].strip() else EMPTY_ASSISTANT_PLACEHOLDER})
            else:
                out.append({"role": "user", "content": [_part_to_api(p) for p in m.parts]})
        return out

    def to_log_records(self) -> list[dict]:
        """Same structure as ``to_messages`` but images replaced by their sha
        (bytes go to ``images/<sha>.jpg``); assistant text exactly as stored."""
        out: list[dict] = [{"role": "system", "content": self.system_text}]
        for m in self._msgs:
            if m.role == "assistant":
                out.append({"role": "assistant", "content": m.parts[0]})
            else:
                out.append({"role": "user", "content": [_part_to_log(p) for p in m.parts]})
        return out


def _part_to_api(p: str | ImagePart) -> dict:
    if isinstance(p, ImagePart):
        return {"type": "image_url", "image_url": {"url": p.data_url}}
    return {"type": "text", "text": p}


def _part_to_log(p: str | ImagePart) -> dict:
    if isinstance(p, ImagePart):
        return {"type": "image", "image_sha": p.sha, "label": p.label,
                "width": p.width, "height": p.height}
    return {"type": "text", "text": p}
