"""System prompts and the freeze pin: preamble + the vendored protocol, every file md5-checked at load."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
PROMPTS = HERE / "prompts"
PREAMBLES = {"robodojo": "runner_robodojo_v1.md", "held": "runner_held_v1.md", "libero": "runner_libero_v1.md"}


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def verify_vendor() -> dict:
    """{relative path: md5} of the vendored robot-harness files; raises if any differs from MANIFEST.md5."""
    out = {}
    for line in (VENDOR / "MANIFEST.md5").read_text().splitlines():
        h, rel = line.split(maxsplit=1)
        rel = rel.strip().lstrip("./")
        got = md5(VENDOR / rel)
        if got != h:
            raise RuntimeError(f"vendored {rel}: md5 {got} != pinned {h} (re-vendor and re-pin, never edit)")
        out[rel] = h
    return out


@lru_cache(maxsize=None)
def system_prompt(kind: str) -> str:
    verify_vendor()
    pre = (PROMPTS / PREAMBLES[kind]).read_text()
    proto = (VENDOR / "PERCEIVER-PROTOCOL.md").read_text()
    return pre.rstrip() + "\n\n---\n\n" + proto


def pin(kind: str, model: str) -> dict:
    """What a freeze records: model id, preamble and protocol md5, the system prompt's md5, every vendored tool."""
    s = system_prompt(kind)
    return {"kind": kind, "model": model, "preamble": PREAMBLES[kind], "preamble_md5": md5(PROMPTS / PREAMBLES[kind]),
            "protocol_md5": md5(VENDOR / "PERCEIVER-PROTOCOL.md"), "system_md5": hashlib.md5(s.encode()).hexdigest(),
            "vendor_source": (VENDOR / "SOURCE.txt").read_text().strip(), "vendor": verify_vendor()}
