"""Run directory writer + log readers.

One episode = one directory ``<root>/<UTC timestamp>_<name>/``::

    config.yaml        the fully resolved Config (no secrets: only the key's env-var name)
    system_prompt.md   the exact system text of the control transcript
    plan.md            planner output (absent when the planner is disabled)
    images/<sha>.jpg   the exact JPEG bytes that were sent, content-addressed (dedup for free)
    messages.jsonl     the control transcript, one message per line, images as {"image_sha": ...}
    turns.jsonl        one record per turn (timings, usage, reply, actions, events, goal, state)
    summary.json       outcome, turns, token totals, cache-read share, latency p50/p90

WHY append + flush per turn: a crashed / Ctrl-C'd / OOM-killed episode must
still leave a readable log, and ``controlr report`` can rebuild the summary
from ``turns.jsonl`` alone (``summarize_turns``).

The stats helpers live here (not in loop.py) because both the loop and the
offline ``report`` command compute the same numbers from the same records.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / numpy / enums / paths into plain JSON types.

    Floats that are NaN/inf become None so every line stays strict JSON."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, allow_nan=False)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, tolerating a truncated last line (crash mid-write)."""
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def utc_stamp(t: float | None = None) -> str:
    dt = datetime.fromtimestamp(time.time() if t is None else t, tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return keep.strip("._") or "run"


class RunLog:
    """Writer for one run directory. Cheap to construct; all writes are
    immediate (no buffering beyond the OS) so the directory is always
    consistent up to the last completed turn."""

    def __init__(self, root: str | Path, name: str, *, save_images: bool = True,
                 stamp: str | None = None) -> None:
        root = Path(root)
        base = f"{stamp or utc_stamp()}_{_safe_name(name)}"
        path = root / base
        k = 1
        while path.exists():          # several episodes within the same second (mock/sweeps)
            path = root / f"{base}_{k}"
            k += 1
        (path / "images").mkdir(parents=True)
        self.path: Path = path
        self.keep_images = save_images     # NB: not ``save_images`` — that name is a method
        self._n_messages = 0
        self._saved: set[str] = set()

    # -- static files ------------------------------------------------------
    def write_config(self, cfg: Any) -> None:
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(to_jsonable(data), sort_keys=False, allow_unicode=True))

    def write_text(self, rel: str, text: str) -> Path:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def write_system_prompt(self, text: str) -> None:
        self.write_text("system_prompt.md", text)

    def write_plan(self, text: str) -> None:
        self.write_text("plan.md", text)

    def write_json(self, rel: str, obj: Any) -> Path:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return p

    def write_summary(self, summary: dict) -> None:
        self.write_json("summary.json", summary)

    # -- images ------------------------------------------------------------
    def save_image(self, part: Any) -> str:
        """Write ``images/<sha>.jpg`` with the exact bytes sent (an ImagePart:
        ``.jpeg`` + ``.sha``). Idempotent; returns the sha."""
        sha = part.sha
        if self.keep_images and sha not in self._saved:
            p = self.path / "images" / f"{sha}.jpg"
            if not p.exists():
                p.write_bytes(part.jpeg)
            self._saved.add(sha)
        return sha

    def save_images(self, parts: Iterable[Any]) -> list[str]:
        return [self.save_image(p) for p in parts]

    # -- append-only streams ------------------------------------------------
    def _append(self, rel: str, records: Iterable[Any]) -> None:
        with open(self.path / rel, "a", encoding="utf-8") as f:
            for r in records:
                f.write(_dumps(r) + "\n")
            f.flush()

    def sync_messages(self, log_records: list[dict]) -> None:
        """Append the not-yet-written tail of ``Transcript.to_log_records()``.
        The transcript is append-only, so the written prefix never changes."""
        new = log_records[self._n_messages:]
        if new:
            self._append("messages.jsonl", new)
            self._n_messages = len(log_records)

    def append_turn(self, record: dict) -> None:
        self._append("turns.jsonl", [record])

    def append_jsonl(self, rel: str, record: dict) -> None:
        self._append(rel, [record])


# ---------------------------------------------------------------------------
# stats (shared by loop summary and offline report)
# ---------------------------------------------------------------------------

def percentiles(values: Iterable[float | None], qs: tuple[int, ...] = (50, 90)) -> dict[str, float | None]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    out: dict[str, float | None] = {}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(v, q)) if v else None
    out["mean"] = float(np.mean(v)) if v else None
    out["max"] = float(np.max(v)) if v else None
    out["n"] = len(v)
    return out


TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "cache_read_tokens",
              "cache_write_tokens", "reasoning_tokens")


def token_totals(usages: Iterable[dict | None]) -> dict[str, int]:
    tot = {k: 0 for k in TOKEN_KEYS}
    tot["calls_with_usage"] = 0
    tot["calls_without_usage"] = 0
    for u in usages:
        if not u:
            tot["calls_without_usage"] += 1
            continue
        tot["calls_with_usage"] += 1
        for k in TOKEN_KEYS:
            tot[k] += int(u.get(k) or 0)
    return tot


def cache_read_share(totals: dict) -> float | None:
    """Fraction of prompt tokens served from cache. ``prompt_tokens`` is the
    total input (cached + uncached) as normalised by the client."""
    pt = totals.get("prompt_tokens") or 0
    return (totals.get("cache_read_tokens") or 0) / pt if pt else None


def summarize_turns(records: list[dict]) -> dict:
    """Latency + token aggregates from per-turn records (the turns.jsonl schema
    written by ``controlr.loop``)."""
    llm = [r.get("llm") or {} for r in records]
    tim = [r.get("timings") or {} for r in records]
    totals = token_totals(r.get("usage") for r in records)
    return {
        "totals": totals,
        "cache_read_share": cache_read_share(totals),
        "latency": {
            "llm_end_s": percentiles(x.get("t_end") for x in llm),
            "llm_ttft_s": percentiles(x.get("ttft") for x in llm),
            "llm_complete_s": percentiles(x.get("t_complete") for x in llm),
            "exec_s": percentiles(x.get("exec") for x in tim),
            "cycle_s": percentiles(x.get("cycle") for x in tim),
        },
    }


# ---------------------------------------------------------------------------
# readers (for `controlr report`)
# ---------------------------------------------------------------------------

def is_run_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and ((p / "summary.json").exists() or (p / "turns.jsonl").exists())


def find_run_dirs(path: str | Path) -> list[Path]:
    """A run dir itself, or every run dir below ``path`` (sweep/bench dirs)."""
    p = Path(path)
    if is_run_dir(p):
        return [p]
    if not p.is_dir():
        return []
    return sorted({q.parent for pat in ("summary.json", "turns.jsonl") for q in p.rglob(pat)})


def load_run(path: str | Path) -> dict:
    """summary.json if present, else a summary rebuilt from turns.jsonl
    (outcome "incomplete"). Always includes ``run_dir``."""
    p = Path(path)
    sp = p / "summary.json"
    if sp.exists():
        try:
            s = json.loads(sp.read_text())
            s.setdefault("run_dir", str(p))
            return s
        except json.JSONDecodeError:
            pass
    records = read_jsonl(p / "turns.jsonl")
    s = summarize_turns(records)
    s.update(outcome="incomplete", turns=len(records), success=None, run_dir=str(p))
    cfg_p = p / "config.yaml"
    if cfg_p.exists():
        cfg = yaml.safe_load(cfg_p.read_text()) or {}
        s["name"] = cfg.get("name")
        s["model"] = (cfg.get("llm") or {}).get("model")
    return s
