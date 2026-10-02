"""Latency matrix: models x image sizes x history lengths x reasoning variants.

What we want to know is the *steady-state* per-turn latency of the control
loop, i.e. a request whose prefix (system + all past turns) is already cached
and whose newest turn is not. So per cell we:

  1. build a transcript with ``history - 1`` past turns (frames + a fixed reply),
  2. append turn ``history`` and make a warm-up call (writes the cache; logged as
     phase "warmup", not scored),
  3. append one more turn and measure, ``repeats`` times (each call adds one turn,
     exactly like the loop does) -> measured calls carry history+1 .. history+repeats
     images, all but the newest served from cache.

Calls per cell = 1 + repeats; ``plan_calls`` reports the total before spending.
Frames come from a recorded run / frame directory (the same frames a
ReplayRobot would serve) or deterministic synthetic frames.

Config (YAML)::

    models: [no-think/claude/claude-haiku-4-5-20251001, ...]
    sizes: [224, 448, 672]
    history: [1, 10, 30]
    variants:                       # reasoning variants; model_suffix appended to the id
      - {name: default}
      - {name: low, extra_body: {reasoning_effort: low}}
    repeats: 3
    max_tokens: 2000                # counts thinking; early stop at STATUS keeps output short
    jpeg_quality: 90
    frames: null                    # dir of frames or a run dir; null -> synthetic
    cache: auto
    out_root: runs
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from controlr.bench.cache_probe import (
    PROBE_REPLY,
    PROBE_SYSTEM,
    default_out_dir,
    frame_source,
    resize_long_edge,
    response_cache_header,
    write_csv,
)


@dataclass
class Variant:
    name: str = "default"
    extra_body: dict = field(default_factory=dict)
    model_suffix: str = ""


@dataclass
class LatencyBench:
    models: list[str]
    sizes: list[int] = field(default_factory=lambda: [448])
    history: list[int] = field(default_factory=lambda: [1])
    variants: list[Variant] = field(default_factory=lambda: [Variant()])
    repeats: int = 3
    max_tokens: int = 2000            # counts thinking tokens: 40 starved Sonnet/Opus (empty replies)
    jpeg_quality: int = 90
    frames: str | None = None
    cache: str = "auto"
    cache_ttl: str = "5m"
    out_root: str = "runs"

    def cells(self) -> list[tuple[str, int, int, Variant]]:
        return list(itertools.product(self.models, self.sizes, self.history, self.variants))

    def plan_calls(self) -> int:
        return len(self.cells()) * (1 + self.repeats)


def load_bench(path: str | Path) -> LatencyBench:
    data = yaml.safe_load(Path(path).read_text()) or {}
    variants = [Variant(**v) if isinstance(v, dict) else Variant(name=str(v))
                for v in data.pop("variants", [{}])]
    if data.get("frames"):
        f = Path(data["frames"])
        data["frames"] = str(f if f.is_absolute() else (Path(path).parent / f).resolve())
    return LatencyBench(variants=variants, **data)


def _call_row(res, **meta) -> dict:
    u = res.usage
    pt = getattr(u, "prompt_tokens", None)
    cr = getattr(u, "cache_read_tokens", None)
    return dict(meta, ttft=res.timings.ttft, ttft_any=getattr(res.timings, "ttft_any", None),
                t_complete=res.timings.t_complete,
                t_end=res.timings.t_end, prompt_tokens=pt, cache_read_tokens=cr,
                response_cache=response_cache_header(res),
                cache_write_tokens=getattr(u, "cache_write_tokens", None),
                completion_tokens=getattr(u, "completion_tokens", None),
                reasoning_tokens=getattr(u, "reasoning_tokens", None),
                cache_share=(cr / pt) if (pt and cr is not None) else None,
                stopped_early=res.stopped_early, error=res.error, http_status=res.http_status,
                reply=(res.text or "")[:200])


def run_latency(bench: LatencyBench, llm, *, out_dir: str | Path | None = None,
                printer: Callable[[str], None] | None = print) -> tuple[list[dict], Path]:
    """Run the matrix; write ``calls.csv`` (every call) + ``table.md`` (medians)."""
    from controlr.llm.caching import apply_cache_markers, cache_style_for
    from controlr.llm.transcript import Transcript, encode_image
    from controlr.protocol.grammar import is_complete

    out = Path(out_dir) if out_dir else default_out_dir(bench.out_root, "bench_latency")
    out.mkdir(parents=True, exist_ok=True)
    src = frame_source(bench.frames)
    rows: list[dict] = []
    if printer:
        printer(f"latency bench: {len(bench.cells())} cells, {bench.plan_calls()} calls -> {out}")

    for model, size, hist, var in bench.cells():
        model_id = model + var.model_suffix
        style = cache_style_for(model_id, bench.cache)
        enc: dict[int, object] = {}

        def frame(i: int):
            if i not in enc:
                enc[i] = encode_image(resize_long_edge(src(i), size), bench.jpeg_quality, f"scene t{i}")
            return enc[i]

        tr = Transcript(PROBE_SYSTEM)
        # per-run/cell nonce in the first user turn: byte-identical requests across runs
        # would be answered from omniroute's response cache (~0.4 s, no counters)
        nonce = f"RUN {out.name} {model_id} {size} {hist} {var.name}\n"
        for i in range(max(0, hist - 1)):
            tr.add_user([(nonce if i == 0 else "") + f"TURN {i}", frame(i)])
            tr.add_assistant(PROBE_REPLY)
        meta = dict(model=model_id, size=size, history=hist, variant=var.name)
        n = max(0, hist - 1)
        for k in range(1 + bench.repeats):
            tr.add_user([(nonce if n == 0 else "") + f"TURN {n}", frame(n)])
            msgs = apply_cache_markers(tr.to_messages(), style, bench.cache_ttl)
            res = llm.complete(model_id, msgs, max_tokens=bench.max_tokens,
                               extra_body=dict(var.extra_body) or None, stop_when=is_complete)
            row = _call_row(res, **meta, call=k, phase="warmup" if k == 0 else "measure",
                            n_images=tr.n_images())
            rows.append(row)
            tr.add_assistant(PROBE_REPLY)
            n += 1
            if printer:
                printer(f"{model_id:45s} {size:4d} h={hist:<3d} {var.name:8s} {row['phase']:7s} "
                        f"end={_f(row['t_end'])} ttft={_f(row['ttft'])} share={_f(row['cache_share'])}"
                        + (f" ERROR {row['error']}" if row["error"] else ""))
            if res.error and k == 0 and res.http_status and 400 <= res.http_status < 500:
                break   # e.g. route rejects images: do not burn the repeats

    write_csv(out / "calls.csv", rows)
    (out / "table.md").write_text(markdown_table(rows))
    (out / "bench.yaml").write_text(yaml.safe_dump(_bench_dict(bench), sort_keys=False))
    if printer:
        printer(markdown_table(rows))
    return rows, out


def _bench_dict(b: LatencyBench) -> dict:
    import dataclasses

    return dataclasses.asdict(b)


def _f(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def _med(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _replayed(r: dict) -> bool:
    return str(r.get("response_cache") or "").upper().startswith("HIT")


def markdown_table(rows: list[dict]) -> str:
    """Medians over measured (non-warmup, non-error, not replayed) calls per cell.
    "ttft" = first content byte (includes thinking on thinking routes), "first byte" =
    first content OR thinking byte, "status" = time until the STATUS line was complete
    (what the loop waits for with early stop); "end" includes the usage grace period.
    A cell whose median cache share is < 0.5 is flagged UNCACHED (e.g. prompt below the
    model's minimum cacheable prefix): it is not a steady-state number."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["size"], r["history"], r["variant"]), []).append(r)
    lines = ["| model | size | history | variant | n | ttft p50 s | first byte p50 s | status p50 s | end p50 s | "
             "prompt tok | cache share | notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, s, h, v), rs in cells.items():
        ok = [r for r in rs if r["phase"] == "measure" and not r["error"] and not _replayed(r)]
        notes = sorted({str(r["error"])[:40] for r in rs if r["error"]})
        n_rep = sum(_replayed(r) for r in rs)
        if n_rep:
            notes.append(f"{n_rep} replayed by the router (excluded)")
        share = _med([r['cache_share'] for r in ok])
        if ok and (share is None or share < 0.5):
            notes.append("UNCACHED")
        lines.append(f"| {m} | {s} | {h} | {v} | {len(ok)} | {_f(_med([r['ttft'] for r in ok]))} | "
                     f"{_f(_med([r.get('ttft_any') for r in ok]))} | "
                     f"{_f(_med([r['t_complete'] for r in ok]))} | {_f(_med([r['t_end'] for r in ok]))} | "
                     f"{_med([r['prompt_tokens'] for r in ok]) or '-'} | "
                     f"{_f(share)} | {'; '.join(notes)} |")
    return "\n".join(lines) + "\n"
