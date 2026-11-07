# Round-2 artifacts

Same 60-recording / 180-view manifest and instruction strings as
`../samint-general/`; all owned IDs, no sealed inputs. `design.json`,
`generated-prompts.json`, and `geometry-manifest.json` were frozen before inference.
Generated prompts are an audit of instruction parsing, never a task-selected bank.

- `records.jsonl.gz`: 5,940 scored records (v2b plus 32 arms, 180 views each).
- `proposals.jsonl.gz`: 180 pre-filter proposal audits, prompt strings, scores,
  independent flags, estimated bounds and measured timing components.
- `masks.tar.gz`: 180 NPZ files, named by manifest-order bundle index and camera.
  Each contains `sam` and `v2b` packed mask arrays, plus the image `shape`.
  Ordering matches the corresponding proposal audit. v2b keys/scores are in that
  audit; SAM categories are not exact asset identities. These arrays contain no GT.
- `mask-hashes.json`: SHA256 of every NPZ member.
- `size-ranges.json`: fixed whole-catalogue size envelopes used by the size check;
  not a prompt inventory. `*` is the unknown-label/global envelope.
- `ablations.csv`: every combination and overall gates/recall cost/timing.
- `cells.csv`: every cell x combination; positive recall cost means lost recall
  against the unfiltered small-prompt union.
- `risk.csv`: every camera/phase group x combination (1,344 rows), unchanged gates.
- `scaling.csv`: controlled count sweep, including two-client request maxima.
- `latency-by-count.csv`: actual unfiltered frame times grouped by prompt count.
- `meta.json`: frozen config/hash, source hashes, startup, timing, server peak
  allocated VRAM, and raw controlled single/concurrent request samples.
- `filter-diagnostics.json`: missing-geometry versus other filter rejection counts.
- `verdict.json`: qualification, exact predeclared closest ranking and paired CIs.
- `no-dispatch-receipt.json`: locked compute4 snapshot, no row/STOP mutation.
- `result-receipt.json`: output hashes, source/merge/CI pins and zero LLM calls.

Decode an NPZ mask array with numpy only: for nonempty packed array `a`,
`np.unpackbits(a, axis=1).reshape((len(a), *shape))`. Empty arrays represent zero
masks. Do not interpret estimated geometry as measured depth/contact. Ground truth
was opened only after predictions and filter decisions; its frozen scoring-file
hashes are in `geometry-manifest.json`.

Regenerate the tables without GPU or model calls from a checkout of the recorded
source (invoke as a module so the `scripts` namespace resolves):

```bash
gzip -dc docs/experiments/data/samint-r2/records.jsonl.gz > /tmp/samint-r2-records.jsonl
uv run python -m scripts.perception.sam.general_r2_summary \
  --input /tmp/samint-r2-records.jsonl \
  --meta docs/experiments/data/samint-r2/meta.json \
  --reference docs/experiments/data/2026-11-07-sam3-task-prompts/summary.csv \
  --out /tmp/samint-r2-summary
```

Published CSV line endings are normalized to LF; numeric records are unchanged.

The measured source snapshot remains under `~/controlr-samint-r2` on compute3;
raw files and masks remain under `~/controlr-samint-r2-results`. Its owned shared
server was stopped after measurement. No camera adoption or dispatch is authorized
by these offline artifacts. Historical task-bank routes remain diagnostic-only.
