# General instruction/catalogue SAM union: complete offline comparison

**Decision: NOT QUALIFIED; no camera A/B registered or queued.** This frozen general
configuration improves recall but fails the prospectively frozen material-harm rule.
Historical task-specific routes/banks remain diagnostic-only and prohibited for board
candidates. No runtime default changed.

The [preregistration](2026-11-07-samint-general-offline-prereg.md) was committed on
main at `ea08171a` before inference. Component commit `e695260c`, frozen deployed
evaluator `5324a662`, full implementation [PR #428](https://github.com/miilv/controlr/pull/428).
Data: [all-task table](data/samint-general/all-task-table.csv), [head table](data/samint-general/head-table.csv), [all views](data/samint-general/summary.csv),
[all harm groups/CIs](data/samint-general/risk.csv), [category details](data/samint-general/category-table.csv),
[verdict](data/samint-general/verdict.json), [raw observations](data/samint-general/general-sam3.jsonl),
[run metadata](data/samint-general/general-sam3.jsonl.meta.json).

## Paired comparison

All cameras and phases; 4–5 independent input layouts per task. Triples below are **v2b / historical
task-bank SAM3 union / general SAM3 union**. Recall is percent at IoU≥0.5; mean IoU
includes misses. Each row pools TP/n across its views; historical mean IoU is weighted
by visible-object count and phantom counts are summed. The bank reference is not an eligible configuration. Its historical confidence thresholds
were 0.25 for imitate, 0.05 for match-conveyor and 0.2 elsewhere; it is a reference,
not an isolated prompt-only intervention. SAM3.1 reference recall is also retained
in the head-table CSV. All 180 views,
including wrists and surveys, determine qualification. Head-frame-zero values matching
the original SAMBAKE presentation are retained in the separate head-table CSV.

| Recorded cell (scoring only) | Views | Object observations | Recall %: v2b / bank / general | Mean IoU: v2b / bank / general | Phantoms: v2b / bank / general |
|---|---:|---:|---|---|---|
| classify_objects | 15 | 93 | 95.7 / 98.9 / 98.9 | 0.883 / 0.907 / 0.907 | 0 / 4 / 99 |
| classify_objects_by_language | 15 | 94 | 68.1 / 97.9 / 97.9 | 0.651 / 0.913 / 0.913 | 0 / 14 / 71 |
| deposit_coin | 21 | 37 | 83.8 / 91.9 / 97.3 | 0.726 / 0.789 / 0.805 | 0 / 9 / 78 |
| fill_egg_holder | 12 | 35 | 2.9 / 97.1 / 97.1 | 0.021 / 0.836 / 0.889 | 0 / 3 / 54 |
| fill_pen_holder | 21 | 67 | 100.0 / 100.0 / 100.0 | 0.919 / 0.916 / 0.916 | 0 / 0 / 117 |
| imitate_sorting_sequence | 15 | 126 | 59.5 / 92.1 / 96.8 | 0.555 / 0.847 / 0.894 | 0 / 6 / 694 |
| match_and_pick_from_conveyor | 15 | 25 | 8.0 / 40.0 / 40.0 | 0.071 / 0.349 / 0.431 | 2 / 33 / 151 |
| organize_table | 21 | 120 | 90.0 / 92.5 / 92.5 | 0.851 / 0.874 / 0.874 | 8 / 84 / 807 |
| pick_from_conveyor_by_image | 15 | 36 | 5.6 / 44.4 / 58.3 | 0.048 / 0.395 / 0.533 | 0 / 6 / 106 |
| pour_by_language | 15 | 60 | 43.3 / 100.0 / 100.0 | 0.417 / 0.971 / 0.971 | 0 / 25 / 118 |
| put_bottles_into_dustbin | 15 | 40 | 92.5 / 97.5 / 100.0 | 0.872 / 0.920 / 0.944 | 1 / 10 / 83 |

Across **180 views / 60 recordings / 11 tasks / 733 primary object observations**:

- Micro recall 68.5% → 93.5% (+25.0 pp).
- Macro task recall 59.0% → 89.0% (+30.0 pp).
- Mean IoU 0.638 → 0.868.
- Phantoms 11 → 2378; 0.06 → 13.21 per image.
- **42 task/camera/phase groups fail** at least one frozen practical-harm gate.
- Rerun v2b vs published reference: 0 metric differences above 1e-6 (see verdict JSON).

The overall gain requirement is ≥5pp for both micro and macro recall. A single group
with recall loss ≥5pp, mean-IoU loss ≥0.02, or ≥0.5 added phantoms/image and ≥2 added
phantoms blocks camera dispatch. All 42 failures are phantom increases; no group breaches the recall or mean-IoU loss
bound. The largest mask gains across views are egg (+94.3pp), pour (+56.7pp), image
conveyor (+52.8pp), imitate (+37.3pp) and match-conveyor (+32.0pp). The general union
also exceeds historical task-bank recall on coin, imitate, image conveyor and
bottles; match-conveyor recall ties, with higher mean IoU. Pen mean IoU falls slightly (0.91875 to 0.91641), below the 0.02 guard bound,
with no recall gain. Every cell adds unmatched masks; organize and imitate add 799
and 694 respectively, while pens add 117. These are the guard failures even where
mask recall is unchanged or improved.

Across 54 layout clusters, paired 95% bootstrap intervals are +18.3 to +32.2pp for
recall and +9.29 to +17.52 phantoms/image; see [uncertainty](data/samint-general/overall-uncertainty.json).
Layout-cluster 10,000-resample paired bootstrap CIs
(seed 0) are descriptive and cannot waive that rule. Every group, including groups
with no visible primary objects, is checked for phantom increases.

## One configuration and input provenance

One SAM3 BF16 checkpoint, SAM threshold 0.2, v2b confidence 0.25, base-first IoU>0.5
de-duplication and positive-area filtering apply identically to every image. No region,
object-count, competition or task filter was used. Prompts comprise all normalized
category/caption labels from the one shared catalogue plus generic grammatical object
phrases extracted from actual logged episode instructions (a deterministic grammatical
approximation, not a learned noun tagger). The catalogue contributes 615 normalized
labels; requests contain 616–622 prompts. The predictor takes RGB,
instruction text and the shared catalogue only. It does not accept task identity,
task configs, checkers, GT, role labels, geometry, placement priors or per-task banks.
SAM outputs are masks/candidate categories, not exact asset IDs.

The exact SAMBAKE RGB/layout SHA256 panel and all 60 recovered instructions were
validated before model load. There are 54 frame-zero layouts plus six pre-grasp survey
observations, three cameras each. IDs remain within 100000–09 / 100020–24. No sealed
input was accessed. The [global catalogue labels](data/samint-general/catalogue-labels.json) and emitted
prompts are retained as provenance. The [result receipt](data/samint-general/result-receipt.json)
freezes all result file hashes and source/CI pins. Historical per-task bank scores are scoring references only.

Scoring is unchanged from SAMBAKE: non-clutter objects with ≥30 visible pixels,
one-to-one greedy silhouette IoU matching, IoU≥0.5 recall and mean IoU including misses.
Phantoms are masks with best GT IoU<0.3, excluding ≥50% robot overlap. They can include
unannotated fixtures or object parts; the metric does not prove every such mask is
hallucinated. Belt GT uses the original approximate phase reconstruction; survey GT
assumes observation motion did not move objects. This is mask evidence, not identity,
pose, belief-write or manipulation-success evidence.

The task-free API remains separate from the legacy inverse-layout fitter, which still
reads task-config candidate pools/priors. No end-to-end board compliance is claimed.

## Resources and disposition

Compute3 only, one free light lock, admission checked with who/GPU telemetry, 30%
CUDA allocator (<10 GB), bounded 10G RAM / 600% CPU scope. Same existing checkpoints;
no model downloads. Torch peak allocated 5.57 GB, reserved 8.54 GB.
Joint proposer latency p50 27.53s, p90 52.56s.
Timing was measured on the shared box under changing contention, not as isolated
throughput. The general bank should not be assigned the small task-bank service
latencies from the earlier benchmark. 180-view inference wall time 1.73 hours. Latency includes both detectors,
prompt generation and de-duplication; baseline records carry the same joint timing
and are not independent v2b latency measurements. **LLM calls/tokens/spend: 0.**

No general camera manifest or queue rows were created because offline qualification
failed. Compute4's intentional STOP is preserved (same inode/mtime). The final locked read
found 50 pre-existing queue rows; SAMINT added zero and did not edit the queue.
The [no-dispatch receipt](data/samint-general/general-no-dispatch-receipt.json) records
the queue SHA and STOP identity. Old staged
per-task panels/wave recipes are explicitly cancelled. No compute4 GPU process was
started; no compute6/7 work or partner process was touched. No queue position or ETA
applies. This experiment was never mixed into IMIT2, POUR or MATCH2 arms.

Egg's first-salt 13/20 settle failures and their reasons remain retained. The bounded
fresh-salt refill authorization is prospective and conditional on a future qualified
general experiment including egg; no refill was run for the cancelled task panel or
this rejected configuration. The settle-failure landmine remains in BACKLOG.

Validation: full local suite **2,291 passed, 206 skipped**; four subsequent summary
unit tests pass. CI status and final resource/queue receipts are recorded in the journal.
