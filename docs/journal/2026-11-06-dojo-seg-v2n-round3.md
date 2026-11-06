# SEG round 3: independent new-class detector

The lead rejected another shared-model class increment after v2e repeated old-class losses.
This round keeps the production v2b file and network exact, trains an independent 32-class
v2n, and appends only non-overlapping extra detections. No relabel arm is included.

## Pre-registration and frozen inputs

The [recipe](../experiments/data/seg-v2n/recipe.json) was committed as `a0b45237` before
training or additive measurements. Naming must reach at least 18/49 archived genuine-dev
image-level asset presences. All four metrics (found, identity, pose10, identity+10) must
have zero category/asset losers. Phantom tolerance: baseline +2 overall, at most +1 per
(set, task). Dev IoU grid: 0.1/0.3/0.5, extra confidence fixed 0.5; qualify old metrics and
phantoms first, then maximize naming, ties choose the lowest IoU. Checkpoint selection is
training-validation fitness, not gate results. Raw base responses must match bitwise.
The gate uses only frame 0, IDs 100000–04 and 100030–49: 162 frames / 975 objects.

Training reuses the complete validated v2e export, projecting its labels onto the 32 new
classes and retaining old-only images as deliberate negatives. No additional rendering
or model download. 5,067 images: 4,816 train / 251 val (100049). There are 3,346 train /
171 val empty-label images. RFS t=0.05 gives 5,842 train lines. Every new class has positive
training support; raw proxy/genuine provenance is inherited explicitly from round 2.
The separate initial checkpoint selects v2e's 32 classifier rows; Ultralytics reports
561/561 weights transferred. The source v2e checkpoint and production v2b are never written.

Training maximum 40 epochs, patience 10, imgsz 960, seed 0, close_mosaic 5, automatic AdamW.
Before any dev measurement, inspection of installed SegmentMetrics.fitness showed that
Ultralytics selects best.pt by the sum of box and mask mAP50-95, not mask alone as originally
stated. The recipe is clarified to match the actual trainer; no dev outcome informed this.
At epoch 27 the best mask and combined-fitness epoch coincided. This is a documented
selection-description correction, not a claim that the trainer was modified mid-run.
AutoBatch first proposed 39, hit the allocator cap, and retried successfully at **19**.
The initial resolved_batch file reports the pre-retry estimate; the epoch-start recorder
is corrected for future runs. The actual batch follows the retry log and 308 batches for
5,842 lines. No physical GPU exhaustion or colleague process termination.

## Implementation and verification

`seg_twin._AdditiveSegModel` composes independent base/new models with disjoint vocabularies.
The base response is returned directly when disabled. Enabled responses retain every base
wire entry as an untouched prefix, append only new-class masks whose IoU with every base
mask is at most theta, and apply no shared NMS or combined detection cap. Provenance travels
beside the unchanged detection tuples. The optional extra classes file must match its
checkpoint. An enabled request without the extra model fails loudly.

`EstimatorConfig.seg_additive` defaults false; IoU/confidence are explicit config fields.
Extra masks must match exact public task candidates, never category fallback. Base entries
have priority during downstream deduplication, matching and proposal consumption. Wrist pools and pad extension stay base-only.
Raw preservation does not prove downstream invariance when joint matching/refitting consumes
additional masks; the complete paired gate remains mandatory.

Local focused tests: 34 passed / 8 optional skips. Compute3's existing perception environment:
59 passed with no skips (including SAM/proposer consumers), including classifier-row selection, empty-label negatives,
wire-prefix preservation, default request compatibility and consumer priority.

## Remote ownership and resource discipline

On resume, no SEG jobs/servers survived; heavy lock free, colleague PID 3613 retained its
2,418 MiB. Training launcher PID1082040; queued gate launcher PID1095758. Heavy work uses
`~/dojo-gpu.lock` only. Four physical cores excluded from affinity; VRAM reserve8 GiB in
training (hard free floor 6 GiB). Disk floor 10 GB retained; ~110 GB free at launch.
Logs/weights/data: `compute3:~/dojo-data/s-v2n-seg`; deployed source `~/controlr-seg-v2e`.
The queued gate acquires the same heavy lock after training, rechecks who/load/GPU/disk,
checks raw parity, then runs a fresh disabled baseline and all three registered IoUs.
No system changes, compute4 access, LLM calls or downloaded weights.

## Training and detector results

Early stopping completed at epoch 37 after ten epochs without improving validation fitness.
Both mask-only and combined validation maxima are epoch 27, so the selection-description
correction did not change the chosen checkpoint. Training occupied about 22 minutes.

- v2b MD5: `2f546196145fc424f76a5e2e47dd3210` (unchanged).
- v2n best.pt MD5: `be0817051b8efd0565218033cb647cec`.
- v2n last.pt MD5: `f1ce5afa880847f2ca9d46e1ec95f52f`.
- Weights remain `compute3:~/dojo-data/s-v2n-seg/runs/v2n/weights/`.

Raw parity passed on **486 gate images / 2,055 base detections**, comparing serialized
standalone-v2b, disabled-composition and enabled base-prefix tuples. Including hitcheck
requests across all three thresholds: 552 checks / 2,772 base detections, zero mismatches.
Each request checks mask bits, key, score, box, order and base detection count.

Naming is identical at theta0.1/0.3/0.5: **24 hit / 13 merged / 12 missed, of 49**.
All 24 hits are within the original43-instance vocabulary; brick2/paper_ball0 remain0/6.
Dolls 15–19 total8/15; laptop 1 2/3; toaster 4 5/6; holder 1 3/3; headset 8 1/3;
mug 14 1/3; mug 17 2/3; wuliangye 1 2/3. V2b0/49, historical v2d13/49,
round2v2e10/49. Unit is image-level scene membership, not localisation or visibility.
The untouched base may retain a wrong name alongside an added correct name; naming is
not a count/phantom or episode-success result. Full rows and image-level predictions:
[hitcheck.json](../experiments/data/seg-v2n/hitcheck.json).

CI passed at `ff9ca54d` (GitHub run 42221716519,9m47s). IoU 0.1, 0.3 and 0.5 complete with zero frame errors. Each has **zero category and asset losers**
on all 975 objects / 90 category groups / 292 asset groups, including 756 old-class objects
in 68 category / 220 asset groups. The old rows reproduce all checked detection/identity/
pose/confidence values exactly. Old totals stay 739/507/622/447. Full-gate totals become
913/639/768/572 (v2b 913/637/767/571): doll 17 gains identity/pose/id+10 1→2 of 3;
doll 19 identity 1→2 of 3. All other asset-group metrics are unchanged.
Phantoms 35→35 with zero delta in every one of 32 set/task groups.

The naming score ties at all registered IoUs, so the lowest qualifying value **0.1** wins
with every registered threshold complete. The [camera A/B spec](../experiments/2026-11-06-seg-v2n-camera-ab.md)
and concrete A/B estimator JSON overlays are prepared: 50 gain pairs across five new-asset
variant cells, 20 strong-guard pairs, repository default decision rule. No camera execution
or adoption is authorized by this offline result. All 650 baseline inputs/calibrations
rehash unchanged. Final 0.5 has the same passing metrics. Four bakes 397/397/402/389 seconds = 26.4 minutes;
training about 22 minutes; total heavy-lock occupancy about 50 minutes. Final input rehash:
650 checked / 0 changed. Checkpoint MD5s unchanged. Launchers 1082040/1095758 exited;
all SEG GPU servers/workers gone, heavy lock tested free, 110 GB disk free. Colleague GPU
PID 3613 retained 2,418 MiB; desktop processes remained. No light lock or compute4 access,
LLM spend, system modification or default-weight change. Final source/output hashes and
old-field comparisons: [provenance](../experiments/data/seg-v2n/final-provenance.json).

Data export/select-head actually executed revision a0b45237; training used train.py from
8c7b9b33 (SHA256 5b3f7604...), whose effective-batch recorder predated the epoch-start fix.
The deployed-source snapshot is taken before gates, after the exporter firewall addition;
all 5,067 existing exported images independently passed that firewall. Training metadata
and future input validation differ without changing the frozen image/label data or model.
Inference code is ff9ca54d; the final later changes are evidence/docs. Full report:
[seg-v2n-additive](../experiments/2026-11-06-seg-v2n-additive.md).
