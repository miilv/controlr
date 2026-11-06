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

Training maximum 40 epochs, patience 10, imgsz960, seed0, close_mosaic5, automatic AdamW.
Before any dev measurement, inspection of installed SegmentMetrics.fitness showed that
Ultralytics selects best.pt by the sum of box and mask mAP50-95, not mask alone as originally
stated. The recipe is clarified to match the actual trainer; no dev outcome informed this.
At epoch27 the best mask and combined-fitness epoch coincided. This is a documented
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
58 passed with no skips (including SAM/proposer consumers), including classifier-row selection, empty-label negatives,
wire-prefix preservation, default request compatibility and consumer priority.

## Remote ownership and resource discipline

On resume, no SEG jobs/servers survived; heavy lock free, colleague PID3613 retained its
2,418 MiB. Training launcher PID1082040; queued gate launcher PID1095758. Heavy work uses
`~/dojo-gpu.lock` only. Four physical cores excluded from affinity; VRAM reserve8 GiB in
training (hard free floor6 GiB). Disk floor10 GB retained; ~110 GB free at launch.
Logs/weights/data: `compute3:~/dojo-data/s-v2n-seg`; deployed source `~/controlr-seg-v2e`.
The queued gate acquires the same heavy lock after training, rechecks who/load/GPU/disk,
checks raw parity, then runs a fresh disabled baseline and all three registered IoUs.
No system changes, compute4 access, LLM calls or downloaded weights.

## Training and detector results

Early stopping completed at epoch37 after ten epochs without improving validation fitness.
Both mask-only and combined validation maxima are epoch27, so the selection-description
correction did not change the chosen checkpoint. Training occupied about22 minutes.

- v2b MD5: `2f546196145fc424f76a5e2e47dd3210` (unchanged).
- v2n best.pt MD5: `be0817051b8efd0565218033cb647cec`.
- v2n last.pt MD5: `f1ce5afa880847f2ca9d46e1ec95f52f`.
- Weights remain `compute3:~/dojo-data/s-v2n-seg/runs/v2n/weights/`.

Raw parity passed on **486 gate images / 2,055 base detections**, comparing serialized
standalone-v2b, disabled-composition and enabled base-prefix tuples. Including hitcheck
requests across all three thresholds: 552 checks / 2,772 base detections, zero mismatches.
Each request checks mask bits, key, score, box, order and base detection count.

Naming is identical at theta0.1/0.3/0.5: **24 hit / 13 merged / 12 missed, of49**.
All24 hits are within the original43-instance vocabulary; brick2/paper_ball0 remain0/6.
Dolls15–19 total8/15; laptop1 2/3; toaster4 5/6; holder1 3/3; headset8 1/3;
mug14 1/3; mug17 2/3; wuliangye1 2/3. V2b0/49, historical v2d13/49,
round2v2e10/49. Unit is image-level scene membership, not localisation or visibility.
The untouched base may retain a wrong name alongside an added correct name; naming is
not a count/phantom or episode-success result. Full rows and image-level predictions:
[hitcheck.json](../experiments/data/seg-v2n/hitcheck.json).

CI passed at `ff9ca54d` (GitHub run42221716519,9m47s). Per-class and phantom gates
remain pending; final qualification and cleanup follow their complete paired outputs.
