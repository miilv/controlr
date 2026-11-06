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
training-validation mask metric, not gate results. Raw base responses must match bitwise.
The gate uses only frame 0, IDs 100000–04 and 100030–49: 162 frames / 975 objects.

Training reuses the complete validated v2e export, projecting its labels onto the 32 new
classes and retaining old-only images as deliberate negatives. No additional rendering
or model download. 5,067 images: 4,816 train / 251 val (100049). There are 3,346 train /
171 val empty-label images. RFS t=0.05 gives 5,842 train lines. Every new class has positive
training support; raw proxy/genuine provenance is inherited explicitly from round 2.
The separate initial checkpoint selects v2e's 32 classifier rows; Ultralytics reports
561/561 weights transferred. The source v2e checkpoint and production v2b are never written.

Training maximum 40 epochs, patience 10, imgsz960, seed0, close_mosaic5, automatic AdamW.
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
have priority during downstream deduplication. Wrist pools and pad extension stay base-only.
Raw preservation does not prove downstream invariance when joint matching/refitting consumes
additional masks; the complete paired gate remains mandatory.

Local focused tests: 32 passed / 8 optional skips. Compute3's existing perception environment:
39 passed with no skips, including classifier-row selection, empty-label negatives,
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

Results and final cleanup are recorded after the queued gate completes.
