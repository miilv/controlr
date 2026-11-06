# SEG round 2: clean dev baseline repair; genuine training renderer blocked

Continues PR #387 / branch `t3/86744994`. No model promoted. No compute4 access,
LLM calls or tokens, weight downloads, system changes or real-robot actions.

## Baseline

The first new GPU-GL bake reproduced the five `press_by_number` errors in
162 dev frames. CUDA recovery did not repair this Python dependency issue.
A CPU-only reproducer showed `ArrayDatatype.getHandler(ctypes.byref(c_uint()))`
raising the same TypeError. Explicitly registering the **installed**
`CtypesParameterHandler.HANDLED_TYPES` fixes the conversion. No installed package
was edited. `bake_seg.py` now initializes each GL worker with that registration.
The pointer-conversion test passes in compute3's Python 3.12 environment.

A second complete bake completed using the corrected code, three workers, GPU GL and W's
unchanged `cfg_link_off.json`, all five sets, dev ids only. Its outputs are under
`~/dojo-data/s-v2e-seg/bake/v2b-clean`; the first failed attempt is `v2b-fresh`.

The corrected baseline has **162 frames / 975 objects / zero errors**: found
913, identity 637, pose10 767, identity+10 571; 35 phantoms. Full baseline counts:
[per-class table](../experiments/data/seg-v2e-dev/v2b-clean.md),
[source/code/output hashes](../experiments/data/seg-v2e-dev/v2b-clean-manifest.json).
The +25 recovered objects are the formerly errored press_by_number rows.

Correction to the round-1 limitation: rows retain `gt_idx`, although the fully
qualified catalogue type/key is absent. This is sufficient to split each
(set, task, category) by asset index. The new `--by-asset` report and its
within-category-cancellation test were added before training. Adoption requires
zero losers at both category and asset-index resolution, not a pooled category
win that hides a lost numeral. [Baseline per-asset counts](../experiments/data/seg-v2e-dev/v2b-clean-per-asset.md).

## Rehearsal hypothesis

The new `audit_rehearsal.py` validates exact base-class prefix, unique-image
support, training variant ids and the validation split. It rejects zero old-class
training support with exit 1; it does not claim deleted images still exist.
Its independent audit reproduces **140 / 511 old classes with zero training
images** in v2d, including every numeral. Full hashed per-asset counts:
[training support](../experiments/data/seg-v2e-dev/v2d-support.json).
This remains evidence of starvation, **not a causal training ablation**.

The [round-2 recipe](../experiments/data/seg-v2e-dev/round2-recipe.json) was
recorded before any training: paired two-epoch directional data ablation, then
an eight-epoch rehearsal candidate with the exact v2b prefix and 32 additions.
The historical v2d recipe used 8 epochs; a two-epoch result would only be a cheap
hypothesis check, not a comparison claiming to reproduce its final weights.
Both candidate and historical narrow arm need re-rendered data: their old
images/labels were deleted. No fresh model has been trained at this checkpoint. Both arms share every
appended-class image and its exact full-export RFS repetition count; the control
omits only old-class-only rehearsal images. This controls new-class exposure.

The initial export has 3483 images (3313 train / 170 val), 543 classes and
32901 instances, all with Isaac instance labels. `--new-keep` excluded 558
images containing 30 other assets. The strict support gate found **54 old
classes still without training images**, and stopped before expansion/training.
[Initial support audit](../experiments/data/seg-v2e-dev/support-initial-round2.json).
A registered targeted repair cycles these missing assets, one per draw, through
stack_blocks variants 12–17 on training ids only. That public task has no
clutter, avoiding the out-of-vocabulary clutter filtering that cost coverage.
The requirement remains every old class ≥1 training image.

## Blocker and shared-machine checks

A single genuine Isaac Sim 5.1 rendering smoke test, queued after the first bake
on the same heavy GPU lock, **segfaulted with exit 139 at renderer startup**.
The stack includes `librtx.scenedb.plugin.so`; zero training frames were produced.
Details: [incident](../incidents/2026-11-06-seg-isaac51-renderer.md).
No system repair was attempted. After this failure the lead relayed Ilia's
updated utilization request: parallel render workers, maximum fitting batch,
≥6 GiB VRAM and four physical CPU cores reserved, no light locks held. A fresh
proxy-domain recipe amendment was recorded **before** the proxy pilot or training.
Two physics workers succeeded; the first two 6.0 RGB workers returned empty
arrays. A bounded single-worker retry produced valid 640×480 RGB and masks.
The committed capture wrapper now checks readiness of both buffers, with a
20-attempt cap; it never accepts empty captures as training data.

Base proxy rehearsal used two workers; focused extras use three on disjoint output
roots and a four-physical-core CPU affinity reservation. The heavy lock is held;
G3 light locks are untouched. The supervisor monitors disk and VRAM every two
seconds and can terminate only the subprocess sessions it created.

The all-class extra pool exposed an injector bug: **146 v2b Rigid vocabulary
keys are stored under Clutter, not Rigid**. All assets are already present; no
asset downloads occurred. Initial extra draws hit misleading “missing asset”
errors. The training-only extra injector now falls back to Clutter when the
normal task asset is absent, preserving the canonical Rigid class key and
writing the required cluttered record type. Ordinary task generation is unchanged.
An actual CPU-generated book extra verifies this record; a focused test checks
both the extra path and rejection on the unchanged native path. A repair pass
retried only our 50 explicitly diagnosed missing-asset failures.

An app restart interrupted the agent, but the remote pipeline remained alive:
launcher PID 654489 held the heavy lock and three focused workers continued with
CPUs 0–27. No duplicate was started. Later a 6.0 worker segfaulted at native
startup (exit -11), leaving completed outputs and saved stages intact. After
confirming the launcher/workers had exited and the lock was free, the unfinished
focused phase resumed from those stages. Its crash log was preserved. No model
checkpoint existed yet. The targeted coverage phase staggers Kit startups by
25 seconds to reduce concurrent startup contention.

Before each job: `who`, `uptime`, `nvidia-smi`; one heavy job under the existing
`~/dojo-gpu.lock`. The pre-existing GPU service (PID 3613, 2418 MiB), Xorg and
GNOME were not touched. Available VRAM during the first bake stayed above the
8 GB requirement in observed checks (22 GB free with its GL workers loaded).
Disk free: 114 GB; no disk-floor override or data deletion. The simulator probe
ended itself; bake servers use the existing wrapper's PID-specific cleanup.
All requested output/deploy paths are under `~/controlr-seg-v2e` and
`~/dojo-data/s-v2e-seg`. Weights remain on compute3. Baseline weight MD5:
`~/dojo-w-ext/seg_v2b.pt`: `2f546196145fc424f76a5e2e47dd3210`.

## Verification

Focused segmenter suite: 34 passed / 5 skipped locally (optional dependencies).
Compute3 focused gate/rehearsal/capture/layout tests: 60 passed, including ctypes pointer conversion and clutter injection.
Focused suite after the training-extra/capture additions: 46 passed / 14 skipped.
Compileall, pyflakes and `git diff --check` pass. Broad suite was bounded at
900 seconds and did not complete (progress reached 74%, with one observed failure:
`tests/test_fkagent.py::test_pixel_loop_pick_place_done_and_fold`). That test
passed independently on retry (1 passed in 8.20 s). No full-suite pass is claimed.
