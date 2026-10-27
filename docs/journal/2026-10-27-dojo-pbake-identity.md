# 2026-10-27 — Perception bake-off, Identity stream: learned image embeddings for identity

Stream: Identity (perception bake-off, evening of 27 Oct). Brief: do DINOv2 / SigLIP2 / CLIP-L embeddings fix identity
among similar objects (class b: the belt camera read as a phone, the garage figure, puppet -> bottle, mugs, stacking
blocks, tools) better than our shape / colour identity? Report: [experiments/2026-10-27-identity-embeddings.md](../experiments/2026-10-27-identity-embeddings.md).

Machines: compute3 light slot 1 only, plus one heavy-1 hold the lead approved (6 min, by_image render). Box at
<= 3 GPU jobs whenever a job of ours started. 0 LLM calls.

## Setup

- **Own venv.** `~/controlr-identity` has `.[dev,perception]` + torch 2.9.1 cu128 + transformers 5.2; triton was
  uninstalled. The uv cache was private and hardlinked, then cleaned.
- **Weights.** `~/dojo-data/pbake-identity/models`, 1.2 GB:
  - DINOv2-S and DINOv2-B as downloaded;
  - SigLIP2-B and CLIP-L as fp16 vision towers. The full checkpoints were downloaded, converted and deleted.
- **Footprint.** Everything of the stream totals ~8.4 GB on compute3, under the 10 GB cap.
- **Excluded ids.** The ids 100010-19 are dropped from every eval through a symlinked D0 root
  (`~/dojo-data/pbake-identity/d0sel`); eval_g1 itself has no id filter.

## 1. Belt identity (component)

See the report's section 1: per head blob on P2's 5.1 belt frames, the frozen DINOv2-S arm 0.46 -> 0.83 on the held-out
layouts. Notes from getting there:
- **The crop matters most.** The masked crop loses the dark parts the segmentation drops on the dark belt: the camera's
  body is 80 % "shadow" (Memory's probe). The raw crop matched against renders composited over the belt background
  works; masked does not.
- **Fragments.** The garage figure's blobs are fragments: up to 11 per frame on 100000, many < 100 px. Most of its
  remaining errors are fragments, or merges with a neighbouring bottle.
- **The lit renders** (a light at the camera) beat flat per-face colours by 0.02-0.04 with DINOv2. The lit arm was frozen.
- **Throughput.** The first full run was slow: ~30 s per frame with 4 models x 6 crop variants x ~13 blobs. Layouts
  100001-03 therefore ran at every 3rd frame.

## 2. Read level

`belt_read` off vs `embed_model="dinov2_s"`, on the same 67 reads (every 3rd frame of 100000-03), scored with P2's
matching. Identity of the matched reads, off -> on: 79/116 -> 106/115. Phantoms 77 -> 114: each fragment of the
garage figure now gets a confident fit of its own.

The lead's lever was to merge the same-asset fragments (`embed_merge`). With it: 104/116 identity, 47 phantoms (fewer
than main), xy median 7.3 -> 5.9 mm, +0.9 s per read. `embed_merge` is therefore on whenever `embed_model` is set.

## 3. pick_from_conveyor_by_image

- **Riders (6.0 proxy, 5 layouts, 208 blobs).** Ours 0.69, the frozen DINOv2 arm 0.68 (no gain), SigLIP2 fused 0.87:
  it reads the basket and the cans.
- **The board.** It stands behind the right arm's links in the home head view, so neither our colour pick (always
  `photo/00000`, which has no map entry) nor any embedding can read it there.
- **Given the picture** (the true `photo.png`), the target rider is the most similar crop: CLIP-L 33/33, SigLIP2 32/33,
  DINOv2 28-29/33. The skill needs a look at the board first.

## 4. eval_g1 on the table

Pending: the eval_g1 off / on pairs are running on compute3 light 1. This section is added to this PR when they end.

## Code (PR)

- **`dojo/perception/embed_id.py`.** `Embedder` (lazy torch), `Gallery` (catalogue renders at the blob's ray, lit or
  flat, colour-mapped), `EmbedIdentity.scores` / `scores_at_poses`, `identity()` (one model per process), `fuse`.
- **`EstimatorConfig`.** New fields `embed_model` (default None = off), `embed_weight` 0.25, `embed_crop` composite,
  `embed_lit` True, `embed_min_px` 40, `embed_merge` False. With the model unset, nothing changes.
- **`belt.belt_read`.** Per blob, the fused pick. A pick we did not propose is fitted on the blob (`_fit_key`).
  `_merge_fragments` (`embed_merge`) merges same-asset blobs within half the asset's extent.
- **B1.**
  - `_polish` adds the embedding's best geometry as a trial and decides by the fused score.
  - The same-geometry identity step adds 1 - similarity, rendered at the fitted pose.
  - `ViewMasks.bg` keeps the background each view was segmented against.
- **`perception_service`.** `DOJO_PERCEPTION_CFG` (JSON) overrides `EstimatorConfig` for an A/B arm.
- **Scripts.** `scripts/perception/identity_bake.py` + `identity_bake_report.py` (component), `belt_id_ab.py` (read level),
  `identity_photo.py` (by_image picture).
