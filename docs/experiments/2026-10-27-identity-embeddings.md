# 2026-10-27 — Identity by learned image embeddings (perception bake-off, Identity stream)

**Question.** Do learned image embeddings (DINOv2, SigLIP2, CLIP-L) fix identity errors among similar objects (bake-off
class b) better than our shape / colour identity? The known failures are on the recorded 5.1 belt frames: the black
camera read as a phone, the garage figure wrong in 16 of 19 reads, the puppet read as a bottle.

**Method.** Each detected head-view object crop is matched against catalogue renders of every candidate asset, by the
cosine similarity of image features (`dojo/perception/embed_id.py`).
- **Renders.** The catalogue's per-face-colour meshes, at the camera's viewpoint, on the ray through the blob's centroid.
  Each place frame is rendered at 8 yaws, lit by a light at the camera and colour-mapped (`color_calib`).
- **Crops.** Three styles, all 224 px:
  - `ca`: the raw crop against renders composited over the background crop;
  - `ma`: both masked on grey;
  - `mt`: both masked, with each render cropped tightly around its own projection (scale normalised).
- **Ours** is what the pipeline does today on the same blobs: the belt read's best hypothesis (silhouette + colour +
  texture), or B1's polish and colour identity.
- **Truth** (scorer only): the GT object whose GT-pose silhouette covers >= 30 % of the blob. Merged blobs (a second
  object >= 20 %) are left out.
- **Models.** The models are open weights, kept as fp16 vision towers (1.2 GB):
  - `facebook/dinov2-small` and `facebook/dinov2-base`: CLS token;
  - `google/siglip2-base-patch16-224`: the pooled head;
  - `openai/clip-vit-large-patch14`: the image projection.
- **Arm selection.** The arm was chosen on layout 100000 and frozen before the held-out layouts 100001-03:
  DINOv2-S, `ca`, lit, fused with our loss as similarity - 0.25 x loss (keys we did not fit count loss 1.0).
- 0 LLM calls. All runs on compute3's light slot 1, plus one 6-min heavy hold for the by_image render.

## 1. Component: top-1 identity per head blob, recorded 5.1 belt frames

Data: Perception-2's copy of the 5.1 belt recordings of match_and_pick_from_conveyor, layouts 100000-03, every 3rd
frame (P2's frames). Blobs >= 40 px. Script: `scripts/perception/identity_bake.py`; tables:
`identity_bake_report.py`.

**Held-out, 100001-03 (149 blobs).**

| arm | camera | garage | ice_cream | puppet | bottle | phone | all |
|---|---|---|---|---|---|---|---|
| ours (belt read) | 7/34 | 1/43 | 19/25 | 8/11 | 19/21 | 13/13 | 69/149 (0.46) |
| **DINOv2-S fused (frozen)** | **31/34** | **26/43** | 22/25 | 9/11 | 20/21 | 13/13 | **123/149 (0.83)** |
| DINOv2-B, `ca` unlit | 29/34 | 25/43 | 22/25 | 8/11 | 18/21 | 13/13 | 117/149 (0.79) |
| SigLIP2-B, `ca` unlit | 23/34 | 20/43 | 17/25 | 11/11 | 20/21 | 13/13 | 106/149 (0.71) |
| CLIP-L, best crop (`ma` unlit) | 21/34 | 9/43 | 18/25 | 9/11 | 21/21 | 12/13 | 92/149 (0.62) |
| DINOv2-S, `ma` / `mt` lit | 10 / 15 of 34 | 10 / 15 of 43 | | | | | 0.57 / 0.62 |

**Per layout (ours -> frozen arm):**
- 100000 (the tuning layout): 26/100 -> 84/100;
- 100001: 25/72 -> 52/72;
- 100002: 39/72 -> 66/72;
- 100003: 5/5 -> 5/5 (a short copy).

Pooled over the 4 layouts: 95/249 (0.38) -> 207/249 (0.83). On the blobs the belt read actually reports to the tracker:
95/178 -> 151/178.

**Confusions** (4 layouts):
- Ours: garage -> phone 45, camera -> phone 30, garage not proposed 19, garage -> camera 11, garage -> ice_cream 11.
- Frozen arm: garage -> camera 15, everything else <= 4.

**What matters.**
- **Crop style matters more than the model.** On the dark belt the segmentation mask drops the dark parts: the black
  camera keeps 56-132 px of its body. A masked crop therefore throws away exactly what identifies it. The raw crop
  against renders over the background keeps the whole object.
- **DINOv2 beats the language-aligned models** on this render-to-render matching. Small and base are equal, so small is
  the pick (22 M parameters).
- **The garage figure stays the weak kind** (69/96). Its dark figure splits into pieces on the dark belt, and a piece
  alone looks like part of the camera.

Speed (one model, the frozen arm): ~0.4 s to render 10 assets x place frames x 8 yaws per blob, plus ~0.2 s DINOv2-S;
about 1 GB VRAM.

## 2. Read level: belt_read with the flag, paired, P2's scoring

Same frames (every 3rd frame of 100000-03: 67 reads, 141 GT riders in view), same code. `belt_read` runs with
`embed_model` None, then with `"dinov2_s"`, with and without `embed_merge`. Scoring follows Perception-2's
`belt_id.py`: reported riders vs the GT riders in view, Hungarian on xy within 10 cm; identity = the asset key.
Script: `scripts/perception/belt_id_ab.py`. The off arm reproduces P2's main numbers on 100000-02 exactly (74/111).

| arm | identity right (matched reads) | camera | garage | puppet | ice_cream | phantoms (67 reads) | xy median | s / read |
|---|---|---|---|---|---|---|---|---|
| off (main) | 79/116 (0.68) | 6/12 | 3/19 | 15/23 | 27/33 | 77 | 7.3 mm | 2.15 |
| `embed_model`, no merge | 106/115 (0.92) | 10/11 | 19/19 | 15/22 | 32/33 | 114 | 5.8 mm | 2.40 |
| **`embed_model` + `embed_merge` (the default)** | **104/116 (0.90)** | 10/11 | 17/19 | 15/23 | 32/33 | **47** | 5.9 mm | 3.01 |

Per layout, identity right, off -> merge:
- 100000: 19/37 -> 28/35;
- 100001: 24/38 -> 36/40;
- 100002: 31/36 -> 35/36;
- 100003: 5/5 -> 5/5.

Phantoms per layout, off -> merge: 36 -> 21, 19 -> 12, 22 -> 14, 0 -> 0.

- **Without the merge, phantoms rise 77 -> 114.** The embedding gives each fragment of the garage figure a confident
  fit, so each fragment becomes its own rider.
- **The merge removes them.** It joins blobs given the same asset within half that asset's largest extent. Phantoms
  drop below main (47) at the same identity. That is why `embed_merge` is on whenever `embed_model` is set.
- **What is left: puppet -> camera 5.** The green puppet rides next to the black camera, and their blobs touch.

## 3. The reference picture (pick_from_conveyor_by_image)

Pending: the frames are recorded and partly scored. This section is added to this PR when the run ends.

## 4. End to end on the table (eval_g1, frame 0, paired)

Pending: the eval_g1 off / on pairs are running on compute3 light 1. This section is added to this PR when they end.

## Verdict

Pending, until sections 3 and 4 are in. On the belt: adopt, with `embed_model="dinov2_s"` (merge on) as the arm for the rendered conveyor A/B.
