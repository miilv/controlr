# 2026-10-28: make_kong glyph reads from images: a twin-trained tile classifier (stream S)

Context: after the legality fix (#227), make_kong's tile matching must come from image evidence (D-C17). The reader
is `dojo.perception.glyph_read` (#246, flag `glyph_read`, default off). 0 LLM calls.

**Done**
- **Crops.** 24.5k from my make_kong renders (variants of 100030-49) and the held-out d0_ext frames / `--test` held
  renders (v0 of 100000-09 / 100020-24).
  - Labels come from face visibility toward each camera: cos > 0.17 = glyph, < -0.09 = back (its own class), grazing
    views left out. Most kong tiles stand on edge, so tilt alone mislabelled backs.
  - The held tile is marked.
  - Held picks: 19 lifted of 58 attempts (compute2), then 51 of 162 (compute4, 27 layouts).
- **Classifier.** ResNet-18 at 128 px, 30 epochs, about 3.5 min on compute3 light7. Read margins per view kind (head,
  wrist, held_head, held_wrist), chosen on val = the 100045-49 draws for 98 % precision.
- **Weights.** compute4:~/dojo-data/segtrain/models/kong_v1b/model.pt (sha256 dc35be09...). Margins: head 0.35, wrist
  0.62, held_head 0.94, held_wrist 0.56.

**Held-out test (v0 of 100000-09 / 100020-24; nothing tuned there)**

| view type | n glyph | top-1 | read rate | right among reads |
|---|---|---|---|---|
| (1) free tiles, head | 1515 | 0.96 | 0.97 | 0.98 |
| (1) free tiles, wrists | 593 | 0.86 | 0.81 | 0.95 |
| backs (all views) | 1535 | 0.99 | 0.98 | 1.00 |
| (2) the tile in the jaws, head (v1b, held margin) | 61 | 0.44 | 0.23 | 0.79 |
| (2) same, v1 (val = 100049 only, shared head margin) | 61 | 0.56 | 0.89 | 0.61 |

- Type (1) is usable.
- Type (2), the draw read from the head camera, is not. The held tile is about 20-30 px and half hidden by the pads,
  and no margin gives a useful read rate at >= 95 % precision.
- With a top-down pick, the holding wrist sees an on-edge tile's face at grazing angles in about 75 % of views.

**Next.** Measure the "present" view: the held tile's face turned toward the other arm's wrist camera at ~15 cm
(`seg_render --present`). Go / no-go for building that skill: >= 95 % precision at >= 70 % read rate (margin on
100045-49 val, then the held-out ids).
