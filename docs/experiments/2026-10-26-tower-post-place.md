# 2026-10-26 — tower post-place leftovers: edge cue and wide boards

**Question.** After [what-is-there](2026-10-26-what-is-there.md), the 5.1 build_tower post-place refine confirmed 19
of 67 well-placed pieces. The failures fell into two kinds:
- pillars standing on the same-colour board (the colour model sees no pillar);
- boards wider than the wrist view (skipped as `truncated`).

Can an edge cue and a constrained fit verify them, without confirming wrong beliefs?

**Code.** `dojo/perception/refine.py`:
- `image_edges` (Canny on L, a*, b*);
- `_edge_fit` (silhouette outline to image edges, truncated chamfer);
- `_partial_fit` (truncated objects);
- the edge cross-check for stacked targets.

**Data.**
- Bridge-isr's 5.1 calibration set: 92 post-place frames from 20 dev-eval stack / tower episodes, scored by
  `dojo/perception/val_replay51.py`, truth rule as in what-is-there. Same-renderer D0 regression with
  `val_refine.py`.
- No LLM calls. CPU only (`DOJO_GL=0`).

## Method

- **Edge fallback.** When the colour fit is not ok, the silhouette outline is fitted to image edges.
  - Edges come from Canny on lightness and on both chroma channels. A white pillar on light wood differs in b*,
    not in grey level, and grey-only Canny found no edge between them.
  - Outline pixels on the robot or the image border are ignored.
  - Accepted at a mean outline-to-edge distance <= 1.0 px, >= 85 % of the outline within 2 px of an edge, and a shift
    <= 30 mm.
- **Edge cross-check.** For a target on a known support whose colour fit IS ok, an edge fit at <= 0.5 px that
  disagrees by > 5 mm wins. A plank on the same-colour board could be "ok" while 10-40 mm off.
- **Truncated objects (boards).** The fit runs over the offset ACROSS the long axis and the yaw only.
  - The box is clipped to the stretch of the axis the camera sees. The far ends project behind it.
  - The position along the long axis keeps the prior's value, and the output says so: `partial.unobserved_axis`.
  - A 3-DOF fit when an end was in view stopped 10-20 mm off along the axis (the end strip is a small part of the
    IoU), so it was dropped. The head view checks the along-axis position.

## Results: 5.1 post-place, well-placed pieces (stack 23, tower 67)

**Robustness test.** The prior is the true pose ± 3 mm / ± 3° ("0"), or that pose displaced 15 / 40 mm in a random
direction (a wrong belief).
- **right** = ok within 5 mm;
- **FALSE** = ok but > 5 mm off (a wrong belief confirmed);
- **partial** = a truncated board, judged on the observed (across) axis only.

| prior | code | stack: right / FALSE / not ok | tower: right / partial ok (across) / FALSE / not ok |
|---|---|---|---|
| 0 | PR #50 | 19 / 1 / 3 | 17 / — / 2 / 48 |
| 0 | **this** | **20 / 0 / 3** | **41 / 25 / 1 (partial, across) / 0** |
| 15 mm off | PR #50 | 13 / 2 / 8 | 14 / — / 5 / 48 |
| 15 mm off | **this** | **15 / 2 / 6** | **30 / 29 / 3 (2 + 1 partial) / 5** |
| 40 mm off | PR #50 | 7 / 1 / 15 | 4 / — / 0 / 63 |
| 40 mm off | **this** | **7 / 1 / 15** | **4 / 18 / 0 / 45** |

- **Edge fits.** At 15 mm off they recovered 15 of 15 pillars, planks and blocks to within 1 mm.
  - The one wrong edge fit (40 mm off, landing on a neighbouring pillar) had a cost of 1.37 px; the 1.0 px bar
    rejects it.
- **Partial board fits.** The across offset was right in 72 of 74. The along-axis offset is not checked by design.

## D0 regression (`val_refine.py`, 352 rows, priors GT ±20 mm, `WITH_OTHERS=1`)

| | ok | ok within 5 mm | ok > 10 mm off | all within 5 mm | worse than prior + 10 mm |
|---|---|---|---|---|---|
| main before #44 | 284 | 0.919 | 17 | 0.744 | 6 |
| PR #50 | 306 | 0.951 | 5 | 0.830 | 2 |
| **this** | **336** | **0.949** | **5** | **0.906** | **2** |

## Caveats / next

- **The prior protocol is ours.** The runtime's expected pose is not logged.
- **A partial ok confirms the across offset and the yaw only.**
  - The runtime must not read it as a full confirmation of the along-axis position.
  - It should check boards along their axis with the head-view render-compare (resync `check`). The same applies to
    a two-pose wrist look at both ends.
- **Pillars on block0 are now verified from the wrist** (edges); the tower failures left at prior "0" are none.
