# 2026-10-26 — tower post-place: edge cue for same-colour supports, across-axis fit for wide boards

**Done:** branch `feat/dojo-perception-towerpp`. All changes are in `dojo/perception/refine.py`.
- **Edge fallback.** `image_edges` runs Canny on L, a* and b*. `_edge_fit` fits the outline to the edges (truncated
  chamfer). It is accepted at <= 1.0 px, >= 85 % coverage and <= 30 mm.
- **Edge cross-check.** For stacked targets whose colour fit is ok, a <= 0.5 px edge fit that disagrees by > 5 mm
  wins.
- **Truncated objects.** `_partial_fit` clips the box to the visible stretch of its long axis and fits the across
  offset and the yaw. The output carries `partial.unobserved_axis`; the along-axis position stays the prior's.
- `val_replay51.py` gained `PP_ONLY` / `PP_SHIFT_MM` (displaced-prior robustness) and stores poses and the fit path.
- Report: docs/experiments/2026-10-26-tower-post-place.md.

**Decisions:**
- No along-axis fit even when a board end is in view. That 3-DOF fit stopped 10-20 mm off and was confirmed at
  iou 0.99, which is worse than saying nothing. The head view should check that axis.
- The edge-fit cost bar was tightened from 2.0 to 1.0 px. The one wrong edge fit seen (40 mm displaced prior, landing
  on a neighbouring pillar) had 1.37 px.

**Found along the way:**
- Grey-level Canny sees no edge between a white pillar and the light-wood board; the difference is in b* only.
- The runtime must treat `partial` as a confirmation of the across offset and the yaw only (BACKLOG).

**Not done / next step:**
- Along-axis verification of boards from the head view or a second wrist pose (runtime side).
