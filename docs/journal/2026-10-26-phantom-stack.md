# 2026-10-26 — elevated hypotheses need two views (phantom pillars in the scan-primary estimate)

**Done:** branch `feat/dojo-perception-phantom`.
- **The rule.** `inverse_layout.elevated_support` / `_multiview_ok` judge an elevated hypothesis per view, on the
  pixels it alone would explain. It needs >= 2 supporting views (>= 50 % foreground there) and more supports than
  refutes (< 25 %).
- **Config.** `EstimatorConfig.elevated_multiview`:
  - "identical" (default): a piece on an identical piece; `_stack_proposals` and the upper levels of
    `_tower_proposals`;
  - "all": every elevated hypothesis; `scan.estimate_with_scan` sets it.
- Report: docs/experiments/2026-10-26-phantom-stack.md.

**Decisions:**
- The default checks only identical-on-identical pieces. Mid-episode estimates (resync, search) do see real stacks,
  and a real block on top seen by the head view alone must not be lost. Episode-start estimates use "all".
- A view's evidence is the hypothesis's EXCLUSIVE pixels. Its silhouette overlaps its support's in most views, so
  plain IoU would count the support as evidence.

**Found along the way:**
- Integration's `save_look_frames` keeps frame 0 and the last look, not the scan frame. Scan-record artefacts cannot be
  replayed (BACKLOG).
- On CPU (`DOJO_GL=0`) the estimate differs from the GPU ROI renderer's enough to change the selection. Replays of
  service runs must use EGL.

**Not done / next step:**
- Replay 100011 / 100014 once their scan frames are saved.
