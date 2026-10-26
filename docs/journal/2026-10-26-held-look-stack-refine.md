# 2026-10-26 — B1 held_look false negatives and stack refine failures on 5.1 replay frames

**Done:** branch `feat/dojo-perception-heldfix`.
- `held_look` (`dojo/perception/j1_adapter.py`) now uses the jaw opening as its primary evidence.
  `measure_opening` renders the finger silhouettes (FK) at 21 openings and scores dark pixels inside them against a
  band along their outer sides. A jaw stopped by an object reads wider than commanded (`held_from_opening`).
- Without a measurement, the colour test decides alone (`held_from_colour`). Its "nothing found" verdict on a dark
  region is absence evidence, capped at 0.3.
- New per-arm fields: `evidence` ("opening" | "colour" | "absent") and `opening` {measured, commanded, score}.
  `VERSION` is now j1-adapter-2.
- `refine` (`dojo/perception/refine.py`), four changes for stacks:
  - a known object under the target is no longer masked out of the foreground (`_under`);
  - the neighbour guard compares objects at the same level only;
  - a known object resting on the target makes the fit `ok=False`, why "covered:<key>", and the prior pose is kept
    (`_rests_on`);
  - a prior floating more than 8 mm above the highest known support under it (`support_z`) gets a second full
    refine, from the prior lowered onto that support. It must be a full refine, not just another start: the local
    masks are built around the prior's projection. The second result is taken when it is ok and its IoU is at
    least 0.15 higher, and it then carries `z_dropped_mm`.

**Found (Bridge-isr's 12 regenerated 5.1 replay checks, truth in their index.json):**
- **held_look said "not held", confidence 1.0, on a lifted block** (100004 x2, 100008 x4).
  - Cause: the dark textured blocks (idx 12, 13) look like the grey finger pads in the fingertip region, and the
    colour test then gave full confidence to "not held".
  - The 6.0 proxy renders these blocks just as dark (L 93 vs 89, 52 vs 66), so the proxy does not hide the case.
  - The measured opening is 0.55 / 0.65 against commands of 0.30 / 0.34, so with the fix both read as held.
- **Post-place refine failed on complete 3-stacks** (100001 s152, 100007 s142).
  - Cause: the xy-only neighbour guard rejected the top block because the block below shares its xy.
  - The support's pixels were also masked out of the foreground.
- **The "label permutation" in 100007 / 100008 is not physical.**
  - stack_blocks labels are exchangeable slots: three unique indices from one size group, labelled in sampling
    order.
  - The scorer is order-free (`is_stacked(in_order=False)`), and its xy threshold comes from block_0's index,
    which is the same for every block of a size group.
  - Perception gave every physical block its correct asset at every stage: initial, scan, agentic, fused.
  - Truth must be matched to belief by asset key + position, not by label.
- **The false "centre 0.8961" in 100007 is not a perception reading.**
  - That run's resync did not re-estimate (diverged: []).
  - The value appears first in the skill's place check, right after a place whose verification passed no
    candidate (`verify passes [0, 0, 0, 0]`). It was then the prior of the post-place refine.
  - Seeded 56 mm too high, the refine finds nothing (iou 0.0, the logged 0.0017). Seeded at the support's top it is
    ok (iou 0.98, xy error 0.4 mm).
  - main already stops adopting a prediction when no candidate passed (908f3cf). The support-height refit covers
    any other floating belief.
- **Landmine (runtime):** `belief.post_place` adopts a refine only when its xy shift is at least 2 mm, so a z-only
  correction is dropped. Added to BACKLOG.

**Results:**
- Replay frames: 8/8 checks now right (6 held, 2 empty jaws). Both stack top blocks refine ok (iou 0.86 / 0.91,
  xy error 0.3 / 0.1 mm). The covered lower blocks are rejected.
- 100007 s142, seeded from the runtime's floating prior (z 0.8961): ok, iou 0.985, xy error 0.4 mm (main: iou 0.0).
- D0 wrist refine validation (`val_refine.py`, build_tower + stack_blocks + push_T, frames 1/2/L1, `WITH_OTHERS=1`):
  0 of 352 rows changed.
- D0 has no closed-gripper frames, so a D0 held_look regression is vacuous: every D0 look has the gripper
  commanded open, and held_look skips those arms.

**Not done / next step:**
- held_look `what` and refine `best`: report the best-matching same-category candidate (identity + conf + margin).
  The contract was agreed with integration-2 through the coordinator.
- Calibrate their thresholds (0.7 / 0.15) on a held set: 6.0 proxy renders of held and empty jaws across the 15
  block assets, plus the 5.1 replay.
