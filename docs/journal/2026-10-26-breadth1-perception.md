# 2026-10-26 — breadth-1 perception list: items 1-2 (swap_blocks, deposit_coin); paused by the lead

**Done:** branch `feat/dojo-perception-breadth1`.
- **swap_blocks.** A slot label pinned at offset (0, 0) on one of several bases is placed at the base's support point;
  only the yaw is fitted (`_support_xy`, `_yaw_only_fit`). Cube xy median 18.4 -> 2.5 mm on the breadth frames.
- **deposit_coin.**
  - The (vertical_coin_stand, slot, coin) pair is added to `slot_settle.json`. Compute3's catalogue is annotated for
    the stand; its original info.json is backed up in `~/dojo-percrnd-data/`.
  - Small single objects (≤ 6 cm) get the wrist-view proposal fallback.
  - Coin found 0 -> 7/10 on D0 frame 0, 0 -> 3/5 on the breadth frames. The stand is still unreliable.
- **Data and scripts.** `d0_coin` (our make_d0 frames); `breadth_eval.py` (breadth look frames vs settled layout
  truth, interchangeable labels matched by position).
- Report: docs/experiments/2026-10-26-breadth1-perception.md.

**Decisions:**
- **Not shipped:** proposing small objects from the look frames' views. It helps when the look targets the object
  (stand 10/10) and harms when it doesn't (4/5 wrong).

**Not done / next step:**
- After the pause: plug_in_charger, organize_table, insert_tubes, conveyor and store_tools (in the report).
- deposit_coin's stand: a targeted look plus a head/wrist consistency check.
- Every catalogue install needs `catalog_check --sync --assets` for `Geometry/vertical_coin_stand/00000`.
