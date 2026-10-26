# 2026-10-26 — objects in a fixture's support slot (make_toast bread, play_Xylophone mallet): B1 had no path for them

**Done:** branch `feat/dojo-perception-toast`.
- `inverse_layout`:
  - `support_slot` / `support_slot_pose`: the generator's slot placement, or the settled pose from
    `slot_settle.json`.
  - Slot groups are kept out of the blob proposals and placed after the base (`_place_support_slots`, per-view check
    on exclusive pixels; empty slot -> `_not_found` "support slot empty").
  - Assembly shapes (`_add_assembly_shapes`): the base and its settled slot objects as one candidate shape.
  - `_cat_key` maps composite keys to catalogue keys.
  - Config: `EstimatorConfig.support_slots` / `slot_assembly`.
- Catalogue on compute3: `Geometry/bread_shelf/00000` and `Rigid/mallet_stand/00000` annotated with their support
  frames (`catalog annotate`). The original info.json files are backed up in `~/dojo-percrnd-data/*_info.json.bak`.
- `d0_toast` / `d0_xylo`: our frames (10 dev layouts each, frame 0 + a wrist look over the base) in
  `~/dojo-percrnd-data`.
- DEV scripts: `val_slots.py`, `val_slots_breadth.py`, `slot_settle_build.py`.
- Report: docs/experiments/2026-10-26-slot-objects.md.

**Decisions:**
- **Settled poses come from our own physics settle of public assets, not from the layout generator.** The generator
  drops the slices from 2 cm; the eval scenes are settled the same way.
- **The assembly runs at proposal time and through refinement**, then becomes the shelf for the records. A re-fit
  AFTER a shelf-only selection made good shelves worse.
- **The slot path needs the catalogue's support frame.** Unannotated bases keep today's behaviour, so other slot
  tasks do not change silently.

**Found along the way:**
- **Catalogue support frames.** Only the checkerboard had them. Every catalogue install (compute3, isr-lab, the GPU
  boxes) needs `catalog annotate` for the slot bases (BACKLOG).
- **CPU vs GPU.** CPU silhouettes (`DOJO_GL=0`) gave different estimates on the same frames. Replays of service runs
  must use EGL.

**Not done / next step:**
- Annotate and settle the other slot tasks' bases: deposit_coin, fill_egg_holder.
- The mallet stand's 180° yaw ambiguity at frame 0: one mallet ended up 148 mm off, at confidence 0.23.
