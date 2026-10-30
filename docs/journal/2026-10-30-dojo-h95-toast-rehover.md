# 2026-10-30: make_toast(_random) camera slips: the pick refine's re-hover hint in a bread rack; `rehover_siblings` (stream H)

**Rows used.**
- Integration's dev camera rows: `c4b_maketoastsg_dev_<id>` and `c4b_maketoastrandomsg_dev_<id>`, be7a798b, dev ids
  100000-04, CANDIDATE.txt with look frames.
- G3 mode gt on the same dev ids, main be7a798b, as the reference:
  - make_toast on raw records (`SIMLAB_DEV_LAYOUT_SUFFIX=`): 5/5;
  - make_toast_random: 3/4.
- No measurement-id rows were opened.

## G3 vs camera: the same grasp

- G3 uses the same grasp families that slip under camera: "box close 2 approach -0 tilt +25/+40 off 0/-46mm", a
  pinch across the slice's 12 mm thickness.
- In G3 they lift with 0.3-2.3 mm of in-hand drift.
- The bread: 117 × 116 × 12 mm, 70 g. Four slices stand 25 mm apart in the shelf, along x.

## The class

**Every camera bread slip follows a refine hint.**
- At the pick hover, the wrist refine did not fit the slice but "saw it elsewhere". `_refine_hover` wrote that pose
  into the world and re-hovered, and the pinch was then planned on it.
- The frame-0 belief equalled the layout record to 0.1 mm in these rows; the hints did not:

| row | pick | hint − initial (dx across, dy along, dz; mm) | outcome |
|---|---|---|---|
| toast 100000 | bread_0 | 3, 11, 0 | ok |
| toast 100000 | bread_1 | 5, -12, 0 | slip, "jaws empty (air)" |
| toast 100000 | bread_1 again | 12, 21, -13 (the slice had moved) | ok |
| random 100000 | bread_0 | 13, -7, 0 | slip (air) |
| random 100000 | bread_1 | -11, 174, 0 | slip (air) |
| random 100000 | bread_0 again | 6, 61, -12 | lifted, slipped on the carry |

- In a rack of identical slices, "seen elsewhere" is most likely a neighbour, or a blend of two.
- About 5 mm across a 12 mm slice is enough to close the jaws on air.

## `rehover_siblings` (new switch, default "follow")

- "keep": at a pick, `Skills._refine_hover` does not follow a hint for an object with an identical sibling (same
  asset key) within `SIBLING_NEAR` (5 cm), and keeps its pose. Place hints, and objects with no such sibling, are
  unchanged.
- The dev B rows run c962c89d, which also covered place hints. For make_toast those land on the toaster only, which
  has no sibling, so there is no difference.
- G3 cannot screen it: G3's refine hook gives no hints. The screen is Integration's dev camera B arm
  (`c4b_maketoast*sg_dev_rsk_<id>`, c962c89d).

## Where else it can act (the guards)

The flag acts at a pick refine's re-hover hint on an object with an identical asset within 5 cm.
- **Same-asset objects within 5 cm at frame 0** (dev layouts 100000-04):
  - make_toast(_random) bread, 24 mm;
  - make_kong mahjong, 46 mm;
  - pour_balls_into_vase spheres, 3 mm.
- **Same-asset objects at all:** 26 tasks, mostly 73-305 mm apart. Stacking and nesting can bring them within 5 cm
  later.
- **Re-hover hints in Integration's dev camera rows** (ids 100000-04, compute4 runs*): make_kong 0 in 5 reports. Pick
  hints on same-asset objects appear in:
  - build_tower ("pick block3", 82 hints / 20 reports, picks and places);
  - stack_bowls ("pick bowl0 / bowl2", 38 / 20);
  - play_stacking_toy ("pick block6", 24 / 10).
- **Never fires in those rows** (0 hints): align_blocks, swap_blocks, make_kong, match_and_pick, tic-tac-toe,
  store_laptop.
- **Unknown, no dev camera rows:** fill_egg_holder (eggs, 56 mm apart at frame 0) and pour_balls.

**Guards for the camera A/B** (5 pairs each): build_tower, stack_bowls, play_stacking_toy, fill_egg_holder. Not
guarded:
- make_kong (0 hints in 5 dev rows);
- pour_balls (it picks the vase; the spheres are never picked);
- stack_blocks, the dolls, insert_tubes and fill_pen (no identical asset within 5 cm at a pick).
