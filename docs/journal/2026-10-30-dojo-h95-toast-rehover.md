# 2026-10-30: make_toast(_random) camera slips: the slice beside the one just pulled out; `rehover_siblings`, `sibling_order`, `slip_shift` (stream H)

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

## The census over 8 A rows (be7a798b, dev 100000-03, both cells)

| | picks | ended in a slip |
|---|---|---|
| after a re-hover hint | 12 | 6 |
| without a hint | 9 | 5 |
| the first slice of a row | 8 | 1 |
| the next slice | 8 | 8 (or the episode ended before it) |

- **The hint is not what decides; the slice is.** The second slice slipped in every row, with a hint or on the
  frame-0 prior. It stood beside the one just pulled out, and the tracker's resync read it at IoU 0.14-0.49 afterwards.
- **Why it moves.** The records are raw: no settled record of these layouts is stable. The slices lean on each other
  and settle into the gap.
- **The pick refine** then fails on the thin slice (IoU 0.16-0.62, "perception not ok"). Neither the stale prior nor
  the rejected fit is within the pinch's margin.
- **Pinch margin (G3 mode B, make_toast 100000-01, every slice believed dx off across its thickness):** 3 mm 2/2,
  6 mm 1/2 (picks ok), 10 mm 0/2 (every pick slips), 13 mm 0/2.
- **make_toast_random** also ends on perception_gross in 4/4 rows (flags "conf 0.45" on every slice): a separate class,
  for perception.

## `sibling_order` (new switch, default "given")

- "far": `slots_and_press` takes next the object farthest from where the taken ones stood. With two of four slices
  needed, that is the far end of the rack, not the neighbour.
- G3 mode gt (it cannot show the disturbance):
  - make_toast 5/5 = 5/5; every row now takes bread_0 + bread_3, which 3 of 5 base rows already did.
  - make_toast_random 3/4 → 2/4 (100003 1.0 → 0.5, a different second slice).
- The screen is camera.

## `slip_shift` (new switch, default "off"): null in G3

- "across": after a pick of a thin object (thinnest extent < 20 mm) that closed on air (rose < 5 mm), `_run_leg`
  retries the pinch 7 mm to one side across that axis, then to the other.
- G3 mode B sweep, make_toast 100000-01, off vs across:

| offset | off | across |
|---|---|---|
| 3 mm | 2/2 | 2/2 |
| 6 mm | 1/2 | 1/2 |
| 10 mm | 0/2 | 0/2 (8 retries, 1 ran shifted) |
| 13 mm | 0/2 | 0/2 (9 retries, 2 ran shifted) |

- Most shifted re-plans found no pair beside the siblings, and the ones that ran slipped too: a missed pinch also knocks
  the slice.
- No-regression in mode gt:
  - fill_pen_holder 100030-34: 0.09 → 0.12 (1 retry, 1 up);
  - insert_key 100000-02: identical (0 retries).
- Kept default-off for the record.

## Camera dev B arm: keep + sibling_order far (9e8bf90d, merged as 710b9a70)

Integration's `c4b_maketoast*sg_dev_rsk_<id>`, ids 100000-04, paired with the A rows (be7a798b):

| id | make_toast A | make_toast B | random A | random B |
|---|---|---|---|---|
| 100000 | 0.25, slip | 0.0, belief_mismatch | 0, perception_gross | 0, perception_gross |
| 100001 | 0.25, slip | 0.5, belief_mismatch | 0 | 0 |
| 100002 | 0.25, slip | 0.5, belief_mismatch | 0 | 0 |
| 100003 | 0.25, perception_gross | 0.25, perception_gross | 0 | 0 |
| 100004 | 0.25, slip | **1.0, success** | 0 | 0 |
| mean | 0.25 | 0.45 (3 up / 1 down) | 0 | 0 |

- **The fix works on the slip class.** Every make_toast B row took bread_0 and then bread_3, the far end of the rack.
  The B picks held in 4/5 rows. In 100003 a perception_gross row still slipped twice; in A, all 5 rows slipped.
  "keep" refused 1-2 hints a row.
- **The loser, 100000, and the belief_mismatch rows.**
  - 100000 went 0.25 → 0.0. Both B picks and places read ok in the belief, but the official score counts no slot.
  - 100001 and 100002 reach 0.5 (both slots) and miss the success: the lever / finish check.
  - These are the class after the grasp: the placement as the checker sees it, and the press.
- **make_toast_random stays 0/5 in both arms: a perception class (W).** The random scenes are routed to the
  "make_toast" vocabulary, whose toaster candidates (00000 / 00001) don't include the random layouts' 00004 / 00002.
  B1 fits the wrong toaster model.
