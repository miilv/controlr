# 2026-10-30: sort_nesting_dolls skill failures in G3, row_stage and row_home_clear (stream N)

The census's second class: skill failures after correct perception (journal 2026-10-30-dojo-row-size-read). G3 mode gt,
c42e011c, row_by_size. Dev draws 100000-04 (settled) and the v1 draws 100030-39, both cells: 30 episodes.

**Baseline: 17 / 30** (std dev 2/5, `_random` dev 3/5, std v1 7/10, `_random` v1 5/10; the rows reproduce b7763a97 /
e783de43 exactly where they overlap). All 13 failures leave a doll unplaced or toppled; the checker's row rule never
failed with every doll placed upright. What the dolls left unplaced share:
- **A start beyond |x| 0.34** (9 of 13 episodes): one arm alone reaches the doll, at the limit of its lift. Its carry to a
  slot hits a doll already in the row ("approach: joint move at t=0.79: hits doll3", "via failed: lift: ik_fail"), or
  no grasp-place pair reaches the slot. The second try repeats the first plan.
- **A start near the arm bases** (y -0.19 to -0.25) or far out (y -0.05): 0-32 of 180 grasps reachable under the centred-grasp
  filter (`AXIS_OFF` 8 mm, `MAX_PICK_TILT` 30 deg).
- **A doll knocked over** (std 100003): the right arm going home after a place swept its hand through the 130 mm doll
  standing in the row's end slot (0.245, -0.21), 0.15 m from that arm's home hand, and rolled it 30 cm off.
- `grasp_slip` / `place_miss` (random 100003, 100038, 100030, 100039).

**The camera rows' "all placed but 0"** (physics-only replays of their action lists on compute3, the episode's
`stages_left 1` reproduced):
- std 100003: the same knock as G3's at step 141-149.
- `_random` 100002: doll2 was held 48 deg tilted in the jaws (an in-hand rotation the belief did not see) and toppled at the
  release; doll4 was placed at 13-18 deg. Held state, stream H's side.

**`RuntimeConfig.row_stage` / `Skills.row_stage`** (default off). A doll whose transfer to its slot ends in `ik_fail` is
set down upright on a staging spot first: within 0.18 m of the middle, 0.11 m off the row's line, y >= -0.27, clear of
every slot. Then it is tried again from there. Dolls that plan go straight to their slot, as today. G3: **18 / 30**
- std 100034 is converted;
- every other episode is identical;
- 15 stagings ran in 12 episodes, and 4 set the doll down.

The other 11 stagings fail at the same pick: with the centred-grasp filter, the far / near starts have 0-32 reachable
grasps. That filter binds next. Lever to try: a relaxed pick for the staging leg only (wider tilt / off-axis grasps; the
doll is re-gripped centred at the staging spot for its slot).

**`RuntimeConfig.row_home_clear` / `Skills.row_home_clear`** (default off). Only rows whose slots all keep 0.17 m from
both home hands; with none, today's rows. At 0.20 m, two `_random` v1 layouts had no row at all. The row changes on 3 of
30 seeds:
- std 100003: 0 → 1;
- `_random` 100035: 1 → 0 (a place_miss in the new row);
- `_random` 100039: 0 → 0.

Net 0 on n = 3. Not a lever as is; it stays off and on the record.

0 LLM calls. compute3: one light lane, 63 G3 episodes plus 3 physics-only replays.
