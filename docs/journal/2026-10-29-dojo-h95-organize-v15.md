# 2026-10-29: organize_table on v15 dev draws: the classes; push_goal "cover" / "cover_obb", all within noise (stream H)

## Method

- G3 mode gt: physics-only twin, GT world, 0 LLM, traced.
- Layouts: my own draws 100030-49, variant 1 (simlab/v15), presettled.
  - 17 ids ran.
  - 100035, 100040 and 100046 failed the presettle ("not stable" / "not at rest") and were skipped by G3.
- Base = main at edd98a6d, no flags. Every arm ran on the same code (`feat/dojo-h95-organize`) and the same 17 ids.
- Per-item counts come from the final poses, with an approximation of `tasks/organize_table.py`.
- The official score is `score_mode="transition"`: the most items that were in place at the same time, with all
  grippers open, at any step. So an item placed and later knocked off still counts. That is why base's official 0.397
  sits above its final-state 0.31.

## Classes (base: 0 successes, mean 0.397)

| item | final ok | what fails |
|---|---|---|
| garage | 1/17 | 9 no full plan ("N grasp-place pairs"), 6 slips, 1 miss |
| alarm | 3/17 | 8 tilted landings on the drawer's rim, 5 slips, 1 no plan |
| keyboard | 4/17 | 11 short in x, 2 in y / yaw |
| mouse | 13/17 | 3 slips (off-centre grasp), 1 no plan |

**How the keyboard fails.**
- The +y strokes work. Then the +x stroke has no plan: "no push stroke along axis 0".
- The face stroke starts about 0.24 m left of the keyboard's centre, by the garage or the drawer.
- `push_to` also aims at the frame's centre within 12 mm. The checker only needs the keyboard's footprint to cover
  the rect x 0..0.2, y ±0.03, axis within 45 deg.

## `push_goal` (new switch, default "centre")

- **"cover":** `push_to` stops once the centre lies where the object's footprint covers the support (or lies inside
  it). Otherwise it aims at the nearest such centre (`library.cover_region`). The ranges are computed on world axes,
  from the rotated footprint's bounding box.
- **"cover_obb":** the same in the object's own frame (`library.cover_goal_obb`).
  - The cover / inside choice per axis is made by size (the longer side against the longer side), so it does not
    change with yaw.
  - An axis that cannot cover at the current yaw aims at the support's centre along that axis.
- `arrange` is the only caller of `push_to` (and of `_on_targets`).

## Arms (17 paired ids, 0 successes in every arm)

| arm | mean | up / down vs base | final mouse / alarm / keyboard / garage |
|---|---|---|---|
| base | 0.397 | | 13 / 3 / 4 / 1 |
| B: cover + push_order lookahead | 0.397 | 2 / 2 | 13 / 3 / 5 / 1 |
| D: B + place_surface mesh + push_stroke press | 0.426 | 2 / 1 | 14 / 8 / 6 / 0 |
| E: cover_obb + lookahead + mesh (no press) | 0.412 | 2 / 1 | 14 / 8 / 5 / 0 |

- D was meant to be B + mesh. It also carries `push_stroke=press` by mistake, so it is confounded.
- **E's losers and winners.**
  - The loser (shared with B and D): 100041 goes 0.25 → 0. The keyboard ends at yaw -17 / -25 off the checker's
    footprint, where base had stopped on it.
  - The winners: 100032 0.5 → 0.75 (keyboard covered at yaw -23 by the centring fallback) and 100033 0 → 0.25.
  - E vs D: 1 up / 2 down. 100043 0.5 → 0.25: E kept pushing a keyboard that already passed, and it turned from
    -24 to -27 deg.
- **Mesh.** It turns the alarm's final state 3 → 8 (as on 10-28, 1 → 5). The official score barely moves, because
  base's tilted alarms counted while they still stood.

## Why the cover goal doesn't help

- **5/17 keyboards never get an x stroke** in any arm ("no push stroke along axis 0"; press strokes don't plan there
  either).
- **5-7/17 end yawed 17-42 deg.**
  - At 25 deg, covering the 0.2 × 0.06 rect leaves ~5 mm of slack across. `push_to`'s 12 mm acceptance is wider than
    that: 100041 "within 12 mm", 100044 "within 2 mm" then 5 mm off.
  - Much past 27 deg the rect can't be covered at all (100047, -42 deg).
  - Every further stroke turns the keyboard more.
- **The next lever is yaw:** a rotating stroke before the translation. longh's off-centre strokes (journal
  2026-10-27-dojo-longh §5) did not move the counts. A tighter tolerance on the centred axis is the cheap variant.
- Prior work I missed in the first grep: longh had already tried a "cover the frame" goal without a gain (BACKLOG).
  B reproduces that null.

## Spend

0 LLM calls. G3 on compute3: 6 jobs of 8-9 episodes, one at a time.
- One exception: for ~1 min at 21:19Z a duplicate ot_D_b ran next to the real one. I had killed a waiting chain
  script just as it launched its job; I then killed the duplicate's process group.
- When you kill a chain's parent bash, its foreground child keeps running: kill the job's process group, by PID.
