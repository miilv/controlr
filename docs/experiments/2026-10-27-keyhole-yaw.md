# 2026-10-27 — insert_key: the lock's yaw from its keyhole

**Question.** In rendered 5.1 insert_key (Pegs' keyR_b13 100003), the lock (`Geometry/key_slot`) was read at
yaw 0.03 deg; the truth was 10.7 deg. The key met the lock top 10.7 deg off the 6 mm channel and fell over. Every
layout needs the lock's yaw within about 3 deg.
- Why is the yaw wrong?
- Can perception read it, from the head at frame 0 and from a wrist look over the lock?

**Data.**
- **insert_key D0 on genuine 5.1** (`d0_key51`): 25 dev layouts (100000-24). Frame 0 plus a wrist straight
  over the lock from 15 cm (`make_d0 --look-label slot --hover-cm 15`). Rendered for this report under compute6's
  heavy lock; no SDF cooking failure in the log.
- **d0_ext insert_key** (6.0 proxy, 10 layouts): frame 0, B1 and the G1 regression.
- **Truth.** The D0 GT. Yaw is scored over the full turn and mod 180.
- 0 LLM calls.

## Cause

- The lock is a copper cylinder, 100 mm across and 55 mm tall, not a square block (its bbox is 100 x 100 mm).
- Its outline is the same at every yaw (shape sym 36), so B1's yaw is arbitrary: 29.6 deg in 100003.
- The agentic perceiver then wrote 0.03 deg. Its H2 refit turned less than 30 deg, which arbitration exempts, and no
  silhouette loss can tell the two apart anyway.
- The place-time refine could not correct it, because the lock was not in the wrist view (visible_frac 0).
- The yaw is visible only in the keyhole: a 9 mm round bulb and an 11 x 6 mm channel, 20 mm long and 30 mm deep.
  - Its long axis gives the yaw mod 180.
  - The bulb end gives the sign.
- The catalogue colour does not help: Copper.mdl is texture-driven, and the outline is a disc either way.

## The read (`dojo/perception/top_feature.py`)

**Template from the mesh.** No task or asset name is used.
- The top face is the up-facing triangles at the mesh's highest level.
- The opening is what the face's own outline encloses but the face does not cover.
- A feature needs all of:
  - a symmetric outline (sym >= 8);
  - a small opening (< 25 % of the top);
  - an elongated opening (axis ratio >= 1.4).
- Across all 406 catalogue candidate keys exactly one asset qualifies: key_slot. Mugs' mouths, the stacking ring
  and plain discs do not.

**Image read.**
- The ROI is the lock top within the opening's reach + 9 mm of the prior centre.
- Pixels darker than their own 4 mm neighbourhood by 20 L count as the opening. A global threshold took the shaded
  half of the top: in the wrist hover the top runs L 60-200, while the opening is L ~45 against ~170 around it.
- These pixels are back-projected onto the top plane and matched to the template:
  - by a truncated symmetric chamfer;
  - coarse over the full turn, with the shift that matches the centroids;
  - then Nelder-Mead.
- The bulb side comes from the opening's end widths (8.8 vs 6 mm), not from the fit. With a free shift, the flipped
  opening slid 6 mm along its axis fits almost as well.
- The reported fields:
  - `ok`: the axis, against the best fit across it;
  - `sign_ok`: the ends' width ratio and <= 1 mm per pixel, so never from the head.

**Wiring.**
- **B1:** sets the yaw of a feature object from the head and wrist views at frame 0 (`EstimatorConfig.top_feature_yaw`,
  default on), and records `_perception.yaw_read`.
- **refine():** after the outline fit, reads the keyhole in the wrist view.
  - `ok`: the pose takes the read yaw.
  - `sign_ok`: the pose also takes the xy the keyhole implies.
  - The output gains `yaw_read` and `yaw_source`: "top_feature" when the sign is read, "top_feature_axis" when it is not.
  - The perception service passes both through.
- **Agentic perceiver:** an agent pose never overwrites a B1 yaw marked `yaw_read` (`seed_refine.keep_read_yaws`).
- **Runtime:**
  - `policy._refine_hook` returns `yaw_source` and leaves the raw reading in `sk.last_refine`.
  - A "top_feature" read is exempt from the turn guards (`refine_max_turn_deg`, `_low_iou_turn`), because flipping
    B1's mod-180 head yaw is the correction.

## Results

**Wrist straight over the lock, genuine 5.1.** 25 layouts, `refine()` from the truth xy + noise and a random yaw (what
B1 or the agent hands over).

| arm | prior noise | yaw, full turn: median / <= 3 deg | xy: median / max / <= 2 mm |
|---|---|---|---|
| main (outline fit only) | 2 mm | 103.5 deg / 0 of 25 (the arbitrary prior) | 4.0 / 6.9 mm / 8 of 25 |
| keyhole read | 2 mm | **0.8 deg / 24 of 25** (worst 3.5) | **0.4 / 0.7 mm / 25 of 25** |
| keyhole read | 10 mm | 0.6 deg / 24 of 25 | 0.4 / 0.7 mm / 25 of 25 |

- The axis was read in 25 of 25 hovers and the bulb side in 23. The bulb side was never wrong.
- The outline fit of the round lock was 3-7 mm off in two thirds of the hovers. The keyhole places it within 0.7 mm.

**Head at frame 0 (B1).** Slot yaw mod 180; the sign is never confident from the head, by design.

| set | main: median / max / <= 3 deg / <= 8 deg | keyhole: median / max / <= 3 deg / <= 8 deg |
|---|---|---|
| d0_key51 (5.1, 23 found) | 11.9 / 29.3 / 2 / 9 | **3.5 / 24.6 / 9 / 20** |
| d0_ext (6.0, 9 found) | 10.6 / 45.2 / 1 / 1 | **2.1 / 5.0 / 7 / 9** |

- The 3 head misses beyond 8 deg on 5.1 are B1 xy misses of the lock (18-25 mm off), not misreads.
- B1 also missed the lock in 2 of 25 5.1 frames and 1 of 10 6.0 frames. That is a separate open item.

**Regression** (G1 frame 0, d0_ext insert_key + plug_in_charger, main vs branch).
- Only the slot's yaw changes; key, charger and socket are identical object for object.
- Slot yaw error: 24 / 45 / 8.5 / 8.4 / 10.6 / 8.5 / 15 / 24.8 / 2.3 deg -> 2.7 / 2.1 / 4.3 / 2.0 / 1.6 / 0.5 / 0.8 / 5.0 / 2.9 deg.
- No other catalogue asset has a top feature, so no other task's B1 or refine output changes.

**What insert_key needs from the skill.** One wrist look straight over the lock, 15-20 cm up, before the place (Pegs).
- `yaw_source == "top_feature"`: insert along the read yaw at the read xy.
- `"top_feature_axis"`: sweep {yaw, yaw + 180}.
- No read: re-hover.

## Spend

- 0 LLM calls.
- GPU: compute6 heavy lock ~8 min for the D0 render; light slot 7 on compute6 / 7 for the evals.
