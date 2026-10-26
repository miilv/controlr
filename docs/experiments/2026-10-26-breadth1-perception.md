# 2026-10-26 — breadth-1 perception fixes: swap_blocks cube bias, deposit_coin coin and stand

**Question.** Integration-2's breadth-1 attribution (proxy mode B, submit_v1) ranked the perception losses. This
report covers the first two items, before the lead's pause:
1. **swap_blocks** (1/5 vs G3 96): both cubes ~16-22 mm off at planning, systematically.
2. **deposit_coin** (0/5 vs G3 69): the coin is never found, and the coin stand is missing or 70-660 mm off.

**Data.**
- **Breadth frames.** Integration's breadth episodes with saved look frames (`~/dojo-data/integ/runs/B_<task>_breadth/`).
  Frame 0 alone, and frame 0 plus the saved look frame (the last look_plan look). Truth comes from our settled dev
  layouts. Identical assets are matched by position (their labels are interchangeable).
- **`d0_coin`.** Our deposit_coin frames: 10 dev layouts, frame 0 plus frame 1 (left wrist hovering 10 cm over the
  stand). Physics-only Isaac 5.1 states, Isaac 6.0 proxy RGB.
- **Code.** "main" = origin/main at db48652. "fix" = this branch. GPU light lane, no LLM calls.
- **Scripts.** `breadth_eval.py`, `slot_d0_eval.py` (`~/dojo-percrnd-data`).

## 1. swap_blocks: the cube pinned on a mat

**Cause.** The config pins each cube at a mat's support point (offset 0, 0) with a random yaw. B1 fitted xy freely.
On the mat's same-height foreground, the fit slid 17-22 mm toward the camera, the same way for both cubes in every
episode.

**Fix.** A slot label pinned at offset (0, 0) on one of several bases sits exactly on the base's support point. Only
the yaw is fitted, by a 3° grid on the ROI IoU (`_support_xy`, `_yaw_only_fit`). The base choice keeps using the
top-face colour.

| breadth frames (5 episodes, 10 cubes) | xy median / max | yaw median / max (mod 90°) |
|---|---|---|
| main, frame 0 | 18.4 / 21.6 mm | 28.2 / 40.5° |
| **fix, frame 0** | **2.5 / 5.0 mm** | 27.4 / 39.0° |
| main, frame 0 + look | 7.7 / 21.6 mm | 19.1 / 44.5° |
| **fix, frame 0 + look** | **2.5 / 5.0 mm** | **6.0 / 24.9°** |

Frame-0 yaw stays coarse: the head view barely constrains a cube's yaw. The pick-hover refine fixes it.

## 2. deposit_coin: a 32 × 11 × 17 mm coin stand with the coin standing in its slot

**Cause.**
- **The coin is pinned in the stand's support slot**, the same class as make_toast's bread. Before this branch the
  pair had no settle data, so the slot path did not apply.
- **The stand is a few head-view pixels**, often next to the home gripper. Blob proposals put it on wrong blobs
  (70-660 mm) or missed it.

**Fix.**
- **The (vertical_coin_stand, slot, coin) pair is added to `slot_settle.json`** (our physics settle of 10 layouts,
  spread 1.3 mm). `Geometry/vertical_coin_stand/00000` is annotated in compute3's catalogue.
- **Small single objects** (largest extent ≤ 6 cm, `_small_group`) also get the wrist-view proposal fallback, which
  before served single-target groups only.

| set | variant | stand found / within 15 mm | coin found / within 15 mm |
|---|---|---|---|
| d0_coin frame 0 (10) | main | 5 / 2 | **0** / 0 |
| d0_coin frame 0 (10) | **fix** | 7 / 2 | **7** / 2 |
| d0_coin frame 0 + look at the stand (10) | main | 3 / 0 | 1 / 0 |
| d0_coin frame 0 + look at the stand (10) | **fix** | 5 / 1 | 4 / 1 |
| breadth frame 0 (5) | main | 3 / 0 (17.5, 68.5, 661 mm) | **0** |
| breadth frame 0 (5) | **fix** | 4 / 2 (0.2, 20.4, 68.5, 0.1 mm) | **3** (0.2, 20.5, 0.1 mm) |
| breadth frame 0 + untargeted look (5) | main | 1 / 0 (611 mm) | 0 |
| breadth frame 0 + untargeted look (5) | fix | 4 / 1 (611, 21, 119, 9.5 mm) | 0 |

- **This is only partly solved.** Whenever the stand is right, the coin is now placed exactly, where before it was
  never found. The stand itself is still unreliable.
- **Confidence does not separate good and bad stands** (0.08-0.60 for both).
- **A version that also proposed from the look frames' views** found the stand in 10/10 when the look targets it
  (6 within 15 mm). It made 4 of 5 stands wrong when the look targets something else, so it was not shipped.
- **Next step:** a look at the stand when it is small and uncertain, plus a consistency check between the head view
  and the wrist view.

## Not started (paused by the lead)

1. plug_in_charger: charger flip / 18° at planning.
2. organize_table: garage z / rotation (B1 frame 0 here: garage 41-100 mm off).
3. insert_tubes: tubes ~130 mm off at planning (B1 frame 0 here: ≤ 2 mm, so the error comes from a later stage).
4. conveyor: frame-0 proposals.
5. store_tools: hammer (here a constant 17.8° yaw offset and 5-18 mm) and wrench (45° in 3/5, 147 mm in 1/5).
