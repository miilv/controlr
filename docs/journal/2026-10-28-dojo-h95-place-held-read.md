# 2026-10-28: the held object's read at the pre-place hover, measured and parked (stream H)

The lead's ask: read the held object's pose in the jaws at the pre-place hover (`held.held_pose`) at the
non-insertion places too (hang, tools, sweep, bowls), extending the insertion held read that plug already uses. Trigger
it only where it pays. Measure it E2-style with M's metric. 0 LLM calls, CPU only.

## Built (branch `feat/dojo-h95-place-held-read`, parked: no PR)

- `RuntimeConfig.place_held_read`: off | swing | doubt | swing_or_doubt | all. It applies with `held_pose` "request".
  - At a place that did not ask for the read, `Skills._held_protocol` asks `provider.held_read_wanted`.
  - **swing:** the centre of mass sits >= 20 mm off the pads' pivot axis across gravity (`grasps.swing_lever`).
  - **doubt:** the grasp's pick refine was not ok, or fit at IoU < 0.85.
  - The read is one held_pose call at the existing pre-place still step, so it costs 0 extra env steps. Non-servo
    places take no looks.
  - `place_read_coarse_mm` (default 0) can also take "ambiguous" fits.
- G3 `--place-held-read` (heldsim) for step cost. 8 unit tests.
- Eval: `scripts/perception/h95_hover_eval.py`, on C's per-step sequences, scored with M's metric.
  - The hover = the arm's last still step before the place's descent.
  - The in-hand transform is patched there and rides on to the release.
  - Arms: rt, oracle (the GT in-hand transform at the hover), read_ok, read_coarse.
  - Reads are cached per place.

## Numbers (C's 38 proxy sequences, 297 events, R10)

| arm | clean@R10 | fixed / broken |
|---|---|---|
| rt (the runtime's belief) | 192 | |
| oracle, every place | 230 | +42 / -4: mostly ttt closes on air, where the "in-hand" GT is the piece in the pool; not a hover read's job |
| oracle, GT object in the jaws only | 195 | +10 / -7 |
| oracle, in the jaws, adopted only if it moves >= 5 mm / 8 deg | 197 | **+10 / -5** |
| real held_pose reads, ok (insertion-grade) | 192 | +0 / -0 |
| real reads, also coarse "ambiguous" (sigma <= 3 mm, IoU >= 0.75) | 192 | +0 / -0 |

- The ceiling fixes: mallet x3, pen_holder x2, mug x1, pliers x1, ttt piece x2 (+1 follow-on).
- It breaks objects the descent itself re-seats. A mug's hook re-orients it, a nesting bowl turns into its base, a ttt
  piece moves after its release. For these the hover pose is not the pose at the release.
- The real reads (101; median 3.1 s, max 15.9 s): 18 ok, 31 ambiguous, 29 no_views, 16 fit_poor, 7 out_of_range.
  - Where the prior was >= 10 mm / 10 deg off (mugs, mallet, broom, pen_holder, bowls, ttt pieces), no read was ok.
    held_pose fits those objects at IoU 0.33-0.83, or sees them in no view.
  - Against GT, the fit was better than the prior in 8 of 72 reads and worse in 11.
  - The 18 ok reads moved beliefs that were already right.

## Decision

The lead's rule was: net >= +6 with <= 1 broken on the proxy set, else park. The ceiling is +10 / -5, so the lever is
parked. The code stays on the branch. If it is revived:
- the read needs a fit that sees mugs, brooms and mallets (the views are the bottleneck);
- it must not adopt where the descent re-seats the object.

`place_read_coarse_mm` must stay 0: Integration's arm 9 showed low-IoU reads that move beliefs wreck contact skills.
