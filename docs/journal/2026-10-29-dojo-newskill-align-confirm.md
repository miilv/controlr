# 2026-10-29 — stream N: align, an unconfirmed fixing slide is measured

**Symptom** (W, camera c4b_alignsg_par_100020, compute4 runs2):
- The parallel slides left cube2 29.8° off.
- The fixing slide ran, and the head-view confirm rejected its expected outcome (`adopted=False`).
- align_in_row still reported "row 0.0 mm / 0.0 deg" and finished.

**Cause.** The fixing slide is a grasp, so its release put the block at the slide's target in the belief. The rejected
confirm left that belief as it was.

**Change** (only with `align_measure` "parallel", the adopted value; "seq" unchanged). When the confirm of the fixing
slide is not adopted, `_fix_unconfirmed` does three things:
1. Measures the block from the arm's hover: wrist refine, yaw included.
2. Logs the measured error.
3. Still off by more than `FIX_TOL`, with at least 60 steps left: slides it once more from the measured pose.

Unit test: `test_an_unconfirmed_fix_slide_is_measured_and_slid_again` (a block 30° off is measured, then slid within
`FIX_TOL`).
