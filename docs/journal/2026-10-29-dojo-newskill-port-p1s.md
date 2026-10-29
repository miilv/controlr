# 2026-10-29 — stream N: Misha's review P1-1 / P1-2 of our ports (failure-path fixes, in place)

Source: robot-harness docs/notes/REVIEW-ILIA-PORTS-2026-10-29.md, findings 1-2.

**P1-1 (pour_order).**
- **The bug.** `pour_in_order` went on to the next pair on `ok`. The vendored pour_bottle (and pour_bottle_lean) return
  ok=True / verified=False when they find no neck grasp, or the grasp fails, so a later pour could come before an
  earlier one.
- **The fix.** The next pair runs only after a pour that poured: verified, or its own trace reached `drain_from`
  (120°) with no error and no in-hand slip. Otherwise it stops and reports the unrun pair (`not_poured`).
- **Plumbing.** `ported()` now passes a liquid pour's deepest roll and its stop reason on (`pour: {max_phi, stop}`).

**P1-2 (classify_into_bins_p2, adopted).**
- **The bug.** The high retry's put-back called v3's `put_back`, which opens the gripper after any lowering, a refused
  one (budget) included. The object then fell from ~6 cm over the rim, or onto a neighbour.
- **The fix.** p2 has its own `put_back_held`: v3's motions, with the release only where the lowering arrived
  (`reached`, as put_into_bin_v4's `_drop` does). With no pick pose, or a refused or short lowering, the object stays in
  the hand.

Tests reproduce the probes: `test_a_later_pour_only_after_one_that_poured` (6 cases),
`test_ported_reports_the_pour_trace`, `test_p2_put_back_opens_only_where_the_lowering_arrived` (refused, short,
arrived), `test_p2_never_calls_v3s_put_back`.
