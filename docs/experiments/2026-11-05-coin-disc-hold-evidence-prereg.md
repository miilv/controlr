# COIN camera A/B pre-registration: disc_hold_evidence

Hypothesis: isolating the disc-only opening-resolution guard avoids false belief
releases during coin handovers without the losses seen in the slot_read bundle.
This is a prospective test, not an adoption recommendation. No measurement-id
individual episodes were diagnosed to design it.

Start only after the lead explicitly releases compute4 following D-C19; the
estimated ending time is not authorization to start. Use the existing official
strict CAMERA evaluator, frozen candidate, layouts, model routes, budgets and
seeds. No changes to the running block. Lead schedules all camera work; this
stream has made zero model calls and will not use Claude routes.

- A: the then-frozen CANDIDATE.txt verbatim, disc_hold_evidence off.
- B: the identical candidate plus RuntimeConfig.disc_hold_evidence="on" only.
  Keep slot_read, held_thin and all other fields identical to A. Record candidate
  file SHA256, effective resolved configs, code SHA, asset/layout hashes, renderer,
  grip source, model routes and actual calls/tokens. Freeze these before scoring.
- Primary cell: deposit_coin v0, scoring ids 100005–09 and 100020–29, three paired
  repeats per id (45 episodes per arm). These ids are scoring-only: retain only
  aggregate paired results for the decision; no episode-level diagnosis or tuning.
- Guards: insert_key, play_tic_tac_toe and stack_blocks, same 15 ids and three paired
  repeats (45 episodes per task per arm). They exercise non-disc thin objects and
  an unrelated thick object. Any geometry-qualified pieces are allowed to trigger
  the switch; guards measure that risk.
- Interleave A/B within each task/id/repeat; swap order on alternate pairs. Freeze
  the ordering seed at 0. Restart from a fresh episode in each arm. Infrastructure
  failures use the evaluator's pre-existing classification; rerun both members
  of an affected pair exactly once, otherwise mark the pair unavailable. Missing
  pairs prevent adoption; do not silently reduce n or add favorable repeats.

Adopt-candidate only if ALL hold: B gains at least three deposit_coin successes
net, has **zero paired success losers**, improves mean strict score by >=0.05,
and the lower endpoint of a paired 95% percentile bootstrap CI on score delta is
positive (resample the 15 layout clusters, keeping all repeats, 10,000 draws,
NumPy default_rng(0)). Each guard must have no paired success loser and mean
score >= A. Report n, per-arm success and score mean/spread, paired wins/losses,
steps/time caps, all model calls/tokens, and CI. No outcome justifies interpreting
G3 or replay improvements as camera success.

Before adoption, perform the standard `m95_score.py audit` on physics-only replays
of both arms: compare harmful (>2 mm added 3-D error) writes by writer, including
release events; require no excess harmful writes in B. Counterfactual event counts
alone do not pass this requirement. Any failed criterion parks the switch. Do not
retune on this panel; further development returns to 100000–04 / 100030–49 and
requires a new preregistration before another scoring test. Lead reviews and
merges; this PR must remain unmerged.
