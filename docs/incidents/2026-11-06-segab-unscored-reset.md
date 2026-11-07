# SEGAB: unscored unstable reset omitted from checkpoint attempt counts

The end-of-box audit found one retained attempt absent from the episode-row
collector. This is an accounting and retry-eligibility issue; it does not
establish a detector effect. The [cutoff report](../experiments/2026-11-06-seg-v2n-camera-ab-endbox.md)
remains pending and labels retry-selected pour statistics provisional.

## Evidence

All paths below are on compute6 under `~/dojo-data/integ-c6/runs2/`:

- `c6b_poursg_segA_b_100003`: START17:12:19Z, DONE17:13:04Z, rc=0,
  lock47s, no episode JSONL. Summary `n=0, unstable=1`; driver prints
  `Unstable envs: [0]`. A nearby PhysX warning concerns non-GPU-compatible
  convex-mesh/particle collision; it is not a confirmed causal diagnosis.
- At 17:13:04Z the lane requeued once as `c6b_poursg_segA_b_100003__retry`.
  START17:13:34Z, DONE17:17:20Z, lock256s, success, score 100.
- Both resolved runtime configs match; frozen input hashes still match.
  The original summary SHA256 is
  `6ad971ccb7762b72d44effcd4d67141f2d08e84fab628fda08ba5bf28545e672`.

The pinned [eval path](../../dojo/runtime/eval.py) appends an unstable marker
and continues when `ep.reset()` returns `stable=false`, before emitting the
normal episode JSONL. The summary counts this marker separately from n.
The [lane](../../scripts/dojo/queues/lane2.sh) requeues when there is no completed
row, without first requiring a classified infrastructure failure.
`segab-1106/report.py` reads only episode JSONLs, so its attempt count omitted
this attempt while selecting the retry as the paired A result.

## Impact and disposition

There are 114 completed scored rows plus this unscored reset, not 114 total
completed attempts. There were 117 STARTs at the frozen cutoff:115 completed
launcher attempts and two still running. Earlier routine checkpoint statements
about no infrastructure failures were limited to parsed rows and missed this
attempt. Full lane logs, original summary/provenance, retry and source hashes
are retained; no artifact was deleted or overwritten to repair the result.

The registration permits one retry of a classified infrastructure failure.
Whether this unstable reset qualifies is unresolved. Do not silently relabel
it or adopt the lever using the retry-selected result. A hypothetical original
failure/zero assignment would change the pour table from A 50/B 40, W/L 2/3, to
A 40/B 40, W/L 3/3; this sensitivity is not a post-hoc scoring decision.

No source, tolerance, reset behavior, retry policy, layout or queue was changed
after the audit. Monitoring ended at its time box; authorized lanes were left
running. The final cohort report needs explicit adjudication and complete
attempt accounting. The general empty-row retry/accounting issue is in BACKLOG.
