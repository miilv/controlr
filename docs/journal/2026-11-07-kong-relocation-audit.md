# 2026-11-07 — KONG relocation write audit coverage

Accepted `watch_relocate` events now retain full-precision `before7` and `after7`
pose lists, copied immediately around the existing belief assignment. Rejected
reads and errors have no pose pair. Search priors, reads, writes, convergence,
try bounds, skill settings and action generation are unchanged.

The audit reader previously ignored all per-writer events once any coarse
`belief_write` existed. It now includes exact relocation events even in those
reports, replacing a same-label/step/rounded-endpoint coarse version event to
avoid double counting. Thus sub-5-mm relocations are covered too. Unrelated
canonical writers remain intact. Audit arm grouping accepts c6b as well as c4b;
this is offline reporting only.

Targeted tests verify exact pose copies, unchanged expected belief updates,
perception-read counts and convergence through a missing read, immutable event
values, rejected-read exclusion, exact/subthreshold writes with and without a
coarse log, and preservation of other writers. Seven focused tests passed.
No GPU or model calls are needed to validate the logging change.

This closes the instrumentation gap identified in the scoped-relocation
registration. Physics-only replays and measured helpful/harmful counts are still
required before a camera verdict; this entry makes no audit-outcome claim.
