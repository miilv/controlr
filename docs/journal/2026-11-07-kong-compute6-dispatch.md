# 2026-11-07 — KONG compute6 preparation and dispatch

The lead merged #410 and [#411](https://github.com/miilv/controlr/pull/411),
then released preparation on compute6. Code is merged main
`ffb2c151f1d48f5bc68f67b10ab4430b0fd4685e`, isolated at
`~/dojo-integ-c6-kong-ffb2c151`. The implementation, exact write logging and
camera design are described in the [preregistration](../experiments/2026-11-07-kong-scoped-relocate-prereg.md).

The new managed Python 3.12.13 venv uses SEGAB's exact frozen requirements.
Its initial PyTorch-first-index resolution shadowed certifi; the corrected
installation considers the existing PyPI/PyTorch indexes together.
Candidate/package parity passes with `diff={}`, the catalogue gate passes,
and all existing resolved candidate fields equal the old lane deployment.
Added fields retain inactive/legacy defaults. The RoboDojo environment,
system, drivers, existing services and lane managers were not modified.
The accepted offline Kit overlay and 600-second watchdog recipe remain.

The two preregistered salt strings and hashes were copied unchanged from git,
verified before generation, and recorded in `~/dojo-data/kong-1107/notes.log`.
All 20 fixed layout identities generated and passed two-pass settling.
Preparation used one free light lock at a time, with admission used VRAM
<=20480 MiB, load <=26, six-core/20-GiB scopes and a 50-GB disk floor.
No compute4 or dc19 path was accessed and no process was killed.

Default imitation demo generation failed 100001 (924 environment steps) and
100004 (967) against the unchanged 900-step cap. The lead authorized all five
same-layout demos at speed 1.25, normal validation, no further escalation,
and exclusion/reporting of any remaining invalid pair. This amendment was
committed before camera collection as `3f089253`. Original attempts/pkls and
the old prep script were archived. **All five passed at 1.25**; no exclusions.
The panel remains ten make_kong, five imitate and five stack pairs.
Preparation completed at **2026-11-06 23:26:07Z**; no model calls were made in prep.

The [committed input freeze](../experiments/data/kong-scoped-relocate/freeze.json)
retains every raw/settled/demo hash, resolved A/B config, checkpoint and source
pin, script hashes and pair identity. Full remote manifest SHA256:
`b5f3bac375b6023ab9b23e66ae6de3fde6da113a7eaf497c49a97de20555135a`.
Operational scripts, attempt logs and full manifests remain in
`compute6:~/dojo-data/kong-1107/`.

At **23:28:21Z**, under admission then queue locks, 40 `c6b_` rows were inserted
at the queue head before 100 remaining tie rows. SEGAB had drained. Both arms
of the current tie pair were already claimed and there were no queued mates
or retries to protect; no queued pair was split. This implements the lead's
amendment to let tie run during prep rather than holding the lanes idle.
The queue before/after hashes and exact inserted rows are in `inserted.json`
and the preregistration. No queue or lane algorithm was changed.

First KONG A START: **23:31:36Z**, heavy1,
`c6b_makekongsg_kongAr1b1_100000`. Its live perception process was verified
through an explicit environment allowlist: new deployment Python,
`AGENTIC_MODEL=cx/gpt-6-luna`, private salted DOJO_DATA, expected background
and shared v2b/kong addresses. Credentials were neither read out nor logged.
Every new row pins cx; unrelated existing rows retain their own settings.

A no-model-call physics write-audit worker started at **23:28:21Z** (initial
PID 686923). It waits for original report/actions, retains their hashes,
replays original prefixes through the final write plus one step on the same
salted settled layout/demo, and snapshots every write step exactly. Admission
uses a free light lock with the same VRAM/load limits. Per-arm harmful/helpful
counts and expected/scored coverage are saved in `audit/summary.json` and
per-run `writes.jsonl`; the worker has a three-hour collection deadline.

Audit outcomes and camera verdict remain pending. Queue insertion, a live
route check and successful preparation are not evidence of a passing lever.
Record measured audit results in the preregistration before any camera verdict.

## Completed audit

At **2026-11-07 01:34:36Z**, all 40 registered arms had completed physics
write audits; the worker exited normally and released its light lock. All
20 pairs are matched. Coverage: **228/228 writes**, **0 uncertain**, 40 stable
replay resets, unchanged frozen input/checkpoint hashes, no scope mismatch
or ineligible attempt. Both guards have zero relocation writes.

The [full counts and artifacts](../experiments/2026-11-07-kong-scoped-relocate-prereg.md)
are committed before a camera verdict. Registered nearest/group counts show
B harmful-write excesses in multiple writers. The relocation writer has
30 B writes (15 helpful / 13 neutral / 2 harmful) versus none in A. Both
relocation flags select GT `mahjong8_0` for belief `mahjong7_0`; a supplemental
same-label check improves 46.31→1.64 mm with unchanged z, while the registered
neighbour association reports 44.3→66.2 mm. The flags remain unmodified and
the association issue is in BACKLOG; this diagnostic does not waive other
writers' excesses or qualify adoption. No camera verdict is issued by KONG.

Reported camera spend is 74 logical cx/gpt-6-luna calls, 763,614 prompt /
25,666 completion / 13,342 reasoning / 588,672 cache-read tokens. Prep and
replays used zero model calls. Two imitation wall caps are retained as policy
outcomes. All attempts, original actions/reports, prefix provenance and replay
GT remain under the dispatched roots. Unrelated lane work was left running.
