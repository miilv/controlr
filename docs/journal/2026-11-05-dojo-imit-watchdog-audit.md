# IMIT round 2: watchdog ownership and candidate infrastructure census

**Result:** the watchdog is ours, but the available evidence points to a Kit/rendered-driver stall, not an identified
repairable policy bug. The candidate family reproducing 8.8 already excludes infrastructure attempts. No watchdog,
controller, perception, default, or timeout change is justified by this audit. PR scope is diagnosis and a reproducible
aggregate-only reporting tool; it does not claim a robustness fix or board gain.

## Access and reproducibility

Read only old compute4 `~/dojo-data/integ-c4/runs2/c4b_imitatesg_*`. Deep dives select dev 100000-04 by directory
name before opening a file. Measurement 100005-09 / 100020-29: project only category, success, rc, steps_used from
JSONL; summary headers supply n / score / unstable. No individual measurement diagnostics, actions, reports, layouts,
demos or reasons inspected. No ids >=100100, sealed paths, compute4 writes, GPU use or process inspection. No LLM calls
on any route (0 tokens / spend). Local environment occupies 96 MB; no dataset/model/video downloaded.

`scripts/dojo/imit_infra_census.py` reports attempts by exact candidate suffix and archive status. It includes retries,
keeps `.codebug` / `.layoutfreeze` separate, rejects other slices/families before opening files, skips symlinks and
emits no ids, scores or episode diagnostics. It is a census, not a board selector. Reproduce without remote writes:

```bash
ssh compute4 python3 - /home/haptics/dojo-data/integ-c4/runs2 < scripts/dojo/imit_infra_census.py
```

## Which watchdog and phase?

`scripts/dojo/integ_eval.sh` monitors its own systemd scope every 20 s, tracking the count of `[integ]` lines plus the
last RoboDojo `env0 step: N /` in the driver log. Defaults:

- `INTEG_WATCHDOG_S=300`: no progress and either a new zenity/display error or scope CPU <0.5 cores.
- `INTEG_WATCHDOG_HARD_S=600`: no progress regardless of CPU.
- TERM scope, grace 90 s, then SIGKILL remaining scope PIDs. Append synthetic `infra_failure` with **steps_used=0**,
  score=0, wall_s=0; exit 4. The zero is not an observed step count. The queue retries layouts without results once;
  it does not automatically retry the frozen layout, which now has a result.

This is neither RoboDojo's step budget nor our driver `--episode-max-minutes` (25 min, checked between loop turns).
It also is not the official-client 900-s timeout / DEMO_KEY problem documented in the rehearsal journal.

All 12 round-1 dev infra rows fired the **300/301-s zenity branch**, despite scope CPU 3.13-11.73 cores. Every log
repeats `Failed to open display` at ~120-s intervals after the last progress. No low-CPU or 600-s hard-limit firing.
Nine have no env step: dev 100003 (base/retry, home/retry, smask) and 100001 (wid/retry, wid_r2/retry). Their driver
has completed scene setup, logged reset and `[render_sync] zero-delay Kit settings active`, then stalls. Three are
mid-episode: dev 100004 base at 1015, r2 at 1011, r2 retry at 879. No report was saved for the interrupted episode.

The perception and twin service logs have listening/ready markers and no Python traceback or fatal error. No bridge
is used on this Integration path. The last log evidence is the RTX-enabled RoboDojo/Kit driver; the no-step cases
narrow it to reset / initial rendering before the first action. The three later stalls cannot be localized to
`render_for_capture`, `get_obs`, a Kit step or policy RPC from these logs alone. CPU is for the whole scope, not the
driver; a listening service is not proof that no later request blocked. **Zenity is a symptom, not a causal stack
trace.** The archives do not establish GPU OOM, a dead process, corrupt demo replay or a particular native call.

Ownership: watchdog/launch integration ours; environment and renderer RoboDojo/Isaac. The underlying fault owner
remains unresolved. Increasing timeouts or suppressing zenity is not a demonstrated repair. G3 cannot validate an
RTX fix. Next causal experiment requires a released genuine-5.1 rendering host on dev ids, bounded Python/native
stack capture around reset/observe/step and per-process Kit logs/VRAM. Do not rerun measurement ids to diagnose it.

## Candidate census (aggregate only)

Active directories, including retries; archived `.codebug` / `.layoutfreeze` excluded below. These are attempt counts,
not independent layouts. All successes are zero; rc is absent on every row (do not infer rc=0).

| suffix | attempts | perception_gross | infra_failure | unstable_episode | plan | time_cap | infra rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| gcon | 13 | 12 | 0 | 1 | 0 | 0 | 0% |
| o2o | 7 | 4 | 1 | 1 | 1 | 0 | 14.3% |
| o2otwin | 8 | 5 | 2 | 1 | 0 | 0 | 25.0% |
| o2owait | 11 | 7 | 2 | 1 | 0 | 1 | 18.2% |
| o2owaitlook | 10 | 7 | 2 | 1 | 0 | 0 | 20.0% |

Every infra row has steps_used=0. Category alone cannot distinguish a watchdog from another infrastructure failure;
confirmed watchdog rate on measurement rows is **not identifiable** within the authorized fields. Infra rate is its
upper bound. Do not transfer the dev 12/27 (44.4%) rate to these rows.

Other ablation suffixes: bundleoff 2, bundleon 2, bundlewid 1; all perception_gross, no infra. Archived layoutfreeze:
gcon 4/4 infra, o2o 3/4 infra + 1 perception_gross, o2otwin 4/4 infra. Archived codebug: o2owait 1 error + 1
perception_gross; o2owaitlook 2 errors. They are not evidence about the active candidate's rate.

Summary headers (weighted by n, excluding unstable rows; watchdog attempts have no summary): gcon n=12 mean 2.083,
o2o n=5 mean 2.0, o2otwin n=5 mean 5.0, **o2owait n=8 mean 8.75**, o2owaitlook n=7 mean 3.571. This independently
reproduces the supplied 8.8 using o2owait. The counted eight have 7 perception_gross and 1 time_cap; 0 infra / 8.
The complete o2owait attempt family has 2 infra / 11 and 1 unstable / 11. `perception_gross` is a coarse scorer
category, not a diagnosis of each episode's causal failure.

The checked-in journals explain historical families and v15 replacement at n>=5
(`2026-10-29-dojo-imitate-twins.md`, `2026-10-29-dojo-v15-measurement-layouts.md`), but do **not** contain the current
board selector or a rule choosing o2owait over o2owaitlook. Asked the lead for that source. Consequently the exact
external selector remains unverified; the numerical reconstruction is strong evidence for the counted family, not
proof of the board's general selection rule. No forbidden coordination files were used to fill this gap.

## Expected effect

Removing infra zeros from the reconstructed counted mean has **zero effect**: they are already absent. Infra can
bias sample coverage, but does not arithmetically explain 8.8 vs 82. A pure fix to these missing attempts has no
measured gain yet. As a sensitivity calculation only, if two newly completed, distinct samples both scored 82,
(8 * 8.75 + 2 * 82) / 10 = 23.4; delta 14.65 cell points, ~0.49 board points at Memory weight 1/30. If both scored
100 the mean would be 27.0 (~0.61 board points). These assume both attempts are distinct missing samples and the
selector includes both; retries/overlap or different outcomes invalidate that calculation. They are not forecasts.
Achieving 82 on the existing counted sample would instead need ~2.44 board points of policy/perception improvement.

## Validation

Two census tests pass locally and on compute3: sensitive fields do not escape, retries aggregate, archives remain
separate, and forbidden slice/family/symlink files containing invalid JSON are never opened. The script ran via
stdin on compute4 and reproduced the aggregate table without writes. Focused census + Integration tooling tests: 11 passed locally, census tests 2 passed on compute3. Isolated deployment
`~/controlr-imit-watchdog`; outputs `~/dojo-data/imit-watchdog`. One physics-only Isaac job at a time under the existing
`~/dojo-gpu.lock`, 123 GB free (10 GB floor). Dev 100003 reset returned in 3.79 s and execution reached 1408 steps,
then was excluded as unstable_layout (0 departures seen). No 300-s reset stall reproduced; this is not a success or
a render validation. Process returned rc=0 and the shared lock was available afterward. No persistent server started.
No motion/perception fix is offered without a causal reproducer.
