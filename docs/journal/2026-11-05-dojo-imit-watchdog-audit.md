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

The lead supplied the board source: local `~/dojo-state-backup/m95/cells.md` (30 Oct 19:48Z) and `cells.py`.
Read the selector as code only, never executed it or loaded its input datasets. Lines 348-409 reject dev/renamed/
invalid-code rows, infra and unstable results, then keep the newest valid attempt per layout/family. Lines 614-638
exclude the unstable imitate layout explicitly, select **o2owait**, retire the other historical imitate suffixes and
rank the candidate at n>=5. This verifies the family and exclusions; `cells.md:3` confirms 0/8 / 8.8.

Crucially, an aggregate join by directory layout key finds **9 distinct o2owait layouts in 11 attempts**. Both infra
attempts have a later noninfra attempt; **zero infra-only layouts remain**. Eight distinct usable rows plus one
unstable layout explain n=8 completely. This join projects the same authorized four fields and emits counts only,
never per-id diagnoses. The census tool includes these distinct-layout counts to avoid mistaking retried attempts
for lost score coverage.

## Expected effect

**Zero direct board effect from fixing the two observed historical watchdog/infra attempts:** their completed
retries already supply the counted results. Dropping infra zeros cannot raise 8.8 because those zeros are already
absent. The 44.4% dev attempt rate and the candidate's 18.2% attempt rate measure operational cost, not missing
counted samples. There is no evidence here that watchdogs explain the 73.25-point cell gap. This does not rule out
unobserved future infrastructure failures or infrastructure-induced changes in completed runs.

No measured score gain is claimed for this reporting PR. A robustness fix may save latency and duplicate GPU work,
but cannot be scored as recovering the existing 8.8-to-82 gap. Matching the G3 ceiling on the eight counted rows would
still require ~73.25 / 30 = **2.44 board points** of improvement in completed episodes; camera and G3 references use
different layout sets, so that ceiling is an opportunity estimate, not a paired causal effect.

## Validation

Two census tests pass locally and on compute3: sensitive fields do not escape, retries aggregate, archives remain
separate, and forbidden slice/family/symlink files containing invalid JSON are never opened. The script ran via
stdin on compute4 and reproduced the aggregate table without writes. Focused census + Integration tooling tests: 11 passed locally, census tests 2 passed on compute3. Isolated deployment
`~/controlr-imit-watchdog`; outputs `~/dojo-data/imit-watchdog`. One physics-only Isaac job at a time under the existing
`~/dojo-gpu.lock`, 123 GB free (10 GB floor). Dev 100003 reset returned in 3.79 s and execution reached 1408 steps,
then was excluded as unstable_layout (0 departures seen). No 300-s reset stall reproduced; this is not a success or
a render validation. Process returned rc=0 and the shared lock was available afterward. No persistent server started.
No motion/perception fix is offered without a causal reproducer.

Compute3 positive control: dev 100002 with one_to_one / wait home / keep_sides completed successfully, score 100,
1408 steps, wall 80.3 s, no infra/instability. Explicit disk guard passed (123 GB free); process rc=0 and shared lock
available after exit. This validates the physics/demo path only; it neither exercises nor repairs RTX capture.

Full local suite: **1745 passed, 187 skipped**, 1095.65 s (`uv sync --extra dev`, then
`uv run python -m pytest -q`). Initial bare `uv run pytest` used the system pytest before the dev extra was installed
and failed collection; the properly provisioned full run above passed. Changed-file pyflakes and diff whitespace
checks pass. Final reporting code `ae17b122` was redeployed in isolation and its two census tests passed on compute3.
