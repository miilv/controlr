# PLAN CPU routes: prospective egg camera comparison

No camera inputs or outcomes were used to write this specification. This is a
plan for lead review, not a camera dispatch or permission to promote.

## Candidate and frozen arms

G3 screen: [ten pairs](2026-11-06-plan-world-g3.md), egg score 23.5→28.5,
five score wins / one loss, paired delta SD 12.91. Both arms have zero full
successes. This is a partial-score signal with substantial uncertainty.
Tools loses 2.5 points; stack holds 100.

Run the same final PR source in both arms, with RuntimeConfig.plan_solver
explicitly frozen to routes:
- A: plan_world = table.
- B: plan_world = objects.

All other RuntimeConfig fields, models/effort, belief writers, asset/catalogue
hashes, calibration, step limits and camera settings are identical and frozen
before outcomes. Keep all egg grasp/wall/drop switches off. Use the lead's
approved non-Claude camera stack and retain its complete resolved configuration
and hashes; reject any resolved Claude route. Neither arm may receive GT poses,
mode-A refreshes or checker feedback as perception. Use the official bridge and
real Isaac rendering. Dynamic cuRobo is a separate candidate: never substitute
plan_solver = curobo into this registration.

## New layouts, selected before outcomes

Gain cell: fill_egg_holder, ten distinct paired settled v6 draws.
Use only owned base IDs **100000–04**, with fresh variants v=1..50.
Never use 100005–09, 100020–29, any ID >=100100, or another team's
100010–19 / 100050–99 rendered namespace. G3 originals are not camera inputs.

Use the archived v6 generator 0434230e:dojo/simlab/gen_layouts.py. Replace only
its RNG seed expression seed * 1000 + attempt with the first four SHA-256
bytes, big-endian unsigned, of:
PLAN-camera-routes-2026-11-06-r1:{task}:{id}:{variant}:{attempt}.
Keep original v6 sampling and placement semantics. Generate in ascending
(variant, id) order and select the first ten passing the unmodified two-pass
settling check. Selection sees settling only, never skill/camera scores.
If fewer than ten survive 250 draws, report sampling failure.

Freeze source/config, exclusions and ten (variant, base ID, SHA-256) records
before either arm runs. Stage variants in separate private roots; pair by the
full tuple. No tuning or rehearsal on selected draws before the A/B.

Guard cells: stack_blocks and store_tools_in_toolbox, ten pairs each. Use their
normal v15 generator with the same salt protocol, owned IDs and first-ten-stable
rule. Freeze separate manifests before outcomes. Do not reuse G3 records.

Alternate A/B and B/A ordering by pair index. Record infrastructure errors;
rerun only infra-invalid pairs, both arms, retaining invalid attempts.
Do not increase step limits, pool away a negative guard or tune between pairs.

## Decision rule

Use the repository default, unchanged:
- Gain: pooled B score at least 5 points above A; success flips wins >=2×losses,
  with at least two net wins, at least ten pairs.
- Each guard: B score no more than 5 points below A; flip losses <= wins.

Report per-pair success/score, paired spread, steps, planning time, route/fallback
counts, failures and all calls/tokens. Report partial-score gains even if the
full-success gate fails; do not relax that gate after results.
Run the harmful-belief-write audit on both arms' physics replays and retain
writer-level counts, since changed trajectories can change downstream views.
No excess harmful writes is an additional promotion requirement.

A rejected positive-delta lever gets at most one freshly registered retest on
layouts never used by it; never replay rejected camera IDs.

## Budget and disposition

10 gain + 20 guard pairs = 60 rendered episodes. The lead must freeze the
non-Claude policy's per-episode call cap and estimate maximum calls/tokens
before dispatch. This stream has made zero model calls.
Keep this specification undispatched pending lead review of the weak G3 signal
and separate dynamic-planner investigation. No default changes.
