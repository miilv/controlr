# Prospective headset camera A/B — blocked, not dispatched

## Mechanical and observer prerequisites

This is a future experiment specification, not a dispatch request or board
selection. No camera jobs, rendering or model calls are authorized. Actual
calls/tokens/spend remain **0/0/$0**. Compute3 driver595.99 cannot run the
Isaac renderer; the lead must separately assign a compatible rendering host.
No compute4 access is implied.

Mechanical qualification is complete on source **10f3e0a8**:
[round2h](2026-11-07-laptop-g3-round2h.md) runs ten fresh pairs per cell,
A0.00 ±0.00→B18.00 ±6.32 points in both. Headset credits0/10→9/10 each;
9 wins/0 losses/1 tie each, full-task successes0. All40 rows valid. Both
7/10-per-cell and paired-gain rules pass, with no older outcomes pooled.
This clears only the mechanical prerequisite.

The remaining observer blocker is **fresh physical held-pose evidence**. The
loaded proof samples object pose while the robot holds, raises and returns
it. Current cached runtime belief cannot establish that following motion;
a pose projected from commanded joints or an assumed rigid attachment is
also not evidence of a hold. G3 mode A used actual simulator observations.
This PR does not provide a camera observer or claim that GT transfers to it.

Before any camera dispatch, separately implement/qualify a held observer and
pin the complete source, observer/model/asset/calibration versions and baseline
board-candidate config hashes. Every object sample consumed by the proof must
be fresh physical image evidence, or explicitly unavailable/rejected. Test
stale and kinematically projected beliefs, missing reads, identity mistakes,
sliding and falling objects; none may establish a hold. Preserve the physical
proof limits and original hard slip/lift gates. Use GT only for offline
measurement/scoring, never policy feedback or belief correction.

The current helper has no timestamp/provenance contract that rejects a
projected held belief. That integration and its mocks/physical validation are
a release prerequisite, not an implementation already present in10f. If it
changes controller behavior, freeze/requalify the complete candidate and amend
this spec **before** camera outcomes. No unresolved pin may be filled from
results. Keep the lead's board selection unchanged throughout.

## Conditions

A uses the lead-pinned actual board candidate. B changes the headset mechanical
bundle only, plus any explicitly declared observer integration required above:

```yaml
hang_cradle: straddle
hang_wide: 'on'
headset_seat: 'on'
headset_carry: staged
headset_grasp: cups
headset_table: mesh
headset_stage: mesh_clear
headset_retry: sites
headset_lift: slow
headset_hold: loaded
```

The same pinned observer implementation, perception models and calibration
must be available to both conditions; any extra reads/actions triggered by B
are part of its candidate and must be counted. Do not describe a perception
change as a pure motion ablation. Record exactly what A inherits rather than
silently enabling new flags in its baseline. G3 held straddle common in both
arms; this camera trial compares the full bundle against the actual baseline.
No laptop route, model/planner substitution, joint-limit, task-budget or board
change. Every new field retains its off/legacy default. Tolerant relays keep
their original path; threaded mug hooks do not acquire headset-only behavior.

## Fresh paired sampling

Gain cells: store_laptop_and_headphones and its random cell, ten pairs each.
Guard cells: hang_mugs and hang_mugs_random, five pairs each. Total30 pairs /
60 episodes. Use a separate camera measurement namespace and never-used salted
records, not sealed sources or the G3 settled files. Proposed v15 variant22,
salt `laptop-camera-strictload-prospective-2026-11-07-88c48b74`; gain ids
100000–09, guard ids100020–24. Confirm ownership/freshness before generation;
if any proposed record exists, amend the entire draw before looking at outcomes.
Settle once and freeze both cell/layout hashes before either condition.

Each A/B pair uses identical settled bytes, source, host, renderer/perception
pins and step budget. Alternate A/B order by selected index. Predeclare
infrastructure-only retry/reserve rules before dispatch; retain every attempt,
never replace a completed policy zero or tune from individual panel failures.
Use the official final partial score, not the skill's internal result category.
Report actual n/exclusions, A/B mean/sample SD on0–100, headset credits, full
task successes, score wins/losses/ties and success flips per cell. Keep G3 and
all earlier panels separate; do not pool selected successful rows.

No calls are budgeted by this document. Inventory the pinned perception/held
observer request path, estimate60 matched episodes, and obtain separate lead
authorization before any model spending. Retain actual attempts, calls, tokens,
latency and spend, including failed/extra reads. Never use agent-session keys.

## Decision and write audit

Gain cells pooled B−A>=5 points; score wins (>=10-point increase)>=2*losses,
with>=2 net wins. Report each cell independently and the7/10 headset-credit
reliability diagnostic. Guards: B mean no more than5 points below A and no
excess full-success flip losses. Defaults stay off and board unchanged unless
subsequently selected through the normal review process.

Run the paired belief-write audit on both camera arms. Require no excess
harmful writes by writer in B. Audit actual read availability/provenance,
projected-versus-observed held poses, extra observer actions and occlusions;
unknown evidence is not a negative read. Include the known headset00002/00004
identity confusion and articulation-root plumbing. Never feed audit GT back
to the policy. On the compatible host, retain a success and typical failure
video plus numeric traces. Nothing here dispatches those jobs.
