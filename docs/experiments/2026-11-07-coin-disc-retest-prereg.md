# COIN once-only fresh-layout camera re-test: disc hold evidence

Prospective registration before generation or camera collection. This consumes
`disc_hold_evidence=on` (#382)'s ONE permitted re-test. The first panel's
positive pooled gain qualified it: deposit_coin 12 → 26 (+14), success flips
2 / 1; all three guards identical. It missed the old net-success-win bar.
See [wave verdicts](../journal/2026-11-06-ab-wave-verdicts.md) and the
[original registration](2026-11-05-coin-disc-hold-evidence-prereg.md).
No further re-test, tuning, optional stopping or outcome-based selection.

## Arms and freeze

Reuse the original disc deployment on compute4 (haptics):
`~/dojo-integ-c4-416a2a03`, clean tracked tree at
`416a2a0374e06628350816a019862caf5156e730`. Merged #382
(`c04335990e2b36c06fa5642998c4c2908d5f59b5`) is its ancestor.
Both arms use the captured candidate `scripts/dojo/queues/2026-11-06-candidate.txt`,
SHA256 `5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e`.
A retains disc_hold_evidence=off; B changes ONLY disc_hold_evidence=on.
Keep slot_read, held_thin and every other resolved field identical.
Both pin `AGENTIC_MODEL=claude/claude-sonnet-5-5` explicitly on every row.
Use submit_v3 / RuleSupervisor, Isaac 5.1 CAMERA belief, seg v2b,
digit_v2, kong_v1b, grip v2, original background and frozen color calibration.
No implementation-agent model probes. Stop on HTTP 429 via existing lane policy.
Freeze complete effective A/B configs, queue rows, code, inputs, catalog,
backgrounds, calibration, models and versions before camera dispatch.
The final manifest and this registration must be on main before queue append.

## Sample and fresh identities

| cell | role | pairs | five-record blocks |
|---|---|---:|---:|
| deposit_coin | gain | 20 | 4 |
| insert_key | guard | 10 | 2 |
| play_tic_tac_toe | guard | 10 | 2 |
| stack_blocks | guard | 10 | 2 |

One repeat per identity, 50 pairs / 100 camera rows. Guard set follows the
original disc panel; 10 pairs per guard mirrors the peg/waist re-test's policy,
expanding the initial three-pair guards rather than weakening regression coverage.
[Literal salts](2026-11-07-coin-disc-retest-salts.json) are fixed before generation.
Private roots: `~/dojo-data/coin-disc-retest-1107/blocks/<block>`;
each uses ids 100000–04 only. Identity is task + salt + id, not numeric id alone.
Fresh salts and new roots must not collide with the historical camera registry;
raw content hashes are checked against original disc inputs before dispatch.
Use unchanged generator and default two-pass presettle for settled cells;
if a task's task_info says settled=false, use raw plus unchanged --raw-check.
Keep all attempts, preparation indices and drops. Exactly one five-record refill
block per cell is reserved; only preparation drops permit ascending valid refill
ids to restore n. If still short, block dispatch; never substitute after outcomes.
Order AB for even zero-based sequential layout index and BA for odd, per cell,
continuing across blocks. Both arms start fresh episodes on the same box.
Infrastructure failures allow one identical-input retry per arm via lane2;
record all attempts; missing terminal pairs block adoption.

## Score decision rule (7 Nov)

The gain B pooled score must be at least A + 5 points. A paired score win is
B − A >= 10 points (0.1); loss is A − B >= 10. Require wins >= 2 × losses
and wins − losses >= 2. Report success flips alongside, not as the gain rule.
Each guard must have B pooled >= A − 5 points and no more success-flip losses
than wins. Full registered n must complete. Report per-cell mean/sample SD,
successes, paired score counts, success flips, infra exclusions and spend.
Adoption also requires the standard physics-only replay harmful belief-write
audit (>2 mm added 3-D error), no excess harmful writes in B by writer.
This task freezes and dispatches only; verdict and write audit are later work.

## Lane and budget

Append complete `c4b_` rows AFTER all queued work under
`flock ~/dojo-integ-c4-state/queue2.txt.lock`; do not rewrite existing bytes.
Existing heavy1/heavy2 `lane2.sh` retain admission gates, 50 GB disk floor,
1980-second job cap / 25-minute episode cap, private locks and retry policy.
Preparation serializes through our wave-start and heavy1 locks, respects the
same VRAM/disk limits, and never opens coord. Only lane2 accesses its boot stamp.
No sealed data, reserved ids >=100100, other team's paths, system changes or
process termination. Retain GPU/occupancy logs; timing is shared-box timing.

Budget estimated only from first disc A/B call logs (`agentic/**/calls.jsonl`,
summing `calls`, never outcomes): deposit_coin 134 calls / 20 episodes;
insert_key 23 / 6; play_tic_tac_toe 12 / 6; stack_blocks 14 / 6.
For 40 gain + 20 episodes per guard: 268 + 76.7 + 40 + 46.7 = **432 logical
Claude calls** (round up to 433). Allow about 650 for adaptive variation;
one infra retry per arm could increase this further. Historical token scaling
predicts 6.17M prompt, 276k completion, 143k reasoning (included in completion),
4.95M cache-read (included in prompt). Claude has a separate weekly limit;
remaining headroom is unknown, so this is not a claim of sufficient quota.
No live calls are needed for preparation. Actual calls/tokens at dispatch and
queue positions/hashes will be appended as a receipt without computing a verdict.
