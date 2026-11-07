# SAMINT egg camera A/B

Gain cells: fill_egg_holder. Design: [data/samint/egg.json](data/samint/egg.json).

These are PARKED dispatch specifications, registered before generation or collection. No queue rows have been sent.

A is the frozen candidate `scripts/dojo/queues/2026-11-06-candidate.txt` (SHA256 5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e),
with `DOJO_PERCEPTION_CFG={"series_count":"seg","glyph_read":"on","pad_row_extend":true}`.
B is A with only the task routing map in the linked design JSON added to that perception JSON.
Both arms use the same merged SAMINT code (including #417 identity instrumentation), submit_v3,
grip v2, v2b/digit_v2/kong_v1b, candidate skill parameters, host-mapped frozen colour calibration,
backgrounds and catalogue; pin their hashes and deployed commit in the manifest before dispatch.
Use `--sam sam3.1_text` in **both** arms, so they share the same resident side service and contention.
A makes zero SAM inference requests. The raw task map is the lever; compare its *resolved* inactive
route, rather than raw config JSON, for parity guards.

Each gain cell uses all **20 pairs**, IDs **100000–100009 and 100020–100029 only**,
on the new salt in the design JSON. One repeat/layout; alternate AB/BA by ascending ID.
No sealed data, no IDs >=100100. Generate with `dojo.simlab.gen_layouts --seed-salt` and
two-pass physics-only settling, select solely on stability; save every attempt. For imitation,
validate demos at speed 1.25, retry invalid demos once at 1.5. If fewer than 20 stable paired
layouts exist, park the cell; no efficacy selection or recycling another salt.
Commit design, exact layout/demo hashes, preparation/deployment provenance and final queue manifest
before the lead dispatches. All cell/arm layouts are identical and SHA256-verified on the target box.

Strong guard: **play_Xylophone**, 10 pairs, first ten stable owned IDs on this spec's salt;
the frozen candidate has >=90% historical success. It retains raw arm configs and is camera-gated.
Parity guard: **classify_objects**, 5 pairs, first five stable owned IDs on the same salt.
Require identical resolved estimator config (excluding inactive task-map entries), identical skill
parameters, zero SAM requests/actions and zero SAM-origin belief writes on *every* guard episode.
Report its scores but gate only parity; the strong guard still has the ordinary score/flip gate.

Every row: `AGENTIC_sam3.1_text=claude/claude-sonnet-5-5`, renderer isaac51, candidate step limits,
25-minute episode/job caps, native watchdog600, existing lane watchdog and infrastructure requeue.
Pin and verify actual model route from transcripts. No new policy switches or dynamic defaults.
Use saved looks and images; no LLM calls in preparation or physics-only audits.

Qualification requires **both** current EXPERIMENTS.md gates. On gain cells B-A pooled score >=5 points,
per-pair score wins (delta >=0.1) >=2 times losses (delta <=-0.1), and >=2 net wins.
Report n, mean, SD, range, paired deltas, score wins/losses and success flips per cell, plus pooled results.
Strong guard: B score >= A-5 points and success flip losses <= wins. Safety incident fails the batch;
HTTP429 stops dispatch. Budget endings, policy failures and wall caps are outcomes. Only proven
infrastructure failures are excluded/requeued, symmetrically and reported. No efficacy early stopping.

This is a perception lever: run `scripts/perception/m95_score.py audit` on physics-only replays
of **both** arms, with #417 `identity_marks` / first-consumed pick or grasp-close identity,
provisional unambiguous <=10mm associations, persistent GT-instance links and fallback counts.
Harmful/helpful means >2mm 3-D error increase/decrease; report z separately. Per writer on active
gain cells, B harmful/writes <= A harmful/writes +5 percentage points when A writes; for a writer
new in B require helpful >=2*harmful. Report counts, denominators, unassociated/phantom writes,
identity changes, tiny misses, consumed identity/pose tolerance and mask phantoms. Do not treat
prompt categories as asset identities. Parity guard writes are reported under parity proof.
Generate a success and typical failure video after collection; report all actual calls/tokens/spend.

**Overlap:** run after the queued IMIT2 three-arm camera A/B and POUR pacing re-test complete.
Never insert SAM into those arms. Freeze the current candidate stated here for this test; if later
adoptions are to be stacked, preregister a new candidate pin before generation, with SAM as the
only A/B difference. Queue manifests stay out of active lane state until the lead allocates time.

Budget: 70 episodes, **about 210 calls** at 3 calls/episode planning mean; expected **6.7 lane-hours**
(450s/gain episode, 160s/Xylophone, 300s/classify; two lanes roughly half, plus prep/queue delay).
The 25-minute cap gives a conservative ceiling of 29.2 lane-hours. Audit/prep costs zero calls.

Call figures are estimates, not enforced caps: retain A’s existing agentic budgets in both arms.
`agentic_max` counts estimates, not LLM turns; one estimate can use multiple calls.
Reserve up to 16 calls/episode (8 turns x 2 estimates) for planning, and report held-look
requests separately; pause for lead budget review if observed consumption exceeds that reserve.
