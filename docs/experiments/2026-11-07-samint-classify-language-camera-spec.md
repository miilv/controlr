# SAMINT classify-language camera A/B

Gain cells: classify_objects_by_language. Design: [data/samint/classify-language.json](data/samint/classify-language.json).

These are PARKED dispatch specifications, registered before generation or collection. No queue rows have been sent.

A is the frozen candidate `scripts/dojo/queues/2026-11-06-candidate.txt` (SHA256 5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e),
with `DOJO_PERCEPTION_CFG={"series_count":"seg","glyph_read":"on","pad_row_extend":true}`.
B is A with only the task routing map in the linked design JSON added to that perception JSON.
Both arms use the same merged SAMINT code (including #417 identity instrumentation), submit_v3,
grip v2, v2b/digit_v2/kong_v1b, candidate skill parameters, host-mapped frozen colour calibration,
backgrounds and catalogue; pin their hashes and deployed commit in the manifest before dispatch.
Use `--sam sam3` in **both** arms, so they share the same resident side service and contention.
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

Every row: `AGENTIC_MODEL=claude/claude-sonnet-5-5`, renderer isaac51, candidate step limits,
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
Reserve 24 calls/episode (8 turns x 3 candidate estimates) for planning, and report held-look
requests separately; pause for lead budget review if observed consumption exceeds that reserve.

## Lead-authorized compute4 preparation and queue amendment

The lead authorizes **pour → classify-language → egg** on compute4, appended behind every
remaining `kongre` row. This prospective authorization supersedes this spec's PARKED label
and its blanket wait for IMIT2/POUR; neither earlier experiment's arms are modified or mixed.
Imitate and conveyor SAM specs remain parked for IMIT2/MATCH2. Only queue preparation/insertion
is authorized while compute4's intentional global STOP is set; the lead alone resumes lanes.
No service, simulator or camera job is started on compute4 by this dispatch preparation.

Camera source is merged #424, **3e08be8f6ba460abbe0b582cdad4eff1dec89053**, independently
deployed to `compute4:~/dojo-integ-c4-3e08be8f`. Preparation code is merged main
**f2c7d9962335fe04716073071f65d546c47e1aa1**, deployed under compute3's own controlr tree;
no generator, settling settings or camera candidate changes are introduced by that prep pin.
Input root: `compute4:~/dojo-data/samint-1107/classify-language/data`. Salt/owned IDs, 20 gain pairs,
10 Xylophone strong pairs and five classify parity pairs remain exactly as designed.
Default two-pass presettle, validity-only selection and all failed attempts are retained.
Any insufficient panel parks; no silent reduction, new IDs or score-dependent selection.

Both arms explicitly set **AGENTIC_MODEL=claude/claude-sonnet-5-5**. The original template's
`AGENTIC_sam3` / `AGENTIC_sam3.1_text` spelling was a documentation substitution error;
those names are not valid route settings and never enter queued rows. Queue validation
checks the exact variable on every row, then collection must verify actual call metadata.
The primary offline audit is fixed to #417 merge **77b4571907f6ce49eec11c0685db8f0c7ba05d64**;
nearest-GT remains supplementary. Apply the current refined per-writer gate to the gain
cell, ordinary strong guard gate, and post-collection config/Skills/zero-request/write
parity proof on classify. No changed audit threshold or retrospective instrument choice.

One shared checkpoint serves both lanes: SAM3 for pour/classify-language, then a **drained
switch** to SAM3.1 text for egg. Existing seg/glyph services and their venv remain unchanged.
The lead-run service helper reuses the pinned lane's exact cap and `boot_gate` definitions,
including the sole `~/coord/lane_gate.stamp` access and 120-second protocol. No other coord
path is accessed. Shared SAM counts once in `sides()` and actual nvidia-smi admission;
the measured opt-in reserves 12 GiB for each SAM camera lane, including the inner launch gate.
The helper refuses to start while the global STOP is set. It never removes a STOP.
The old KONG rows remain intact: absent SAM at the first SAMINT row yields a safe requeue/stop,
then the lead starts the shared service and resumes both lanes. Egg's checkpoint mismatch
likewise stops/requeues before execution until the lead performs the drained switch.

A frozen manifest, package/source/asset hashes, canonical configs and Skills snapshots,
whole-pair schedule and 70-row file will be committed on main **before** the locked append.
Insertion receipts record positions and queue hashes; STOP stays set throughout insertion.


Parity proof must cover all attempted vocabulary estimates, including every
`vocab_explain.candidates` entry, not only the final selected vocabulary. Resolve
the frozen routing map for each candidate name. If any guard candidate resolves
to active SAM, zero-request exclusion is not proven and parity fails; never hide
an exploratory SAM call behind an inactive final route or a discarded record.
The candidate's vocabulary-explain policy remains unchanged in both arms.
