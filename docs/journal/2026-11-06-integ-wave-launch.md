# Integration wave preparation

The lead released compute4 after D-C19 and requested current main, with Claude
Sonnet 5.5 for all camera perceiver calls. These supersede the wave plan's code
pin and cx routes before collection. Main dfcdaf6f was deployed to a new directory;
the catalogue gate passed and all 316 RuntimeConfig rows validated. The candidate
SHA256 matches the prospective freeze. Active queue was empty; old deferred queue
is left untouched. Physics preparation is separate from camera dispatch.

The operational lane script had only a free-memory guard. Its versioned copy adds
used-memory admission at 16 GiB, a 50 GB disk floor, and serialized service startups.
It uses explicit merged-deployment paths and private locks; legacy reserve-file
reads are removed for this booking. No sealed artifacts or other team's directories
are required by the runner. The status tool reads launcher logs only.

Physics results, layout manifests and actual starts remain operational evidence
under `~/dojo-data/wave-1106` on compute4; no camera result is claimed here.

Lead amendments before collection: RoboDojo's standard environment device is CPU,
including historical physics runs; this startup line alone is not an invalid GPU
signal. CUDA-no-device errors remain infrastructure failures. Use one Integration
lane alongside the other team's one Isaac, and the explicitly authorized shared
`~/coord/lane_gate.stamp` boot protocol (120 s); all other coord paths stay forbidden.

For the paired tie comparison the lead amended existing M15 demo provenance before
collection: require status=ok / validation.ok, current settled layout mtime no newer
than demo pkl mtime, and freeze current SHA256 of both for identity across arms.
All 15 M15 demos pass these checks, with no exclusions. Nanosecond mtimes and SHA256s
are recorded in compute4 `~/dojo-data/wave-1106/m15-demo-provenance.json`. Historical
absence of a loaded-layout hash in demo_index does not block this paired comparison.
No scoring demos were regenerated.

## Toy physics head review

Both five-record jobs completed with rc=0, no unstable episodes, infrastructure
failures or CUDA errors. A: 3/5 successes, mean score 0.84. B: 4/5, mean 0.92;
one B-only success, zero A-only losses. The common failed id 100004 scores 0.6 in
both arms (A place_miss; B not_home). B records 63 held reads / 38 adoptions;
46 residual readings have lateral median 0.30 mm, maximum 0.62 mm, angular
maximum 1.98 degrees. Per-view attempts/adoptions: hover 50/34, look 2/1,
show 11/11. Original terminal object traces, held events and summaries are retained
under compute4 `~/dojo-data/ab-wave-1106/physics/` and summarized at
`~/dojo-data/wave-1106/physics-report.json`.

The lead approved the toy physics head: `peg_held_read` is eligible for its
pre-registered camera comparison. This is smoke validation, not camera adoption.
The interrupted first A attempt is preserved separately as
`toy-off.partial-pre-amendment`; it was stopped solely because the original plan
incorrectly rejected the normal CPU environment device. It is not pooled.

## Conveyor physics head review

All four jobs completed rc=0, five records each. In both blocks A and B each
scored 4/5 successes and mean 0.80; pooled A=8/10, B=8/10. There is one B-only
success and one A-only success in block 1; block 2 has identical paired outcomes.
All jobs report zero instability, infrastructure failures, CUDA errors and missing
meshes. Waist interception log counts are block 1 A=0/B=2 and block 2 A=1/B=1
(later attempts can use waist even with the first-choice switch off).

The lead approved conveyor camera eligibility: rare waist activation, no G3 gain,
no systematic new physical failure. Run the 30 pre-registered camera jobs; camera
adoption remains subject to its paired decision rule and write audit.

The lead then authorized immediate one-lane camera dispatch after merged #392,
in plan order while skipping families whose preparation is incomplete, to revisit
later. Conveyor measurement RAW checks take priority; the other salted records
remain generated and hashed while their settling/demos await their own booking.

Before any waist camera data, required original RAW checking rejected gain id
100008 as `dropped: raw record not at rest`. The lead prospectively amended the
family: exclude 100008; try unused measurement ids 100025 through 100029 in that
order with the same raw check, and take the first passing record. Record all tried
ids and results. If none passes, run nine gain pairs with the same +5-point score
and wins/losses rule, stating n=9. No waist outcomes have been collected, and no
threshold or physics setting changes. Launch the next ready family (peg) meanwhile.

The first replacement candidate, 100025, has no prior waist A/B name in our
camera registry and passed the unchanged RAW check (`raw ok`). No later candidate
was tried. Gain n remains ten. The derived dispatch schedule replaces only the
100008 pair, retaining its queue position and applying the existing parity arm
order to 100025. The original pre-registration files remain unchanged; the amended
schedule and queue are `~/dojo-data/wave-1106/wave-amended.json` and
`camera-amended.txt` on compute4. No waist camera episodes existed at amendment.

## First camera dispatch

Merged runner PRs #391/#392 are deployed at
`~/dojo-integ-c4-416a2a03`, SHA `416a2a0374e06628350816a019862caf5156e730`.
Catalogue and full candidate/package parity pass. The prior sealed deployments
were not modified. Sixty peg rows were appended under queue2's existing lock;
no old deferred rows were appended. One lane started
`c4b_playstackingtoysg_pegAr1_100005` at 05:22:19Z, using Claude Sonnet 5.5,
the frozen calibration/backgrounds and healthy shared v2b/digit_v2/kong_v1b
services. The first job completed rc=0 at 05:28:04Z (346-second occupancy),
success, strict score 1.0; this single arm is not comparative evidence.
Its actual spend is four Claude calls, 51,201 prompt / 3,022 completion tokens
(1,489 reasoning, 44,475 cache reads). No 429 occurred.

The complete peg manifest freezes 30 input identities and verifies both arms'
loaded/raw hashes before dispatch:
`~/dojo-data/wave-1106/layout-manifest-peg.json`, SHA256
`5fbbf5522c69f1656477fcaa889a3855c6148382b4439ff52ff57b1d59b47036`.
The lane pauses after its first job for the serial conveyor preflight reservation,
then resumes its unchanged queue unless global 429 STOP is present.

All 57 prospective salted records were generated and hashed: ten physics RAW
records validated and the 47 fresh camera records remain unsettled, with their
17 fresh demos still pending. Per the lead's skip-unready amendment, relocate,
key and tie are deferred until that preparation is booked; no replacement layout
selection or synthetic settled/demo records are used.

Conveyor's five image guard records all passed RAW validation. Peg B resumed at
05:33:33Z with unchanged inputs and queue order. Ready waist/disc manifests and
queues are frozen separately; the full launch evidence and preparation shortfall
are recorded in [the launch report](../experiments/2026-11-06-dojo-ab-wave-launch.md).

## PREP amendments before reloc/key/tie collection

PREP attempted the 47 fresh camera settles on the pinned 416a2a03 deployment,
holding the heavy1 lane lock for every job and honoring the shared 120-second
boot stamp, 16 GiB used-VRAM admission and 50 GB disk floor. Forty-two passed.
Insert-key records key-gain1/100000,100001 and key-gain2/100001,100002,100003
failed the unchanged settled-record stability check and are excluded.

Before camera collection for these families, the lead authorized a bounded key
refill: generate key-gain3 (insert_key, 100000–04), select valid ids in ascending
order to restore ten gain pairs, and try key-gain4 only if still short; then stop
and report. Guards are unchanged. Salts were fixed before generation:

- `queue-wave-20261106-key-gain3`, SHA256
  `a096bcb74f74544545e69be2141eecc3c12720aad8c193b0d814d9e64fd80053`.
- `queue-wave-20261106-key-gain4`, SHA256
  `57021c3517641457453f7d465ad62a14b90adff3a4ef5107be3869915ea94f79`.

The lead also amended failed/unattempted fresh imitation demos to `--speed 1.5`,
matching the prior dev generation recipe. Keep completed validated speed-1.0
demos; record speed per record and use the identical pkl for both camera arms.
Existing M15 scoring demos remain unchanged. Exclude remaining failures and, if
needed, refill from one additional salted block of the same task, lowest valid
ids first, then stop. The original speed-1.0 attempts and indices are retained.

The lead approved correcting eight tie-primary queue rows whose DOJO_DATA
pointed to tie-guard1 despite loading tie-primary1 layouts. The staged tie queue
now uses the matching tie-primary1 root for demo lookup. All 188 original queue
RuntimeConfigs and 94 registered pair deltas validated. PREP does not append to
queue2. Timestamped amendments, chosen salts, exclusions and attempt logs live
under compute4 `~/dojo-data/wave-1106/`, including
`prep-lead-amendments.json` and `prep-amendment-journal.md`; these amendments were
recorded there before amended preparation. Final refill/demo results and
manifest hashes remain pending.

The two authorized key refill blocks each yielded two stable records: gain3
100000/100003 and gain4 100001/100004. Before any key camera data, the lead then
authorized **nine gain pairs**, keeping the five original valid records and these
four additions, with all ten charger guard pairs unchanged (38 camera jobs).
The decision rule is unchanged: pooled B at least A + 5 gain-cell points,
success flips wins at least twice losses with at least two net wins, and the
existing EXPERIMENTS guard bound. State n=9; no further generation is authorized.
