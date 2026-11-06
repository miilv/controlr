# Post-D-C19 wave launch and physics head

**Question:** can the pre-registered camera wave be dispatched with validated physics,
frozen inputs and safe compute4 sharing? Camera adoption results remain pending.

**Setup:** [prospective plan](2026-11-06-dojo-ab-wave.md), captured CANDIDATE SHA256
`5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e`.
Physics/generation pin `dfcdaf6f3e27839b443e96b2ed2e9ee258d592ac`; camera/runner pin
`416a2a0374e06628350816a019862caf5156e730`. No controller/runtime code differs between
the two pins. PRs [#391](https://github.com/miilv/controlr/pull/391) and
[#392](https://github.com/miilv/controlr/pull/392) add operational admission/status tooling.
Catalogue and full candidate/package parity pass on the final deploy. The unit suite
passed 1,781 tests with 187 skips; both runner PRs have green CI.

## Prospective amendments

The lead fixed these before collecting the affected camera data:

- Both camera arms use the candidate's `claude/claude-sonnet-5-5` route.
- RoboDojo's standard CPU environment device is valid physics; CUDA-no-device errors
  still invalidate a job. An interrupted pre-amendment toy A attempt is retained separately.
- One Integration call at a time alongside the other team's single Isaac. Admission checks
  used VRAM <=16,384 MiB and the explicitly authorized `~/coord/lane_gate.stamp` age >=120 s;
  the cap is rechecked at stamp claim. A 50 GB disk floor and HTTP 429 STOP remain active.
- Existing M15 imitation demos are accepted with validation/status ok, current layout mtime
  <= demo pkl mtime, and frozen SHA256s of both. All 15 pass, with no exclusions.
- Original conveyor gain id 100008 failed RAW validation (`not at rest`). Before any waist
  camera data, the lead authorized trying unused measurement ids 100025–29 in order,
  using the first passing record (or n=9 if none passes). Only 100025 was tried: no prior
  waist registry name, RAW check passed. Gain n stays ten. The replaced pair retains its
  queue slot, with the usual parity arm order. All five image-conveyor guard RAW checks pass.
- Dispatch may skip families whose preparation is incomplete and pick them up later.

The original prospective schedule stays unchanged in git. Derived schedule/queue files on
compute4 are `~/dojo-data/wave-1106/wave-amended.json` and `camera-amended.txt`.

## Physics head

All six jobs completed rc=0, with no instability, infrastructure failures, CUDA errors or
missing meshes. Score spread below is sample SD across the five dev records, not a CI.

| Job | n | Successes | Mean score ± SD | Gate review |
|---|---:|---:|---:|---|
| toy off | 5 | 3 | 0.84 ± 0.22 | baseline |
| toy on | 5 | 4 | 0.92 ± 0.18 | lead approved camera eligibility |
| waist block 1 off | 5 | 4 | 0.80 ± 0.45 | baseline |
| waist block 1 on | 5 | 4 | 0.80 ± 0.45 | one flip each way |
| waist block 2 off | 5 | 4 | 0.80 ± 0.45 | baseline |
| waist block 2 on | 5 | 4 | 0.80 ± 0.45 | paired outcomes identical; lead approved eligibility |

Toy has one B-only success, no A-only loss. B records 63 held reads / 38 adoptions;
46 lateral residual readings have median 0.30 mm and maximum 0.62 mm; angular maximum
1.98 degrees. Attempts/adoptions by view: hover 50/34, look 2/1, show 11/11.
Conveyor pooled A=B=8/10; one B-only and one A-only success. Waist interception log counts
are block 1 A=0/B=2 and block 2 A=1/B=1. No G3 gain is claimed for conveyor.
These are smoke gates, not camera adoption evidence. Valid jobs occupied 1,479 lane-seconds
(0.411 lane-hours), excluding the interrupted attempt, generation/validation and boot waits.
Full per-record outcomes, method/interception logs, held events and terminal object traces
are retained in `~/dojo-data/wave-1106/physics-report.json` and its referenced job directories.

## Frozen inputs and dispatch

All 57 salted raw records were generated and hashed. The ten physics RAW records validate.
The 47 fresh camera records remain unsettled and their 17 fresh demos remain pending;
relocate, key and tie are skipped until that preparation is booked. No generated replacement
or successful subset was selected for those families.

Complete ready-family manifests under `compute4:~/dojo-data/wave-1106/`:

| Manifest | Input identities / camera arms | SHA256 |
|---|---:|---|
| layout-manifest-peg.json | 30 / 60 | `5fbbf5522c69f1656477fcaa889a3855c6148382b4439ff52ff57b1d59b47036` |
| layout-manifest-waist.json | 15 / 30 | `0ca206d496e96e441927b4ac5d9682bac696d3e76a4c045ec5221c32de09bd1d` |
| layout-manifest-disc.json | 19 / 38 | `c16509960683504ecf5a7026bb90adcbef1493bbd1ea9361f3e5599dea62ee56` |

Each records raw/loaded md5 and SHA256, validation indexes and both-arm preflight hashes.
Salt strings, salt SHA256 and generated-record md5s are in
`layout-manifest-preparation.json` (explicitly incomplete for pending settled/demo inputs).
Model/checkpoint/calibration hashes and resolved runtime are in `freeze-416a2a03.json`;
M15 mtime/hash evidence is in `m15-demo-provenance.json`.

Sixty peg camera rows were appended under queue2's existing lock; no old deferred rows were
appended. Shared v2b, digit_v2 and kong_v1b services are healthy. First START lines:

```text
2026-11-06T05:22:19Z c4b_playstackingtoysg_pegAr1_100005
2026-11-06T05:33:33Z c4b_playstackingtoysg_pegBr1_100005
```

The first A job completed rc=0, success, score 1.0 in 346 lane-seconds. Its spend was four
Claude calls: 51,201 prompt tokens, 3,022 completion tokens, 1,489 reasoning tokens,
44,475 cache-read tokens. No 429 occurred through this launch snapshot. The serial pause
between A and B was the approved conveyor RAW preflight reservation; queue pair order
was preserved. No camera A/B conclusion or adoption is made from this launch snapshot.
Waist/disc are frozen and staged for subsequent booking/review; fresh families remain pending.

One-lane prospective camera occupancy, excluding preparation/resource/review waits:
waist 2.15 h, relocate 4.00 h, key 2.00 h, peg 4.14 h, disc 3.35 h, tie 14.64 h.
The lane's 1,980-second hard cap can make elapsed time much longer. Reconcile all pairs,
retry/infra attempts and write-audit gates before any adoption.

Read-only status (launcher completion/rc and planning ETA, not validated statistical pairs):

```bash
ssh compute4 '~/dojo-integ-c4-416a2a03/.venv/bin/python ~/dojo-integ-c4-416a2a03/scripts/dojo/queues/wave_status.py --schedule ~/dojo-data/wave-1106/wave-amended.json'
```

Sealed artifacts and other-team directories were not accessed or modified. At dispatch,
compute4 retained approximately 1.2 TB free; local task writes stayed below 1 GB.
