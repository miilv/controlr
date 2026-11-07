# 2026-11-07 — write-audit transport and persistent identity instrument

**Measurement-only fix. Registered peg and KONG verdicts are unchanged.** Peg remains NOT ADOPTED, its re-test
exhausted. The supplementary results below neither retrospectively qualify a lever nor change a camera score.
No live LLM calls, camera/simulator/replay jobs, compute4/compute6 writes, lane/queue changes or sealed-data reads.

## Instrument changes

`RuntimePolicy._held_pose` now logs `before7` as the **prior attachment at the current measured TCP**, rather than the
cached world object pose from before a show move. `before_sync7` retains that raw cached pose, and `arm` identifies the
holding arm. The synchronized prior is computed without mutating the live belief. The original attachment assignment,
two observation calls, and single live `sync_robot` call stay in their original order. Only report content changes.

A FakeLLM-driven, three-turn kinematic mock episode compares logging on/off. It asserts identical numeric actions,
bit-identical object poses, attachments and joint states after every turn, identical provider statistics, and identical
observation/synchronization counts. The transported cached before moves >10 mm, but each logged fitted correction is
exactly 3 mm. No new experiment switch or controller default is introduced.

`AuditIdentity` replaces per-write nearest-neighbour selection in `m95_score.py audit`:

- Initial-record, saved snapshot and write-time proximity can associate an unambiguous object within **10 mm**, with
  a **>2 mm nearest/runner-up separation**. Distant nearest guesses do not become anchors. Initial evidence is not
  borrowed from later `agentic2`/searched records.
- A first consumed pick, or saved grasp-close snapshot with an explicit pick target, finalizes the physical identity.
  It may supersede a provisional initial association because interchangeable detector labels can be renamed before
  their first use. `identity_prior_gt_label` records that transition. No later retry can rebind a finalized identity.
  Evidence is processed chronologically, never backdated to earlier writes.
- Unpicked objects retain their tight association. Without an identity the old nearest/group rule remains an
  explicitly marked fallback; its use is counted. If an already-anchored GT object disappears, the audit fails rather
  than silently substituting a neighbour.
- Rows retain source, anchor step/distance, GT label and fallback flag. `association.json` reports source/fallback
  counts and missing body geometry. The **2 mm threshold, 3-D body-centre calculation, z reporting, and harmful/helpful
  definitions are unchanged**. This changes object association, not the threshold or camera/controller behavior.

Tests cover transported multi-instance objects, initial label renaming, immutable retry identities, chronological
initial evidence, saved grasp-close evidence, missing persistent GT, nearest fallback without accidental persistence,
and CLI fallback reporting while retaining the body-centre metric.

## Supplementary re-scores only

Inputs are the exact original ledgers and saved trajectories: peg's 80 re-test arms / 1,696 writes and the KONG Sonnet
primary's 40 arms / 249 writes. Original reports/actions remain untouched. Normalized local analysis reports preserve
the frozen ledger's write coverage (including its supplemental legacy events) and retain the original consume/probe
identity evidence. No trajectory was executed again.

Before changing association, the unmodified old scorer reproduced **every original verdict and every rounded before/
after 3-D and z error**: maximum numerical discrepancy **0.0 mm** across both ledgers. Frozen group maps and bbox extrema
of the existing `Footprints` vertex samples are retained in [body caches and results](../experiments/data/audit-identity/).
These extrema preserve exactly the original body-centre computation; there is **zero missing body geometry**. They are
public asset-derived measurements, not new poses supplied to a controller.

| panel / writer | original H / helpful / neutral | supplementary H / helpful / neutral | focal nearest fallbacks |
|---|---:|---:|---:|
| Peg B `held_pose`, 147 writes | 22 / 11 / 114 | **0 / 13 / 134** | **0 / 147** |
| KONG Sonnet B `watch_relocate`, 29 writes | 2 / 16 / 11 | **0 / 18 / 11** | **0 / 29** |

A has zero writes for those two focal writers. Across **all** writers, the persistent instrument uses nearest fallback
on **77/1,696 peg writes** and **26/249 KONG writes**; each fallback is exposed in its row. All 26 peg show identities
agree with the independently checked first-pick physical identities from the [mechanism investigation](2026-11-07-dojo-peg-mechanism.md).
The two previously flagged KONG `mahjong7_0` relocations associate with `mahjong7_0`, not standing neighbour `mahjong8_0`.

### Legacy peg before-pose reconstruction caveat

Future reports directly log the exact synchronized prior; historical reports lack it. For supplementary peg analysis
only, old show-write before poses are transported from the last non-show read's TCP to the write's current saved TCP.
**15/26** show priors use exact saved anchor flanges; **11/26** lack that exact anchor frame and use commanded-anchor FK.
Those eleven are explicitly marked **approximate**, with raw/corrected poses, anchor steps and source hashes retained.
The 121 hover writes require no show-transport repair.

| peg B held writes | harmful | helpful | neutral |
|---|---:|---:|---:|
| 121 hover + 15 exact-anchor shows | 0 | 7 | 129 |
| 11 commanded-anchor reconstructions (supplementary point estimates) | 0 | 6 | 5 |
| total | 0 | 13 | 134 |

Thus six of the thirteen supplementary helpful labels depend on the commanded-anchor reconstruction. This is **not**
a fresh exact gate audit, and no error bound on that approximation is silently assumed. KONG has no held-show
normalization and no such reconstruction. Association-only peg re-scoring would still count show transport as an
improvement; both timestamp and identity repairs are needed to interpret attachment corrections.

Other writer results are retained, not hidden. In particular, supplementary peg `perception` harmful counts are A49 /
B50 and KONG Sonnet `perception` A10 / B12. No aggregate acceptance rule is re-applied here; neither the old registered
verdicts nor their audit snapshots/pre-registrations are edited.

## Reproduction and validation

The committed `scripts/perception/audit_supplementary.py` reads local copies and writes only local analysis outputs:

```bash
uv run python scripts/perception/audit_supplementary.py \
  --sources runs/peg-mechanism \
  --manifest docs/experiments/data/peg-retest/verdict-camera.json \
  --writes runs/peg-mechanism/audit/writes.jsonl \
  --geometry docs/experiments/data/audit-identity/compute4-body-bounds.json \
  --out runs/audit-identity/peg-supplementary --repair-legacy-held

uv run python scripts/perception/audit_supplementary.py \
  --sources runs/audit-identity/kong \
  --manifest docs/experiments/data/kong-scoped-relocate/audit-sonnet.json \
  --writes docs/experiments/data/kong-scoped-relocate/audit-sonnet.json \
  --geometry docs/experiments/data/audit-identity/compute6-body-bounds.json \
  --out runs/audit-identity/kong-supplementary
```

Source copies follow `episodes/<run>/*_{report,actions}.json.gz` and `replays/<run>_<id>.json.gz`. They came read-only
from the already-authorized peg audit/runs2 roots on compute4 and KONG `audit-sonnet`/`integ-c6/runs2` roots on compute6.
[Supplementary summary/provenance](../experiments/data/audit-identity/supplementary.json) retains full writer counts,
fallback counts, each input hash, and each historical transport reconstruction. Registered verdicts remain in their
original files. The helper fixes `harm_mm=2`, requires exact write frames (`back=0`), and refuses a missing geometry
cache. New prospective audits use the corrected direct logs, not this historical-reconstruction option.

Focused tests: **27 passed** before the final documentation commit. Full suite and PR CI are required before merge.
This PR changes no robot/Isaac/loop/safety execution code; no prohibited live/sim validation is needed for the report fix.
