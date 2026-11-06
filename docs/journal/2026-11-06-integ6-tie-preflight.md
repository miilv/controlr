# Compute6 tie relocation preflight

INTEG6 was assigned the 108-row tie family from the camera wave, keeping both
arms of every pair on compute6. The lead subsequently requested two camera
lanes, each retaining the 16 GiB-used admission gate, alongside independent
physics streams on the light locks. Sustained GPU power above 700 W requires
reducing to one camera lane. Initial preflight stopped before camera dispatch:
the brief required a lead decision on stack or perception-data differences.
No compute4 queue mutation or camera/model request was made in that phase.

## Deployment and validation

The exact wave commit `416a2a0374e06628350816a019862caf5156e730` was streamed
from `git archive` into `compute6:~/dojo-integ-c6-416a2a03`; its SHA is recorded
in `.deployed_commit`. Existing deployments and the read-only RoboDojo install
were left intact. The deployment uses managed Python 3.12.13, frozen uv sync
with dev/perception extras, followed by compute4's existing
`~/dojo-integ-c4-state/venv_extras.sh`. Package reference was freshly extracted
from compute4's **416a2a03 lane venv**, rather than inferred from its old controlr.
The extras script resolves timm without a pin; it installed 1.0.31 versus the
lane's 1.0.30, so the new deployment explicitly pins timm to 1.0.30.
Final `candidate_parity --packages-ref` has **zero package differences**, with
the intended lane background environment set on both paths. It exits 1 with
only `missing_models=[seg,glyph,glyphk]`; candidate parity compares the two
launcher paths on one box and does not certify cross-box background/catalogue
contents. The separate hash inventory and failed catalogue gate remain blockers.

All 108 queue RuntimeConfigs construct successfully on the pinned deployment;
every row requests `isaac51` and `claude/claude-sonnet-5-5`. This validation makes
no API calls and does not start Isaac. Operational evidence is under
`compute6:~/dojo-data/wave-1106-c6`.

## Differences requiring the lead's decision

The [complete per-path difference inventory](2026-11-06-integ6-parity-differences.json)
records SHA256 on each box, with null for absent files. It covers these scopes;
it is not a claim of complete simulation binary or asset-content identity.

- Catalogue: compute4 has 2,427 files, compute6 has 1,734. Exactly 693 files are
  missing on compute6 and `index.json` differs. Index MD5 is
  `2cd7455cbdce2ab38e4e9d586e28db61` on compute4 versus
  `6e90c3499d3a7427b0c577fc6ada1539` on compute6. The catalogue gate fails,
  including missing `Geometry/basket/00002`, `00003`, `00004` for the classify
  guard. Support-frame and articulation-link checks pass.
- Lane backgrounds: compute6 lacks the entire `~/dojo-data/bg_folds_51_ilia`
  directory (96 files on compute4), including scene backgrounds, frozen
  `frozen_d0_100040_49` calibration frames and its colour-calibration JSON.
  The shared catalogue's `color_calib_51.json` is byte-identical, but it is not
  the candidate's explicitly selected colour map.
- Side models: compute6 lacks `v2b/best.pt`, `v2b/classes.json`,
  `digit_v2/model.pt` and `kong_v1b/model.pt` under `~/dojo-data/segtrain/models`.
- RoboDojo conda Python differs: compute4 3.11.17, compute6 3.11.18.
  Installed distribution versions match except msgpack: 1.2.3 versus 1.2.4.
  Both report Isaac Sim 5.1.0. This is separate from the deployment venv.
- RoboDojo and tracked submodule revisions match: root `f57e1633`,
  XPolicyLab `e08ba788`, IsaacLab `afca7b09`, cuRobo `895c6517`.
  XPolicyLab untracked directories differ: compute4 `policy/harness/` and
  `policy/harness_strict/`; compute6 `policy/dojo_replay/`. No tracked diff
  was found in these repositories.
- HF assets reference the same `269457a3` commit, but materialized asset trees
  occupy 39 GiB versus 13 GiB. Their contents have not been exhaustively hashed;
  no sealed evaluation assets were read.
- compute4 has Isaac 6.0 installed and bootstrap role `physics`; compute6 lacks
  6.0 and has role `full`. These jobs explicitly select the 5.1 renderer, so
  no 6.0 service would be used. No role or system setting was changed.

Both GPUs are RTX 5090 with driver 580.178.04. Direct box-to-box SSH in both
directions failed `Host key verification failed.` Transfers therefore use a
streaming agent relay; no SSH configuration was changed.

## Data identity and hand-over barrier

The frozen tie manifest has the requested SHA256
`c6ec0687f671772c3fa7f582e682bec6202dacf0926127b02421cdc01963a83e`.
Its 32 selected input records resolve to 96 unique raw, settled, demo and index
files, totaling 39,193,873 bytes. All 96 source SHA256s match; repeated index
references have no conflicting hashes. Only the enumerated tie inputs and
preflight metadata were requested from compute4.

An initial transfer used `--keep-old-files` at the standard compute6 data root;
it refused existing files, preserving them. Existing indices differ, so that
root must not be used for the migrated wave. A second copy isolates the exact
inputs under `~/dojo-data/wave-1106-c6/inputs`, preserving `simlab/` and
`ab-wave-1106/` relative paths. Future queue paths and DOJO_DATA must explicitly
select this copy, including each salted block's private demo root.
All 96 isolated destination SHA256s match the manifest. The 15 retained M15
provenance entries agree with its loaded-layout and demo hashes and retain
`accept=true` / `validation_ok=true`. The original provenance JSON is retained
alongside the manifest. A `c6b_` queue is staged as `c6-queue.proposed.txt` only;
it has not been placed in an active lane queue. Its unsalted records require
inheriting DOJO_DATA at the isolated input root.

At preflight, all 108 tie rows remained in compute4 queue2; its active job was
`c4b_inserttubessg_pegAr1_100009`. No rows were removed or archived, and
`lead-notes.log` was not changed. Re-check started pairs under the queue lock
when actually handing over: this observation is not a reservation.

There are no compute6 START lines, camera calls or camera spend. Idle power
samples were 10.47 W (382 MiB initially, 207 MiB later). A five-minute logger
and camera services have not been started while parity is blocked. The wave
estimate is 14.64 lane-hours for tie, or about 7.32 ideal wall-hours on two
camera lanes **after release**, plus setup/resource waits and retries. There
is no committed completion ETA while the parity decision is pending.

## Lead release and compute6 lane preparation

The lead accepted the Python 3.11.17/3.11.18 and msgpack 1.2.3/1.2.4
differences and explicitly prohibited modifying the RoboDojo environment.
Both arms remain on compute6. The deployment's `.deployed.json` carries
`host=compute6`, `family_prefix=c6b_`, and a `simulation_environment` object
with both versions and the lead's acceptance. Existing runtime serialization
places this complete metadata object in normal episode rows' `code` field.
The lane also saves it in each job's `provenance.json` before starting, covering
synthetic watchdog infrastructure rows that do not carry eval's `code` field.
No episode serializer or controller code changes are needed.

The lead authorized syncing the exact 794 differing perception files from
compute4: 693 missing catalogue files plus its index, 96 backgrounds/calibration
files and four side-model files. Inputs remain the isolated manifest-exact
96-file copy from initial preflight. No RoboDojo environment or system setting
was changed.

The queue runner gains explicit state/run/data/stream environment overrides,
preserving compute4 defaults. The `integ-c6` profile skips compute4's coord
stamp entirely, accepts the two existing heavy slot locks, and retains the
16 GiB-used gate, free-VRAM check, 50 GB disk floor, startup serialization,
same-task stagger, HTTP 429 global STOP and one infrastructure retry.
Light locks remain independent. Shared services use the pinned camera deploy;
the versioned helper records ownership instead of stopping scopes by pattern.

The power watcher samples every five minutes. Two consecutive readings above
700 W lower the lane cap to one and create `STOP.heavy2`: heavy2 finishes its
current job, then exits. It never signals another stream's process. The exact
queue hand-over helper takes compute4's existing queue lock, keeps both arms
of a partially started pair there, and archives removed rows with before/after
queue hashes in `lead-notes.log`.

Local validation: 1,786 tests passed, 187 skipped; the eight focused lane,
power, queue hand-over and wave-status cases passed against the final helpers.
Shell syntax, Python compilation and `git diff --check` passed. No simulation
or model call was used by these checks.
