# COIN2-CAM: camera A/B of the thin-object contact profile (pre-registration)

Written before any camera episode and before queueing. Lead approval of the camera slot: 7 Nov, after the G3 panel
([COIN2 contact](2026-11-07-coin2-contact.md): 49 → 76 under camera-like belief errors, GT guard 83 → 100).

## Question and arms

Does the profile raise deposit_coin's camera score without regressing the guards?

| arm | config |
|---|---|
| **A** | the frozen candidate `scripts/dojo/queues/2026-11-06-candidate.txt` (strict v2 grip source, `grip_check on`), as in the disc re-test |
| **B** | A + `disc_align=walls, disc_snap=on, disc_lift=shallow, disc_seat=table, disc_release=staged` (`disc_reserve` off) |

Every row: `--setenv AGENTIC_MODEL=claude/claude-sonnet-5-5`, checked on every row before queueing; private salted
`DOJO_DATA`; `--renderer isaac51`; 25-minute episode / job caps; the existing lane wrapper and watchdogs. The A / B
resolved RuntimeConfigs differ in exactly the five fields, checked for every pair (`rows-check.json`).

## General-system statement (hard rule)

- The `disc_*` fields are general geometry options of the slot-insertion skill for **any thin flat object standing in
  a fixed slotted holder** (`precision.disc_into_slot`, reached from `insert_into` by object geometry: a thin axis
  under 5 mm with equal other extents, standing in a fixed object's footprint). They are not keyed to any task,
  route or instruction; the same fields act on every scene and are inert where no such object exists.
- They read no task config and no checker: the slot comes from the receptacle's **perceived** pose and its
  **catalogue** functional frame; the holder's wall width at the squeeze height from the **catalogue mesh**
  (`wall_intercept`); the plane's yaw and offset are **measured** by the wall probe from the robot's own jaw reading.
  `tests/dojo/test_coin_contact_rules.py` (PR #432) asserts the module and the fields' comments name no task and that
  the module imports no task / layout / checker code.
- The **2.6 mm slot width** quoted in the G3 report is an analysis number measured from the stand's catalogue mesh to
  explain the physics; no code reads it. The robot motion constants (squeeze depth, gaps, steps, rates) were measured
  in our twin. `WALL_A` (the intercept) is used only when an object's mesh is missing.
- No evaluator-termination signal is used (no retry is decided from the episode's end).

## Cells and layouts (fresh salt; prepared on compute3)

[design.json](data/coin2-camera/design.json) and [prep.sh](data/coin2-camera/prep.sh) were on main (`da7c89a6`)
before generation. Salt `coin2-cam-20261107-ea2cb662ebd5ef4df5cf0e19c81b1f24`; unchanged generator; default two-pass
presettle. Ids only in 100000–09 / 100020–29; never ≥ 100100, sealed data or old salts.
**Selection by stability only** ([select.py](data/coin2-camera/select.py)):

| cell | role | pairs | pool | rule |
|---|---|---:|---|---|
| deposit_coin | gain | 20 | 100000–09, 100020–29 | every stable id (at least 18, else no dispatch) |
| stack_blocks | strong guard | 5 | 100000–09 | first 5 stable by id |
| insert_key | parity guard | 5 | 100000–09, 100020–29 | first 5 stable by id |
| play_tic_tac_toe | parity guard | 5 | 100000–09 | first 5 stable by id |

Order AB for even zero-based index within a cell, BA for odd. 70 rows, one repeat per identity.

## Decision rules (7 Nov camera rule, fixed now)

- **Gain (deposit_coin):** B pooled ≥ A + 5 points; paired score wins (B − A ≥ 10 points) ≥ 2 × losses and wins −
  losses ≥ 2. Success flips reported alongside.
- **Strong guard (stack_blocks):** B pooled ≥ A − 5 points and no more success-flip losses than wins.
- **Parity guards (insert_key, play_tic_tac_toe):** gated by a parity proof from the logs: identical resolved
  config apart from the five fields, and zero `wall_probe` events and zero `belief_write` events with writer
  `wall_probe` in every guard episode. Their camera scores are reported, not gated.
- **Write audit (adoption also needs it):** physics-only replays of both arms, `m95_score.py audit` with the #417
  persistent-identity instrument, harm > 2 mm added 3-D error. `wall_probe` is a writer new in B: it passes when
  its helpful writes are ≥ 2 × its harmful ones; writers present in both arms pass when B's harmful rate is at most
  5 points above A's (EXPERIMENTS.md, rule for pre-registrations from 7 Nov 07:00Z).
- Full registered n must complete; infra failures get one identical-input retry per arm via lane2; a missing pair
  blocks the verdict.

## Host, lane and budget

compute4 (haptics, shared with the partner team). Deployment of merged main by the MOVE34 / KONG recipe
([deploy-c4.sh](data/coin2-camera/deploy-c4.sh)): source from `git archive`, an independent copy of the working
`416a2a03` lane venv re-pointed to it, package / candidate parity, catalogue gate, per-job Kit environment through
`BASH_ENV` ([kit-env.sh](data/coin2-camera/kit-env.sh)). Rows append **after** every row already queued, under
`flock ~/dojo-integ-c4-state/queue2.txt.lock` ([append.py](data/coin2-camera/append.py)); existing bytes are never
changed. The `STOP` file in `~/dojo-integ-c4-state` is not touched: the lead resumes the lanes. No `~/coord` access,
no other process touched.

Budget from the disc re-test's call logs: deposit_coin 6.7 calls / episode, insert_key 3.8, stack_blocks 2.3,
play_tic_tac_toe 2.0 → 268 + 38 + 23 + 20 ≈ **350 logical Claude calls** (~5 M prompt tokens). The profile adds no
perception call (the probe reads proprioception), so per-episode calls should not grow. Lane time ~300 s per row:
~6 lane-hours. The Claude route has its own weekly limit; a 429 stops via the lane policy.

## Frozen inputs (before queueing)

- **Selection** ([selection.json](data/coin2-camera/selection.json)): deposit_coin all 20 (100000–09, 100020–29; 0
  dropped); stack_blocks 100000–04 (0 dropped); play_tic_tac_toe 100000–04 (0 dropped); insert_key 100001, 100002,
  100004, 100007, 100008 (13 of 20 pool records failed the unchanged second stability pass: the known key landmine).
  Layout and index hashes: [layout-hashes.txt](data/coin2-camera/layout-hashes.txt).
- **Deployment:** merged main `b3e173a6f344313db1e914a4ddeb20e6146c8b30` (contains #427 and #432) at
  `compute4:~/dojo-integ-c4-b3e173a6`, source from `git archive`; venv copied from `dojo-integ-c4-416a2a03` and
  re-pointed; `uv pip freeze` identical ([freeze.status](data/coin2-camera/freeze.status)); dojo imports from the new
  source; `candidate_parity` true ([parity.json](data/coin2-camera/parity.json)); catalogue gate OK.
- **Rows:** [queue-coin2cam.txt](data/coin2-camera/queue-coin2cam.txt), 70 rows, sha256
  `9c15c8d5ae8feeef2f02bb1ca0709e0ee7cf3352dc3cd28f4332b8fceec63be1`; [rows-check.json](data/coin2-camera/rows-check.json):
  every row `claude/claude-sonnet-5-5` (exactly one `AGENTIC_MODEL`), `DOJO_GRIP_SOURCE=v2`, `grip_check on`; every
  pair's resolved configs differ in exactly `disc_align, disc_lift, disc_release, disc_seat, disc_snap`.
- Spend before dispatch: 0 logical calls, 0 tokens.

## Receipt

RECEIPT
