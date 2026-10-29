# 2026-10-29: D-C19 readiness: the candidate on the official path, parity, freeze manifest, dress-rehearsal smoke (stream S)

Under D-C19 (robot-harness coord/DECISIONS.md), our candidate is scored once, blind, through RoboDojo's official eval
client behind Misha's strict v2 server. That number replaces every figure we quote. This entry is the readiness work.
It used compute4 dev ids only (stack_blocks and arrange_largest_number 100000) and spent a few agentic calls
(submit_v3's agentic budget).

## The blocker found and fixed (PR #343, 09e2eb49)

`scripts/dojo/official_eval.sh` could not run the candidate as CANDIDATE.txt defines it:
- no `--seg` / `--glyph` / `--glyph-kong` side servers, so no seg_propose and no glyph reads;
- no `--setenv`, so `DOJO_PERCEPTION_CFG` never reached the perception service;
- AgentPolicy by default instead of RuntimePolicy.

A D-C19 run would have measured another system.

**What it does now:**
- `--candidate CANDIDATE.txt` takes the file verbatim, split like Integration's lane2.sh queue line, and defaults to
  RuntimePolicy.
- `--setenv` reaches our processes only, never robot-harness's; credential-shaped keys are refused.
- The side servers start with lane2.sh's commands and model dirs, before robot-harness's server. One that does not come
  up ends the call there: fail loud, which decides PKG_NOTES' "degrade vs fail-loud" item for the scored run.
- `--dry-run` prints the resolved call as JSON.

Also in #343:
- G5 review #6: `ServerPerception` forwards every op to a local client. Without one, `check` says `_unavailable` and
  resync logs `unverified`, and the driver refuses local-only features.
- G5 review #7: `search_identity`, default `"first"`.

## Parity: integ_eval (lane2.sh) vs official path

`scripts/dojo/candidate_parity.py` compares the two paths:
- the effective RuntimeConfig, through each harness's own `build_config`;
- the policy class;
- the candidate environment;
- the side-server addresses and weight md5s;
- the perception data (backgrounds, colour map, calibration set).

**The first run found a second gap.** lane2.sh runs `DOJO_BG_FOLDS=~/dojo-data/bg_folds_51_ilia`, the 5.1 set with the
scene backgrounds. official_eval.sh defaulted to `~/dojo-perception-data/bg_folds_51`, which on compute4 has no frozen
calibration set at all. With the lane's value exported, the diff is **zero** on compute4 for the current CANDIDATE.txt.

`DOJO_GRIP_SOURCE=v2` reaches both paths verbatim and is inert on the official one, because the strict v2 server
sends `ee_joint_state_official` itself. The diff output notes this as expected.

## Freeze manifest (dry run)

`scripts/dojo/freeze_manifest.py`, run on compute4 from the deploy of 0c1feba4, wrote
compute4:~/dojo-data/official/s-dc19/manifest_dryrun.json:
- **Code and tools:**
  - parity zero;
  - RoboDojo f57e1633, Isaac Sim 5.1.0.0 (RoboDojo python 3.11.17);
  - our venv: python 3.12.13, torch 2.11.0+cu128, ultralytics 8.4.191;
  - robot-harness 38cb6c67 (a pinned archive with a COMMIT file).
- **Side-server weight md5s:**
  - seg v2b `best.pt` 2f546196…, `classes.json` 5560c910…;
  - glyph digit_v2 df7da507…;
  - kong_v1b 300f319c….
- **Perception data:** calibration set frozen_d0_100040_49 in bg_folds_51_ilia, files md5'd; colour map
  color_calib_51.json, md5 7ac852aa….
- **Routes, names only:** agentic perceiver `claude/claude-sonnet-5-5` (the code default; AGENTIC_MODEL unset),
  agentic "budget" × 3, held_look "b1_then_agent".

## Dress-rehearsal smoke (official path, the candidate verbatim, compute4)

Each call ran RoboDojo's eval client with the strict J2 server (`--strict --dev`), our driver with RuntimePolicy
submit_v3 + the candidate config, our perception service and the twin.

| call | side servers | episode | RoboDojo result |
|---|---|---|---|
| stack_blocks 100000 | seg v2b, glyph digit_v2, kong_v1b up in 2-6 s | 229 / 550 steps, ended (latch), 122 s | success, 100 |
| arrange_largest_number 100000 | same | 626 / 1050, policy_done, 208 s | fail, 15 (belief_mismatch) |

- seg_propose ran on the segmenter's masks (n_dets 4 and 11).
- In the arrange call, the series count came from the segmenter (4 digits), and all four digits were read by the glyph
  server (`digit_source` "glyph").
- The arrange failure matches this layout on the integ path: 3 of 4 recent integ rows on arrange 100000 fail too
  (0.0 / 0.15 / 0.4, perception_gross or belief_mismatch; 1 success).
- No bridge error, no policy error, rc 0 for both calls.

**Box use:** Integration lent one lane. The lock was `~/dojo-integ-c4-heavy4.lock`, not the coord lock `~/dojo-gpu.lock`
shared with Misha. The run started at ≥ 19 GB free VRAM, and `caps.restore` was touched at the end.

## Open for D-C19 itself (not changed here: they touch the sealed-block rules)

- **Ids:** official_eval.sh refuses ids >= 100100, and so does `dev_eval.py`'s seed patch (`paths.is_reserved_id`). The
  D-C19 ids are 100100-100119. The scored run needs an explicit joint-block mode that reads only the sealed directory
  and its md5 manifest.
- **Unsettled layouts:** official_eval.sh stages only `<id>.settled.json`. make_toast and the two conveyor tasks run raw
  in D-C19.
- **Lock:** the default `DOJO_GPU_LOCK` is the coord lock on compute4. The scored run uses a lane lock agreed with
  Integration and Misha's side.
- **Model route:** the agentic perceiver uses the Claude route, which has its own weekly limit. Check the headroom
  before tranche A.
