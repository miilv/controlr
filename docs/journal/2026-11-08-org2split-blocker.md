# ORG2SPLIT dispatch blocker — 2026-11-08 05:06Z

No rows moved; neither queue nor running job was changed. No deployment or model calls ($0).

## Selection checked

80 queued rows (40 whole A/B pairs): largestnum, largestnumrand, storetools each 100000–100009;
insertkey 100003, 100004, 100005, 100007, 100009, 100020, 100022, 100024, 100025, 100026.
Families are `c6b_<cell>sg_org2{A,B}r1_<id>`. All 80 had zero start/done matches in both
compute6 lane logs at inspection. Gain, strong guard, classify and fillpen remain on compute6.
The source SHA in the rows is `83887dae2ee50d582520b60d13191279a5a58f32`;
intended compute4 deployment is `~/dojo-integ-c4-83887dae` (not created).

## Exact failing preflight

The required per-row diff permits only `--repo` and family prefix changes. Every selected row also contains
absolute compute6 inputs/output paths:

- `DOJO_DATA=/home/robotics/dojo-data/org2-cam/org2-cam-1`
- `DOJO_COLOR_CALIB=/home/robotics/dojo-data/bg_folds_51_ilia/frozen_d0_100040_49.color_calib.json`
- `--layouts /home/robotics/dojo-data/org2-cam/org2-cam-1/simlab/layouts/<cell>/<id>.settled.json`

compute4's existing lane wrapper passes these verbatim: lane2.sh line 177 only expands a tilde in
`--setenv`, line 258 passes the environment and remaining arguments to integ_eval.sh. It has no
robotics-to-haptics translation. Thus a deploy-path-only rewrite cannot produce host-local input paths.
Moreover the explicit compute6 read allowlist excludes `~/dojo-data/org2-cam`, so staging and verifying
these inputs from their source is outside this assignment. The compute4 write allowlist excludes that
stream as well. No out-of-allowlist existence checks or input reads were attempted.
The CLASSIFY3 recipe additionally reads `~/dojo-env.sh` and writes its own data stream; it cannot be
run verbatim under this assignment's explicit root restrictions.

The lead must amend the allowed roots and permit host-path substitutions (and the required Kit environment)
before dispatch can proceed. This is a blocker receipt, not a preregistered move; a dated prereg amendment
must still be committed before any future move, before the 07:40Z freeze deadline.

## Queue snapshots (read only; no before/after mutation)

- compute6: 150 rows, SHA256 `1f2ad4142ac17e5e8736a0fa15e5ce3d4a1aff13227a80bf423692c7096c4333`.
- compute4: 4 queued COIN2-CAM rows, SHA256 `1e133b659d0c78ad04825de1a454b59d83063a0ac43f9f0e87bb61131df36159`.

These are unlocked read snapshots; lanes may naturally consume rows. compute4's logs showed heavy1/heavy2
active on COIN2-CAM at inspection (last starts 04:53:46Z / 04:51:00Z, subsequent readiness releases).
No restart was issued. Venv freeze, source parity, candidate parity and catalogue checks were not run;
none is claimed green. Every selected row already specifies `AGENTIC_MODEL=claude/claude-sonnet-5-5`.
THE HARD RULE and experiment configs are unchanged.
