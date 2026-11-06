# Wide-object conveyor interception: camera A/B pre-registration

Hypothesis: when a belt rider has no graspable top, trying its mesh waist before a shallow slice/rim grasp increases successful interceptions. `belt_waist_first` is default off; `belt_wrist_veto` stays on in both arms. This protocol is prospective: no camera episode has run on these records.

## Fixed inputs and freshness

Gain cell: `match_and_pick_from_conveyor`, ten pairs, one pair on each record in `data/conv-waist-1105/gain{1,2}/`. Guard cell: `pick_from_conveyor_by_image`, five pairs on `data/conv-waist-1105/guard/`. These are fifteen distinct new layouts, not repeats of old trajectories. The numeric loader ids are reused **dev** ids 100000–04; layout identity here is (block, id, SHA256), not id alone. There are no unused numeric ids outside the historical A/Bs within the authorized dev slice. This protocol therefore claims fresh layouts, not never-used numeric ids.

Generated on compute3 with the generator at base commit `20903bc2`, using independent salts `conv-camera-gain1-20261105`, `conv-camera-gain2-20261105`, and `conv-camera-guard-20261105` respectively. The exact bytes and their SHA256 manifest are committed alongside this spec. None was loaded into a camera episode or used for tuning. G3 tuning uses a separate salt, `conv-waist-dev-20261105`, on variant records 100030–39.

Before any camera data, run `dojo.simlab.presettle --raw-check` separately on each block, under the existing GPU lock and disk guard. These tasks use RAW records: never substitute settled poses. Retain the records and `raw_index.json`, even on failure. If a record fails `raw ok`, stop: amend this pre-registration with a new independently salted replacement block before reading any camera outcomes. Do not silently drop it or select a replacement based on score.

## Arms and execution

Pin one PR commit for BOTH arms. Save the commit, resolved `RuntimeConfig`, all service environment settings, asset/catalogue/background hashes, layout hashes, and agentic model in the run manifest before starting. Baseline is preset `submit_v3` plus `belt_wrist_veto=on`, `track_expire=off`, `belt_confirm_zone=off`. A sets `belt_waist_first=off`; B sets it `on`. Every other field and service setting must be identical. This is an explicit dev baseline, not a claim to reproduce every field of the frozen scoring candidate.

Use genuine Isaac 5.1 rendering on compute4 only after the lead releases the D-C19 machine reservation (estimated 6 Nov 06:00Z is not itself a release). Until then no compute4 writes, processes or GPU work. Use a separate deployment and output directory, one episode per `integ_eval.sh` job, shared GPU lock, unchanged 10 GB disk guard, and automatic service cleanup. Load the committed files from separate block directories; never overwrite the shared layouts. No scoring or sealed records are inputs.

Perception uses `AGENTIC_MODEL=cx/gpt-6-sol` in both arms; label all output **cx route**. No `claude/` or `cc/` calls. Use the same scene background and calibration in both arms, including `Dynamic_conveyor_00000` for gain and `Dynamic_conveyor_00001` for guard; fail preflight if absent. Preserve all camera frames, actions, report events and dev GT probes for diagnosis. Alternate order AB / BA by manifest row index across all 15 pairs. One repeat per layout; no early stopping on gains.

Planned 30 camera episodes: 20 gain, 10 guard. Agentic budget is at most three estimate invocations per episode (90 total); each invocation can make multiple model calls. Record actual calls and prompt/completion/reasoning/cache tokens from the reports and service logs, plus GPU time; do not report the invocation bound as a call bound. Do not change the model route mid-block. A service/bootstrap failure before any action may be retried once on the same arm; retain both attempts. A timeout or skill failure after actions counts as an outcome. An incomplete pair makes the adoption decision pending.

## Decision rule (pre-registered default)

On all ten gain pairs, B must have a pooled score at least five percentage points above A, flip wins at least twice flip losses, and at least two net wins. On the guard cell, B's pooled score must be no more than five points below A and flip losses must not exceed wins. Publish each pair, pooled means, paired score deltas and spread, successes, wins/losses and n. G3 improvements are not camera adoption evidence.

If the lever changes any belief writes indirectly (different wrist views/timing), run both action traces through physics-only `scripts/perception/m95_score.py audit`. B must not add excess harmful writes per writer against A on the same episodes (>2 mm added 3-D error; z separately). Report helpful/harmful counts, even if the camera outcome passes. This lever itself does not write beliefs.

Any threshold failure rejects adoption. A rejected lever with positive pooled delta may receive one further test, only on new layouts under a new prospective registration. Never repeat the same layouts to overturn rejection. A passing result establishes this dev configuration on the cx route; the lead decides its applicability to other routes and the scoring candidate, without modifying the sealed run.

## Failure attribution

Use dev GT probes and frame/action replays, not report category names alone. Separate unsupported/stale track closes (ghost), correct-rider physical grasp/interception failure (including wide cone), identity/match errors selecting a different rider, and timing/run-out. Record secondary causes (for example a ghost attempt followed by budget exhaustion), plus ambiguous cases instead of forcing a class. Wrist vetoes are avoided attempts, not successful picks. Count episode outcomes and attempt counts separately.
