# Conveyor: wide-object waist-first interception candidate

## Evidence and limits

CONV stream, base `20903bc2`. Goal: close the camera/G3 gap on `match_and_pick_from_conveyor`, keeping the adopted wrist veto. This PR is an **unvalidated, default-off implementation candidate**, not an adoption or score-gain claim.

Only dev rows were opened. `compute4:~/dojo-data/integ-c4/runs2` has no `c4b_matchpick*` directories. The lead confirmed the later veto-on A/B rows exist only on scoring ids; none was read for diagnosis. The additional permitted directory `~/dojo-data/imitate_c/match_replays/work_match_and_pick_from_conveyor` is empty. The points-table PR #386 supplies aggregate context only (camera 50, historical G3 96; wrist veto adopted, track expiry parked).

Existing camera dev evidence: `c4b_matchbelt2_{off,emb}_100000–04`, ten episodes at `7f29156`, **before the wrist veto**. Reduced source telemetry is committed in [pre_veto_dev.jsonl](../experiments/data/conv-waist-1105/diagnosis/pre_veto_dev.jsonl). The two arms vary embedding, not this PR's lever. Per the original event/replay diagnosis:

- 3/10 successes (off 100000/100002, emb 100002).
- 5/10 unsupported/stale-track (“ghost”) failures: off 100001/100003/100004 and emb 100001/100004. This includes the off 100003 unsupported wrist look with a real rider only 38.3 mm from the close; do not interpret all five as literally empty space hundreds of mm from any object. Four of the five have nearest-object distances 193–357 mm. Ghost attempts also occur in successful episodes.
- 1/10 wrong-rider/match failure: emb 100003 ultimately closes nearest other1 (20.8 mm), not target1 (61.1 mm).
- 1/10 physical miss/slip at the match: emb 100000, target1 nearest (50.9 mm), rose 19 mm, then no second intercept. The evidence does not establish a cone-specific class in this camera sample.
- 0/10 primary timing failures under that partition; emb 100001 also exhausts its match wait after two ghost attempts. Timing is a secondary cause there.

The earlier `c4b_matchbelt_*` family used missing belt scene backgrounds and diagnosed “nothing moves”; it is a separate known blind-perception class, not evidence about wide-object grasp selection. Misha's G5 item 8 mechanism (unmatched tracks never expiring while arms move, unlimited extrapolation, duplicate newcomer) is documented in `2026-10-29-dojo-belt-track-expire.md`; `track_expire` already implements its proposed fix. The board records a negative camera screen for it. Keep it off; do not treat another tracker patch as an established gain.

Historical compute3 `~/dojo-data/cells_c/match/v15_base`: **18/20**, dev variant ids 100030–49, two failures: 100037 camera slip and 100038 ice-cream slip. Both initially used an antipodal grasp (`phi … tilt 25`); camera retry used a waist and rose 37 mm, cone retry used a shallow slice and rose 11 mm. Full reduced rows: [historical_g3.jsonl](../experiments/data/conv-waist-1105/diagnosis/historical_g3.jsonl). That differs from the older 24/25 ceiling; neither is a current camera-veto dev failure distribution. Wrong match, ghost and primary timing failures are each 0/20 under perfect perception.

## Change

`RuntimeConfig.belt_waist_first` / `Skills.belt_waist_first`, `off|on`, default off; G3 exposes the same switch. On a rider with no top grasp, try mesh waist grasps before the existing slice/antipodal/rim plans. A surface grasp can be reachable while slipping off a wide scoop; waist candidates test mesh contacts below it. If no waist interception is feasible, keep the existing surface search. Narrow tops keep the existing path. Under the new switch, retry method exclusions also apply to the fallback waist search so a failed waist is not immediately reintroduced.

This is generic object geometry, with no task/asset/seed special case. It changes grasp selection only: no new perception writes, GT in the controller, tracker change, or action skill. Both conveyor tasks share the planner, so the camera protocol includes an image-conveyor guard. `belt_wrist_veto=on` remains explicit in both arms; no defaults changed.

## Validation and infrastructure blocker

Seven new unit cases cover real mesh waist contacts on a synthetic wide-cap cone, selection priority, unreachable-waist fallback, unchanged narrow-top search, retry exclusions, config default and validation. Existing belt replay/wrist/expiry tests are included in focused validation. Full-suite and CI results are recorded in the PR.

Deployed through `scripts/deploy.sh controlr-conv-1105`. Fresh salted G3 variants 100030–39 were generated at `compute3:~/dojo-data/conv-1105/simlab/layouts/match_and_pick_from_conveyor`, salt `conv-waist-dev-20261105`, `_simlab.variant=1`. Planned both arms on the exact same RAW records with wrist veto on. Five of these ten fresh draws are ice cream (not selected by outcome).

**G3 and raw stability validation blocked:** on 5 Nov around 22:59Z compute3's `nvidia-smi` reported “couldn't communicate with the NVIDIA driver”; `/dev/nvidia0` was absent. Isaac logged `cudaErrorNoDevice`, “No CUDA devices found”, and switched to software PhysX. No resulting score is valid evidence of this lever. The first raw-check process exited status 9 without a result; a scoped raw-check retry and the B job were cancelled while waiting. The A job was stopped by its own PID (TERM then KILL because Kit did not exit). No other process, driver, system setting or shared source was changed. No running CONV server was left behind. Logs remain under `~/dojo-data/conv-1105/`.

After compute3 is restored: raw-check the ten private raw records under `~/dojo-gpu.lock` in a resource-limited scope, then run G3 A/B in <=10-layout jobs (same lock and 10 GB disk guard). Reject software fallback. Record per-seed outcomes and grasp methods before considering a camera run. This remains a mandatory unresolved PR gate, not a waived check.

## Prospective camera test

[Pre-registration](../experiments/2026-11-05-conv-waist-camera-prereg.md): ten gain pairs on fresh salted layouts plus five image-conveyor guard pairs, genuine 5.1, cx route in both arms. Exact layouts and hashes are committed before camera data. Default pooled +5-point / wins >=2×losses / >=2 net wins and guard bounds apply, plus belief-write audit. Numeric loader ids are reused dev ids; the layouts themselves are new. No scoring-only ids are substituted to satisfy a “fresh numeric ids” interpretation.

Compute4 was read-only throughout, restricted to the authorized old dev reports and explicitly permitted empty replay directory. No sealed directories or scoring episode contents were read. No live model calls: **0 calls, 0 tokens**. Local artifacts/venv remain well below 2 GB; remote private data was below 1 MB at shutdown. This PR must not be merged/adopted until the physics gate and camera decision have actual results.
