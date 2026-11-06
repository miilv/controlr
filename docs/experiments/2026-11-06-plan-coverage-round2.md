# Object-world planning: coverage diagnosis and bounded recovery

**Question:** can the object-world proposer produce enough validated paths to make a new skill-ceiling comparison informative?

**Decision:** the coverage gate failed. Final tools fallback is **12.9% (4/31)**, egg **87.0% (240/276)**. Defaults stay off; the new four-cell paired score study was not dispatched. This follows the [original GPU study](2026-11-06-plan-curobo-g3.md) and [HANG/sweep extension](2026-11-06-plan-extension-g3.md); their high fallback rates leave the proposed planning lever insufficiently exercised. Their score results do not refute object-aware planning.

[Machine-readable counts, stage flags and provenance](2026-11-06-plan-coverage-round2-results.json).

## Protocol and scope

Physics-only G3 on compute6, with the final strict egg recheck moved before admission to the authorized compute3 physics lane. No camera-policy or model calls. The engineering gate was registered before recovery measurements: **fallback below 20% separately on tools and egg**, counting every requested plan, speculative candidate and native validation rejection. Only after both pass would a fresh ten-pair comparison start on tools, egg, HANG and stack. Repeated engineering pilots on these development layouts are tuning evidence, not independent success-rate estimates.

The five-cell legacy diagnostic uses one dev layout per cell: 100000 for tools, egg, stack and sweep, 100031 for HANG. Recovery uses tools/egg 100000 and 100001, one episode per layout per source. These IDs are development-only; no 100005–09, 100020–29 or IDs >=100100. Layout files and hashes are pinned in the [first](2026-11-06-plan-round2-coverage-config.json), [v2](2026-11-06-plan-round2-v2-coverage-config.json), [v3](2026-11-06-plan-round2-v3-coverage-config.json), [v4](2026-11-06-plan-round2-v4-coverage-config.json), [v5](2026-11-06-plan-round2-v5-coverage-config.json) and [strict final](2026-11-06-plan-round2-v6-coverage-config.json) registrations, plus the [compute3 host registration](2026-11-06-plan-round2-v6-compute3-config.json). The final controller source is `ab2fa47c36e754f5a1307e530c55d5ba3a73779e`.

## What a failure label means

The installed cuRobo v2 exposes result success, feasibility, pose errors and collision constraints. It has **no MotionGenStatus enum**. The adapter now retains those native flags and the last stage, then assigns a stage-derived category:

- Start collision: direct cuRobo self/world constraint query, or a rejected start escape prefix.
- Goal collision: failed IK fits the configured pose tolerance but violates collision constraints.
- IK failure: IK does not return a successful solution; collision at a distant approximate solution alone does not establish goal collision.
- Graph / trajectory optimisation failure: respective unsuccessful stage after successful IK.
- Timeout: actual worker deadline; retained progress cannot restart the deadline.
- Native validation rejection: GPU proposal returned, but current joint bounds, endpoint or original collision checks rejected it. This is distinct from a cuRobo solver failure.

A cached fallback points to the earlier actual failure and makes no new GPU query. Counts below separate those events. Numeric request capture is opt-in (`plan_capture=requests`, default off), including existing belief geometry and proposed joints. No eval environment or simulator pose is available to the GPU worker. Capture also records native endpoint checks and, on the final source, the actual legacy fallback result.

The original 100 GPU episodes retained **1,442 untyped `no_plan` failures**. Their result flags and request payloads were discarded; their native causes cannot be recovered from the logs. We preserve that limitation instead of fabricating retrospective status codes:

| Historical cell, ten layouts | All fallbacks | Cached failures | Actual worker timeouts | Untyped no-plan | Native reject / validation budget |
|---|---:|---:|---:|---:|---:|
| tools | 275 | 193 | 10 | 72 | 0 |
| egg | 2,850 | 1,546 | 5 | 1,264 | 35 |
| stack | 162 | 121 | 8 | 33 | 0 |
| HANG | 743 | 683 | 10 | 36 | 14 |
| sweep | 47 | 7 | 2 | 37 | 1 |

The sweep remainder is a native validation-time budget expiration. Historical cached events are inherited failures, not thousands of independent solver deadlines.

## Corrected legacy diagnostic panel

Source `59c48ccf`, default legacy solver settings, one layout per cell. Every fallback in this fresh panel has retained stage evidence. An earlier instrumentation probe had a malformed standalone start-state diagnostic and is kept separately in the data, not substituted for this corrected panel. This is a diagnostic panel, not the ten-layout paired score study.

| Cell | Requests | Accepted | Start collision | Goal collision | IK fail | Graph fail | Trajopt fail | Actual timeout | Native reject | Cached fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tools | 12 | 3 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 7 |
| egg | 17 | 11 | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| stack | 32 | 23 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 7 |
| HANG | 104 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 96 |
| sweep | 10 | 4 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |

There are 175 requests, 43 accepted proposals and 132 fallbacks: 114 cached, five actual timeouts, four start collisions, three graph failures, one goal collision and five native rejects. Four HANG rejects hit the holder; one violates native joint bounds. Counts include hypothetical candidate plans. Acceptance alone does not mean a proposal was selected and executed.

## Recovery changes

`plan_world=table`, `plan_solver=routes`, `plan_profile=legacy` and `plan_capture=off` remain the defaults. `plan_profile=recover` is a separate, explicit experimental axis:

- Ten-second request budget, 64 IK seeds, four attempts, sixteen graph iterations and one trajectory seed. Each retry uses a distinct deterministic bank: one quarter near native IK, the rest across current native bounds. GPU tolerances are 2 mm / 0.03 rad, inside the unchanged native 3 mm / 0.035 rad acceptance check.
- Tighten the public GPU robot's broad URDF joint ranges to the current native working envelope before constructing solvers and costs. Bounds only tighten. Do not accept a forbidden branch to improve coverage.
- Explicitly omit an attached object's own world label. Use existing mesh surface covers, current-opening native finger proxies and conservative columns from existing height fields. Where regular held-box samples matter to native height-field checks, add those samples without removing mesh coverage.
- An initially colliding request can propose a bounded repair: <=15 mm initial world penetration, no self-contact, <=0.3 rad departure, <=0.03 rad increments, <=12 iterations / three seconds. No sphere/obstacle-pair penetration may increase beyond 10 micrometres numerical slack. The final check fixes an edge case where maximum depth per sphere could decrease while that sphere starts touching another obstacle. Upward, tool-axis and inward retract proposals also pass the GPU prefix check. Nothing executes unless the entire prefix and tail pass the original validator.
- A timeout closes its owned worker; the next request may start a fresh worker. The legacy profile retains its earlier timeout cache behavior. Startup remains separately bounded at sixty seconds.

All ordinary step rates/easing, safety envelope checks and explicit contact/joint primitives remain in place. The GPU model and the native validator use different sampled geometry; neither provides continuous collision certification or other-arm avoidance.

The suspected attachment propagation bug was ruled out: 21 inspected kinematic model handles share the same sphere storage. The actual fixes concern missing moving-finger geometry, incompatible joint ranges, start contacts, seed selection and timeout caching.

## Coverage progression

| Pilot | Source | Tools fallback | Egg fallback | Gate |
|---|---|---:|---:|---|
| 1 | 1f186059 | 16/31 (51.6%) | 250/263 (95.1%) | fails |
| 2 | 9eeb7756 | 4/31 (12.9%) | 29/54 (53.7%) | fails |
| 3 | 59c48ccf | 4/31 (12.9%) | 22/48 (45.8%) | fails |
| 4 | a4a89797 | 4/30 (13.3%) | 21/52 (40.4%) | fails |
| 5 | 385ab1df | 4/31 (12.9%) | 240/276 (87.0%) | fails |
| 6 | ab2fa47c | 4/31 (12.9%) | 240/276 (87.0%) | fails |

The strict final panel retains all failures:

| Cell | Start / prefix | Goal collision | IK fail | Graph fail | Trajopt fail | Timeout | Native reject | Cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| store_tools_in_toolbox | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| fill_egg_holder | 1 | 37 | 199 | 0 | 0 | 0 | 3 | 0 |

Objects-only final engineering outcomes are reported for completeness. These two-layout tuning means must not be compared to historical ten-layout controls. There is no new A arm or success-ceiling estimate.

| Cell / host | Success | Score ± SD | Mean / median steps | Total planning s | Query / validation s | Startup s | Mean / max query s |
|---|---:|---:|---:|---:|---:|---:|---:|
| store_tools_in_toolbox / compute6 | 1/2 | 75.0 ± 35.4 | 430.0 / 430 | 352.0 | 29.82 | 9.84 | 0.96 / 1.34 |
| fill_egg_holder / compute3 | 0/2 | 10.0 ± 0.0 | 404.5 / 404.5 | 263.7 | 223.20 | 18.35 | 0.81 / 1.05 |

The final panel accepts 27 tools and 36 egg proposals. Actual deadline and cached-failure counts are retained above. 12 accepted egg requests include a bounded start repair. The earlier v4 egg 40.4% result is retained; the final stricter proxy/seed bundle changes prior motions and exposes many more failed place candidates. This is a regression, not a basis for selecting the best pilot after the fact.

Earlier passes never excluded speculative IK failures from the denominator. Goals and geometry change as earlier motions alter the episode, so request counts differ between pilots; these are whole-episode coverage probes, not a fixed request-bank comparison.

## Why residual failures matter

Direct sphere/cuboid queries found a shallow start overlap at link3 (~11 mm) and a native-clear goal with link2/3 overlap (~38 mm). These are not solely finger padding. The native arm's point samples and the GPU volume model disagree around fixtures. Keeping the native final validator catches proposals the GPU proxy misses; accepting every GPU success would misstate usable coverage.

A separate diagnostic removes scene obstacles from twelve failed IK requests from the fourth egg pilot. It never executes an action. Repeated on the distinct-seed source `385ab1df`, including refreshed four-seed native IK hints, only **one of twelve** obtains successful IK; the other eleven retain roughly **7–42 mm** position residuals, above the 2 mm criterion, with no world/self collision constraint. Those eleven also have no converged native IK when the hints are refreshed with four native random seeds. This identifies fixed-goal / workspace limitations under the tested bounds, seeds and budget; it is not a proof of global unreachability. The one scene-sensitive request demonstrates that obstacle-induced IK difficulty also remains.

Goal selection must cooperate with route planning, and native/GPU geometry needs agreement near the basket and holder. A route to an unchanged infeasible endpoint cannot solve the skill's choice of endpoint. Relaxing the final validator, enlarging limits or omitting failed candidate goals from the fallback denominator would conceal that issue.


Final fallback endpoint and actual legacy-route evidence:

| Cell / layout | Native IK failed | Native endpoint collision | Native endpoint clear | Actual legacy outcome |
|---|---:|---:|---:|---|
| store_tools_in_toolbox / 100000 | 0 | 0 | 0 | no fallback |
| store_tools_in_toolbox / 100001 | 4 | 0 | 0 | ik_fail: 4 |
| fill_egg_holder / 100001 | 129 | 95 | 11 | collision: 103, ik_fail: 129, reached: 3 |
| fill_egg_holder / 100000 | 0 | 0 | 5 | reached: 5 |

## Reproduction, artifacts and limits

Inside the approved PLAN light-lock/admission wrapper, run each registered source with its frozen layouts and unique output directory:

```bash
scripts/simlab/rdpy.sh -u -m dojo.skills.g3 run \
  --task fill_egg_holder --seeds 100001,100000 \
  --layouts ~/dojo-data/plan/layouts/fill_egg_holder \
  --skill-attr plan_world=objects --skill-attr plan_solver=curobo \
  --skill-attr plan_profile=recover --skill-attr plan_capture=requests \
  --trace --out ~/dojo-data/plan/round2/<unique-output>
```

Compute3 robot YAML differs only in absolute home paths; URDF and cuRobo source fingerprints match compute6. The initial compute3 setup attempts lacked the settled-layout selector and produced no episode; their logs are retained separately.

Raw requests, native flags, actions, source receipts and traces remain under `compute6:~/dojo-data/plan/round2/`. Timing is observed shared-host wall time, not a controlled latency benchmark. Final tools timings come from compute6 and egg from compute3 and are kept separate. Worker startup and query/native-validation time are reported separately from total skill planning. Existing [tools success/failure](2026-11-06-plan-curobo-g3.md#reproduction-and-diagnostic-media) and [HANG replays](2026-11-06-plan-extension-g3.md) illustrate round-one source trajectories; they are not round-two outcome evidence or camera-policy trials.

Zero LLM calls, tokens or model spend. No compute4 use, system changes or foreign-process signals. Compute3 had one admitted physics job at a time, with `who`/GPU checks, >=24 GiB free VRAM at admission and a guard stopping only checked owned descendants if free VRAM fell below 16 GiB. Code and temporary data writes there stayed inside `~/controlr-plan`; completed data was copied into the PLAN data root on compute6. Local writes remain below 1 GB. All 28 compute6 admissions meet the gate: max used VRAM 20422 MiB, min available RAM 37283 MiB, max load1 17.08. There are 34 engineering G3 episodes across diagnostic and recovery phases, with 5549.1 summed recorded episode wall seconds, excluding app startup, layout loading, queue and probe overhead. Compute6 robot/cuRobo source fingerprints match round one; compute3 has the documented home-path difference only in robot YAML, and all nine checked egg/robot geometry files match. Compute3 admitted its measured job with 28,670 MiB free VRAM and retained at least 24,789 MiB free throughout; the guard never triggered. All forty compute6 and two compute3 recorded workers exited, and neither host has a GPU process belonging to the PLAN checkout. Final test receipts are in the journal.
