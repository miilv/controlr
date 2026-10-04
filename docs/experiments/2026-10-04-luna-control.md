# 2026-10-04 — Can GPT-6 Luna (no thinking) control the rotation pick-to-box as well as Sonnet 5.5?

**Hypothesis:** a fast GPT model with thinking off (`cx/gpt-6-luna`, `reasoning_effort: none`,
`service_tier: priority`) reaches Sonnet 5.5's success on the same seeds and plans at a shorter turn.

**Setup.** `configs/sim_waffle_yaw_luna.yaml` = `sim_waffle_yaw` with only the control model changed
(manual byte-identical to the Sonnet runs). Seeds 0–3 × 2 repeats, the rotation round's pinned plans
(`planner.plan_file=runs/rot_plans/seed<k>.md`, 0 planner calls). Baseline: Sonnet 5.5 (default
effort ≈ high) in [contacts-and-speed](2026-10-02-contacts-and-speed.md), seeds 0–3 × 1, 4/4. Code:
branch `feat/luna-eval` off 4fc4491. LLM calls went from compute3 through an ssh tunnel to the dev
box's proxy (`CONTROLR_LLM_PROXY`), because compute3's own route to the router uploaded at
42–138 KB/s that day ([journal](../journal/2026-10-04-luna-latency.md)). Two arms differ from the
baseline: the model **and** the effort (Sonnet thought, Luna did not).

**Budget and spend.** Budget: unlimited (Ilia). Round: 151 control calls, 0 planner calls; with the
aborted `low` round (3 episodes, 32 calls) 183 calls, 1.22 M prompt tokens (82 % cache reads), 6.2 k
completion tokens (1.6 k reasoning, all in the `low` episodes). Probes and the latency stand: ≈ 200
calls (journal). GPU ≈ 25 min; the Isaac server stopped after every episode.

## Results

| seed | run | outcome | turns | STOPs (shown) | `unstable` | LLM s p50 / p90 | turn cycle p50 s | cache share | usage-less turns | turns with reasoning |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `20261004T103138Z` | **success** | 23 | 0 | — | 2.94 / 3.55 | 4.33 | 0.85 | 0 | 0 |
| 1 | `20261004T103334Z` | **success** | 25 | 1 | — | 2.63 / 4.93 | 4.23 | 0.86 | 1 | 0 |
| 2 | `20261004T103605Z` | safety_stop | 17 | 2 | — | 2.65 / 4.08 | 5.46 | 0.82 | 0 | 0 |
| 3 | `20261004T103751Z` | safety_stop | 18 | 2 | — | 2.49 / 4.20 | 3.83 | 0.78 | 0 | 0 |
| 0 | `20261004T103924Z` | safety_stop | 13 | 0 | yes | 2.67 / 3.87 | 4.94 | 0.75 | 0 | 0 |
| 1 | `20261004T104044Z` | **success** | 20 | 0 | — | 2.41 / 2.91 | 3.84 | 0.78 | 1 | 0 |
| 2 | `20261004T104219Z` | safety_stop | 9 | 1 | yes | 2.63 / 5.16 | 5.49 | 0.81 | 1 | 0 |
| 3 | `20261004T104329Z` | safety_stop | 26 | 2 | — | 2.62 / 4.21 | 4.05 | 0.84 | 0 | 0 |

(The STOP that ends an episode is not shown to the model, so "shown" is one less than
`episode.max_stops` = 3 for the wall/packet endings.)

**Luna 3/8** (seed 0 1/2, seed 1 2/2, seed 2 0/2, seed 3 0/2) vs **Sonnet 4/4** (one per seed).
LLM p50 2.4–2.9 s vs Sonnet 2.2–3.4 s; turn cycle p50 3.8–5.5 s vs 5.5–6.3 s. Successful episodes
took 20–25 turns (Sonnet 18–23). Luna never thought at `none` (0 of 151 calls).

Aborted first round (`low`, compute3's slow route; latency unusable, outcomes valid since the robot
waits): seed 0 and seed 1 both `unstable` at a re-grasp `GRIP close` (`20261004T094646Z`,
`20261004T095130Z`); seed 2 stopped by hand after 1 turn.

## Failure modes

- **PhysX blow-up at `GRIP close` (2 of 5; 4 of 7 counting the `low` round).** Sonnet's final round
  had none. Luna closes low and close to the mat after small descents
  (`MOVE ee_delta 0 0 -10 0` / `STATUS OK lowering incrementally toward packet midpoint` →
  `GRIP close`, seed 2 rep 2; [video](../video/luna_seed2_grasp_unstable.mp4)). The squeezed-object
  landmine is in the sim, but Luna's grasp poses trigger it far more often.
- **Carry into the near box wall (seed 3, both repeats).** The held packet hit the wall three times;
  after each STOP Luna backed off and lifted a little, then lowered again at nearly the same y:
  `MOVE ee_delta 0 0 -60 0` / `STATUS OK lowering packet into the box opening cautiously` → STOP →
  `MOVE ee_delta 0 -35 20 0` / `STATUS OK withdrawing from the wall to reassess box opening
  clearance` → `MOVE ee_delta 0 60 0 0` → STOP ([video](../video/luna_seed3_wall_failure.mp4)).
  Sonnet in the same situation lifted 50 mm and carried further in before lowering.
- **Open gripper pushing the packet (seed 2 rep 1):** two "the gripper pushed against the packet"
  STOPs while descending in 10 mm steps next to the packet, then a third.
- Style: replies are terse; one episode appended `**` to every STATUS note (harmless to the parser).

[Success video](../video/luna_seed0_success.mp4) (seed 0, 23 turns).

## Conclusions (given n)

- 3/8 vs 4/4 — the Sonnet arm has n = 4 and thought, so this is not a model comparison yet; it
  shows Luna at `none` can do the task (seeds 0, 1) and fails seeds 2–3 in ways Sonnet did not.
- Luna is not faster per call than Sonnet on this router (2.4–2.9 vs 2.2–3.4 s p50); its turn cycle
  is shorter mostly because the sim was faster per turn and no turn thought. The per-call floor on
  `cx/` is ≈ 1.1–1.4 s to the first token (journal).
- `service_tier: priority`: no measurable effect; whether the Codex backend applies it is unknown.

## Found along the way → BACKLOG

compute3's slow route (and `CONTROLR_LLM_PROXY`), the per-turn transcript re-upload, the usage-chunk
wait (~0.3 s/turn), Luna's grasp blow-ups.

## Next experiment

1. Same seeds × 2 for Sonnet 5.5 at `reasoning_effort: low` and Luna at `low` — separates "model"
   from "thinking" (≈ 400 calls).
2. Harness latency: execute when the action line is complete (no STATUS/usage wait) and
   `jpeg_quality: 75`, measured on replay first.
