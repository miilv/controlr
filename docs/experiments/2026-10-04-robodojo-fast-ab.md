# 2026-10-04 — RoboDojo general_pickup: do faster turns (effort low, smaller JPEGs) and bigger motion per env step help?

**Hypothesis:** (a) Sonnet `reasoning_effort: low` + 336 px + JPEG 75 cuts turn latency without
hurting success; (b) moving the joints up to 0.12 rad per env step instead of 0.05 halves the
env-step cost of each move (probe: max landing error 3 mm), so fewer episodes run out of
RoboDojo's 200-step budget — the failure mode of the [pilot](2026-10-04-robodojo-pilot.md).

**Setup:** `general_pickup`, eval seed 0, layouts 0–9, 1 repeat, no planner, control model
`claude/claude-sonnet-5-5`; controlr `3760c8a` (PR #9), RoboDojo `266130a`, compute2. Arms:

| Arm | Config | Effort | Frames | Motion per env step (arm / gripper) |
|---|---|---|---|---|
| base | `configs/robodojo.yaml` | default (≈ high) | 448 px, JPEG 90 | 0.05 rad / 0.25 |
| low | `configs/robodojo_fast.yaml` + `--set robot.params.arm_step_rad=0.05 --set robot.params.grip_step=0.25` | low | 336 px, JPEG 75 | 0.05 rad / 0.25 |
| fast | `configs/robodojo_fast.yaml` | low | 336 px, JPEG 75 | 0.12 rad / 0.5 |

Arms ran one after another (base, fast, then low); fast was stopped after 3 episodes (below).

**Spend:** 681 control calls (base 207, low 294, fast 180 + the interrupted 4th episode), 0 planner;
prompt 2.67 M / 4.00 M / ~2.4 M tokens, cache-read share 0.86 / 0.91 / ~0.9. Today's total with the
pilot: ≈ 840 calls.

## Results

| Arm | Success (RoboDojo `_result.json`) | LLM p50 per turn | Reasoning tokens | Reply length (median words) | Grasp closes per episode (median) |
|---|---|---|---|---|---|
| base | **5/10** | 8.2 s | 117 k | 18 | 1 |
| low | **3/10** | 3.7 s | 11 k | 51 | 3.5 |
| fast | **0/3** (stopped) | 3.7 s | — | 50 | 9 |

base vs low: Fisher p = 0.65; base vs fast: p = 0.23 — neither is a finding at this n.

Per layout (✓ success; outcome otherwise; env steps used of 200):

| Layout | base | low | fast |
|---|---|---|---|
| 0 | ✓ 144 | fail 185 | max_turns 164 |
| 1 | ✓ 150 | ✓ 141 | max_turns 156 |
| 2 | ✓ 154 | fail 196 | max_turns 108 |
| 3 | fail 181 | fail 176 | — |
| 4 | fail 191 | fail 189 | — |
| 5 | fail 193 | step limit 200 | — |
| 6 | ✓ 85 | ✓ 70 | — |
| 7 | fail 186 | fail 179 | — |
| 8 | fail 178 | ✓ 173 | — |
| 9 | ✓ 104 | fail 190 | — |

Run-to-run noise is large: the base arm is the pilot's exact configuration, and on layouts 0–4 it
went 3/5 here (L0, L1, L2) vs 2/5 in the pilot (L0, L4).

Runs: `runs/ab_base/`, `runs/ab_lowslow/`, `runs/ab_fast/`; RoboDojo results in
`runs/robodojo_eval/20261004T181412Z_*`, `…185004Z_*`, `…191021Z_*`.

## Failure modes

- **fast: grasps slip.** Every episode re-grasped 4–10 times and hit the 60-turn limit with
  steps left (L0: 39 left). Typical: "The grip is closing down to 9 mm, which means the fingers are
  squeezing past the thin scissors and slipping" (L0 t27), "Every grasp so far has slipped" (t48).
  The faster joint motion and/or the faster gripper close (2 steps instead of 4) loses objects that
  the slower arms hold — which of the two was not separated.
- **`low` does not remove the reasoning, it moves it into the reply.** Hidden reasoning fell from
  117 k to 11 k tokens, but the median reply grew from 18 to 51 words of prose before the command
  (the manual says "no prose"). Turns are still 2.2× faster (3.7 vs 8.2 s p50).
- **low: more re-grasps** (median 3.5 vs 1 closes per episode) and the env-step budget again
  (5 of 7 failures used ≥ 176 steps).
- **base: the model gives up** with steps left after one or two failed grasps (L7 FAIL after 10
  turns), and every base failure is a FAIL with ≥ 178 steps used.

## Conclusions (given n)

- Effort low + smaller JPEGs: 2.2× faster turns (3.7 vs 8.2 s), success 3/10 vs 5/10 — not
  distinguishable at n = 10, so the speed-up is not free of doubt; it also makes Sonnet write
  prose.
- 0.12 rad / 0.5 gripper per env step: landing accuracy in free space is fine (probe), but with
  objects in hand it made grasps fail (0/3, 9 closes per episode). Saving env steps this way does
  not pay; `robodojo_fast.yaml` keeps it only as a documented arm.

## Found along the way → BACKLOG

- Separate arm speed from gripper speed (`arm_step_rad=0.12, grip_step=0.25`), or move fast only
  while the gripper is open (approach) and slowly while holding.
- `run.sh` cleanup: Isaac ignored SIGTERM and kept 5 GB of GPU memory → SIGKILL fallback (PR).

## Next experiment

Before more arms: repeats. base 5/10 vs pilot 2/5 on the same config says per-episode variance
dominates; ≥ 2 repeats × 10 layouts per arm (≈ 400 calls per arm) before comparing settings.
