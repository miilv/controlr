# 2026-10-27 — perceiver models: Sonnet vs Opus vs Fable, turns, "agent always", and the arbitration setting

Stream: Perceiver-models (perception bake-off), branch `feat/dojo-pbake-pmodels`. Question: which perceiver model and
settings fix the most gross misses without breaking correct objects? The agentic perceiver and its 2026-10-25
measurement: [perception-options](2026-10-25-perception-options.md). Everything here is offline, at frame 0, paired
(the same frames and the same B1 hint for every arm). GT is read only by the scorer.

## Summary

1. **The model matters less than one post-processing setting.** Today's arbitration (`seed_refine.arbitrate`) reverts
   10 of Sonnet's 37 raw fixes to prevent 3 breaks (Opus: 13 of 43 for 2).
   - The cause: an unrefined agent point (a head-plane hit with a coarse yaw) scores a worse silhouette (0.58-0.88) than
     B1's refined fit to the wrong blob (0.33-0.70).
   - The new `arbitrate(risky_b1=0.2)` (default off) lets the agent stand where B1's own calibrated risk is >= 0.2 and
     keeps the comparison elsewhere.
   - Effect on every arm: +7 to +12 fixed, broken +0 to +2, within tolerance +4 to +5.
2. **Models, on 94 scenes / 401 objects** (B1: 84 gross), with risky_b1 0.2:

   | arm | fixed / broken | gross after |
   |---|---|---|
   | Opus 5.5 | 42 / 1 | 43 |
   | Sonnet 5.5, run 1 | 36 / 4 | 52 |
   | Sonnet 5.5, run 2 | 37 / 3 | 50 |
   | Sonnet, 16 turns | 36 / 2 | 50 |

   - Opus's edge is consistent but modest. Opus alone fixes 7 objects, against 1-2 that only one Sonnet run fixes.
     Sonnet's run-to-run difference is 3-4 objects each way. Opus is one run (its repeat died at the 429).
   - Opus costs 47 s vs 31 s median per perceive call, 1.3x the LLM calls and 1.5x the tokens.
   - Fable 5.1 (63 scenes, cut by the 429) is Sonnet-level: 27 / 2 vs 25-26 / 2. It is the slowest: 99 s median and
     209 s p90 per perceive call.
   - 16 turns adds nothing: the median is 3 LLM calls, and 8 turns almost never binds.
3. **"Agent always" on the same view buys nothing.** A second Sonnet call, with the first answer as its hint, nets 1
   fixed / 1 broken on 46 scenes for +155 calls. The gate after the first call fires on 36 of 46 scenes, which hold 25
   of the 26 gross objects left. The leftovers are not visible at frame 0: push_T under the grippers, stack edge blocks,
   pack items B1 never found.
4. **store_laptop** (Integration's genuine-5.1 A/B): the laptop's "77-83 mm off" is mostly a frame mismatch. The
   catalogue mesh drawn at the sim's pose sits about 80 mm in front of the rendered laptop; at +(0, 0.08, -0.025) m in
   the object frame it fits the image.
   - The laptop's 180° flip and the headset identity (00002 vs 00004) are B1-side. No model fixes them.
5. **Recommendation:** Sonnet 5.5 (Opus 5.5 if the latency budget allows), 8 turns, budget gate, `risky_b1 = 0.2` after a
   rendered A/B. Not Fable.

## 1. Data and protocol

B1 (main, `dev_d0 b1`: the 2-fold backgrounds and colour map) was run on 217 candidate frame-0 scenes, then the sets were
chosen as follows. Ids are 100000-09 and 100020-49 only.

| set | source | scenes / objects | B1 gross | chosen |
|---|---|---|---|---|
| hard19 | d0, 6.0 proxy | 19 / 70 | 10 | the 2026-10-25 hard set minus 100010-19 (frames byte-identical, md5) |
| exthard | d0_ext, 6.0 proxy | 36 / 165 | 29 | every B1-gross scene + controls: organize 9, plug 6, general_pickup 4, store_tools 4, insert_key 2, align 3, pen holder 2, stack_by_language 4, press 2 |
| d0hard | d0 | 29 / 136 | 33 | the other 24 B1-gross scenes + 5 controls |
| coin | deposit_coin frames (compute3 copy) | 10 / 30 | 12 | all |
| laptop | Integration's c4_laptop_v2 / v3 frame 0, genuine 5.1 | 10 frames | — | the runtime's own B1 hint, and the scan observation where the episode had one |

- **Every arm** = a fresh agent per scene (`dev_d0 run`), then the runtime's post-processing:
  - snap to B1, then B1's unmentioned objects kept;
  - H2 (`dev_d0 refine`);
  - arbitration (`dev_d0 arbitrate`).
- **H2 and arbitration ran on CPU silhouettes** (`DOJO_GL=0`). The box sat at 5 GPU processes from other jobs, so the
  GPU slot was blocked. All arms were recomputed the same way; GL vs CPU differs by at most 1 object on hard19.
- **Scorer:** dev_d0's (Hungarian per category on xy). Tolerances are eval_g1's where dev_d0 had none. Gross = beyond
  max(15 mm, 2 x tol) or not found. A free-text kind maps to the nearest B1 object's kind, as `merge_into_record` does.
  The class-name match is now ordered: it used a set, whose order changes per process, and moved counts by about 1 between
  runs.
- **Models via omniroute** (`claude/` routes), router-default thinking, `max_tokens` 6000; no reply hit it.

## 2. Reproduction (hard19, 70 objects)

| | B1 gross | B1 within tol | Sonnet + H2 gross |
|---|---|---|---|
| 2026-10-25 (B1 at the J1b pin) | 17 | 44 | 6 |
| today (main B1) | 10 | 51 | 7 (3 fixed / 0 broken) |

- B1 now gets all 6 old tower misses right itself.
- The agent's fixes reproduce per object (pack hammer, car, toothbrush), and so do the leftovers (3 push_T target_t
  under the home grippers, 2 stack 100020 edge blocks, the pack shoe).
- Every model and setting ends at 7 here.

## 3. Results (94 scenes / 401 objects; B1: 290 within tolerance, 84 gross, 20 not found, 0 phantoms)

| arm | arbitration | within tol | gross | not found | fixed / broken / still gross | phantoms | LLM calls (med per perceive) | prompt / completion tokens | s per LLM call med / p90 | s per perceive med / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Sonnet 8 turns, run 1 | today | 305 | 59 | 15 | 27 / 2 / 57 | 13 | 307 (3) | 4.36M / 265k | 9.3 / 17.3 | 31 / 57 |
| Sonnet 8 turns, run 2 | today | 303 | 59 | 13 | 28 / 3 / 56 | 10 | 305 (3) | 4.43M / 265k | 9.1 / 17.4 | 31 / 54 |
| Sonnet 16 turns | today | 306 | 57 | 13 | 29 / 2 / 55 | 16 | 316 (3) | 4.63M / 268k | 9.9 / 17.2 | 30 / 60 |
| Opus 5.5 | today | 310 | 55 | 14 | 30 / 1 / 54 | 14 | 400 (4) | 6.46M / 333k | 10.8 / 19.1 | 47 / 80 |
| Sonnet run 1 | risky_b1 0.2 | 310 | 52 | 15 | 36 / 4 / 48 | 13 | | | | |
| Sonnet run 2 | risky_b1 0.2 | 310 | 50 | 13 | 37 / 3 / 47 | 10 | | | | |
| Sonnet 16 turns | risky_b1 0.2 | 306 | 50 | 13 | 36 / 2 / 48 | 16 | | | | |
| Opus 5.5 | risky_b1 0.2 | 314 | 43 | 14 | 42 / 1 / 42 | 14 | | | | |
| raw agent (Sonnet 1 / Sonnet 2 / Opus) | none | 306 / 308 / 305 | 52 / 49 / 44 | | 37/5, 39/4, 43/3 | | | | | |

**Fable 5.1 on the 63 scenes / 284 objects it completed** (every arm on the same subset; B1: 63 gross):

| arbitration | Fable | Sonnet run 1 | Sonnet run 2 | Sonnet 16 | Opus |
|---|---|---|---|---|---|
| today (fixed / broken) | 20 / 1 | 20 / 1 | 19 / 2 | 22 / 1 | 22 / 0 |
| risky_b1 0.2 (fixed / broken) | 27 / 2 | 25 / 2 | 26 / 2 | 27 / 1 | 31 / 0 |

Fable's cost on those 63 scenes: 5 LLM calls per perceive (median), 19.9 s per call median (p90 45 s), 99 s per perceive
(p90 209 s), 5.34M prompt / 458k completion tokens.

**Per task** (risky_b1 0.2; gross after, with fixed / broken):

| task | objects | B1 gross | Sonnet run 1 | Opus |
|---|---|---|---|---|
| pack_objects_into_box_random | 65 | 22 | 10 (12/0) | 8 (14/0) |
| organize_table | 81 | 17 | 10 (8/1) | 8 (9/0) |
| deposit_coin | 30 | 12 | 10 (3/1) | 5 (7/0) |
| put_bottles_into_dustbin | 35 | 7 | 3 (5/1) | 2 (5/0) |
| general_pickup | 4 | 3 | 0 (3/0) | 1 (2/0) |
| store_tools_in_toolbox | 20 | 2 | 0 (2/0) | 0 (2/0) |
| build_tower | 48 | 4 | 3 (1/0) | 3 (1/0) |
| stack_bowls | 9 | 2 | 1 (1/0) | 1 (1/0) |
| push_T / stack_blocks / plug_in_charger / align / stack_by_language | 16 / 33 / 12 / 12 / 12 | 3 / 5 / 4 / 1 / 1 | unchanged (plug: 1 charger broken by every arm) | unchanged |
| insert_key, press_by_number | 4, 10 | 0 | 0 | 0 |

- **Coin, per arm (fixed / broken):** Sonnet run 1 3 / 1, Sonnet run 2 6 / 1, Sonnet 16 turns 6 / 1, Opus 7 / 0,
  Fable 6 / 0. Sonnet's two runs differ by 3 here alone.
- **Run-to-run noise (risky_b1, per object):**
  - Sonnet vs Sonnet: fixed by both 33, by only one 3 / 4; broken 3 + 1 / 0.
  - Opus vs either Sonnet run: fixed by both 35; only Opus 7, only Sonnet 1-2; broken 1 vs 3-4.
- **Phantoms:** every agent arm adds 10-16 objects the scorer matches to nothing (B1: 0). Most are organize_table
  free-text kinds ("figurine") left beside the object they meant, plus coin-stand duplicates. They don't count in fixed /
  broken, but they reach the belief.
- **H2** is neutral on gross and +3 within tolerance.

## 4. Gate and "agent always"

- **The budget gate on B1's own output** (`needs_agent`: a missing label or a risk >= 0.2) would skip 29 of 94 scenes.
  Those hold 13 of the 84 gross objects; on them the agent fixes 3-4 and breaks 1. The runtime runs the agent at
  frame 0 anyway.
- **"Agent always" at after_looks**, proxied by a second Sonnet call on the same frame 0 with the first call's output as
  its hint (as the runtime passes its current record). On 46 scenes (cut by the 429):
  - raw: 4 fixed / 3 broken against the first call;
  - after H2 and arbitration: 1 / 1, for 155 more LLM calls.
- **The gate after the first call** (`dev_d0 paired --gate-after`, `needs_agent` on the merged record, approximated)
  fires on 36 of the 46 scenes, which hold 25 of the 26 gross objects left. It does not miss the remaining errors here;
  another call on the same view just can't fix them.

## 5. store_laptop_and_headphones (Integration's rendered A/B, 0/5 + 0/5)

- **Laptop position.** The catalogue mesh `Articulation/laptop/00000` drawn at the GT pose lands about 80 mm in front of
  the rendered laptop. GT here is the settled record, which equals what `objects_gt` reports: integ's 82.4 mm is
  recomputed from it.
  - At GT + R·(0, +0.08, -0.025) m with the GT yaw, it fits the image: overlays on 100000 and 100003, and a silhouette
    fit picks dy 0.07-0.09 / dz -0.025 on all 10 frames.
  - So the agents' and H2's laptop (55-104 mm "off") is about right, and the 80 mm is a root-vs-mesh frame offset. It
    went to Integration, since the twin and skills may carry it.
- **Laptop yaw.** The GT yaw fits the image. B1 is 170-180° off on all 10 frames, and every runtime record stage keeps
  it.
  - After H2 and arbitration the yaw is right (< 10°) in 3/10 frames for Sonnet, Sonnet 16 turns and Opus, and 0/4 for
    Fable.
  - The others stay flipped, or carry no yaw, in which case the merge keeps B1's.
- **Headset identity.** GT is 00004 on 100001/02/04. B1 picks 00002, with the alternative 00004 at margin -0.05..+0.08.
  Every arm keeps B1's asset: the agent never sees candidates, and snap and arbitration keep B1's asset.

## 6. Spend and reproduction

- **LLM:** 2,018 calls via omniroute (`claude/`), 519 completed perceive calls. 96 calls were 429 rejections: the Claude
  account's limit was reached at 18:24Z, reset after 61 h; see the [incident](../incidents/2026-10-27-claude-route-rate-limit.md).
  Tokens: 29.6M prompt (25.0M cache reads), 1.86M completion (1.26M reasoning).

  | arm | calls | prompt tokens | completion tokens |
  |---|---|---|---|
  | Sonnet run 1 | 337 | 4.78M | 298k |
  | Sonnet run 2 | 305 | 4.43M | 265k |
  | Sonnet 16 turns | 353 | 5.17M | 304k |
  | Opus | 438 | 7.10M | 368k |
  | Fable | 385 | 5.85M | 514k |
  | second call | 180 | 2.26M | 111k |
  | Opus repeat (all 429) | 20 | 0 | 0 |

  Outputs that fell back to B1 after a 429 (Fable 22, second call 12, Opus repeat 10) are excluded.
- **GPU:** B1 only, on compute4 light2.
- **Reproduce** (compute4 `~/dojo-data/pmodels`):
  - `cases` (dev_d0 case dirs), the `only_*.txt` scene lists, `tools/` (case builders and analysis);
  - `python -m dojo.perception.agentic.dev_d0 run|refine|arbitrate [--risky-b1 0.2 --out-suffix _arbR]|paired`.
