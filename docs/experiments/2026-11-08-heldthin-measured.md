# 2026-11-08 — held-thin gate by measurement: the agent second look on re-rendered decision frames

Data: [agent rows, B1 rows, key B1 rows](data/heldthin-measured/).

Follow-up to [held-thin offline](2026-11-08-heldthin-offline.md), which left `held_thin=agent` passing the gate only
in expectation. Here the 44 decisive states (B1 opening-only "not held" on a held or dropped 1.9 mm coin) were
**re-rendered and re-perceived**.

**Setup.**
- **Render (compute7, partial use):** `seq_capture` replays of the recorded actions of the 34 coin episodes and the
  10 key episodes, genuine Isaac 5.1 frames at every decision step.
  - Same box class (i9-14900KF + RTX 5090) and RoboDojo commit (`f57e1633`) as the recording box.
  - Fidelity checked: the coin's re-rendered pose equals the compute3 physics replay at the checked state.
  - One job at a time, `nice -n 10`, 8 cores, ≤ 16 GB, ~1 min per episode; the desktop session was not touched.
- **Perception:** `j1.held_look_impl` exactly as the perception service calls it, on the re-rendered frames and the
  recorded official state.
  - Mode `b1` (0 calls) reproduces the recorded verdicts: 43 / 44 identical ("not held"; one now reads "held").
  - Mode `agent` is the second look of `held_thin=agent`: one call per state, `claude/claude-sonnet-5-5`,
    44 calls, no 429.
- **Truth:** the physics replays (dev only).
- Code: local `feat/coin2-heldthin` (`held_thin=agent`, `scripts/perception/heldthin_frames_eval.py`).

## Result

**Agent second look on the 44 decisive states (one call each):**

| truth (physics) | agent "held" | agent "not held" (conf 0.9) |
|---|---:|---:|
| coin held (17) | 14 | **3** |
| coin dropped (27) | 0 | 27 |

**`held_thin=agent` over all coin `held_look` decisions** (false DROP of truly held · missed true drops):

| set | today (R0) | `held_thin=on` (R1) | `held_thin=agent` (R2), measured | gate |
|---|---|---|---|---|
| COIN2-CAM B | 6 / 10 · 0 / 6 | 0 / 10 · 4 / 6 | **2 / 10** · 0 / 6 | ≤ 1 / 10 and no extra misses: **FAIL** |
| COIN2-CAM A+B | 10 / 19 · 0 / 26 | 0 / 19 · 13 / 26 | 2 / 19 · 0 / 26 | |
| all 94 | 17 / 36 · 0 / 58 | 0 / 36 · 27 / 58 | 3 / 36 · 0 / 58 | |

- The agent's errors on held coins: B 100001 step 87 (right wrist), B 100007 step 210 (left wrist), disc re-test B
  4 / 100004 step 210. In the first two frames (inspected) the coin is plainly visible edge-on between the
  fingertips. On these states it errs 3 / 17 on held coins, against 0 / 19 on its own earlier coin states.
- B1's colour share in the fingertip region (`fg_frac`) does not separate held from dropped either: held
  0.10–0.46, dropped 0.00–0.52. No threshold was fitted on these 44 states.
- **Keys without `_official`** (B1 vision only, re-rendered, 30 `grip_check` states, 0 calls): 19 / 19 held keys
  read "held" (no false drops); 8 / 11 truly empty read "not held". The 3 others are the key resting in its lock,
  where my rigid-follow truth is doubtful. The opening resolves a key; if `_official` is ruled out, keys do not
  inherit the coin's problem. Tiles and pieces were not re-rendered here.

## Conclusions

- **Gate decided by measurement: `held_thin=agent` FAILS** on COIN2-CAM B: 2 / 10 false drops against ≤ 1 / 10. It
  removes 14 of the 17 false drops overall and misses no true drop, but one Sonnet look still discards 3 of 17 held
  coins, and two of those three are in B. Per the brief, **no re-test prereg is written.**
- Limit, with numbers: on coins B1's opening carries no information (10.8 mm resolution against 1.9 mm), the
  colour share overlaps, `_official` reads 0.0, and the agent's error on held coins is ~18 % (3 / 17) on the exact
  states. A rule that drops only when two independent looks agree would need the agent's errors to be independent
  across looks, which is unmeasured here (the 44-call budget is spent).
- **Candidates for the next step** (not done):
  - two agent looks (both must say "not held"), or the agent given both wrist views; ~44–88 calls to measure on the
    same frames, which are now kept on compute7;
  - a dedicated "object between or beyond the fingertips" read (the coin protrudes beyond the tips when held);
  - keep `held_thin=agent` for objects the opening cannot resolve, accepting ~2 / 10, if the lead prefers that to
    today's 6 / 10 for a re-test.
- **Spend:** 44 logical Sonnet calls (`claude/claude-sonnet-5-5`), no 429. compute7: one render job, 44 episodes,
  ~45 min, peak 5.7 GB / ~1.3 cores; nothing else touched.
