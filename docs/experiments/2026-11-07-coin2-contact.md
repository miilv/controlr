# 2026-11-07 — COIN2: does a thin-object contact sequence hold the coin under the camera's belief errors?

**Hypothesis.** The camera loses deposit_coin in physics, not only in belief: the believed plane of the coin is a few
degrees off, the pads twist it against its stand's 2.6 mm slot and it slips out on the lift. A contact sequence
built for that — squeeze the holder's walls to measure the plane, close on the coin as the fingertips clear the
walls, lift shallow, re-seat on the table, release in stages — holds it.

**Setup.** G3 on compute3 (light5 / light6, scoped 8 G / 600 %, `DOJO_GRIP_SOURCE=v2`), mode B (no in-hand
re-measure) with the coin's and the stand's believed yaw offset together. Dev screen: the disc re-test's 20 camera
layouts (`disc-gain1..4`, ids 100000-04). Panel: 20 fresh salted layouts, pre-registered
([prereg](2026-11-07-coin2-g3-panel-prereg.md), on main at `878c1e2b` before any panel episode). Code: branch
`feat/dojo-coin2` (PR #427); screen at `0207298d` / `a2ad3592` (baselines), panel at `e3d0e722`. Journal:
[2026-11-07-dojo-coin2](../journal/2026-11-07-dojo-coin2.md).

**Budget and spend.** 0 LLM calls, 0 tokens. Physics only: 40 camera-episode replays, 260 screen episodes, 4 probes,
80 panel episodes + their replays; at most two COIN2 jobs at a time.

## 1. Camera: where the coins are lost (40 re-test episodes, physics replays, GT for scoring only)

Replayed scores match the recorded ones 40 / 40. First close at the holder: never closed 10, lost in the lift 14,
lost on the carry 9, held to the release 7. End states: holder 16, flat on the table 11, high in the slot 9, bank 3.
Fingertips 3.9 mm over the holder's top at the close (median), the coin's centre 18.1 mm below the TCP. Belief at the
pick: plane yaw error 3.9 deg median (p75 9.9, max 17), offset along the normal 0.8 mm median
([samples](data/coin2/camera_pick_errors.json)); the slot allows about +-1.4 deg. held_look against GT: 7 / 17 held
coins read "dropped", 0 / 39 empty hands read "held".

## 2. Dev screen (20 camera layouts each, G3 mode B unless noted)

| arm | belief yaw | n | success | score | SD | steps median / max | failures |
|---|---|---:|---:|---:|---:|---|---|
| baseline, mode A (GT) | 0 | 20 | 18 | 92.0 | 24.6 | 168 / 279 | slip 1, place 1 |
| baseline | 0 | 20 | 17 | 88.0 | 29.3 | 164 / 248 | slip 1, place 2 |
| baseline | +10 | 20 | 3 | 19.0 | 35.8 | 66 / 253 | slip 14, place 3 |
| baseline | -12 | 20 | 1 | 6.0 | 22.6 | 54 / 256 | slip 17, place 1, ik 1 |
| snap | +10 | 20 | 17 | 85.0 | 36.6 | 198 / 256 | slip 2, place 1 |
| snap + lift | +10 | 20 | 17 | 85.0 | 36.6 | 168 / 252 | slip 2, place 1 |
| snap + lift + seat | +10 | 20 | 17 | 87.0 | 32.0 | 168 / 252 | slip 1, place 2 |
| snap + lift + seat + release | +10 | 20 | 17 | 87.0 | 32.0 | 172 / 256 | slip 1, place 2 |
| snap + lift + seat + release | -12 | 20 | 6 | 30.0 | 47.0 | 70 / 258 | slip 11, place 1, ik 2 |
| all five (+ wall probe) | 0 | 20 | 19 | 96.0 | 17.9 | 212 / 266 | slip 1 |
| all five | +10 | 20 | 19 | 96.0 | 17.9 | 218 / 266 | place 1 |
| all five | -12 | 20 | 18 | 92.0 | 24.6 | 215 / 279 | slip 1, step budget 1 |

- Snap alone fixes +10 deg (cover 12 mm instead of 9 mm; its 3 failures were +-12 deg tilted grasps) but not -12 deg.
- The wall probe (3 squeezes, 54 steps median) squares the hand: fitted apex error 0.3 deg (median) at 0, 2.2 at +10,
  4.0 at -12 deg — biased toward 0 when the true apex lies at or past the outer squeeze — and the snap close tolerates
  what is left. Fitted offset along the closing axis 0.06-0.27 mm (true 0).
- Lift / seat / release add nothing measurable at n = 20 here; they stay in the profile because the screen chose the
  all-five arm (the only one that held at both signs), not because each was shown to help.
- Steps: the full profile uses ~50 more steps (median 212-218 of 300); one -12 deg episode ran out of steps.

## 3. Pre-registered fresh panel

20 fresh salted layouts (`coin2-panel-a` 100000-09, `coin2-panel-b` 100020-29; 20 / 20 stable), one camera pick
error each ([offsets](data/coin2/panel/), sha256 `46b505ac...` / `5191fdf2...`; 9 of 20 with >= 5 deg yaw error),
deployment `e3d0e722`. All 80 episodes ran once, rc 0, no infra failures. [Verdict data](data/coin2/panel/verdict.json).

| run | A pooled | B pooled | delta | A / B SD | paired CI of delta | score wins / losses | successes A -> B | steps median A / B | registered rule |
|---|---:|---:|---:|---|---|---:|---|---|---|
| gain: mode B + camera errors | 49.0 | **76.0** | +27.0 | 47.9 / 42.8 | [-3, +55] | 9 / 3 | 9 -> 15 | 133 / 214.5 | **pass** |
| guard: mode A (GT beliefs) | 83.0 | **100.0** | +17.0 | 35.1 / 0.0 | [+4, +33] | 4 / 0 (flips 4 / 0) | 16 -> 20 | 195.5 / 236.5 | **pass** |

B's gain score 76.0 (bootstrap CI of the mean [56, 91]) against the bar: 79.6 (PyRUA-RSI seed 0; REPRO's number on our
layouts is not published yet for this cell). Failure categories A / B: grasp_slip 8 / 1, place_miss 2 / 1, ik_fail
1 / 3. B's losses: three picks judged loose (in-hand drift 9-13 mm) whose put-down left no rim grasp (the disc path has
no further retry), one coin lost at the in-air handover, and one layout with a 10 mm offset where neither arm planned
(A too; the probe fit was not bracketed there).

### Stages (GT physics replays of all 80 panel episodes; replayed scores match 80 / 80; [summary](data/coin2/panel/stages-summary.json))

| | gain A | gain B | guard A | guard B |
|---|---|---|---|---|
| first holder close: held to release / lost in lift / lost on carry / never rose | 12 / 5 / 2 / 1 | 15 / 1 / 0 / 4 | 19 / 1 / 0 / 0 | 17 / 0 / 0 / 3 |
| fingertips over the holder's top at the grip (median) | 3.8 mm | 0.95 mm | 3.7 mm | 1.0 mm |
| coin centre below the TCP (median) | 17.75 mm | 14.85 mm | 17.75 mm | 14.8 mm |
| slip in the pads while held (median) | 5.3 mm | 2.65 mm | 1.95 mm | 0.5 mm |
| end: in bank / flat on table / holder / elsewhere | 9 / 7 / 2 / 2 | 15 / 5 / 0 / 0 | 18 / 1 / 1 / 0 | 20 / 0 / 0 / 0 |
| release: slot-plane error / offset across the slot (median, coins in the bank) | 0.0 deg / 1.2 mm | 1.4 deg / 1.15 mm | 0.0 deg / 1.0 mm | 1.05 deg / 1.25 mm |

- B's first closes are the snap (the pads first close on the walls 4.6 mm under their top, then rise and grip with the
  fingertips ~1 mm over it): ~3 mm more of the coin, half the slip; lift / carry losses 7 -> 1.
- What is left: a first snap that does not take the coin (never rose: 4 / 20 gain, 3 / 20 guard). With GT beliefs
  the retry recovers all three; under belief error the put-down leaves no planned rim grasp and the disc path stops.
- Wall probe (gain B): 18 / 20 fits accepted; |yaw error| 1.37 -> 0.45 deg (median); large errors (15-17 deg) only
  down to 4.6-12 deg (the fit's bias toward the +-10 deg squeezes); the 2 refused fits had 4-10 mm normal offsets.
  Offset estimates 2.5-2.8 mm for true 3.1-4.1 mm. 55 steps (median).
- Mistaken "held" decisions: none possible in G3 mode B (a held coin is seen at its true pose); on camera 7 / 17 held
  coins were read "dropped" (section 1), unaffected by this change until the camera A/B.

## Conclusions (given n)

- On physics with the camera's belief errors, the profile moves deposit_coin from 49 to 76 on 20 fresh pairs
  (9 / 3 score wins; registered gain rule passes), and from 83 to 100 with GT beliefs (guard passes). 76 is near the
  79.6 bar (CI [56, 91]); n = 20 cannot tell 76 from 80.
- The physical mechanism is the snap close plus the wall probe; the screen's -12 deg arm shows the probe is what makes
  the snap robust to the error's sign (6 / 20 without, 18 / 20 with).
- The camera gap is not only this: 10 / 40 camera episodes never closed on the coin (perception / plan), and held_look
  drops held coins. G3 says nothing about those.
- Limit seen: the retry after a snap that did not take the coin; the wall fit's bias at large errors; ~50 extra steps
  (B median 214 of 300; one step-budget end in the screen).

**Found along the way → BACKLOG:** tilted grasps under snap; the probe fit's bias near a squeeze turn (a fourth
squeeze at the apex); held_look's false "dropped" on held coins; the camera A/B of the profile.

**Next:** the camera A/B (slot request in the PR / reply).
