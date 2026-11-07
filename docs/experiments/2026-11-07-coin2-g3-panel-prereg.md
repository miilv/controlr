# COIN2 fresh paired G3 panel: the thin-disc contact profile under camera-like belief errors (pre-registration)

Written before any panel episode; the profile below was fixed from the dev screen on the re-test's 20 camera layouts
(report: [2026-11-07-coin2-contact](2026-11-07-coin2-contact.md)), which this panel never reuses.

**Question.** With the coin's believed pose as wrong as the camera's, does the profile (B) hold, carry and insert the
coin near the bar on physics, against today's skill (A)? The bar: REPRO's PyRUA-RSI number on our dev layouts
(`docs/experiments/data/rsi-yardstick/per_cell.json`) when published; until then 79.6 (their seed 0).

## Inputs

- Fresh layouts: two salted blocks ([salts](data/coin2/panel-salts.json), fixed before generation): ids 100000-09
  and 100020-29 (OUR_MEASUREMENT_SLICES), unchanged generator, default two-pass presettle on compute3. Every stable
  record is used; unstable ones are dropped and listed, never refilled after outcomes. Fewer than 10 stable: no panel.
- Belief errors: each layout gets one camera pick error from
  [camera_pick_errors.json](data/coin2/camera_pick_errors.json) (32 samples: first edge-pick belief vs GT on the
  re-test's 40 episodes), index `int(sha256(f"{salt}|{id}"), 16) % 32`. The coin and its stand get the same yaw error,
  and an xy shift `dn` along the coin's GT normal plus `du` along its plane (GT from the settled record). The
  resulting `offsets.json` is frozen with its hash before the first episode.
- Code: one deployment (below); both arms differ only in `--skill-attr`.

## Arms (G3, compute3, light5 / light6, `DOJO_GRIP_SOURCE=v2`)

| run | belief | A | B |
|---|---|---|---|
| gain | mode B + the per-layout camera error | all disc switches off | PROFILE |
| guard | mode A (GT beliefs) | all disc switches off | PROFILE |

PROFILE = `disc_align=walls, disc_snap=on, disc_lift=shallow, disc_seat=table, disc_release=staged` (= `coin_contact.PROFILE`; `disc_reserve` off), passed as `--skill-attr`. Deployment: `~/controlr-coin2-e3d0e722` (branch `feat/dojo-coin2` at `e3d0e722f1c5792bdf42a92165fe66ae1e7ae282`). Chosen on the dev screen because it was the only arm that held at both +10 deg (19 / 20) and -12 deg (18 / 20); snap without the wall probe was 4 / 13 at -12 deg. The screen ran on `0207298d`; `e3d0e722` adds the general `insert_thin` entry, the thin-axis generalisation (identical for discs) and takes the wall fit's intercept from the holder's catalogue mesh (0.0362 vs the measured 0.0347: it moves only the fitted offset, by ~0.1 mm).

## Decision rule (fixed now)

- Gain: B pooled score >= A + 5 points; paired score wins (B - A >= 10 points) >= 2 x losses and wins - losses >= 2.
- Guard: B pooled >= A - 5 points, and B has no more success-flip losses than wins.
- "Near the bar": B's gain-run pooled score with its sample SD and a paired bootstrap CI (10,000 resamples,
  `default_rng(0)`), reported against the bar; it is a statement, not a gate.
- Infra failures (`category infra_failure`): one identical re-run; a missing pair blocks the verdict.

## Stage measures (every episode, from `m95_replay.py --every 1 --full` + `scripts/skills/coin2_stages.py`)

Fingertip clearance over the holder at each close; coin centre below the TCP at the close and after it settles; hold
loss stage (lift / carry / handover / release); slip in the pads; release pose in the bank's frame (slot-plane error,
offset across the slot, coin bottom vs the bank's top); end state (in bank / high in the slot / holder / table);
steps used; `wall_probe` fit vs the true error when `disc_align` is in the profile. Mistaken "held" decisions are a
camera-path metric (G3 mode B sees a held coin at its true pose); reported from the camera episodes only.

No LLM calls. Spend: physics only, ~80 episodes.
