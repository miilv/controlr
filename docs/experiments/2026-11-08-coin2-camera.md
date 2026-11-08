# 2026-11-08 — COIN2-CAM verdict: the thin-object contact profile on camera — FAIL / NOT ADOPTED

Pre-registration: [2026-11-07-coin2-camera-prereg](2026-11-07-coin2-camera-prereg.md) (on main before queueing,
`97929058`; receipt `683e0341`). G3 basis: [COIN2 contact](2026-11-07-coin2-contact.md). Data:
[verdict.json](data/coin2-camera-verdict/verdict.json). Deployment `compute4:~/dojo-integ-c4-b3e173a6`
(merged main `b3e173a6`, contains #427 / #432). 70 rows, ran 01:5xZ–05:55Z on 8 Nov, lanes heavy1 / heavy2, all rc 0.

**Hypothesis.** The profile (`disc_align=walls, disc_snap=on, disc_lift=shallow, disc_seat=table,
disc_release=staged`) that took deposit_coin from 49 to 76 in G3 under camera-like belief errors raises the camera
score.

## Verdict (registered rule)

| cell | role | n | A pooled | B pooled | Δ | A / B SD | W / L / T (±10 pts) | successes A → B | flips W / L | rule |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| deposit_coin | gain | 20 | 13.0 | 10.0 | **−3.0** (CI −13, +4) | 22.7 / 10.3 | 3 / 3 / 14 | 1 → 0 | 0 / 1 | **FAIL** (Δ < +5; wins 3 < 2 × 3) |
| stack_blocks | strong guard | 5 | 80.0 | 80.0 | 0.0 | 44.7 / 44.7 | 0 / 0 / 5 | 4 → 4 | 0 / 0 | pass |
| insert_key | parity guard | 5 | 32.0 | 15.0 | −17.0 | 38.0 / 0.0 | 0 / 1 / 4 | 1 → 0 | 0 / 1 | parity proof holds (score reported, not gated) |
| play_tic_tac_toe | parity guard | 5 | 100.0 | 100.0 | 0.0 | 0 / 0 | 0 / 0 / 5 | 5 → 5 | 0 / 0 | parity proof holds |

- CI: paired bootstrap of the mean Δ, 10,000 resamples, `default_rng(0)`.
- Parity proof: the resolved configs differ only in the five fields (`rows-check.json`). The 20 guard episodes have
  zero `wall_probe` events and zero `belief_write` with writer `wall_probe`. The key's −17 therefore comes from the
  rest of the stack's run-to-run variation (one success in A), not from the lever.
- **Infra:**
  - `c4b_tttsg_coin2camAr1_100003` was put back before it started and re-ran cleanly.
  - `c4b_tttsg_coin2camBr1_100003` was in flight when the shared perception side reached its 48 h limit (~05:10Z).
    The lane moved it to `.sidedown`; it re-ran cleanly at 05:24–05:38Z. The `.sidedown` attempt (2 calls) is an
    infra attempt; its outcome is not used.
  - Every pair is complete, with no other infra exclusions and no time caps.
- **Route:** from every row's `agentic/**/calls.jsonl`: 70 / 70 rows, every record with calls, are
  `claude/claude-sonnet-5-5` (no other route, no unlabelled record with calls). Grip source `v2` on 70 / 70.
- **Spend:**
  - 309 logical Sonnet calls in the 70 rows (deposit_coin A 123 / B 109; key 17 / 16; stack 13 / 10;
    tic-tac-toe 10 / 11), plus 2 in the infra attempt;
  - tokens: 4.43 M prompt (3.50 M cache read), 190 k completion (97 k reasoning);
  - estimate was ~350.

## What happened (physics replays of the 40 coin rows on compute3, GT for scoring only; replayed scores 40 / 40)

| | A | B |
|---|---|---|
| first close at the holder: held to release / lost in lift / lost on carry / never rose / no close | 1 / 4 / 4 / 3 / 8 | **6** / 1 / 1 / **6** / 6 |
| fingertips over the holder top at the grip; coin centre below TCP; slip while held (medians) | 4.05 mm; 17.7 mm; 8.8 mm | 1.2 mm; 14.95 mm; 4.8 mm |
| end: bank / high in slot / table / holder / elsewhere | 1 / 3 / 9 / 5 / 2 | 1 / 3 / 9 / 6 / 1 |
| held_look "not held" while GT held | 4 of 9 held looks | **6 of 10** held looks |

- **The contact sequence did its physical job:** lift / carry losses 8 → 2, coins held to the release 1 → 6, half
  the slip.
- **The camera stack then lost those coins:** held_look read 6 of B's 10 truly held coins as "dropped". On those, the
  disc path stops ("the disc fell out on the carry") and the finish opens the hands over the table. G3 cannot show
  this: there a held coin is seen at its true pose.
- **The snap missed more often on camera:** 6 of 14 closes never lifted the coin (G3 panel 4 / 20). After the
  put-down, the disc path has no re-plan (BACKLOG).
- Perception never located the coin in 6 B episodes (4 `perception_gross`); A 8 without a close.
- **Steps:** the probe costs 52–78 steps (median 54). B reaches its insert late (held_look at steps 178–257), and 2
  B episodes ended on the step budget.

## Write audit

- **Registered full audit** (physics-only replays of both arms, `m95_score.py audit`, #417 identity, all writers):
  **not run**. The gain rule failed, so the lever cannot be adopted whatever the audit shows.
- **Supplementary, the new writer only (gain cell, B):** 24 `wall_probe` writes (coin + stand, 12 probes) scored
  against the replay GT at their write steps:
  - harmful (> 2 mm added 3-D error) 0, helpful 2. The registered new-writer rule (helpful ≥ 2 × harmful) holds, but
    the 3-D centre metric barely sees what this writer changes;
  - plane yaw error 2.39° → 0.66° (median): 16 writes improve it by more than 1°, 2 worsen it (episode 100000:
    1.1° → 3.3°);
  - one large error (17.8°) only came down to 9.6°, the fit bias at large errors noted in BACKLOG.

## Hard-rule statement

- **Wall probe (`disc_align`):** general. It acts on any thin flat object standing in a fixed slotted holder,
  selected by object geometry, never by a task name. The holder's wall width comes from the catalogue mesh; the plane
  from the robot's own jaw readings. **It reads `<arm>_ee_joint_state_official`** (strict-v2 gripper data).
- **held_look's thin-object rule** (`held_thin` / `disc_hold_evidence`): **not in B**; off in both arms, as in the
  candidate. It is the lever aimed at exactly the false drops above, and it was rejected twice before.
- **Slot targets:** from the bank's perceived pose × its catalogue functional frame, with a mesh fine fit. The
  identity comes from perception. A formal "accepted identity" provenance record (the F strawman) does not exist yet.
- **`grip_check`** (in the candidate, on in both arms) also reads `_official`.
- Until the partner Codex rules on `_official` reads, `disc_align` is **dev-only** and not adoptable into a scored
  config, whatever its score.

## Conclusions (given n)

- FAIL / NOT ADOPTED. This consumes the profile's first camera test. A re-test is allowed once, on fresh layouts,
  under a new pre-registration.
- The G3 gain did not transfer, for a reason G3 cannot see: on camera the profile holds more coins, and the held-state
  read discards them. The next coin lever is the held-state evidence on thin objects in the hand (e.g. not deleting a
  held thin object on one "not held" read), together with a re-plan after a snap that missed. Not the contact
  sequence itself.
- The wall probe is accurate on camera (yaw to ~0.7°) but costs ~54 of 300 steps.
