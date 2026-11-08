# 2026-11-08 — false "dropped" on thin held objects: offline scoring of held-state mechanisms (COIN2, next lever)

Follow-up to the [COIN2-CAM verdict](2026-11-08-coin2-camera.md): the contact profile held 6 coins to the release
and `held_look` threw 6 of 10 truly held coins away. **Question:** which general held-state mechanism removes false
"dropped" on thin objects without missing true drops?

**Setup.**
- Data: recorded events of 80 deposit_coin camera episodes, with physics replays of every step on compute3 (GT for
  scoring only; replayed scores match 40 / 40 and 40 / 40):
  - COIN2-CAM: 40 rows, 45 coin `held_look` decisions;
  - the 7 Nov disc re-test: 40 rows, 49 decisions.
- Guards: the 30 COIN2-CAM guard rows (keys, tic-tac-toe pieces, blocks), 25 of them replayed (5 tic-tac-toe
  replays were still running).
- Spend: 0 LLM calls. Physics only: 2 sequential jobs on compute3 light5.
- Code: local branch `feat/coin2-heldthin` (`7c69e802`), not pushed (repo freeze).

## 1. Where the false drops come from

Coin `held_look` verdicts (94) against GT:

| source / evidence | said "not held", truly held | said "not held", truly dropped | said "held", truly held |
|---|---:|---:|---:|
| B1, jaw opening from the finger silhouettes | **17** | 27 | 0 |
| agentic perceiver (Sonnet) | 0 | 31 | 19 |

- Every false drop is a B1 opening-only negative: "measured 0.0 = commanded 0.0", the coin 15–25 mm from the TCP,
  the hand at its lifted pose.
- B1's true drops look the same: same evidence, same scores (0.59–0.94), same hand pose. For a 1.94 mm object the
  silhouette opening cannot tell held from empty.
- The opening's resolution is `opening_resolution_gap` (10.8 mm at a full close), derived from the vision opening
  margin and the finger geometry, not from the coin.
- The agent was right on all 50 of its coin verdicts.
- The opening is a **vision** measurement: it does not read `_official`.

## 2. Mechanisms, scored (confusion: false DROP of truly held / missed true drops)

| rule | COIN2-CAM B | COIN2-CAM A+B | disc re-test A+B | all 94 |
|---|---|---|---|---|
| R0 today (simulated; reproduces the recorded decisions) | 6 / 10 · 0 / 6 | 10 / 19 · 0 / 26 | 7 / 17 · 0 / 32 | 17 / 36 · 0 / 58 |
| R1 `held_thin=on` (opening-only negative on a thin object → keep) | 0 / 10 · **4 / 6** | 0 / 19 · 13 / 26 | 0 / 17 · 14 / 32 | 0 / 36 · **27 / 58** |
| R2 `held_thin=agent` (→ an independent agent look decides), agent = oracle | 0 / 10 · 0 / 6 | 0 / 19 · 0 / 26 | 0 / 17 · 0 / 32 | 0 / 36 · 0 / 58 |
| R2 with the agent's measured coin record (0 / 50 wrong; 95 % upper error 5.8 %) | ≤ 0.6 / 10 expected | | | ≤ 1.0 / 17 · ≤ 1.6 / 27 expected |
| R3 seen elsewhere, via the runtime's later refine at the believed drop pose (recorded) | does not separate: refine "ok" on 3 truly held and 3 truly dropped; "not ok" on 7 held and 4 dropped | | | |
| R4 proprio without `_official` (strict command only) | no information: the command is what the policy sent | | | |
| R4 proprio with `_official` (`grip_check`) | coin: inconclusive on every close (no verdict; the reading is 0.0 held or empty) | | | |

- **R1 fails the gate** (it misses about half the true drops): this is why `held_thin` / `disc_hold_evidence` lost
  twice.
- **R2 meets the gate only under an assumption:** that the agent is as accurate on the 44 opening-negative states
  as on its own 50 coin verdicts. Those 44 states have no recorded frames (B1 answered, so no agent workspace was
  saved), so the agent could not be re-queried on them offline.
- Cost: one extra agent call per thin opening negative (44 in 80 episodes, ~0.55 per episode).
- **R3:** today's wrist refine sees a 1.9 mm coin no better than B1. A useful "seen elsewhere" would need a dedicated
  head / wrist read of the holder and table, which is not in the recordings.
- **R4 with `_official`:** the threshold below which the reading carries nothing is ~2.5 mm, from the finger model:
  reading = (q + 0.01) / 0.054 − 0.2 with q = w / 2 − 0.4 mm, against the 0.2 control lag. On the guard objects it
  is right (rigid-follow GT): blocks 21 / 21, tic-tac-toe pieces 25 / 25, keys 27 / 30 (the 3 exceptions are a key
  resting in the lock, probably my GT test). So `_official` covers thin objects above ~2.5 mm, never a coin.
- **Re-plan after a missed snap:** the camera runtime already re-plans the insert at program level (B 100001 picked
  twice). G3's runner does not, which is why the G3 panel's missed snaps ended early. Not built here.

## 3. Non-thin regression and generality

- In the guard recordings every held-state decision (102) was made by `grip_check`, which reads `_official`.
  Vision `held_look` never ran on a key, piece or block, so R1 / R2 change no guard decision: no regression, but also
  no test of R2 on other thin objects.
- **If the Codex ruling forbids `_official`,** `grip_check` goes away and every object falls back to vision held
  state. R2 then covers every object thinner than 10.8 mm (keys, tiles, game pieces), and its accuracy there is
  unmeasured.
- **Rule:** "thin" is the object's smallest catalogue extent (through its accepted identity) below the opening's
  resolution under the command; catalogue-wide, no category key, no coin constant. R2 itself reads no `_official`.

## 4. Gate status and the limit

- The offline gate (false drops ≤ 1 / 10, no more missed true drops, no non-thin regression) is **met by R2 only in
  expectation**. R1 fails it; R3 / R4 give no usable signal on the coin.
- **Measurement that would close it:** re-render the 44 decisive states and re-run B1 + the agent on the exact
  frames.
  - `dojo.perception.seq_capture` replays of the recorded actions on a 5.1 rendering box, at the `held_look` steps of
    the ~40 episodes involved;
  - ≤ 44 agent calls on `claude/claude-sonnet-5-5`;
  - roughly 2–4 rendering lane-hours.
- Per the brief, **no camera re-test prereg is written until that measurement passes**.
- **Proposed re-test once it passes** (the family's one re-test): B = the COIN2 profile + `held_thin=agent`, A = the
  candidate; same cells, rules and id slices as COIN2-CAM with a fresh salt; ~350 + 44 Sonnet calls. `disc_align`
  stays dev-only until the `_official` ruling; a ruling against it means re-testing without the wall probe (the
  −12° G3 screen: the profile without the probe 6 / 20, with it 18 / 20).
