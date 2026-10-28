# 2026-10-28 — legality: the card digit path (audit P2-6, G2-2, G2-4)

Brief: the "card digit" fix stream of the legality work. Sources (robot-harness, read-only):
`docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md` finding 6, `docs/notes/G1F-VERIFICATION-2026-10-28.md` "For
Ilia" items 3 (G2-2) and 5 (G2-4), with `REVIEW-G2-B1-PERCEPTION-2026-10-28.md` findings 2 and 4 behind them. Offline:
fakes only, 0 LLM calls, no GPU, no ssh.

## 1. Findings, checked on main 1e0f7b0

All three reproduce in our current code (the audit's file:line refs are to robot-harness's import of db48652).

- **P2-6: the agent's corrected digit never reached the planner.** The record path is `agentic.record.agentic_estimate`
  -> `j1.finish_answer` -> `record.merge_into_record` -> `runtime.estimate.objects_from_record` -> `planner.card_digit`.
  Five places lost the correction:
  - `merge_into_record` skipped a same-pose entry (`moved < 1e-3: continue`), so a `{"keep": 0, "value": 8}` answer
    left no trace (`merge_log=[]`);
  - a moved entry took the pose and the asset, never `value`;
  - `ObjectState` had no digit field, so `card_digit` always fell back to `int(o.idx)`, B1's index;
  - `record_objects` (the agent's hint) carried no digit at all;
  - the bridge's server route (`record_from_perceived`) also dropped `value`.
- **G2-4: keep and unmentioned copies dropped B1's digit.** `answer_to_output`'s keep and `add_unmentioned` copy only
  `OBJ_FIELDS`, which has `value` but not `attributes`, where B1 puts the digit. On the J1 route a normal keep answer
  delivered cards with no digit, and the press driver stops before motion. `snap_to_b1` has the same gap for an agent
  point without a value.
- **G2-2: a digit was exported without glyph evidence.** `_slot_identity` scores every candidate with
  `texture_weight * (0.25 if tx is None else tx)`. With no texture score (no GL, a flat or hidden glyph, a missing
  textured GLB), all nine cards tie on the 0.25 penalty and the first candidate wins. j1_adapter then exports that
  catalogue digit (`attributes.digit`, the J1 `value`), and `card_digit` used the index. The record did flag
  `ambiguous` (margin 0 < 0.05), but that flag is about the pose and nobody reads it for the digit.
- **More places that read a digit from an unread index**, found on the way and fixed in the same PR:
  - the runtime's search fills a not-found card with `same[0]`, a candidate's index;
  - the LLM agent's scene (`dojo/agent/scene.py` "shows N", `robo.Obj.digit`) printed a digit from the index;
  - J1 budget mode's tracker pairs a tracked object with B1's re-read and takes the re-read's fields, so the next
    B1-gated call replaced an agent correction with B1's digit.

## 2. Fix

One module, `dojo/perception/digits.py` (stdlib only), holds the evidence rule, the record / J1 fields and the
arbitration rule. Everything else calls it.

- **B1: identity confidence is separate from pose.** `_slot_identity`, and the glyph Hungarian path for non-slot
  texture-critical groups, record `per_view["glyph"]` = `{key, read, why, texture, margin, confidence}`.
  - A read needs a texture score for the chosen candidate AND a margin over the runner-up above
    `EstimatorConfig.glyph_min_margin` (new field, default 0.0: ties are unread).
  - Identity confidence = min(1, margin / `ambiguity_margin`).
  - The record export adds `_perception.digit` / `digit_conf` / `digit_source` ("texture" | "config" | "unread") /
    `digit_unread` ("no_texture" | "tied" | "not_read") to number_card entries.
  - An unread digit also sets `ambiguous`, which asks for the view request / scan trigger / agent gate that ambiguity
    already drives.
  - The asset choice and the pose are unchanged.
- **J1 B1 export** (`j1_adapter._digit_export`): `attributes.digit` only for a read card. An unread card gets
  `digit_unread` and no digit, so the J1 schema has no `value` for it. `carry_prior_digit`: a card this call cannot read
  keeps the previous call's read for the same label (a read beats no read; `merge_prior`'s margin comparison stays as
  it was). As in `merge_prior` since #224, a prior entry `prior_unusable` rejects (no confidence or pose of its own)
  re-identifies nothing.
- **Agentic J1** (`j1.py`):
  - keep / snap / unmentioned copies carry the paired B1 object's digit (`_b1_digit`);
  - the agent's own `value` carries its confidence (`_value_conf`: its `confidence`, else 0.8);
  - `settle_digits` applies THE RULE once after H2 / arbitration. The result goes out as `value` (+ `digit_conf`,
    `digit_source`) or as `digit_unread`, and the asset follows the digit (`digits.asset_for_digit`);
  - rule events go into the call's problems / notes;
  - the hint shows B1's digit, or `value=unread (why)`;
  - the tracker re-applies the rule when it pairs a tracked agent reading with a B1 re-read (`digits.keep_agent_digit`).
- **Agentic record path** (`record.py`):
  - `record_objects` carries the digit, so the hint shows it and the copies keep it;
  - `merge_into_record` calls `apply_digit` on every matched pair, the same-pose ones included. It writes the digit
    fields, moves the card's `category_idx` to the asset that shows the digit, and logs
    `{"action": "identity", "digit_from", "digit_to", "idx_from", "idx_to", "source", "was"}`;
  - a found card takes the agent's digit or is unread.
- **World / planner:**
  - `ObjectState.digit / digit_conf / digit_source`, filled by `objects_from_record` (`digits.from_record`). A card
    entry with no digit fields at all (the runtime's search, an old record) counts as unread;
  - `planner.card_digit` returns the observed digit; None for `digit_source == "unread"`; `int(o.idx)` only for cards
    that did not come from perception (GT / G3, the skills' own rows);
  - the press route returns `Plan("none", {"unknown_digits": [...]}, "press count unknown: ...")` when a counted
    button's nearest card has no digit, or there is no card. That is the planner's existing way to say it cannot act:
    `solve` / the supervisor finish the episode without pressing.
- **Bridge server route:** `value` -> `_perception.digit` (`digit_source` "j1"), and the index follows. A card without
  a value is unread. This sits next to #224's orientation rule (a row needs a measured quat).
- **LLM agent scene:** `scene.shown_digit`. The scene says "digit unread" instead of a placeholder digit, and
  `robo.Obj.digit` is None.

### THE RULE (`digits.resolve`)

1. The agent's `value` is **accepted** when it is a digit (0-9, int or one-digit string) that a candidate shows, at
   confidence >= 0.5 (`AGENT_MIN_CONF`). Otherwise it is ignored, and the event (`agent_invalid` / `agent_low_conf`)
   is logged.
2. An accepted value is the digit:
   - it agrees with B1 -> source "agent+texture", the higher confidence;
   - B1 has no read -> "agent", logged `agent_reads`;
   - it differs from B1's read -> "agent", logged `agent_over_b1` with both digits and B1's confidence.

   Why the agent wins the conflict: its hint shows B1's digit and runner-up margin, so a different value is a
   deliberate correction. B1's texture compare is also the side known to fail under occlusion: J1 press L4 re-read a
   card covered by a hovering arm as 1 at margin 0.003; the truth was 5.
3. Without an accepted value, B1's read stands, or the digit is unknown.

## 3. Tests

`tests/dojo/test_legality_card_digit.py`. Fakes only (FakeChat, synthetic records, monkeypatched B1 scorers). The
first eleven tests below (19 cases) run in CI (numpy only). The last three need scipy / cv2 (the perception extra) and skip in CI,
like their neighbours; I ran them with an existing perception venv.

| test | covers |
|---|---|
| `test_digit_map_agrees_with_the_catalogue` | `digits.DIGIT_OF` = `catalog.CATEGORY_ATTRIBUTES` |
| `test_glyph_evidence_reads_only_a_scored_texture_with_a_margin` | G2-2's evidence rule, record fields, search-filled entries unread |
| `test_digit_rule` (9 cases) | THE RULE: agree, correction over B1 (logged), low-confidence / invalid values ignored, unread + agent, nobody read it |
| `test_observed_digit_reaches_the_planner_and_unknown_is_reported` | world + planner: observed digit over the index; unread -> `Plan("none")`, "press count unknown" |
| `test_the_agent_scene_shows_the_observed_digit_or_none` | `shown_digit` |
| `test_bridge_server_records_carry_the_value_and_no_value_is_unread` | bridge server route |
| `test_keep_and_unmentioned_copies_keep_the_b1_digit` | G2-4 through `perceive_impl` + the vendored J1 schema: values [5, 3] (before [None, None]); hint |
| `test_an_explicit_agent_correction_survives_the_copies` | `{'keep': 0, 'value': 8}` -> 8, asset 00008, conflict logged; snap copy; low-confidence correction ignored |
| `test_record_merge_refuses_a_digit_no_candidate_shows` | `record.apply_digit`: rule 1's candidate check; a correction moves the index |
| `test_b1_j1_export_omits_an_unread_digit` | j1_adapter export + `carry_prior_digit` |
| `test_a_tracked_agent_digit_survives_the_next_b1_gated_call` | tracker (budget mode) |
| `test_audit_same_pose_card_correction_reaches_the_planner` (scipy) | the audit's probe: B1 index 5, `{'keep': 0, 'value': 8}`, same pose -> identity log entry, idx 8, planner presses 8 (before: `merge_log=[]`, 5). On a record without digit fields both cards are unread: values for both -> 8 and 3; a value for one -> the other count unknown |
| `test_conflicting_and_uncertain_readings_follow_the_rule` (scipy) | uncertain correction -> 5; unread + no value -> `Plan("none")`; unread + value 7 -> 7 |
| `test_b1_slot_identity_without_glyph_evidence_exports_no_digit` (cv2, 3 cases) | the real `_slot_identity`: flat glyphs (G2-2's probe), a tie, a read |

Against the unfixed modules (only `digits.py` added, so that the file imports), every integration test fails: 13
failed, 11 passed (the pure rule tests). `uv run pytest -q`: 1174 passed, 102 skipped before the new file. With
it: +19 passed / +5 skipped in the base env, and 24/24 pass with the perception deps. `uvx pyflakes` on the touched
files: only the two findings main already had. Rebased onto main 7f3466f (#222-#231 of the other legality
streams). Conflicts were only in the bridge docstring / row builder (#224's orientation rule, kept, with the digit
fields on its rows), BACKLOG and DEVELOPMENT. After the rebase the related suites pass in both envs.

## 4. Behaviour changes (all default; they stop inventing a digit, or stop dropping a read one)

1. B1 records:
   - new `_perception` digit / glyph fields on number cards;
   - `ambiguous` is now also true for a card whose digit is unread;
   - new `EstimatorConfig.glyph_min_margin = 0.0`.

   Poses and `category_idx` are unchanged.
2. j1_adapter J1 output:
   - no `attributes.digit` (so no J1 `value`) for an unread card;
   - new `digit_unread` / `digit_conf` / `digit_source` keys (the J1 schema strips them as unknown keys);
   - a card this call cannot read takes the previous call's read for its label.
3. Agentic J1 output:
   - keep / snap / unmentioned cards carry `value`;
   - the agent's `value` is arbitrated, and the asset follows it;
   - a value below confidence 0.5, or that is not a digit, is dropped;
   - rule events appear in `notes`;
   - the hint text shows the digits;
   - the budget tracker keeps the agent's digit.
4. Agentic record path:
   - the hint shows B1's digits (before: none);
   - same-pose identity corrections are applied;
   - a card's `category_idx` follows its digit, so the twin sees the same card;
   - new `identity` merge-log entries, and `identity_rejected` when no candidate of the label shows the agent's
     digit.
5. Planner: an unread digit, or a counted button with no card, gives `Plan("none")` (before: B1's placeholder index,
   or 1 press for a missing card).
6. Bridge server route: the J1 `value` is the digit (before: the asset index); no value -> unread.
7. LLM agent scene: "digit unread" / `digit=None` for an unread card.

## 5. Can rendered numbers change?

press_by_number, mode B, rendered. I could not measure how often a card is unread offline: no local run keeps the
per-call card records.

- **Agentic perceiver off (B1 only):**
  - **Identical** whenever B1 scored the chosen card's texture and it beat the runner-up by any margin. That covers
    every card read with GL in dev G1 (20/20 correct); the plan and presses are the same.
  - **Changes** only for a card with no texture score or an exact tie. Before, the press count came from the first
    candidate's digit, right only by chance. Now the episode finishes without pressing ("press count unknown"), after
    whatever extra view / scan / gate the `ambiguous` flag triggers.
  - So the success rate drops only by the episodes where that guess happened to be right.
  - Without GL (`DOJO_GL=0`, `J1_GL=cpu`) every card is unread, so B1-only press_by_number now never presses. Before,
    it pressed counts it never read.
- **Agentic perceiver on (`agentic: budget`):**
  - **Yes, in both directions.** The agent's first message now shows B1's digits (or `value=unread`). The requests
    differ from before, so earlier runs do not replay byte-identically.
  - An agent `value` that differs from B1's read now sets the count (before: B1's index).
  - An agent value for a card B1 could not read now supplies the count.
  - An unread card with no agent value stops the episode.
  - How often the agent wrote a different `value`, I could not count offline. The agent call logs on the boxes (answer
    JSON in each workspace) would show it. From now on, every override is an `identity` entry in
    `_perception.agentic.merge` and a `digit ...: rule=agent_over_b1` problem.
- **G3 / mode gt:** unchanged. GT objects carry no digit fields, so `card_digit` uses the GT index.
- **J1 route (robot-harness strict server with our adapters):**
  - B1 adapter: unread cards now come without `value`;
  - agentic adapter: keep / unmentioned cards now come with `value`, which removes G2-4's stop before motion.

## 6. Deferred

- **Re-reading an unread card.** The `ambiguous` flag asks for the existing view request. A dedicated "look at the card
  and re-read its glyph" step is not built, and conservative `fuse` keeps identities: a look does not re-read a
  card. It needs a runtime change (policy.py, the lead's stream).
- **The runtime's record pipeline: a later record replaces an earlier read.** initial -> scan -> agentic0 -> fused ->
  searched -> agentic2: each later record supersedes the earlier one in `init_from_record`. The scan-primary estimate
  pools frame 0's views into its texture score, so it loses a frame-0 read only when the extra views turn a margin
  into a tie. Even so, nothing says "a read beats no read" across these records, as `carry_prior_digit` does on the
  J1 route. That is a policy.py change.
- **`glyph_min_margin` calibration.** 0.0 only rejects ties. The known cases are a wrong read at 0.003 (J1 press L4)
  and right reads at 0.043-0.053, so 0.01 would separate them, but n is tiny. It needs the card margins of the dev G1
  frames.
- **arrange_largest_number's `number` tiles** (glyph identity, digit = idx % 10 in the agent's scene). B1 records their
  glyph evidence now, but `digits.DIGIT_OF` covers number_card only, so their digit is still taken from the index.
- **merge_prior** (another stream): when the prior's identity wins on margin, it copies `attributes` from the prior.
  If the prior card was unread and the current one read, the current digit stays with the prior's asset. Rare, and
  that stream's function.
- **Partial presses.** With one card unknown, the planner presses nothing. Pressing the known prefix could earn
  partial credit if the checker gives any. Not checked.
