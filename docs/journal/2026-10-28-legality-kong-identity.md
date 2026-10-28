# 2026-10-28: make_kong identities from image evidence only (legality audit P1-2)

Branch `fix/legal-kong-identity`. Audit: robot-harness `docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md`,
finding 2. 0 LLM calls, 0 GPU jobs, no episodes.

## Finding
make_kong has 42 glyph candidates per tile, more than B1's slot identity read handles (`slot_identity_max` 12). So B1
read no glyph and placed every tile at its config slot with a placeholder asset. `inverse_layout.index_structure` then
gave each config-tied group one placeholder index, distinct per group:
- each hand triple / the pair: `same`;
- push tile 5+k with triple k: `same_index_as_label`;
- mahjong9_0 with the pair.

`objects_from_record` turned these into ordinary asset keys, and `declare_kong` picked the hand tiles whose key equals
the discard's. Which push tile the opponent knocked over was a legal image read (`task_state.kong_discard`). Which hand
tiles match it came from the task config, i.e. the answer. `task_state` also hard-coded the same tie
(`PUSH_TO_TRIPLE`, in the watch's event detail; nothing consumed it). The agent's scene printed the tied indices too
(`asset N`), so the LLM agent saw the relation as well.

## Fix
- **`dojo/perception/identity_marks.py`** (new, pure python). `identity_marks(record, task)` replaces
  `index_structure` in `estimate()`. It marks entries B1 placed from the config without choosing the asset: config
  slots, labels stacked on another, support slots, fixed-pose fixtures.
  - Such an entry with several candidates and no read gets `_perception.identity = "unknown"`.
  - Its `category_idx` stays the placeholder (the first candidate); the twin and the meshes need an asset.
  - Of the config it reads only the candidate pool sizes, never `select_mode`.
  - `identity_known(p)` is False for "unknown" and for the old "structural" mark (records written before this fix).
- **`_slot_identity`** marks `identity: "read"` when the best candidate beats the runner-up by more than
  `SLOT_READ_MARGIN` (1e-4). A face the views do not see (a tile's back) scores every candidate the same: no read.
- **`ObjectState.identity_known`** (default True: GT and every non-slot B1 entry). `objects_from_record` sets it from
  the mark.
- **`library.kong_match` / `declare_kong`**. They match only tiles whose identity is known. An unread discard, or no
  read hand tile, stops the skill at stage "roles" before any arm moves. The note reads "no image evidence for the
  matching tiles: the discard mahjong5_0's glyph was not read from the images". The draw's pair preference also counts
  known identities only.
- **`task_state.kong_discard`** returns the knocked label and the scores, no "triple". `PUSH_TO_TRIPLE` became
  `PUSH_LABELS`, the four slot labels.
- `index_structure`, `IDENTITY_READ_MAX` and `test_index_structure_ties_config_groups` (which required the ties) are
  gone.

## Other consumers of the config's identity relations (reviewed)
- `same_index_as_label`: after this PR no policy-side code names it. Only the generator (`dojo/simlab`, dev) does,
  e.g. fasten_screws' nut k / bolt k.
- `mode == "same"` in `_options` and in B1's identity stage. It constrains the members of ONE select list to one
  asset, and that asset is read jointly from the pixels (`argmin` of the summed costs). It applies only when a label
  group equals the list. make_kong's 22 tiles form one wider group, so it never applies there. The tasks it does apply
  to (stack_bowls, align_blocks, insert_tubes, ...) do not score which objects share an asset. Left as is.
- **`same_as_label` in `quota_lists` / `_quota_copy_assets`** (imitate_sorting_sequence, `quota_copy_assets` default
  on). The far list (aim0..4) is constrained to the assets of our list (t0..4), each once. The pairing itself is then
  chosen by fit cost, and the label-level aim_k = t_k tie is not used. The instruction says "place the corresponding
  objects", so a set-level correspondence is arguably instruction-derived. But it does narrow far, poorly read objects
  by a config relation. **Needs the lead's ruling.** It is the one reviewed exception in
  `test_policy_code_reads_no_config_identity_relation`.
- **The kong draw's tie-break** (wall stack closest to the hand row in y). make_kong's wall tiles lie face down, so no
  glyph read can tell the stacks apart. The fixed config layout puts mahjong9_0 (the pair's tile) on the nearer stack,
  so this geometric rule always draws the right tile. It is a fixed-slot prior, not an identity relation, but it
  coincides with the `same_index_as_label` tie. Today the skill stops at "roles" before the draw. **Flag for the
  lead's ruling** before any glyph read makes the draw reachable.
- Agentic perceiver, `seed_refine`, `j1_adapter`, the planner: none reads these relations. The agentic merge keeps
  the "unknown" mark. An agent that changes an unknown tile's asset leaves it "unknown" (conservative; follow-up in
  BACKLOG).

## Tests (`tests/dojo/test_legality_kong.py`, fakes only)
- (a) `test_hidden_config_tie_changes_no_identity_and_no_kong_selection`. Identical synthetic B1 slot records (42
  candidates) and the same kinematic scene, under three hidden configs: the real ties, none, rotated ties. The real
  path runs: `identity_marks` -> `objects_from_record` -> `declare_kong`, with a provider that gives the skill
  perception's identities. Records and identities are identical (all "unknown", index 0). All three outcomes are
  identical: `{"ok": False, "stage": "roles"}`, `match == []`, every hand tile still standing. The audit's probe:
  key equality alone would "match" all 14 placeholder tiles.
- `test_an_unread_placeholder_never_matches_a_read_glyph`: a read discard whose index equals the unread placeholders
  matches nothing.
- (b) `test_glyphs_read_from_the_images_drive_the_selection`: read glyphs select triple g for each discard glyph g,
  under every config.
- (b) `test_read_glyphs_declare_the_kong_against_the_config_tie`: end to end on the kinematic env. The config ties
  mahjong5_0 to triple 0, the read glyph says triple 2. The skill tips triple 2 and leaves triple 0 standing; the
  kong is ok.
- `test_identity_marks_contract`, `test_policy_code_reads_no_config_identity_relation` (AST scan of
  perception / runtime / skills / agent / bridge).
- With the perception extras (cv2 / scipy; they skip in CI):
  - `test_index_structure_is_gone_and_estimate_marks_without_the_config`: monkeypatched `task_map`, tied vs untied:
    identical records;
  - `test_slot_identity_reads_only_with_a_margin`;
  - `test_kong_discard_reports_the_knocked_tile_only`.

  Run locally with another worktree's perception venv.
- Mutation checks: with key-equality selection restored in `kong_match`, (a) and the placeholder test fail. With a
  config tie written into `identity_marks`, (a) fails.

## Behaviour change and numbers
- **make_kong, perception modes** (integ / proxy / official / bridge, RuntimePolicy and the agent's `declare_kong`).
  With no glyph read, the skill holds for the discard, then stops at stage "roles" without moving an arm. Expected
  episode score 0. Before: 10/10 = 100.0 on 5.1 mode B (experiments 2026-10-27-kong-mode-b), and 80 in breadth3 r2a
  (robot-harness ledger row 160).
  - If that row ran this path, the dev-mix averages lose its 2.0 points: 35.8 -> 33.8 and 27.4 -> 25.4. That is
    the audit's computation, on top of the other findings' corrections.
  - The LLM agent may still try its own route through `robo` (it sees the frames and the failure note): unmeasured.
  - Nothing was re-run here. The next rendered round measures it.
- **Not changed.** G3 and the runtime's gt mode (GT identities are known). Every other task: no other code reads the
  mark, and `_slot_identity`'s choice of asset is unchanged. The twin input changes only where `index_structure` used to
  write. That needed a tied group whose labels all sit at config slots with more than 12 candidates: make_kong's tiles,
  which now all carry the first candidate instead of one placeholder per group (same geometry). No catalogue here, so the
  pools of the other tied configs were not enumerated. Where the configs list their indices, the pools are small
  (fasten_screws' 5 nuts, stack_bowls' 8 bowls).
- The watch event's `detail` no longer carries "triple". make_kong's G1 identity scores now compare placeholder 0
  instead of the per-group placeholders. Neither was a read.

## Deferred
- **An image-based kong match** (flag, default off). The natural route: read the face-up discard's glyph after the
  watcher lays it down, and read the hand tiles' visible faces (`slot_identity_max >= 42` already reads them at frame
  0 with `texture_cost`). Then match on the reads. Not done here: it needs rendered frames to measure (the hand faces
  are ~20 px tall in the head view; S's segmenter reads mahjong glyphs at 0.19), and this stream runs no GPU jobs.
  BACKLOG.
- The agent scene prints the placeholder `asset 0` for unknown identities. It should print "unread". That changes the
  agent's prompt for every task with unread slots, so it needs its own A/B. BACKLOG.
