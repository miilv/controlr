# 2026-10-28: scoreboard: missing random cells count as 0 and are listed; imputation only on request (audit finding 5)

Legality work, scoreboard stream. Source: robot-harness `docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md`,
finding 5. `average` in `dojo/scoreboard/board.py` silently used the standard score for every missing `<task>_random`
cell. Fed standard-only dev rows, it reported "random = standard" as if the random half had been measured.

**Done** (`dojo/scoreboard/board.py`):
- `average(per_task, key="score", random_missing="zero")`:
  - A cell with no `key` value is missing. It counts as 0 and is listed in `missing_cells` (every dimension, in
    `cells()` order).
  - `random_missing` decides only a missing `<task>_random`:
    - `"zero"` (default): it counts as 0.
    - `"standard"`: it takes the task's standard score and is listed in `imputed_cells`. It stays in `missing_cells`,
      because imputed is not measured.
    - `"error"`: `ValueError` naming the cells.
  - The choice is in the output as `random_missing`.
- CLI `--random-missing {zero,standard,error}` (default zero). The markdown states the choice and has `Missing` /
  `Imputed` columns. Entries with gaps list their missing cells. A `--ref` that is not on the board now says so instead
  of printing 0.00.
- Callers: `main()` is the only one in controlr (grep over `dojo/`, `scripts/`, `tests/`, `docs/`). robot-harness's
  `scoreboard/board.py` is a copy of this file and still imputes. That is their file; it needs the same change or a
  re-import.

**Numbers checked** (`cells()` weights: Generalization 1/120 per cell, Precision / Long-Horizon / Open 1/40, Memory
1/30):
- A lone standard `stack_blocks=100` gives Average 0.8333 with the random cell missing or zero. It gives 1.6667 only with
  `stack_blocks_random=100` given, or with `random_missing="standard"`. These are the audit's numbers; its board has the
  same weights.
- The audit's probe is reproduced from robot-harness `results/ledger.jsonl`, using the rows SB-VERIFICATION-2026-10-27 §3
  reconstructed: ours 73-97 / 150-174, theirs 20-21 + 11-12.

  | | random imputed = standard | random = 0 |
  |---|---|---|
  | breadth3 r1 mix | 29.0250 | 27.3250 |
  | breadth3 r2a mix | 35.8250 | 33.5000 |
  | breadth3 r2a, our rows only | 27.3583 | 25.7000 |
- The public board, fetched once today: 63 entries; 3 have missing cells. ACT and SmolVLA miss 8 non-random cells each,
  so nothing changes for them. DeepSeek-Flash has no random cell at all: 2.99 imputed, 2.75 now. The top 20 are
  unchanged. The default `--ref PyRUA-Lean` is not an entry name there (the site has `PyRUA_Lean`). The old output
  printed its Average as 0.00 and its "Ref" column as zeros; now the summary line says "not on the board".
- RoboDojo's own `scripts/internal/summarize_result.py` fills a Generalization cell only from 25 standard + 25 random
  episodes. It has no half cell and no imputation.

**Dated note on a past journal (not edited, per the journal convention):** `2026-10-27-dojo-integ3.md` §11, "Projection
(not a score)".
- **29.0** was computed with random = standard. Its inputs line says "Generalization random = std", but the same line
  also says "0 elsewhere". With the random cells of the 7 Generalization tasks that have a standard row at 0, it is **27.3**.
- The "**33.5** if his 6-task row is credited to general_pickup and pour_balls" alternative is 29.0 + (100 + 80) / 40.
  That is the per-task split of robot-harness ledger line 10, the legacy non-strict row (measured finger width, not
  contract), as SB-VERIFICATION §3 identified for the round-2a 40.3. With random = 0 it is **31.8**. With line 9's
  strict split (general_pickup 80, pour_balls 60) it is 32.5 imputed / **30.8** with random = 0.
- None of these is a measured random score or a leaderboard score. The legality audit also classifies these rows as M*
  (mixed) dev rows (findings 1, 3, 4). The round-2a numbers live in robot-harness `coord/STATUS-ilia.md`, already
  corrected there (9b42433).

**Tests** (`tests/dojo/test_scoreboard_board.py`; the six `average` tests fail on main's `board.py`):
- `test_a_lone_standard_cell_with_its_random_cell_missing_counts_the_random_half_as_zero`
- `test_the_random_cell_counts_only_when_measured_or_explicitly_imputed`
- `test_error_mode_refuses_missing_random_cells_and_names_them`
- `test_missing_cells_of_every_dimension_are_listed_and_count_as_zero`
- `test_cells_weights_match_the_dimension_means`
- `test_breadth3_round1_dev_mix_reproduces_the_audit` (29.0250 / 27.3250)
- `test_board_cli_lists_missing_cells_and_records_the_choice` (a fake bundle through `main()`)

**Not done:** robot-harness's copy of the board (theirs). The `--ref` default name was left as it is, now that its
absence is visible.
