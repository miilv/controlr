# 2026-10-29: stream N, solve_equation with perception's placeholder tile

Stream W found that solve_equation's equation tiles reach the runtime as slot entries on the pads. B1 also puts one
on the EMPTY pad: a placeholder with no identity (glyph "low_margin", `identity_known` False, digit None). In all 15
of W's held-out layouts, the empty pad's placeholder was the one tile that reads nothing. `complete_equation` used to
look at it until its look budget ran out, then stopped with "values unknown".

Changes (`dojo/skills/tiles.py`):
- An unread equation tile gets at most two wrist looks (`LOOKS_PER_TILE`), then the next one does.
- When no pad is empty and exactly one tile still reads nothing, that tile is the placeholder and its pad is the empty
  slot. The answer tile is placed there with the placeholder ignored as an obstacle. GT rows (G3) have no placeholder,
  so mode gt is unchanged.
- Unit test: `test_the_unreadable_placeholder_on_the_empty_pad_is_the_empty_slot`.

Not shipped: the numeral pad-contact filter, branch feat/dojo-newskill-numeral-contact (02d67ce + 36b9a337 memo).
- With it: G3 arrange v1 100030-39 4/9, mean score 0.62.
- Without it (1ed92d3): 3/7, mean 0.63.
- No gain, and it adds planning time.
