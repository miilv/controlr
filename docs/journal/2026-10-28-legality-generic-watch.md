# 2026-10-28 — the watch read is generic, not a make_kong branch (Legality stream, single-fixed-system audit b3)

**Before:** `perception_service.watch` answered only `if vocabulary == "make_kong"`: a one-task tool keyed on the
instruction-derived vocabulary.

**Now:** `task_state.actor_change(head_before, head_after, vocabulary)`, the same for every vocabulary. It watches every
STANDING movable object at a public config slot (`fixed_slot_labels`, a `straight*` place tag, type Rigid) and reports the
one whose head-view box changed most when it beats the next by the old margin (5 Lab units): a standing object knocked
over by something other than our arms. A vocabulary with fewer than two such objects is unsupported (the watcher stops,
as before). `kong_discard` is kept as the same read over the four push tiles.

**Behaviour:** over today's catalogue (compute3, public configs) only make_kong has standing movable slot objects (14 hand
tiles + 4 push tiles; the two "other" tiles lie down), so every other vocabulary stays unsupported, as before. For
make_kong the read now also watches the hand tiles; the knocked push tile, its score and the confidence are unchanged
(test). make_kong stops at "roles" since #227 anyway (no glyph read), so no rendered number can move.

**Tests:** `tests/dojo/test_legality_watch.py` (needs cv2: skips in CI, passes locally with a perception venv): the
generic read finds the knocked push tile with the old scores and confidence, reports nothing when nothing changed, a
vocabulary without standing slot objects is unsupported, and the watch op names no task. 0 LLM calls.
