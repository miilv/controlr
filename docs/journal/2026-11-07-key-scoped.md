# 2026-11-07 — KEY: task-scoped handover and lean

The [global key re-test](2026-11-06-ab-wave-verdicts.md) improved insert_key
43.3→70.0 over nine valid pairs (successes 3→6), but charger fell 80→70
(successes 8→7). It remains rejected. This implements the new scoped lever,
following KONG #410, with no box work or model calls.

## Consumption and shared paths

`RuntimeConfig.key_handover` and `key_lean` live in `dojo/runtime/policy.py`.
The runtime `_setup` / `_mk` factory previously copied both fields directly
onto every Skills instance, including every rebuild after perception/search.
`dojo/skills/g3.py:SKILL_ATTRS` also permits direct skill-level overrides;
those developer-only G3 switches remain global and unchanged.

`dojo/skills/library.py:insert_into` handles both key insertion (with a turn)
and charger insertion (active insert frames on power-strip outlets). Failed
or unavailable in-air reorientation can use `_insert_by_rest_handover`, then
`regrasp.rest_handover` → `plan_rest_handover`. That shared planner reads
`Skills.key_handover`: v2 tries four rather than two lengthwise giver picks,
adds an upright fallback when no as-lifted handover works, checks sixteen
rather than six taker places and tries 8/5/3/12-cm vias rather than 8 cm only.
Consequently the global handover switch can change charger execution even
though its name says key.

`regrasp.turn_held` reads `Skills.key_lean`. When the full turn plus three
hold steps exceeds the remaining budget, on truncates the move to at least
one waypoint, reserving three holds. Off leaves the executor to reject an
unaffordable turn. `rest_handover` and `turn_in_lock` share that consumer;
`library._turn_if_not_turned` returns immediately for a zero turn (the normal
charger path). Charger therefore shares the handover switch's live path;
sharing insertion alone does not establish that lean caused its regression.
No causal attribution between the two bundled switches follows from the A/B.

## Change and verification

New default `key_scope_task=""` preserves all legacy global behavior. The
only accepted nonempty value is `insert_key`. With that scope, exact matching
against the instruction-routed vocabulary enables configured key parameters;
every other route, including unknown routes, resolves `v1/off`. Scope alone
does not enable either switch. The gate is in the common Skills factory so
rebuilds inherit the same rule, and uses no task/layout metadata. Existing
consumers, G3 skill attrs, presets and candidate are unchanged.

CPU regression tests cover all four switch combinations and every catalog /
regex vocabulary, including unknown and near-match names. Runtime tests use
real key/charger/stack/kong/imitation instructions with misleading launcher
metadata; they compare JSON parameter bytes and actual regrasp consumer line
traces/results against legacy on for key and legacy off for the others, before
and after rebuilding Skills. Synthetic block geometry tests dispatch parity,
not physical key/charger competence. Full-suite and CI results are recorded
in the PR. No motion executor, robot driver, safety or Isaac code changed;
no remote smoke or dispatch was performed, per the task's box-work constraint.

The [camera preregistration](../experiments/2026-11-07-key-scoped-prereg.md)
freezes fresh salts and their hashes before generation, ten gain pairs, five
charger/five stack guard pairs, and at most two pre-camera gain refill blocks
for invalid generation/settles. It selects the current score-based rule
(>=10-point pair wins/losses), not the earlier success-flip gain rule.
The lead owns launch and review; no merge or default promotion here.
