# Laptop stream: headset seating

Read CLAUDE, DEVELOPMENT, ARCHITECTURE, EXPERIMENTS; BACKLOG laptop/headset
items; October 29 hang/laptop/overhang journal; 62007227; PR #350; branch
histories (`origin/feat/dojo-gennew-laptop` predates the store module,
`wip/store-laptop` retains the closed-slab approach); November 5 points table;
and both compute3 driver/render incidents.

The live task checker sources confirm headset-only 20 points, laptop 80 as
close + upright + bottom in rack. Close alone cannot score. The existing
store module's opening prose incorrectly suggests three independent tiers.
Prioritise the headset. A default-off RuntimeConfig / Skills / G3 switch
`headset_seat=on` filters slipping narrow cradle grasps, squeezes for eight
steps, and seats the measured bridge before release. No task-name dispatch,
checker access, rendering or LLM calls in the action path.

Pre-registration: [panel](../experiments/2026-11-07-laptop-g3.md).
Private salted variant-11 records, pilots separate from panel. Deployment
`~/controlr-laptop`, all outputs `~/dojo-data/laptop`. Light7 serial jobs,
8G/600% user scopes, admission logs, timeout with KILL tail. Root service
and colleague processes untouched. Local pyflakes reports only pre-existing
unused imports and a pre-existing placeholder-free f-string.

The first local pytest invocation used a global pytest because dev extras
were missing; `uv sync --extra dev` corrected the environment. Focused
headset/laptop tests pass. Full suite started with `uv run pytest -q`.
