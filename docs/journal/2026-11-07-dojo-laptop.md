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

The first candidate's 41 mm pinch holds pilot 100031 at lift (3.7 mm drift)
but shifts 113.4 mm in carry. The pre-panel amendment halves the cradle carry
rate; it still shifts 97.6 mm in that pilot and scores zero. Candidate frozen
at deployed `58dcf38e` for the panel; later changes are tests/docs only.
The G3 deployment stamp is explicitly set before the final pilot/panel;
initial pilots have null G3 `code.commit` (deploy.sh writes `.deployed.json`,
whereas G3 reads `.deployed_commit`). No ambiguous pilot is in the panel.

Local full suite: 2057 passed / 191 skipped (1089.38 s), started on the first
candidate before the carry amendment. Final candidate/config/defaults-focused
suite: 10 passed. Final source also receives PR CI validation. Main was merged
for current documentation; the running physics deployment remained frozen.

Completed panel: ten pairs per cell, A 0 ± 0 -> B 2 ± 6.32 points in each;
1 score win / 0 losses / 9 ties each, full success 0/10 in every arm.
No infrastructure failures or missing meshes. No camera qualification (pooled
+2 < +5), no camera spec/dispatch or board update. Own scopes/processes exited.
Footprint 130 MB deployment / 2.6 MB data; zero model calls/tokens/spend.
