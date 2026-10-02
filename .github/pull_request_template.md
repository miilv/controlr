## What and why

<!-- 1-3 sentences; link the BACKLOG item / experiment report -->

## Checklist (docs/DEVELOPMENT.md)

- [ ] `uv run pytest -q` green; touched robot/loop/safety/isaac → Isaac tests on compute3 green
- [ ] Config defaults and prompt/feedback format unchanged — or the change is deliberate and logged in the journal
- [ ] Changed message serialisation → prefix stability test + a live `cache_read` check
- [ ] Live calls: within budget, spend stated below
- [ ] Behaviour changed → ARCHITECTURE / DEVELOPMENT / EXPERIMENTS updated in this PR
- [ ] BACKLOG: closed items struck, new ones added; journal / experiments written
- [ ] compute3: Isaac server stopped, no leftovers in `~/controlr*`

**Spend (if there were live calls):** <!-- control/planner calls, tokens, GPU time -->
