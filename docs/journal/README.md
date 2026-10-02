# Work log

Dated entries about tasks: what was done, what was decided, what was found, what's left.
**The only place for time-bound content** — stable docs (ARCHITECTURE, DEVELOPMENT, RUNBOOK,
EXPERIMENTS…) carry no dates or statuses and so don't rot; the journal is dated and doesn't rot
either: an entry honestly describes the past and needs no cleanup.

## Convention

- File: `YYYY-MM-DD-<slug>.md`, one task/session = one file; never edited after the fact.
- Written by the agent at the end of significant work (rule 10 in CLAUDE.md) or by a human.
- Experiment results don't go here but to [../experiments/](../experiments/); code reviews to [../reviews/](../reviews/).
- Skeleton:

```markdown
# YYYY-MM-DD — <the task in one line>

**Done:** what changed (PRs, commits).
**Decisions:** what was chosen and why (for future "why is it like this?").
**Found along the way:** landmines/debt — duplicate into BACKLOG.md.
**Not done / next step:** where to continue.
```
