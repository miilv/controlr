# Postmortems

Every incident (a broken machine, corrupted data, unexpected spend, damage to hardware) = a file
`YYYY-MM-DD-<slug>.md`, written within a day, blameless. This is the agent's memory of how our
particular environment breaks — read it before similar actions.

```markdown
# YYYY-MM-DD — <what broke, in one line>

**Symptom:** what was seen, when it started.
**Root cause:** the underlying cause, not the symptom.
**How it was fixed:** steps, commands, how long it took.
**What changed so it won't repeat:** commits / rules / BACKLOG items.
**Lessons:** what we didn't know / got wrong.
```
