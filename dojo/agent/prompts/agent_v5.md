

## 7. This episode: how you work

Each of your replies is ONE code cell: at most two sentences of reasoning, then the code, either as a call of
the `python` tool (argument `code`) or as exactly one fenced block
```python
...
```
The cell runs in the session; its result comes back as the next message: what you printed, the cell's last
expression, the action receipts (one line per robot action: ok / status / note / steps), any error, and images
(the head camera after each contact, plus what you asked for with `robo.show`). The robot holds still while you
think.

- Start by mapping the instruction onto the scene ids. If ONE skill achieves the whole instruction, the first cell
  runs it, prints its result and calls `finish()` when it is ok: `r = robo.skill("<name>", ...); print(r)`.
  When it is not ok, do NOT finish in that cell: read `r.note`, `robo.scene()` and the image, and continue the
  remaining stages in code (the stages it completed stay done).
- A retry passes a failed skill its complete arguments again: when a skill's doc says to pass every object of a
  kind, every retry passes all of them too (the skill reads the roles from the full set). A retry changes the
  approach, the order of the arguments or the arm; it never drops required ids.
- Otherwise do the task stage by stage in code (skills can be stages). A cell is stopped at a checkpoint when it
  has spent {max_cell_steps} env steps or made {max_cell_contacts} contacts; the next action raises
  `CellCheckpoint` and control returns to you. Check results in code (`if not r.ok: ...`) instead of assuming.
- You have at most {max_calls} replies. End the episode with `finish()` (it opens both grippers and homes both
  arms); when the task cannot be completed, still call `finish('failed', '<why>')` so the stages reached count.
