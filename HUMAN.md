# HUMAN.md — how to give the agent tasks in controlr

> For the human operator. Rules the agent already knows — [CLAUDE.md](CLAUDE.md);
> process — [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); experiments — [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 0. Your role

You provide what isn't in the repo: the **question** (what we want to learn), the **budget**
(calls / money / GPU hours), the **definition of done**, and the **decisions** with consequences
(hardware, money, the real robot). The agent is strong and fast, but it isn't a mind reader, and
by default it spends whatever it isn't told not to.

## 1. Two kinds of tasks

**Code** (feature, fix, refactor):
> While carrying, the packet catches the box rim and SafetyEnvelope doesn't see it
> (experiments/2026-10-02-rotation-yaw §6.2). Add a check of the arm links against the box body
> with ~50 mm inflation. Don't touch the grammar or the prompt. Done = unit + isaac test + PR.

**Experiment** (required: question, setup, budget, what counts as the result):
> Hypothesis: a second, top-down camera reduces carry failures. waffle + rotation=yaw, seeds 0–3
> × 2 repeats, Sonnet 5.5, plan pinned per seed. Arms: scene vs scene+top (tiled). Budget
> ≤ 300 calls, 0 planner. Done = a report in docs/experiments + videos of 1 success and 1 failure.

## 2. Plan first

For anything non-trivial: "Make a plan, don't change anything yet." Useful questions for a plan:
- "How many calls / dollars / GPU hours will this take, and why that many?"
- "What can be checked on mock/replay for free?"
- "How will we know the difference isn't noise (n, seeds, repeats)?"
- "What changes in the prompt/format, and does it break comparability with earlier runs?"

## 3. What to check in the result

- **Experiment report**: n, spend, failure modes with quoted model replies, videos. "2/4" is not a finding.
- **Diff**: did config defaults or the prompt change (that changes every future comparison)?
- **compute3**: Isaac server stopped, nothing of anyone else's touched, disk not bloated.
- **Loose ends**: BACKLOG struck/extended, journal/experiments written.

## 4. Red zones (your call)

The real UR3 (only with you at the e-stop) · apt/drivers/reboot on compute3 · live runs over
budget · changing the default model/action format · deleting anything on shared machines.
