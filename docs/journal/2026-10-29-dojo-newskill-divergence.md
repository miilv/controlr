# 2026-10-29 — stream N: a divergence while building fresh Skills is a re-plan

W's census (#10): 3 of 5 sweep_blocks_random camera rows (compute4 runs2 numon 100005 / 07 / 09__retry) ended with
category "error": `Divergence: divergence on ['cube_2']` (also 'broom' and 'broom_shovel'). This is the runtime's own
`belief.Divergence`, not PhysX.

**Where it escaped.** `_supervise` catches a divergence raised by a skill. After a re-plan or a failed attempt, though,
it builds fresh `Skills` (`make_skills`). Their constructor reads the world (`provider.refresh` -> `resync`), and a
divergence found there was outside every `try`.

**Change** (behaviour-preserving: before, these episodes ended on the exception).
- Inside `_supervise`, `make_skills` treats such a divergence as a re-plan: the belief is already updated, so it
  builds again (event `divergence_replan` with `at: make_skills`).
- This happens at most `MAKE_SKILLS_TRIES` (3) times. Then comes a safe stop (`divergence_stop`: `finish`, arms home,
  grippers open) that keeps what is already scored.

Tests: `tests/dojo/test_divergence_make_skills.py`.
