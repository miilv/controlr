# 2026-10-27 — perception bake-off: Perceiver-models stream

Report with all numbers: [../experiments/2026-10-27-perceiver-models.md](../experiments/2026-10-27-perceiver-models.md).

**Done**
- `dev_d0`:
  - `paired`: per object vs a base, fixed / broken / still gross / not found / phantoms, per task, calls / tokens /
    latency per arm, and the budget gate's split, on B1 or `--gate-after` a first call;
  - `--hint-tag`: a second agentic call on the first call's output;
  - `arbitrate --risky-b1 --out-suffix`;
  - the scorer: eval_g1 tolerances for tasks dev_d0 had none for, a rotation error everywhere, the record merge's
    free-text kind rule, and a deterministic class-name order (a set made counts move by ~1 between processes).
- `seed_refine.arbitrate(risky_b1=...)`, default off: the agent stands where B1's calibrated risk is >= the value.
- `perceive_impl` returns `call_s` / `commands` / `errors` in `_agent` (return_stats callers only).

**Found**
- Today's arbitration reverts 10 of Sonnet's 37 raw fixes to prevent 3 breaks. `risky_b1 0.2` gives +7..+12 fixed on
  every model.
- Opus > Sonnet by about 5 fixed / 2 broken of 84; Fable ≈ Sonnet at 3x the latency; 16 turns and a second call on the
  same view add nothing.
- store_laptop: a laptop root-vs-mesh offset of about 80 mm (BACKLOG). B1's laptop yaw flip and the headset identity are
  B1-side.
- omniroute's Claude account hit its weekly limit (incident, DEVELOPMENT pitfall).
- Ops: the GPU cap was taken by others' jobs, so H2 / arbitration moved to CPU silhouettes (`DOJO_GL=0`, about 1 core per
  pass).

**Next:** the B1 fixes the lead assigned (identity margin swap with eval_g1 on all D0 sets; the laptop yaw flip). Then a
rendered A/B of `risky_b1` once an LLM route is back.
