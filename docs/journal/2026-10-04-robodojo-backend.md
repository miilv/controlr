# 2026-10-04 — RoboDojo backend (option B) and the first live pilot

**What was built** (PR #8): controlr as RoboDojo's controller, both ARX X5 arms. RoboDojo's eval
client owns the episode; our XPolicyLab policy `controlr` is a shim that connects to
`controlr robodojo-serve` and serves observe / execute / check_goal. Multi-arm contract additions
(`RobotSpec.arms`, `RobotState.arms`, `Action.arm/step/tcp_target`, `Observation.notes`,
`Robot.episode_over`), the `MOVE L … R …` grammar, `CartesianEnvelope` for backends that plan the
joints, the `system_v1` manual. How it works and how to run it: [ROBODOJO.md](../ROBODOJO.md).

**Why this direction** (Ilia chose B over "RoboDojo as a controlr Robot"): RoboDojo's own
layouts, step limits, success latch and `_result.json` stay untouched, so numbers are comparable
with the leaderboard and a submission is one step away; controlr's loop, logs and reports still run
unchanged because the shim exposes `TASK_ENV` through the `Robot` ops.

**compute2 setup (all under `/root/controlr-robodojo`, ~95 GB):** own Miniconda + env, RoboDojo
`266130a` with its IsaacLab / cuRobo / XPolicyLab forks, Isaac Sim 5.1 via pip. Landmines found:
HF rate-limits anonymous per-file downloads (~650 files / 5 min) → sparse git-lfs clone;
RoboDojo's asset check misses `Assets/Room`; `curobo.yml` must be generated from templates;
XPolicyLab's ws server needs websockets ≥ 14 while RoboDojo pins 12 → separate `policy_venv`;
`pkill -f`/`pgrep -f` over ssh matched its own shell twice (the DEVELOPMENT pitfall — use a
`[x]yz` pattern or PIDs).

**Validation:** unit suite green (+15 RoboDojo tests); Isaac 6.0 suite on compute3 green
(431 passed) for the loop/safety change; real-sim fake-LLM smoke and motion probe on compute2
(rotations and translations land within ~1 mm; the folded start pose blocks downward motion by a
joint limit, reported to the model as WARN).

**Pilot:** `general_pickup` layouts 0–4, Sonnet 5.5 → 2/5, 151 calls
([report](../experiments/2026-10-04-robodojo-pilot.md)). The 200-env-step budget is what runs out.

**Policy change today:** per-task call budgets dropped from CLAUDE.md / EXPERIMENTS / HUMAN
(Ilia: soft cap ~5k calls/day).
