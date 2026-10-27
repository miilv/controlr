# 2026-10-27: do integ_eval numbers survive the official path? (RoboDojo's client + the strict server, 25 paired layouts)

**Question:** every team number comes from our own harness: `integ_eval.sh --renderer isaac51`, the proxy, G3. The
leaderboard scores what RoboDojo's own eval client measures, with our system behind robot-harness's strict server. Do
the two agree per layout? Where they disagree, which path is wrong?

**Hypothesis:** they agree.
- Physics is bitwise deterministic across render cadences (journal 2026-10-26-bridge-isr A2).
- Both paths run RoboDojo's EvalEnv code for reset, step accounting, the success latch and the score.
- The differences are in what the policy reads and in the process plumbing (journal 2026-10-27-dojo-official §1.2).

**Setup:**
- **Official leg:** `scripts/dojo/official_eval.sh` (#91).
  - RoboDojo's `src/eval_client/main.py`, unchanged, through `dojo/simlab/dev_eval.py` (SeedManager reads our settled
    layouts). Verdicts come from RoboDojo's own `_result.json`.
  - robot-harness's strict J2 server + `harness_strict_j2` shim, unmodified, pinned (541b8df), with the J2 launcher's
    md5 checks.
  - Our bridge driver (`dojo.bridge.driver`), the perception service with the 5.1 calibration, and the twin it spawns.
  - Several episodes per eval client.
- **integ leg:** `integ_eval.sh --renderer isaac51` (one RTX EvalEnv, our driver in RoboDojo's python, no server).
- **Both legs:**
  - AgentPolicy on `submit_v1`: agent `cx/gpt-6.1-sol`, agentic perceiver Sonnet 5.5, the 5.1 calibration.
  - The same box (compute7, RTX 5090, genuine 5.1 rendering), the same code (e14e504 = main 2a6e30d + the harness), the
    same settled layouts 100000-04, paired by layout.
- **Pairing:** `python -m dojo.bridge.official compare` (verdicts, steps, the first executed action where the two differ).

## Results

| task | official (`_result.json`) | integ_eval | agree | disagreement and cause |
|---|---|---|---|---|
| press_by_number | 5/5, 100 | 5/5, 100 | 5/5 | steps identical on 4/5 |
| stack_blocks | 5/5, 100 | 5/5, 100 | 5/5 | official twin verify was OFF (cause 3); re-run with verify below |
| swap_blocks | 5/5, 100 | 5/5, 100 | 5/5 | the twin rejects every swap candidate on BOTH paths (12/12 segments `none_ok`) |
| make_kong | 0/4 + 1 unstable | 1/5, 20 | 3/5 | 100003: cause 2. 100002: the agent's retry passed 8 of 22 tiles on the official run and all 22 on integ (Sol sampling; the retry line, #182, takes the drop rate from 9/10 to 0/10) |
| insert_tubes | 3/5, 76 | 4/5, 88 | 4/5 | 100000: a step-budget knife edge. With 52 steps left, the 3rd tube's place needed 45 plus the 30-step finish reserve, so it was refused; on integ a few more steps were left and the tube went in before BudgetExceeded |
| **total** | **18/24 + 1 unstable** | **20/25** | **22/25** | |

Plus a 0-LLM smoke (RuntimePolicy, agentic off, press 100000-01): 2/2 = 2/2.

**Noise level:**
- Frame-0 perception differs by 0.1-0.3 mm between the paths, the same spread as two integ runs of one layout.
- So executed actions differ from the first segment on, and knife-edge decisions can flip. press 100001: a number card
  at x = -0.0001 vs 0.0000 flipped a look's arm.

## The four causes found, all fixed

| # | cause | which path was wrong | effect | fix |
|---|---|---|---|---|
| 1 | RoboDojo's `{arm}_ee_joint_state` is the APPLIED gripper control (it follows the blocked fingers: ~0.3 on a held block); the strict server sends the pure command | integ / proxy / G3 (dev) | B1 held_look less sure, Sonnet escalations, false "not held"; the robot mask draws the fingers open. plug 1/5 -> 4/5 with the command (report 2026-10-27-grip-source-ab) | #104 + #132: `DOJO_GRIP_SOURCE=command` is the default |
| 2 | an episode the task marks unstable (`mark_env_unstable`: make_kong's discard, ttt's opponent, imitate's demonstrator) is DROPPED by RoboDojo's client | integ / proxy | counted as failures | #173 |
| 3 | the fleet venv pins lacked tqdm / transforms3d (the twin-layout template's deps) | official / bridge on compute5-8 | twin verify disabled in every bridge episode | #100 + the lead installed them in all fleet venvs |
| 4 | integ's `actions.json.gz` came from the policy's Bridge, which misses the batch the latch ends | integ (dev output) | kit replays and action comparisons saw a prefix (26 of 240 actions) | #95 |

Also checked in the code, no difference found:
- step accounting (`take_action_cnt` on both), the success latch (is_episode_end is idempotent), the action format
  (float32 joint dicts, bit-exact through JSON);
- the end of an episode: the shim's `finish()` books an unfinished episode as failed. main.py alone would book an early
  return as a SUCCESS, because `success` starts True; the shim guards against that, and any shim of ours must too;
- the strict server rounds proprio to 1e-5 (below perception noise);
- the render cadence (the shim renders every step) does not change physics.

## Reading

- integ_eval is a faithful stand-in for the official client once causes 1-4 are fixed. The per-layout differences that
  remain are render / perception noise on knife-edge decisions and LLM sampling.
- No disagreement came from the eval client's own accounting or from the bridge.
- Side findings, routed:
  - swap_blocks' twin verify never passes a candidate on either path: 96 rollouts and ~72 s of wall time per episode
    for no selection.
  - The tubes finish reserve can cost the last insertion (Precision).
  - Rendered ttt is 0/9 `not_home` in the grip A/B while G3 mode B is 5/5 (Long-Horizon).

**Spend** (phase A + B, 54 episodes, from the reports):
- Sol: 206 calls (2.07 M prompt, 1.76 M cached; 18.9 k completion). Most went to make_kong's recovery cells, 7-11 per
  episode; press / stack / swap used 1 per episode.
- Sonnet (agentic perceiver): 234 calls.
