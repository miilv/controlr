# 2026-10-24: M0, reproducing PyRUA-Lean on compute2 (stopped early)

Goal: reproduce PyRUA-Lean (official RoboDojo-Sim #1, 55.26 Score / 50.35 % SR) on our infrastructure. This gives us
a floor, checks that our setup matches the official numbers, and becomes the runtime-agent baseline to beat.
Background: [research/robodojo-teardown/pyrua-lean.md](../../research/robodojo-teardown/pyrua-lean.md). Glue:
[baselines/pyrua/](../../baselines/pyrua/).

**Where it stopped.**
- The install works and a router smoke test passed.
- One `stack_blocks` episode passed end to end through RoboDojo's own eval client.
- The 10-task slice was launched. Then the lead cancelled it on cost grounds (new cap: 30 LLM calls for the whole task), and compute2 dropped off the network minutes after the slice started. Its results were not collected.
- No per-task comparison with the official numbers exists yet.

## 1. Setup (everything under `/root/dojo-pyrua` on compute2)

- **Code**: `DAGroup-PKU/PyRUA-Lean`, pinned at `c686120`.
  - This is the revision the adapter's `install.sh` names as evaluated.
  - `src/` and `integrations/xpolicylab/PyRUA_Lean/` are identical to the official-evaluation commit `077941b`; only docs differ up to HEAD `ffad6d4`.
  - The adapter is used unmodified.
- **Venv** `/root/dojo-pyrua/venv`:
  - uv-managed CPython 3.11.17;
  - torch 2.8.0 + torchvision 0.23.0 (cu128);
  - what upstream `install.sh` installs, under its constraints (numpy<2, opencv-headless<4.12, torch untouched): pyrualean editable, `sam3 @ RLinf/sam3@c1f6b1e`, einops, pycocotools, psutil, httpx, mcp<2, uvicorn.
  - One difference from upstream: XPolicyLab's *dependencies* are installed rather than `pip install -e XPolicyLab`, because that would write egg-info into the shared install. The server runs with `PYTHONPATH=RoboDojo`, which is what upstream's `setup_eval_policy_server.sh` does anyway.
- **SAM3 checkpoint**:
  - `facebook/sam3` is gated on HF. Our token (`~/.cache/huggingface/token` on the workstation; compute2 has none) is valid, but the gate answers 403.
  - Instead I used upstream's documented fallback, `download_assets.sh modelscope`.
  - The file is **byte-identical** to the HF original: sha256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, size 3,450,062,241. This matches the LFS hash in HF's public model metadata. So this is not a deviation.
- **Shared install**: the only write is a symlink, `RoboDojo/XPolicyLab/policy/PyRUA_Lean -> /root/dojo-pyrua/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean`. Upstream's `install.sh` expects exactly this link.
  - RoboDojo's eval client also writes `eval_result/RoboDojo/<task>/PyRUA_Lean/...`, as it does for every policy.
- **Launcher** `baselines/pyrua/run_task.sh`. It does what upstream's `eval.sh` does: starts XPolicyLab's ws policy server with policy `PyRUA_Lean`, then RoboDojo's `scripts/robodojo.sh client`. Our differences:
  - it uses a venv and the shared prefix conda env, not named conda envs;
  - it waits until the GPU has ≥ 12 GB free;
  - it cleans up all processes on exit;
  - it copies RoboDojo's `_result.json` and mp4s into the run dir.

### Landmine: Python 3.11.0rc1 segfaults torch.jit

`uv venv -p 3.11` picked up compute2's system **Python 3.11.0rc1**. On that interpreter `torch.jit.script` segfaults (`sam3/model/utils/sam1_utils.py:33`) while SAM3 builds. The policy server died silently: its log had nothing, and `dmesg` showed a python segfault. The eval client then logged only `keepalive ping timeout` and `policy server not ready`.

The fix is `uv venv --python-preference only-managed -p 3.11`. With it, SAM3 loads in 6.7 s and holds 3.75 GB allocated.

## 2. LLM through omniroute

- **Endpoint and credentials**:
  - The model is `cx/gpt-6-astra` with `reasoning_effort: high`, PyRUA's own `deploy.yml` setting.
  - PyRUA reads `PYRUALEAN_BASE_URL`, plus the key from the variable named by `PYRUALEAN_API_KEY_ENV`. So the launcher points it at `OMNIROUTE_BASE_URL` / `OMNIROUTE_API_KEY` from `/root/controlr/.env`.
  - `OPENAI_*`, `OPENROUTER_API_KEY` and `ANTHROPIC_*` are unset.
- **Request format** (`router_smoke.py`: PyRUA's body with tools, `tool_choice: auto`, a PNG `image_url` data URI, `max_tokens 8192`): works. The reply had `finish_reason=tool_calls` and the right answer read from the image.
- **Effort**:
  - The catalog lists effort tiers on `cx/gpt-6-astra`, which is chat-completions.
  - Probes on a counting problem gave about 2.5k reasoning tokens at `high` and 1.7-1.9k at `low` (n=1-2 each).
  - `none` still reasoned (305 tokens).
  - The `-high` id suffix behaved like the field.
  - So the field is accepted and seems to have some effect, but the effect is weak. I kept the field, as PyRUA does.
- **Replay**: two byte-identical requests both got fresh answers on this route (~20 s each, different usage). So the omniroute response replay noted in DEVELOPMENT.md did not show up for `cx/`, and no nonce was added.
- **429 / backoff**:
  - Upstream already waits out 429s (5 s doubling to 120 s, with jitter, for as long as the 90-min episode budget allows; `openai_runtime.py:56-64,438-456`). This is the fix for their 39/162 lost episodes.
  - The launcher raises the 5xx/transport retries from 10 to 30 (`PYRUALEAN_OPENAI_RETRIES`), since one probe got a dropped connection.
  - No code change was needed.
- **Caching**: the provider's automatic prefix cache works through the router. In the smoke episode 88 % of input tokens were cached.

## 3. Smoke episode: stack_blocks, seed 0, layout 0

Run dir: `/root/dojo-pyrua/runs/20261024T182729Z_stack_blocks_s0`.

| | |
|---|---|
| RoboDojo verdict | `_result.json`: success_rate 1.0, score 100 (official PyRUA stack_blocks: 100 / 100) |
| LLM calls | 10 (10 cells, no `finish`: the success latch ended the episode during cell 10, "truncated") |
| tokens | 192,214 input (168,192 cached), 1,496 output (245 reasoning) |
| env steps | 140 (PyRUA reports a limit of 700) |
| wall | 111 s total, 105 s policy; 4.5-11 s per LLM call |

The model followed the manual's workflow:
1. `locate` all three blocks by text.
2. Move the right arm above the target at pitch 90.
3. `locate` again from the wrist.
4. Descend to `top_z − 1.7 cm`, close, and check `holding`.
5. Place on the base block and `home`.
6. For the third block, project the stacked block's head-camera centroid onto its known height, so the placement point is not read from the table plane.

The first cell raised an exception: `locate` of a name it could not find raises. The model wrapped the call in try/except and carried on.

This episode is far cheaper than the paper's average: 10 calls and 0.19 M input tokens against 25.7 calls and 750 k. That fits stack_blocks being one of the easy tasks (19.3 calls on solved episodes in the paper).

## 4. The 10-task slice: cancelled, results not collected

- The slice (`run_slice.sh`: 2 tasks per dimension, seed 0, layouts 0-4, sequential) started at 18:39:52Z with stack_blocks.
- About 3-13 minutes later compute2 became unreachable over ssh, and then dropped off the tailnet.
- The lead's stop order (cost cap: 30 calls in total) arrived while the machine was unreachable, so I could not kill the slice.
- If compute2 came back with internet access, `run_slice.sh` may have kept running. Check before anything else on compute2: `pgrep -af "glue/run_slice|glue/run_task|eval_client/main|setup_policy_server"`.

Load on compute2 when it went down: one Isaac eval client and one SAM3 policy server, both mine.
- Isaac took ≈ 8.2 GB VRAM. It launches with `--num_envs 10` even with `eval_batch: false`.
- The SAM3 server took ≈ 4-6 GB.
- The GPU had 20.7 GB free before the start.
- During the first smoke, RAM was 39 of 62 GB used (my Isaac plus a teammate's Isaac).
- The two segfaults above dumped core. Check `/tmp` and `/var/crash` for large core files.

## 5. LLM spend (actual)

| What | Calls | Input tokens | Output tokens |
|---|---|---|---|
| router smoke (tools + image) | 1 | 2,582 | 20 |
| effort / replay probes (text) | 12 (1 dropped) | ~0.8k | ~11k (mostly reasoning) |
| smoke episode stack_blocks L0 | 10 | 192,214 (168,192 cached) | 1,496 |
| **verified total** | **23** | **≈ 0.20 M** | **≈ 13 k** |
| slice, stack_blocks L0-4 (not verified) | ~0-50 | ≤ ~1 M | ≤ ~10 k |

The actual slice numbers are in `/root/dojo-pyrua/runs/20261024T183952Z_stack_blocks_s0/episodes/*/openai_events.jsonl`, and in later run dirs if the slice went on. Run `python3 /root/dojo-pyrua/glue/summarize.py <run dirs>` on them.

## 6. Official per-task numbers for the slice

From the leaderboard bundle `robodojo-benchmark.com/assets/index-DA5jHulR.js`, re-extracted today. The whole PyRUA_Lean block is in `baselines/pyrua/official_pyrua_per_task.json`.

| Task | Dim | Official SR / Score |
|---|---|---|
| stack_blocks | G | 100 / 100 (random: 100 / 100) |
| put_bottles_into_dustbin | L | 88 / 92.2 |
| build_tower | P | 4 / 23.2 |
| insert_tubes | P | 50 / 66.4 |
| play_tic_tac_toe | L | 94 / 97.2 |
| make_toast | G | 16 / 26 (random: 12 / 17) |
| swap_blocks | M | 40 / 40 |
| cover_blocks | M | 100 / 100 |
| general_pickup | O | 86 / 86 |
| solve_equation | O | 72 / 72 |

The official run used the hidden eval seed 3 with 50 episodes per task. Our seed-0 layouts 0-4 would give n=5 per task, so ±20-pp swings are noise.

## 7. Runbook

```bash
# once: glue to compute2, install (idempotent)
scp -r baselines/pyrua/* compute2:/root/dojo-pyrua/glue/
ssh compute2 'bash /root/dojo-pyrua/glue/setup.sh > /root/dojo-pyrua/setup.log 2>&1'

# router smoke: 1 call (tools + image)
ssh compute2 'set -a; . /root/controlr/.env; set +a; python3 /root/dojo-pyrua/glue/router_smoke.py --model cx/gpt-6-astra --effort high'

# any task/seed: layouts 0..N-1 of the seed (LIVE CALLS: ~10-30 per episode)
ssh compute2 'cd /root/dojo-pyrua && (setsid nohup bash glue/run_task.sh --task stack_blocks --seed 0 --eval-num 1 > task.log 2>&1 < /dev/null &)'
#  -> /root/dojo-pyrua/runs/<stamp>_<task>_s<seed>/{meta.txt, policy_server.log, eval_client.log,
#     episodes/<task>_l<layout>_s<seed>-<t>/ (PyRUA: result.json, messages.json, openai_events.jsonl, images/),
#     robodojo_result/ (_result.json, mp4s per camera)}

# slice (sequential); TASKS / SEED / EVAL_NUM env override the defaults
ssh compute2 'cd /root/dojo-pyrua && (setsid nohup bash glue/run_slice.sh > slice.log 2>&1 < /dev/null &)'

# summary vs official (+ per-episode CSV)
ssh compute2 'python3 /root/dojo-pyrua/glue/summarize.py /root/dojo-pyrua/runs/*_s0 --csv /root/dojo-pyrua/episodes.csv'

# stop everything of ours
ssh compute2 'pkill -f glue/run_slice.sh; pkill -f glue/run_task.sh'   # the launcher's trap kills its server and client
```

Notes:
- `run_task.sh` waits while the GPU has less than `MIN_FREE_MB` free (default 12000).
- One Isaac at a time is the safe default.
- The episode record says `reasoning_effort: xhigh`. That is upstream's known misreport: the openai runtime ignores it and sends `deploy.yml`'s `high`.

## 8. Next, under the new cap

The lead redirected the work to an infra parity check at zero LLM cost:
- run PyRUA's non-LLM pieces (host IK and motion, SAM3 `locate`, the XPolicyLab bridge) against our RoboDojo install;
- replay a short scripted move to confirm IK accuracy and step accounting.

It was not started because compute2 is down. A starting point: drive `pyrualean`'s `robodojo_xpl_rgb` robot with a fixed cell (`move_to` above a block, `locate`, `set_gripper`, `home`) instead of the LLM. Inject it through the openai runtime's `transport` hook, which returns canned tool calls, so the rest of the path stays unchanged.
