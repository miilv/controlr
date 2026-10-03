# Environment: variables, machines, paths

New variable → `.env.example` (no value) + a row here, in the same PR.

## Environment variables

| Variable | Needed where | What |
|---|---|---|
| `OMNIROUTE_BASE_URL` | anywhere with live calls | OpenAI-compatible router (`.../v1`) |
| `OMNIROUTE_API_KEY` | same | router key; only in `.env` locally and `~/controlr/.env` on compute3 |
| `CONTROLR_ISAAC_AUTHKEY` | compute3 | shared secret client↔Isaac server; unset → the client generates one per launch (fallback `controlr-isaac-dev` is for a dev box only) |
| `CONTROLR_ISAAC_READY_FILE` | compute3 | server readiness flag file (set by `remote_run.sh`) |
| `ISAAC_SIM_ROOT` | compute3 | default `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0` |
| `PHANTOM_ROOT` | compute3 | default `/home/physicalai/phantom-icra-2027/phantom` |
| `CONTROLR_LIVE=1` | tests | enable live tests (`@pytest.mark.live`) — costs money |
| `CONTROLR_ISAAC=1` | tests on compute3 | enable Isaac tests (`@pytest.mark.isaac`) |
| `CONTROLR_ISAAC_TEST_PORT` | tests | test server port (default 7821, not the main 7801) |
| `CONTROLR_ISAAC_IMG_DIR` | tests | where Isaac tests save frames for inspection |

## Machines

| Host (`~/.ssh/config`) | What it is | What we use it for | Rules |
|---|---|---|---|
| local | dev box, no GPU | code, unit tests, mock/replay, reports, videos | — |
| `compute3` (`physicalai`) | Ubuntu 24.04, py3.12, RTX 5090 32 GB, Isaac Sim 6.0, PHANTOM | Isaac backend, all runs | shared machine; only `~/controlr*`; the GPU is ours |
| `compute2` (`isr-lab-4`, root) | Ubuntu 22.04, RTX 4090, Isaac Sim 5.1 in docker | spare host; candidate for RoboDojo (Isaac 5.1) | shared lab machine; only `/root/controlr*` |
| `nuc` | NUC at the real rig (PHANTOM deployment) | future real-UR3 backend | don't touch without a task |

## Paths

| What | Where |
|---|---|
| deployment | `compute3:~/controlr` (`scripts/deploy.sh`, venv `.venv`, `.env`) |
| runs | `runs/` locally and `compute3:~/controlr/runs` (`remote_run.sh` syncs back) |
| Isaac Sim | `compute3:/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0` |
| PHANTOM | `compute3:~/phantom-icra-2027/phantom`, locally `~/skoltech/research` (never modified) |
| real UR3 | **UR3 CB3** (no built-in wrist F/T; TCP force only estimated from joint currents) + Robotiq 2F-85 + RealSense D435; IP and the rest — `configs/hardware.yaml` in PHANTOM (which still says `e-series` — wrong) |
