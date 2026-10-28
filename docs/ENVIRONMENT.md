# Environment: variables, machines, paths

New variable → `.env.example` (no value) + a row here, in the same PR.

## Environment variables

| Variable | Needed where | What |
|---|---|---|
| `OMNIROUTE_BASE_URL` | anywhere with live calls | OpenAI-compatible router (`.../v1`) |
| `OMNIROUTE_API_KEY` | same | router key; only in `.env` locally, `~/controlr/.env` on compute3, `~/data2/ilia/.env` on isr-lab (owner-approved, Mikhail's side may use it; point `DOJO_AGENT_DOTENV` / `AGENTIC_ENV_FILE` at it) and `~/controlr/.env` on our own GPU boxes (owner-approved: `scripts/bootstrap_gpu_box.sh` copies the `OMNIROUTE_*` lines only, mode 600, values never printed or logged). A dojo process reads it from ONE source (below) |
| `AGENTIC_ENV_FILE` | the agentic perceiver (`dojo.perception.agentic`: perception service, the J1 agentic worker) | the dotenv file its `OMNIROUTE_BASE_URL` / `OMNIROUTE_API_KEY` come from (only those two names are read). Set: both values from this file; unset: both from the environment; set while `OMNIROUTE_*` is in the environment: refused. `integ_eval.sh` / `official_eval.sh` default it to `$HOME/controlr/.env`, `isr_env.sh` to `~/data2/ilia/.env` |
| `DOJO_AGENT_DOTENV` | the agent (`dojo.agent.llmio`: `AgentPolicy`, grounding, replay probe) | the same rule for the agent's client. No implicit fallback: before, llmio read `~/skoltech/controlr/.env` and then `~/controlr/.env` by itself. `official_eval.sh` / `bridge_episode.sh` default it to `$HOME/controlr/.env` (`bridge_episode.sh` passes it to the driver only), `isr_env.sh` to `~/data2/ilia/.env`; on a dev box name the file yourself (`DOJO_AGENT_DOTENV=~/skoltech/controlr/.env`) |
| `OPENROUTER_API_KEY` | live runs with `llm.backend=decisions` | OpenRouter key for the Decisions API (`decisions.api_key_env`; omniroute has no decisions route); same storage rules as the router key |
| `CONTROLR_ISAAC_AUTHKEY` | compute3 | shared secret client↔Isaac server; unset → the client generates one per launch (fallback `controlr-isaac-dev` is for a dev box only) |
| `CONTROLR_SO101_AUTHKEY` | the SO-101 rig and the controlr process driving it | shared secret client↔so101 server ([SO101.md](SO101.md)); on the rig in `~/.so101_authkey` (0600); fallback `controlr-so101-dev` is for a dev box only |
| `CONTROLR_ISAAC_READY_FILE` | compute3 | server readiness flag file (set by `remote_run.sh`) |
| `ISAAC_SIM_ROOT` | compute3; physics GPU boxes | default `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0`; on a physics box `~/dojo-isaac6/isaac-sim-6.0` (compute3's build, copied; set by `~/dojo-env.sh`) |
| `PHANTOM_ROOT` | compute3 | default `/home/physicalai/phantom-icra-2027/phantom` |
| `CONTROLR_LLM_PROXY` | local shell, `scripts/remote_run.sh` | optional; becomes `HTTPS_PROXY` for the remote `controlr` process only (LLM calls through an ssh reverse tunnel when compute3's own route is slow — RUNBOOK §3) |
| `CONTROLR_CODEX_AUTH` | where direct Codex calls run | optional path of controlr's own ChatGPT/Codex token file (default `~/.controlr/codex_auth.json`, 0600, created by `scripts/codex_login.py`); a secret, ONE holder only (refresh tokens rotate), never in git/runs/logs |
| `CONTROLR_LIVE=1` | tests | enable live tests (`@pytest.mark.live`) — costs money |
| `CONTROLR_ISAAC=1` | tests on compute3 | enable Isaac tests (`@pytest.mark.isaac`) |
| `CONTROLR_ISAAC_TEST_PORT` | tests | test server port (default 7821, not the main 7801) |
| `CONTROLR_ISAAC_IMG_DIR` | tests | where Isaac tests save frames for inspection |
| `CONTROLR_ROBODOJO_HOST` / `_PORT` / `_AUTHKEY` | compute2 (`scripts/robodojo/run.sh`) | where `controlr robodojo-serve` listens for the RoboDojo shim, and their shared secret (a random one per run) |
| `CONTROLR_ROBODOJO_READY_FILE` | compute2 | readiness flag of `robodojo-serve` (set by `run.sh`) |
| `RD_HOME` | compute2 | RoboDojo install dir for `scripts/robodojo/*.sh` (default `/root/controlr-robodojo`) |
| `BENCH` | the GPU box of a bridge run (`scripts/dojo/bridge_episode.sh`, `python -m dojo.bridge.driver --bench`) | robot-harness's install dir; the strict server's socket is `$BENCH/run/<id>.sock` (default isr-lab's `/home/anywherevla/data2/bench`; [BRIDGE-RUNBOOK.md](BRIDGE-RUNBOOK.md)) |
| `ROBOT_START` | same | robot-harness start script for the strict server with `joint_steps` (default `$BENCH/bin/robot-start-j2`) |
| `BRIDGE_LOCK_WAIT_S` | same | how long `bridge_episode.sh` waits for `~/dojo-gpu.lock` (`DOJO_GPU_LOCK`; default 3600 s) |
| `BOOT_DATA_DIRS` / `BOOT_CATALOG_DIRS` | local (`scripts/bootstrap_gpu_box.sh`) | optional: which dirs the data / catalog relays copy from compute3 (space-separated, relative to the homes); default = the full set ([RUNBOOK-new-gpu-box](RUNBOOK-new-gpu-box.md)) |
| `ASSETS_PINS` | same | optional: another RoboDojo Asset list than `scripts/infra/pins/robodojo_assets.txt` (partial installs, tests) |
| `SMOKE_RESERVED_OK` | a GPU box (`box_steps.sh` smokes) | `1` lets the smokes that replay G2 on layout 110007 run (smoke_twin's contact-restore check, smoke_dev_eval, smoke_proxy_eval). 110007 is in the reserved >= 100100 range (D-C10a), so only with the joint-run owners' OK; default: skipped |
| `SHARED_HOME` | a GPU box (`~/dojo-bootstrap/config.env`, set by `bootstrap_gpu_box.sh --shared-home`) | `1`: the box's `$HOME` is shared with other people: no lingering, no `~/.config/uv/uv.toml`, our uv in `~/dojo-bootstrap/tools/uv` |
| `BOX_DRIVER_OK` / `PREFLIGHT_NET` | same (preflight) | driver majors accepted for Isaac Sim 5.1 rendering (default `570 580`); `PREFLIGHT_NET=0` skips the network probes (unit tests) |
| `DOJO_REPLAY_TRAJ` | a 5.1 box (`dev_eval.sh --policy dojo_replay`) | trajectory JSON the XPolicyLab policy `dojo_replay` replays (bootstrap smoke) |
| `DOJO_GRIP_SOURCE` (deprecated alias `DOJO_GRIP_OBS`, until 2026-10-28) | any dev harness (integ_eval / proxy_eval / G3) | what the policy reads as `{arm}_ee_joint_state`: `command` (default since 2026-10-27): the strict server's contract, the last commanded value; `robodojo`: RoboDojo's own reading, the APPLIED gripper control, which follows the measured fingers within 0.2 of the range (`dojo/simlab/grip_source.py`; recorded per row as `grip_source`) |
| `DOJO_SAM_ADDR`, `DOJO_SAM_MODELS` | a GPU box running `python -m dojo.perception.sam_masks serve` | the SAM server's unix socket (estimator flags `sam_fg` / `sam_propose` / `series_count=sam` use it; unset = plain B1) and the weights dir (default `~/dojo-data/models`: `sam3.1/`, `sam2.1/{small,large}/`; `scripts/perception/sam/setup_venv.sh`) |
| `BOOT_RUNS_DIR` | local (`bootstrap_gpu_box.sh`, `bootstrap_fleet.sh`) | where the local logs, `summary.tsv`, fleet dirs and the fleet key go (default `runs/bootstrap`; tests use a tmp dir) |
| `FLEET_WAIT_S` / `FLEET_POLL_S` / `FLEET_TICK_S` | same (fleet) | how long a box waits for a fleet peer to have a set before fetching from the origin itself (default 5400 s), how often it looks (15 s), how often the fleet prints progress (10 s) |
| `ISAAC6_SRC` | same | where compute3's Isaac Sim 6.0 is copied from (default `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0`) |
| `DOJO_EMBED_MODELS` | wherever perception runs with `EstimatorConfig.embed_model` set | dir with one subdir per image-embedding model (`dinov2-small`, `dinov2-base`, `siglip2-base-patch16-224-vision`, `clip-vit-large-patch14-vision`; `dojo/perception/embed_id.py`); default `~/dojo-data/pbake-identity/models`. Needs torch + transformers in the perception venv (DEVELOPMENT §4) |
| `DOJO_PERCEPTION_CFG` | the perception service (`dojo.runtime.perception_service`) | optional JSON of `EstimatorConfig` overrides for an A/B arm, e.g. `{"embed_model": "dinov2_s", "embed_merge": true}` or `{"stereo_check": true, "stereo_reject": "slot_bases"}`; unset = the defaults |
| `DOJO_BOX_ROLE`, `DOJO_GPU_LOCK`, `DOJO_GPU_LIGHT_LOCKS`, `DOJO_SCOPE_BUNDLE` / `_PROXY` / `_BATCH` / `_G3` / `_LIGHT`, `DOJO_MIN_AVAIL_GB` | a GPU box (`source ~/dojo-env.sh`) | the box's role (full / physics), the heavy lock and the four light-lane slots (`DOJO_GPU_LOCKS="$DOJO_GPU_LIGHT_LOCKS"` for g3's light lane), the `systemd-run --user --scope` settings, the memory floor for heavy jobs; the same facts in `~/dojo-box.json` |

### Credentials in processes (`controlr/keyguard.py`)

- **One source per process.** A process that calls the router takes `OMNIROUTE_BASE_URL` and `OMNIROUTE_API_KEY` both
  from the file its variable names (`AGENTIC_ENV_FILE`, `DOJO_AGENT_DOTENV`) or both from the environment, never a URL
  from one and the key from the other, and never from a home-dir `.env` nobody named. The dojo run scripts that name
  the file refuse to start when `OMNIROUTE_*` is exported as well (exit 2; unset it or the file variable).
- **Children that do not call the model get no credentials.** `keyguard.child_env` builds their environment: the
  parent's minus `OMNIROUTE_*`, `AGENTIC_ENV_FILE`, `DOJO_AGENT_DOTENV`, `OPENROUTER_*` / `ANTHROPIC_*` / `OPENAI_*`,
  `CONTROLR_CODEX_AUTH` and any name with a KEY / TOKEN / SECRET / PASSWORD / AUTH / COOKIE word (`HF_TOKEN`,
  `SSH_AUTH_SOCK`, ...), except our localhost authkeys (`SIMLAB_TWIN_AUTHKEY`, `SIMLAB_RENDER_AUTHKEY`,
  `DOJO_PERCEPTION_AUTHKEY`, `HARNESS_SHIM_AUTHKEY`), which those services need. A denylist, not an allowlist: the
  children read PATH, HOME, CUDA / EGL / DISPLAY, `RD_*` / `DOJO_*` / `SIMLAB_*` / `J1_*` and whatever RoboDojo's and
  Isaac's launchers read. Used by the B1 worker (`j1_adapter.install_env`), the twin (`dojo.bridge.twin`), the agent
  tools (`agentic.tools_rd`, `agentic.libero`) and the eval gate (`DOJO_EVAL_GATE`); the agentic worker
  (`install_env(model_calls=True)`) keeps only the router source. A new spawn in dojo: `env=child_env({...})`.
- **Nothing a provider sends back carries a key into a log**: error bodies and exception texts are redacted whole,
  then cut; response headers are allow-listed; served ids are safe labels (ARCHITECTURE.md, LLM client).

## Machines

| Host (`~/.ssh/config`) | What it is | What we use it for | Rules |
|---|---|---|---|
| local | dev box, no GPU | code, unit tests, mock/replay, reports, videos | — |
| `compute3` (`physicalai`) | Ubuntu 24.04, py3.12, RTX 5090 32 GB, Isaac Sim 6.0, PHANTOM | Isaac backend, all runs | shared machine; only `~/controlr*`; the GPU is ours; Wi-Fi only (no cable yet); machine-wide VPN `controlr-vpn` (sing-box tun, router + RU direct; RUNBOOK §3a) |
| `compute2` (`isr-lab-4`, root over tailscale ssh, tailnet 100.64.0.5) | Ubuntu 22.04, RTX 4090 24 GB, driver 580, i9-13900KF 32 threads, 62 GB RAM | dojo GPU box, role full (`scripts/bootstrap_gpu_box.sh compute2 --role full --shared-home`): `~/dojo-robodojo` (`/root/controlr-robodojo` is a symlink to it; the 4 Oct install is `/root/controlr-robodojo.old-20261004`) | someone else's workstation: a desktop session, a Unity editor and a `vlm-api` service share the GPU, and `/root` is shared (`--shared-home`). Only `~/dojo-*`, `~/controlr*`, `~/dojo-data`; <= 1 rendered + <= 1 other GPU job of ours. The tailnet path is a DERP relay (~1 MB/s); the box's own internet is fast (HF / pip direct) |
| SO-101 rig (`sp@192.168.1.22`, LAN; reached via `ssh mac`) | Ubuntu 24.04 laptop, i7-1370P, no NVIDIA GPU; SO-101 follower on `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` (LeRobot id `follower_v1`), wrist camera `/dev/v4l/by-id/usb-WebCamera_WebCamera_2502131600-video-index0`; LeRobot 0.4.2 in conda env `lerobot` | `scripts/so101_server.sh` from `~/controlr-so101` ([SO101.md](SO101.md)); controlr runs elsewhere (the rig cannot reach the router) | the owner's personal machine: only `~/controlr-so101` and `~/.so101_authkey`; no installs; another agent (Codex) may use the arm — one serial client at a time |
| `nuc` | NUC at the real rig (PHANTOM deployment) | future real-UR3 backend | don't touch without a task |
| `compute4` (`haptics-f0a0`, user `haptics`, tailnet 100.64.0.35) | Ubuntu 24.04, RTX 5090 32 GB, driver 580-open (Canonical-signed modules, Secure Boot on), i9-14900KF 32 threads, 62 GB RAM, 1.8 TB | provisioned by `scripts/provision_box.sh`, then `scripts/bootstrap_gpu_box.sh` (role auto → full) | registered on Mikhail's tailnet account; ours by Ilia's arrangement; `~/dojo-*` conventions |
| new GPU boxes (RTX 4090, RTX 5090s) | set up by `scripts/bootstrap_gpu_box.sh HOST` or, several at once, `scripts/bootstrap_fleet.sh HOST...` ([RUNBOOK-new-gpu-box](RUNBOOK-new-gpu-box.md)) | role `full` (driver 570/580: Isaac Sim 5.1 renders: official dev_eval, the bridge) or `physics` (5.1 physics-only + Isaac Sim 6.0 render: proxy_eval); both: perception, twin, G3 | compute3's conventions (`~/dojo-*`, `~/dojo-gpu.lock` + `~/dojo-gpu-light{,2,3,4}.lock`, `source ~/dojo-env.sh`); facts and capabilities in `~/dojo-box.json`; `~/controlr/.env` = OMNIROUTE_* only; add each host here once it is known |

## Paths

| What | Where |
|---|---|
| deployment | `compute3:~/controlr` (`scripts/deploy.sh`, venv `.venv`, `.env`) |
| runs | `runs/` locally and `compute3:~/controlr/runs` (`remote_run.sh` syncs back) |
| Isaac Sim | `compute3:/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0`; physics GPU boxes: `~/dojo-isaac6/isaac-sim-6.0` (a copy) |
| GPU box facts | `~/dojo-box.json` (schema `dojo-box/1`: role, specs, capabilities, locks, scopes, pins), `~/dojo-env.sh`, `~/DOJO_BOX.md`, `~/dojo-robodojo/READY` |
| RoboDojo | `compute2:/root/controlr-robodojo` (`RoboDojo/` clone + assets, `miniconda3/envs/robodojo`); controlr at `compute2:/root/controlr` |
| PHANTOM | `compute3:~/phantom-icra-2027/phantom`, locally `~/skoltech/research` (never modified) |
| real UR3 | **UR3 CB3** (no built-in wrist F/T; TCP force only estimated from joint currents) + Robotiq 2F-85 + RealSense D435; IP and the rest — `configs/hardware.yaml` in PHANTOM (which still says `e-series` — wrong) |
