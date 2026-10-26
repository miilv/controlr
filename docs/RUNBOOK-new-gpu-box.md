# Runbook: new GPU boxes, productive in about an hour

One command, run from the local box, turns a fresh ssh-reachable GPU machine into a dojo box:
- RoboDojo (Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab) at compute3's exact commits;
- every Asset the dev tasks need;
- controlr main with the perception venv, plus `~/controlr/.env` (OMNIROUTE_* only);
- the perception catalogue with its shape caches, our dev layouts, the D0 references;
- on a physics box, compute3's Isaac Sim 6.0 build for the proxy render;
- the GPU-lock and scope conventions, `~/dojo-box.json`, and smoke tests that prove each layer works.

A second command does the same for several boxes in parallel. HF and compute3 are each asked once, and the boxes
copy from each other.

Code:
- [`scripts/bootstrap_gpu_box.sh`](../scripts/bootstrap_gpu_box.sh): one box, the local driver;
- [`scripts/bootstrap_fleet.sh`](../scripts/bootstrap_fleet.sh): several boxes;
- [`scripts/infra/box_steps.sh`](../scripts/infra/box_steps.sh): the remote half, with
  [`box_json.py`](../scripts/infra/box_json.py) and the pins in [`scripts/infra/pins/`](../scripts/infra/pins/).

Machine conventions: [dojo/README.md](../dojo/README.md). Background and pitfalls: journals
[2026-10-24-dojo-simlab](journal/2026-10-24-dojo-simlab.md) and [2026-10-25-dojo-infra](journal/2026-10-25-dojo-infra.md).

## 0. A clean Ubuntu box: the system layer first

A freshly installed Ubuntu 22.04 / 24.04 with no NVIDIA driver needs root work before the bootstrap:
[`scripts/provision_box.sh`](../scripts/provision_box.sh). Do this once per box.

What the human does at the box: install Ubuntu, create a sudo user, enable ssh, and join the tailnet.

```bash
SUDO_PASS=... scripts/provision_box.sh --ip 100.64.0.N --user haptics boxname   # one new box: writes the ssh alias
SUDO_PASS=... scripts/provision_box.sh box1 box2 box3                           # several (aliases exist), in parallel
scripts/bootstrap_fleet.sh box1 box2 box3                                       # then the usual bootstrap
```

- **ssh key:** it installs this box's ssh key with one password login. The password goes over ssh stdin only; it is never printed, logged or put on a command line.
- **driver:** it installs the HWE kernel meta plus Canonical-SIGNED prebuilt NVIDIA modules (`linux-modules-nvidia-580-open-generic-hwe-<ver>`), then `nvidia-driver-580-open`.
  - It removes `nvidia-dkms-580-open` if apt pulled it in. A DKMS module is signed with a local key that Secure Boot rejects until someone enrols it at the console (MokManager); the signed modules load with Secure Boot on, and nobody needs to enter the BIOS.
  - The meta packages keep kernel and modules in step on later kernel updates.
- **tools and settings:** build tools, git-lfs, ffmpeg, `loginctl enable-linger`, and persistent inotify watches.
- **reboot:** only if the driver isn't live yet, or a newer kernel is waiting. It then checks `nvidia-smi`. Re-runs are idempotent.
- **driver branch:** 580, because Isaac Sim 5.1 renders on 570/580 and crashes on 595. compute4 (RTX 5090, 580.178.04, kernel 7.0.0-38) runs it with Secure Boot on.

## 1. The commands

```bash
cd ~/skoltech/controlr                    # the main checkout (its .env is the one copied)

# one box
scripts/bootstrap_gpu_box.sh HOST --only preflight     # ~10 s, read-only: GO / NO-GO + the role it would pick
scripts/bootstrap_gpu_box.sh HOST                      # everything (role auto); re-run the same command after any failure
scripts/bootstrap_gpu_box.sh HOST --dry-run            # every command, local and remote; no ssh at all

# several boxes (2 tomorrow, 5+ later): parallel, peer-to-peer
scripts/bootstrap_fleet.sh --only preflight HOST1 HOST2 ...   # read-only GO / NO-GO table
scripts/bootstrap_fleet.sh HOST1 HOST2 ...                     # everything; the final table: one row per box
scripts/bootstrap_fleet.sh --dry-run HOST1 HOST2:physics       # the plan + every box's dry run

# afterwards
ssh HOST cat ~/dojo-box.json              # role, specs, capabilities, smoke results (what agents read to pick a box)
ssh HOST cat ~/dojo-robodojo/READY
```

Before you start:
- `ssh HOST true` must work non-interactively (key auth). An alias in `~/.ssh/config` makes the logs readable.
- `ssh compute3 true` must work. compute3 is the source of the layouts, D0, the catalogue and Isaac Sim 6.0. Its data
  streams through the local box (`ssh compute3 tar | ssh HOST tar`): nothing is stored locally, and compute3 is only
  read.
- No LLM calls anywhere in the bootstrap.

Logs:
- local: `runs/bootstrap/<host>/<stamp>.log` + `summary.tsv`; fleet runs: `runs/bootstrap/fleet-<stamp>/`
  (`logs/<host>.log`, `table.txt`, `boxes.json`);
- on the box: `~/dojo-bootstrap/logs/<step>.log`; the smokes also write `~/dojo-data/infra/smoke/<step>/run.log`.

### Options

| Option | Use |
|---|---|
| `--role auto` (default) | The preflight reads the driver: 570/580 means `full`, anything else (595) `physics`. If the 5.1 render smoke then fails on a `full` box, the box switches itself to `physics`: it fetches Isaac Sim 6.0 and runs the proxy smoke. Re-runs keep that decision; `--role full` retries 5.1. |
| `--role full` / `--role physics` | Force a role. In a fleet, use `HOST:full` / `HOST:physics` per host. |
| `--with-isaac6` | A `full` box gets Isaac Sim 6.0 and the proxy smoke too (+31 GB). |
| `--assets-from hf\|compute3\|HOST` | Where the 13.5 GB of Assets come from: HF via the git-lfs batch API (default), compute3 (relayed through this box), or a box that has them. In a fleet, only the seed uses this; the others copy from a box that has them. |
| `--assets all` | Every Asset except eval data: 40.6 GB instead of 13.5 GB. The default list covers all 54 task configs. |
| `--env-file PATH` / `--no-secrets` | Which `.env` to take OMNIROUTE_* from (default: the main checkout's), or skip it. |
| `--prefix /data` | The big disk is not `$HOME`. `~/dojo-robodojo`, `~/dojo-data`, `~/dojo-perception-data` and `~/dojo-isaac6` become symlinks into `/data`. Venvs and the uv / Kit caches stay in `$HOME` (keep ≥ 30 GB there). |
| `--probe basic` | With `--only preflight`: run ONLY `nvidia-smi`, `df`, `free`, `nproc` on HOST and evaluate here. Use it for another team's box (isr-lab). |
| `--only S1,S2` / `--skip S` / `--force S` | Run a subset, leave a step out, or redo a done step. `--force data,catalog` refreshes those from compute3. |
| `--min-disk-gb N`, `--ignore-preflight` | Relax the disk gate, or go on after a NO-GO. Only do this deliberately. |
| fleet: `--seed HOST` | The box that fetches from HF / compute3 first (default: the first GO host). |
| fleet: `--p2p direct\|relay` | How the boxes copy from each other (§4). Default `direct`, which falls back to `relay` per pair. |
| fleet: `--addr HOST=user@ip[:port]` | How the other boxes reach HOST, when its `~/.ssh/config` name means nothing to them. |
| fleet: `--keep-fleet-key` / `--remove-fleet-key HOST...` | Keep the box-to-box key after the run, or remove it later. |

## 2. Roles

| Role | When | What runs there | Smokes |
|---|---|---|---|
| `full` | driver 570 / 580, and the 5.1 RTX smoke passes | official `dev_eval.sh` (RoboDojo's eval client), the bridge's strict episodes ([BRIDGE-RUNBOOK](BRIDGE-RUNBOOK.md)), genuine 5.1 frames, twin, G3, perception | twin, render (5.1), dev_eval, perception, bridge `--fake` |
| `physics` | any other driver (595), or the 5.1 render smoke failed | like compute3: 5.1 physics-only (twin, G3, `RoboDojoRunner(physics_only=True)`) + compute3's Isaac Sim 6.0 build in `~/dojo-isaac6` for `proxy_eval.sh` (5.1 physics, 6.0 render), perception. No official dev_eval and no strict bridge episode. | twin, proxy_eval (6.0 render), perception, bridge `--fake` |

Isaac Sim 6.0 is compute3's own build (`6.0.0-rc.59+release.41464.5f2772bc.gl`, the one the proxy calibration was
fitted on), copied without its shader caches (31 GB). It is not the pip `isaacsim==6.0.x`. `~/dojo-env.sh` sets
`ISAAC_SIM_ROOT` to it, which `scripts/simlab/isaac6.sh` reads.

## 3. Preflight: what it checks

The preflight is read-only. With `--probe full` (the default, for our boxes) it runs `nvidia-smi`, `df`,
`/proc/meminfo`, `command -v`, `ldconfig -p`, `systemctl --user show-environment` and `curl` HEAD requests. The
script is streamed over ssh, nothing is written, and nothing tries sudo. With `--probe basic` it runs only `nvidia-smi`,
`df`, `free` and `nproc`. Exit 0 means GO, exit 1 means NO-GO. Facts go to `summary.tsv`, and the fleet tabulates them.

| Check | FAIL when | Fix |
|---|---|---|
| gpu / vram | no `nvidia-smi`; < 12 GB VRAM (WARN < 20 GB) | — |
| driver | role `full` forced and the driver major is not 570 or 580 (auto: a WARN, and the role becomes physics) | Ask for 580, or let auto pick physics. On driver 595, Isaac Sim 5.1 segfaults in `librtx.scenedb` at boot (compute3). |
| disk | < 300 GB free under `$HOME` (or `--prefix`) | `--prefix` on the big mount. The install needs ~65 GB: env 24, Assets 13.5, data 5, caches; physics +31 GB. |
| ram / cpu | < 32 GB RAM (WARN < 60 GB, or < 25 GB available); WARN < 12 threads | — |
| os | glibc < 2.35 (Isaac Sim 5.1 wheels) | Ubuntu 22.04 or newer |
| tools | missing git, curl, tar, flock, python3, gcc/g++/make/cmake (IsaacLab builds) | `sudo apt-get install -y build-essential cmake git curl python3` |
| tools (WARN) | missing git-lfs, ffmpeg, rsync | Without them the bootstrap uses the static git-lfs release, imageio-ffmpeg's binary, `cp` and relayed copies. |
| python (WARN) | system python3 is a release candidate (Ubuntu 22.04: 3.11.0rc1 segfaults SAM3 / torch.jit) | Nothing to do: every venv here uses uv-managed CPython (`python-preference = "only-managed"`). |
| vulkan | role full and no NVIDIA Vulkan ICD (a container without the graphics capability); physics: WARN (the 6.0 render needs it too) | Fix the container. |
| egl (WARN) | no `libEGL_nvidia` | Perception falls back to the slow CPU path, and the perception smoke fails. |
| network | huggingface.co, pypi.nvidia.com, download.pytorch.org, github.com, repo.anaconda.com or astral.sh unreachable | proxy / VPN |
| WARN only | GPU busy, inotify < 524288, `/dev/shm` < 8 GB, no systemd user manager, no tailscale | Usually fine. `sudo sysctl fs.inotify.max_user_watches=524288`, `sudo loginctl enable-linger $USER`. |

Reference results (2026-10-25):
- compute3, role full: NO-GO on driver 595 and disk (92 GB). Role auto: physics, NO-GO on disk only.
- isr-lab (basic probe): RTX 4090, driver 580.178.04, role full; NO-GO on disk only (8 GB free).

## 4. Fleet: who fetches what, and peer-to-peer copies

1. Every host is preflighted in parallel. NO-GO hosts are reported and left alone.
2. The plan. The seed (`--seed`, else the first GO host) fetches the Assets from HF, and data and catalogue from
   compute3. The first `physics` box fetches Isaac Sim 6.0 from compute3.
3. Every other box runs its installs (venv, RoboDojo) right away. Each copy step (assets, data, catalog, isaac6)
   waits until SOME box has that set, then copies it from the least busy holder. A box that finished a copy serves the
   next ones, so the fan-out widens.
   - A seed that fails hands its fetch to the next box that asks.
   - A box that waits longer than 90 min (`FLEET_WAIT_S`) fetches from the origin itself.
   - So HF is asked once for 13.5 GB, not once per box (7 boxes from HF would be ~95 GB).
4. Copies:
   - `--p2p direct` (default): the fleet makes an ephemeral ed25519 key, `runs/bootstrap/fleet_key` here (runs/ is
     not in git). Every GO box gets `~/dojo-bootstrap/fleet_key` (mode 600, through ssh stdin, never printed) and one
     `restrict` line in `~/.ssh/authorized_keys`. The receiving box runs `rsync -a --partial` from the holder's
     address, `user@hostname:port` from this box's `ssh -G HOST` (or `--addr`). Before each pair it checks the key,
     rsync on both ends and reachability. If any is missing, or the holder sits behind ProxyJump, that copy is
     relayed through this box (`ssh A tar | ssh B tar`).
   - At the end the key lines are removed again with `sed -i`, which rewrites the file atomically and never truncates
     it. `--keep-fleet-key` keeps them; `--remove-fleet-key HOST...` removes them later.
   - compute3 is never a direct peer: no key goes there. Its data is always relayed through this box.
5. The final table: host, GPU, driver, role, go/no-go, every step (OK / FAIL / blk = blocked / `-` = not this
   role), wall time, ready; and each box's role reason. `boxes.json` collects every `~/dojo-box.json`.
6. One code commit for every box: the fleet fetches `origin/main` once and passes its SHA.

Coordination state lives in `runs/bootstrap/fleet-<stamp>/{seed,have,busy,failed,bad}/` on the local box (flock'ed).
A fleet re-run starts a new plan, and every box skips what it has done already.

## 5. Steps, sizes and wall time

Measured on compute3 (Skoltech network: HF ~15-25 MB/s, compute3 → local ~50 MB/s). An unknown network can be
2x slower; box-to-box copies inside one datacenter are much faster.

| Step | What | Download | Wall |
|---|---|---|---|
| preflight | read-only checks; role auto | — | < 1 min |
| conventions | uv 0.12.1; `~/dojo-gpu.lock` + `~/dojo-gpu-light{,2,3,4}.lock`, `~/dojo-data/<stream>/`, `~/dojo-env.sh` (paths, locks, `DOJO_SCOPE_*`, `ISAAC_SIM_ROOT`), `~/DOJO_BOX.md`, `~/dojo-box.json`; uv `python-preference = "only-managed"`; `loginctl enable-linger` (no sudo; WARN if refused) | 20 MB | < 1 min |
| secrets | the local `.env`'s `OMNIROUTE_*` lines → `~/controlr/.env`, mode 600. Key names are logged, values never. Other keys never leave. | — | seconds |
| code | `git archive <sha>` → `~/controlr` (+ `.deployed_commit`; `.env` survives redeploys) | 10 MB | < 1 min |
| venv | `~/controlr/.venv`: uv-managed CPython 3.12 + compute3's perception venv (`pins/controlr-venv.txt`), PyOpenGL 3.1.9 over pyrender's pin | ~0.4 GB | 1-3 min |
| robodojo | Miniconda, RoboDojo f57e163 + IsaacLab afca7b0 + cuRobo 895c651 + XPolicyLab e08ba78; RoboDojo's own install functions (torch 2.7 cu128, `isaacsim[all,extscache]==5.1.0`, IsaacLab, cuRobo); `policy_venv`; a drift report vs compute3's `pip freeze` | ~10-14 GB (24 GB env) | 10-20 min |
| assets | HF dataset at 269457a, sparse git-lfs (`pins/robodojo_assets.txt`, 449 paths, 2.75k LFS files), or a copy from a peer / compute3. The step accepts a copy without any git or HF call. | 13.5 GB | HF 10-15 min; peer 1-5 min |
| data | `dojo-data/simlab/{layouts,traj,g2_c3b}`, `d0`, `d0_51`, `d0_ext`, `skills{2,3}/catalog` | 1.9 GB | 2-6 min |
| catalog | `dojo-perception-data/{catalog,bg_folds,bg_folds_51,task_config,looks}` | 2.9 GB | 2-6 min |
| isaac6 | physics: compute3's `isaac-sim-6.0` minus `kit/cache`, `kit/logs` → `~/dojo-isaac6/isaac-sim-6.0`, then VERSION + bundled-python check | 31 GB | compute3 10-15 min; peer 2-6 min |
| catalog_post | shape caches (`sil.npz`, `shape_cache.json`) for assets that lack them | — | < 1 min |
| smoke_perception | EGL context on the NVIDIA driver + `eval_g1` on push_T 100000 frame 0 (genuine 5.1). Expected: T 0.5 mm, pad 0.7 mm | — | 1-3 min |
| smoke_bridge | CPU: `scripts/dojo/bridge_episode.sh --fake` with RuntimePolicy (the launcher path on the fake strict server, 0 LLM calls). Expected: end=ended, 172/550 steps | — | < 1 min |
| assets_post | `RoboDojo/Assets` link, `curobo.yml` regeneration, `check_assets` over all task configs | — | 1-2 min |
| smoke_twin | physics-only Twin: boot, load + settle, 40-step rollout, snapshot/restore + 40; G2 contact restore bitwise, or within 0.01 mm (WARN) | — | 2-4 min |
| smoke_render | full: an Isaac Sim 5.1 RTX render of D0 stack_blocks 100000 (3 cameras), compared with isr-lab's genuine 5.1 frame 0 (MAD) | — | 5-15 min (the first RTX boot builds the shader cache) |
| smoke_dev_eval | full: the official harness (`scripts/simlab/dev_eval.sh`, RoboDojo's `main.py`) on dev layout 110007, replaying G2's `stack` trajectory with the XPolicyLab policy `dojo_replay`. Expected: 1 episode, success 1.0 | — | 4-8 min |
| smoke_proxy_eval | physics: `scripts/simlab/proxy_eval.sh` (physics-only 5.1 episode + the 6.0 render service in one scope, the heavy lock), replaying the same trajectory. Expected: success, 263/550 steps, ~0.12 s/obs | — | 3-8 min (first 6.0 RTX boot) |
| ready | `~/dojo-robodojo/READY` + the final `~/dojo-box.json` (capabilities from the smokes) | — | < 1 min |

Order:
1. preflight → conventions → secrets → code.
2. In parallel: venv and robodojo on the box, and the copy steps (assets, data, catalog, isaac6).
3. venv → smoke_bridge; venv + catalog → catalog_post → smoke_perception. Perception is usable early.
4. robodojo + assets → assets_post → smoke_twin → (full) smoke_render → smoke_dev_eval / (physics) smoke_proxy_eval
   → ready.

The critical path is robodojo plus the Isaac smokes, about 45-60 min. A fleet adds little: the peers' copies overlap
their RoboDojo installs.

## 6. Watching, resuming, re-running

- Remote steps run detached (`setsid nohup`). A dropped ssh connection or a killed driver loses nothing: run the same
  command again.
  - Done steps are skipped: each has a fingerprint marker of its pins / commit / role.
  - A running step is re-attached.
  - A failed one is re-run, and its internal checks skip finished parts. `git lfs pull` fetches only patterns not
    applied yet; rsync and the relays copy only what is missing or changed.
- A fleet re-run: the same `scripts/bootstrap_fleet.sh HOST...` command; ready boxes go through in a minute and serve
  copies to the others.
- Status by hand: `ssh HOST bash dojo-bootstrap/lib/box_steps.sh poll robodojo assets`.
- Redo a step: `--force smoke_twin`. Rewrite `~/dojo-box.json` by hand: `ssh HOST bash dojo-bootstrap/lib/box_steps.sh boxjson`.

## 7. When a step fails

| Step | Symptom | Fix |
|---|---|---|
| preflight | NO-GO on disk | `--prefix /bigdisk`, or ask for disk. `--min-disk-gb` only for a deliberate partial install. |
| conventions | `WARN: loginctl enable-linger failed` | `sudo loginctl enable-linger <user>` (an admin). Without it, `systemd-run --user` scopes (proxy_eval, integ_eval, g3) need an open ssh session. |
| secrets | `no OMNIROUTE_* in ...` / skipped | `--env-file` the right `.env`; the agent stream needs it, the rest does not. |
| robodojo | `fetch ... origin <sha>` fails | GitHub lost the pinned commit (force-push). Copy compute3's checkout and its git dir: `ssh compute3 'tar -C ~/dojo-robodojo/RoboDojo -cf - <sub> .git/modules/<sub>' \| ssh HOST 'tar -C ~/dojo-robodojo/RoboDojo -xf -'`, then re-run. |
| robodojo | `isaacsim` / `isaaclab` pip errors | Read `~/dojo-robodojo/install.log` and the step log. Re-running resumes (pip cache in `~/dojo-robodojo/.cache/pip`). A missing compiler means preflight `tools`. |
| robodojo | `policy_venv ... rc` assertion | A system release-candidate Python was picked: `rm -rf ~/dojo-robodojo/policy_venv`, re-run. |
| assets | `git lfs pull` retries | HF rate limit or network. The step retries 20 times, 2 min apart, and keeps what it fetched. Faster: `--assets-from compute3`, or wait for a fleet peer. |
| assets | `LFS pointer files left` | Incomplete download or copy. The applied list is cleared, so the re-run pulls every pattern again. |
| assets | `HF main moved past the pin` | The step fetches the pinned revision. If HF refuses it, decide whether to re-pin `HF_REV` (in `box_steps.sh`). |
| assets / data / catalog / isaac6 | `fleet: waiting for a box that has ...` for long | The seed is slow or stuck: read its log (`runs/bootstrap/fleet-*/logs/<seed>.log`). A failed seed hands over by itself; after 90 min the waiting box fetches from the origin. |
| any copy | `direct ... not possible: relaying through this box` | Boxes cannot reach each other (NAT, firewall, ssh alias without a resolvable HostName). It works, slower. `--addr HOST=user@ip:port` with an address the boxes can reach makes it direct. |
| isaac6 | `VERSION ... != 6.0.0-rc.59...` | compute3's Isaac Sim 6.0 changed. Decide whether to re-pin `ISAAC6_VERSION` (both scripts) or copy from a box that has the old one. |
| assets_post | `no curobo.yml` | `update_embodiment_config_path.py` failed: run it by hand in `~/dojo-robodojo/RoboDojo` with the env python. |
| smoke_twin | `contact restore bitwise False` with max > 0.01 mm | Twin restore is not exact on this machine. Physics still works. Report it before trusting verify-before-execute. Within 0.01 mm (float noise, compute4: 0.0001 mm) the step passes with a WARN. |
| smoke_render | segfault in `librtx.scenedb` | 5.1 RTX does not run here (5090 + driver 580 had never been tried; compute3's crash may be Blackwell + Kit 107.3, not the driver). With role auto, the box switches to physics by itself and fetches Isaac 6.0. With `--role full`, re-run with `--role physics`. |
| smoke_dev_eval | `no _result.json` | Read `~/dojo-data/infra/smoke/dev_eval/*/eval_client.log` and `policy_server.log`. |
| smoke_dev_eval / smoke_proxy_eval | success 0.0 (WARN) | The harness ran but the replay failed. Compare with G2b (cross-machine contact divergence, journal): the box works, but the twin-vs-eval gap is larger here. |
| smoke_proxy_eval | `MemAvailable ... refusing to start` / `systemd-run` errors | proxy_eval.sh needs ≥ 25 GB available and a systemd user manager (linger, see conventions). |
| smoke_perception | `EGL is not on the NVIDIA driver` | Mesa took the EGL context: install the driver's EGL (`libnvidia-gl-<ver>`), or check `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`. |
| smoke_bridge | `driver rc != 0` | `~/dojo-data/infra/smoke/bridge/out/fk-smoke/` (`bridge_episode.log`, `fake_server.log`). CPU only: a venv / code problem, not the GPU. |

## 8. After: what is where, and how streams use it

```
~/dojo-box.json         role, GPU / driver / CPU / RAM / disk, capabilities (from the smokes), locks, scopes, pins
~/dojo-env.sh           source it first: RD_HOME, DOJO_DATA, locks, DOJO_SCOPE_*, ISAAC_SIM_ROOT (physics), catalogue
~/DOJO_BOX.md           the rules of the box
~/dojo-robodojo/        RoboDojo + miniconda3/envs/robodojo + policy_venv + hf_git/Assets   (read-only for streams)
~/dojo-robodojo/READY   install summary + smoke results
~/dojo-isaac6/          physics: Isaac Sim 6.0 (compute3's build)
~/controlr/             main at deploy time (.deployed_commit), .venv with the perception extras, .env (OMNIROUTE_*, 600)
~/dojo-data/<stream>/   data; simlab/layouts (<id>.settled.json preferred), d0, d0_51, d0_ext, skills{2,3}/catalog
~/dojo-perception-data/ catalog (+ color_calib_51.json), bg_folds, bg_folds_51
~/dojo-gpu.lock         the heavy GPU lock (flock); ~/dojo-gpu-light{,2,3,4}.lock the light lane
~/dojo-bootstrap/       bootstrap state, logs, lib (remove it when the box is retired)
```

- **Picking a box.** `ssh HOST cat ~/dojo-box.json`, or the fleet's `boxes.json`:
  - `capabilities.official_dev_eval` / `bridge_strict`: the official client runs here;
  - `isaac60_render`: proxy_eval works;
  - `isaac51_physics`: twin and G3;
  - `perception_egl`, `systemd_scopes`, and `llm_env` (whether `.env` is present).
- **The GPU and memory rules** are compute3's (dojo/README.md):
  - One heavy GPU job holds `~/dojo-gpu.lock`, ≤ 30 min per hold, then a 90 s yield.
  - Up to four physics-only light jobs take one `~/dojo-gpu-light*.lock` slot each, e.g.
    `DOJO_GPU_LOCKS="$DOJO_GPU_LIGHT_LOCKS" scripts/skills/g3.sh ...`.
  - Check `nvidia-smi` first; kill only your own PIDs.
  - Big jobs run in `systemd-run --user --scope -q $DOJO_SCOPE_BUNDLE ...`. The settings: bundle 24G / 1400%, proxy
    16G / 1200%, batch 20G / 1200%, g3 10G / 600%, light 8G / 600%; start only with ≥ 25 GB available.
- **Stream code:** `CONTROLR_HOST=HOST scripts/deploy.sh dojo-<stream>` (or the stream's own deploy script). That
  venv has no perception extras; add them with `ssh HOST bash dojo-bootstrap/lib/box_steps.sh venv dojo-<stream>`.
  The agent reads `~/controlr/.env` by default (`dojo/agent/llmio.py`).
- **Calibration on a 5.1-rendering box (role full).** Perception uses the genuine-5.1 set:
  `DOJO_D0=$DOJO_DATA/d0_51`, `DOJO_BG_FOLDS=$DOJO_PERCEPTION_DATA/bg_folds_51`,
  `DOJO_COLOR_CALIB=$DOJO_CATALOG/color_calib_51.json`. `integ_eval.sh` defaults to the 6.0-proxy files, which is
  right on a physics box.
- **The official harness:** `scripts/simlab/dev_eval.sh --task T --policy P --layouts ~/dojo-data/simlab/layouts`.
  The bridge: [BRIDGE-RUNBOOK](BRIDGE-RUNBOOK.md) §2, with `HARNESS_ENV` pointing at this box's `BENCH` / `RD_HOME`.
- **The eval firewall still applies.** `Assets/Eval_Layout` and the eval-paired imitate `Traj` are not even
  downloaded (or copied) here.

## 9. Pins (and how to move them)

| Pin | Where | Source |
|---|---|---|
| RoboDojo + 3 submodule SHAs | `scripts/infra/pins/robodojo_commit.txt` | compute3 `~/dojo-robodojo/robodojo_commit.txt` (verbatim) |
| HF dataset revision | `HF_REV` in `box_steps.sh` | compute3 `hf_git` HEAD |
| Asset list | `scripts/infra/pins/robodojo_assets.txt` | compute3 `hf_git/.git/info/sparse-checkout` minus eval-paired Traj |
| controlr venv | `scripts/infra/pins/controlr-venv.txt` | `uv pip freeze` of compute3 `~/dojo-perception/.venv` |
| RoboDojo env (report only) | `scripts/infra/pins/robodojo-env.freeze.txt` | `pip freeze` of compute3's env (the robodojo step diffs against it) |
| Isaac Sim 6.0 | `ISAAC6_VERSION` in `box_steps.sh` and `bootstrap_gpu_box.sh`, source `ISAAC6_SRC` | compute3 `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0/VERSION` |
| uv, git-lfs | `UV_VERSION`, `GIT_LFS_VERSION` in `box_steps.sh` | compute3 |

A pin change changes the step fingerprint, so the next run redoes that step (and the steps that depend on it).

## 10. Deliberately not done

- No `apt`, no driver, sysctl or sudo changes. The preflight prints the commands.
- No pyrua / SAM3 env (gated weights, torch). Its env must also use uv-managed CPython: Ubuntu 22.04's
  python3.11 is 3.11.0rc1, which segfaults torch.jit.
- No catalogue build from Assets. The catalogue is copied, because its colour maps and annotations come from compute3
  runs. `python -m dojo.perception.catalog build/annotate/lowres/precompute` is the way to extend it.
- No pip `isaacsim==6.0.x` for the render proxy. compute3's exact build is copied, because the proxy calibration was
  fitted on it.
- robot-harness itself (Mikhail's lane) is not installed: the bridge's strict episode needs his checkout + `HARNESS_ENV`
  (BRIDGE-RUNBOOK §2). The bridge smoke here is our side only (`--fake`).
