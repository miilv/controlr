# Runbook: a new GPU box, productive in about an hour

One command, run from the local box, turns a fresh ssh-reachable GPU machine into a dojo box. That means
RoboDojo (Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab) at compute3's exact commits, every Asset the dev tasks
need, controlr main with the perception venv, the perception catalogue with its shape caches, our dev layouts,
the D0 references, the GPU-lock conventions, and smoke tests that prove each layer works.

Code: [`scripts/bootstrap_gpu_box.sh`](../scripts/bootstrap_gpu_box.sh) (local driver) and
[`scripts/infra/box_steps.sh`](../scripts/infra/box_steps.sh) (remote half), with pins in
[`scripts/infra/pins/`](../scripts/infra/pins/). Machine conventions: [dojo/README.md](../dojo/README.md).
Background and pitfalls: [journal 2026-10-24-dojo-simlab](journal/2026-10-24-dojo-simlab.md) and
[2026-10-24-dojo-perception](journal/2026-10-24-dojo-perception.md).

## 1. Before you start (local box)

- `ssh HOST true` must work non-interactively (key auth). An alias in `~/.ssh/config` makes the logs readable.
- `ssh compute3 true` must work: compute3 is the source of the layouts, D0 and the catalogue. The data streams
  through the local box (`ssh compute3 tar | ssh HOST tar`), so nothing is stored locally and compute3 is only read.
- The code deployed is `origin/main` (fetched at start; `--ref` to change it). The remote library comes from the
  checkout you run the script from.
- No LLM calls and no `.env`: the bootstrap never copies secrets (see §7).

## 2. The commands

```bash
cd ~/skoltech/controlr
scripts/bootstrap_gpu_box.sh HOST --only preflight     # ~10 s, read-only: GO / NO-GO, writes nothing on HOST
scripts/bootstrap_gpu_box.sh HOST                      # everything else (--role full is the default)
scripts/bootstrap_gpu_box.sh HOST --dry-run            # every command it would run, local and remote; no ssh at all
```

For two boxes, use two terminals, one command per host. The logs are per host. The two relays share compute3's
uplink, so they run slower.

Options:

| Option | Use |
|---|---|
| `--role physics` | The box cannot render with Isaac Sim 5.1 (driver not 570/580). It becomes a compute3-like physics-only box, and the render and dev_eval smokes are skipped. |
| `--prefix /data` | The big disk is not `$HOME`. `~/dojo-robodojo`, `~/dojo-data` and `~/dojo-perception-data` become symlinks into `/data`. Venvs and the uv / Kit caches stay in `$HOME` (keep ≥ 30 GB there). |
| `--assets all` | Every Asset except eval data: 40.6 GB instead of 13.5 GB, about 30 min more. The default list already covers all 54 task configs. |
| `--only S1,S2` / `--skip S` / `--force S` | Run a subset, leave a step out, or redo a step that is already done. |
| `--min-disk-gb N`, `--ignore-preflight` | Relax the disk gate, or go on after a NO-GO. Only do this deliberately. |

## 3. Preflight: what it checks

The preflight is read-only. It runs `nvidia-smi`, `df`, `/proc/meminfo`, `command -v` and `curl` HEAD requests, and
the script is streamed over ssh. Exit 0 means GO, exit 1 means NO-GO.

| Check | FAIL when | Fix |
|---|---|---|
| gpu / vram | no `nvidia-smi`; < 12 GB VRAM (WARN < 20 GB) | — |
| driver | role full and the driver major is not 570 or 580 | Ask for 580. On driver 595, Isaac Sim 5.1 segfaults in `librtx.scenedb` at boot (compute3). The alternative is `--role physics`. |
| disk | < 300 GB free under `$HOME` (or `--prefix`) | `--prefix` on the big mount. The install needs ~65 GB: env 24, Assets 13.5, data 5, caches. |
| ram / cpu | < 32 GB RAM (WARN < 60 GB, or < 25 GB available); WARN < 12 threads | — |
| os | glibc < 2.35 (Isaac Sim 5.1 wheels) | Ubuntu 22.04 or newer |
| tools | missing git, curl, tar, flock, python3, gcc/g++/make/cmake (IsaacLab builds) | `sudo apt-get install -y build-essential cmake git curl python3` |
| tools (WARN) | missing git-lfs, ffmpeg, rsync | Optional. Without them the bootstrap uses the static git-lfs release, imageio-ffmpeg's binary and `cp`. |
| vulkan | role full and no NVIDIA Vulkan ICD (a container without the graphics capability) | Kit RTX cannot start: fix the container, or use `--role physics` |
| egl (WARN) | no `libEGL_nvidia` | Perception would use the slow CPU path, and card digits fail. The perception smoke fails too. |
| network | huggingface.co, pypi.nvidia.com, download.pytorch.org, github.com, repo.anaconda.com or astral.sh unreachable | proxy / VPN |
| WARN only | GPU busy, inotify < 524288, `/dev/shm` < 8 GB, no systemd user manager (`proxy_eval.sh` / `integ_eval.sh` scopes), no tailscale | Usually fine. `sudo sysctl fs.inotify.max_user_watches=524288`, `loginctl enable-linger $USER`. |

## 4. Steps, sizes and wall time

Measured on compute3 (Skoltech network: HF ~15-25 MB/s, compute3 → local ~50 MB/s). An unknown network can be
2x slower.

| Step | What | Download | Wall |
|---|---|---|---|
| preflight | read-only checks | — | < 1 min |
| conventions | uv 0.12.1; `~/dojo-gpu.lock`, `~/dojo-data/<stream>/`, `~/dojo-env.sh`, `~/DOJO_BOX.md`; uv `python-preference = "only-managed"` | 20 MB | < 1 min |
| code | `git archive origin/main` → `~/controlr` (+ `.deployed_commit`) | 10 MB | < 1 min |
| venv | `~/controlr/.venv`: uv-managed CPython 3.12 + compute3's perception venv (`pins/controlr-venv.txt`), PyOpenGL 3.1.9 over pyrender's pin | ~0.4 GB | 1-3 min |
| robodojo | Miniconda, RoboDojo f57e163 + IsaacLab afca7b0 + cuRobo 895c651 + XPolicyLab e08ba78 (`pins/robodojo_commit.txt`); RoboDojo's own install functions (torch 2.7 cu128, `isaacsim[all,extscache]==5.1.0`, IsaacLab, cuRobo); `policy_venv`; drift report vs compute3's `pip freeze` | ~10-14 GB (24 GB env) | 10-20 min (compute3: 9) |
| assets | HF dataset at 269457a, sparse git-lfs (`pins/robodojo_assets.txt`, 449 paths, 2.75k LFS files) | 13.5 GB | 10-15 min |
| data | relay: `dojo-data/simlab/{layouts,traj,g2_c3b}`, `d0`, `d0_51`, `d0_ext`, `skills{2,3}/catalog` | 1.9 GB | 2-6 min |
| catalog | relay: `dojo-perception-data/{catalog,bg_folds,bg_folds_51,task_config,looks}` | 2.9 GB | 2-6 min |
| catalog_post | shape caches (`sil.npz`, `shape_cache.json`) for any asset that lacks them | — | < 1 min |
| smoke_perception | EGL context on the NVIDIA driver + `eval_g1` on push_T 100000 frame 0 (genuine 5.1). Expected: T 0.5 mm, pad 0.7 mm | — | 1-3 min |
| assets_post | `RoboDojo/Assets` link, `curobo.yml` regeneration, `check_assets` over all task configs | — | 1-2 min |
| smoke_twin | physics-only Twin: boot, load + settle, 40-step rollout, snapshot/restore + 40, G2 contact restore must be bitwise | — | 2-4 min |
| smoke_render | full Isaac Sim 5.1 RTX render of D0 stack_blocks 100000 (3 cameras). Compared with isr-lab's genuine 5.1 frame 0 (MAD) | — | 5-15 min (the first RTX boot builds the shader cache) |
| smoke_dev_eval | the official harness (`scripts/simlab/dev_eval.sh`, RoboDojo's `main.py`) on dev layout 110007, replaying G2's `stack` trajectory with the XPolicyLab policy `dojo_replay`. Expected: 1 episode, success 1.0 | — | 4-8 min |
| ready | `~/dojo-robodojo/READY` (GPU, pins, smoke results) | — | < 1 min |

Order: preflight → conventions → code. Then venv, robodojo and assets run on the box in parallel, while the data
and catalog relays run here. venv + catalog → catalog_post → smoke_perception (perception is usable early).
robodojo + assets → assets_post → smoke_twin → smoke_render → smoke_dev_eval → ready. The critical path is
robodojo/assets plus the Isaac smokes, about 45-60 min (role physics: about 35 min).

## 5. Watching, resuming, re-running

- Local log: `runs/bootstrap/<host>/<stamp>.log`. Box logs: `~/dojo-bootstrap/logs/<step>.log` (the smokes also
  write `~/dojo-data/infra/smoke/<step>/run.log`). On failure the driver prints the last 40 lines of the step's log.
- Remote steps run detached (`setsid nohup`). A dropped ssh connection or a killed driver loses nothing: run the
  same command again. Done steps are skipped (each has a fingerprint marker of its pins/commit). A running step is
  re-attached. A failed one is re-run, and its internal checks skip finished parts: `git lfs pull` only fetches
  patterns not yet applied, pip skips what is installed.
- Status by hand: `ssh HOST bash dojo-bootstrap/lib/box_steps.sh poll robodojo assets`.
- Redo a step: `--force smoke_twin`. Add assets later: edit the pins list (or `--assets all`) and re-run. Only the
  new patterns are pulled.

## 6. When a step fails

| Step | Symptom | Fix |
|---|---|---|
| robodojo | `fetch ... origin <sha>` fails | GitHub lost the pinned commit (force-push). Copy compute3's checkout and its git dir: `ssh compute3 'tar -C ~/dojo-robodojo/RoboDojo -cf - <sub> .git/modules/<sub>' \| ssh HOST 'tar -C ~/dojo-robodojo/RoboDojo -xf -'`, then re-run. The pin check then sees the right HEAD. |
| robodojo | `isaacsim` / `isaaclab` pip errors | Read `~/dojo-robodojo/install.log` and the step log. Re-running resumes, and the pip cache is in `~/dojo-robodojo/.cache/pip`. A missing compiler means preflight `tools`. |
| robodojo | `policy_venv ... rc` assertion | A system release-candidate Python was picked: `rm -rf ~/dojo-robodojo/policy_venv`, then re-run. uv is set to managed-only. |
| assets | `git lfs pull` retries | HF rate limit or network. The step retries 20 times, 2 min apart, and keeps what it fetched. |
| assets | `LFS pointer files left` | The download is incomplete: re-run the step. |
| assets | `HF main moved past the pin` | The step fetches the pinned revision. If HF refuses that, decide whether to re-pin `HF_REV` (in `box_steps.sh`) to the new main. |
| assets_post | `no curobo.yml` | `update_embodiment_config_path.py` failed: run it by hand in `~/dojo-robodojo/RoboDojo` with the env python. |
| smoke_twin | `contact restore bitwise False` | Twin restore is not exact on this machine. Physics still works. Report it before trusting verify-before-execute. |
| smoke_render | segfault in `librtx.scenedb` | 5.1 RTX does not run here. Isaac Sim 5.1 on a 5090 with driver 580 had never been tried (compute3's crash may be Blackwell + Kit 107.3, not the driver). Re-run with `--role physics` and report. The box is then compute3-like: perception and twin work, and the official harness does not. |
| smoke_dev_eval | `no _result.json` | Read `~/dojo-data/infra/smoke/dev_eval/*/eval_client.log` and `policy_server.log`. |
| smoke_dev_eval | success 0.0 (WARN) | The harness ran but the replay failed. Compare with G2b (cross-machine contact divergence, journal): the box works, but the twin-vs-eval gap is larger here. |
| smoke_perception | `EGL is not on the NVIDIA driver` | Mesa took the EGL context: install the driver's EGL (`libnvidia-gl-<ver>`), or check `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`. |
| data / catalog | relay failed | compute3 or the box was unreachable. Re-run: only missing or changed files are copied (size + mtime manifest). |

## 7. After: what is where, and how streams use it

```
~/dojo-robodojo/        RoboDojo + miniconda3/envs/robodojo + policy_venv + hf_git/Assets   (read-only for streams)
~/dojo-robodojo/READY   install summary + smoke results
~/controlr/             main at deploy time (.deployed_commit), .venv with the perception extras
~/dojo-data/<stream>/   data; simlab/layouts (<id>.settled.json preferred), d0, d0_51, d0_ext, skills{2,3}/catalog
~/dojo-perception-data/ catalog (+ color_calib_51.json), bg_folds, bg_folds_51
~/dojo-gpu.lock         the team GPU lock (flock); ~/dojo-env.sh; ~/DOJO_BOX.md (the rules)
~/dojo-bootstrap/       bootstrap state, logs, lib (remove it when the box is retired)
```

- Every shell or job on the box: `source ~/dojo-env.sh`. It sets `RD_HOME`, `DOJO_DATA`, `DOJO_GPU_LOCK` and the
  catalogue paths. Repo scripts default to compute2's `/root/controlr-robodojo` otherwise. As root, a
  `/root/controlr-robodojo` symlink covers those defaults.
- Stream code: `CONTROLR_HOST=HOST scripts/deploy.sh dojo-<stream>` (or the stream's own deploy script). That venv
  has no perception extras. Add them with `ssh HOST bash dojo-bootstrap/lib/box_steps.sh venv dojo-<stream>`.
- The GPU and memory rules are compute3's (dojo/README.md): one GPU-heavy job holds `~/dojo-gpu.lock`, ≤ 30 min per
  hold, then a 90 s yield; check `nvidia-smi` first; kill only your own PIDs; big jobs run in
  `systemd-run --user --scope -p MemoryMax=24G -p CPUQuota=1400%`.
- On a 5.1-rendering box, perception uses the genuine-5.1 calibration: `DOJO_D0=$DOJO_DATA/d0_51`,
  `DOJO_BG_FOLDS=$DOJO_PERCEPTION_DATA/bg_folds_51`, `DOJO_COLOR_CALIB=$DOJO_CATALOG/color_calib_51.json`.
  `integ_eval.sh` defaults to the 6.0-proxy files.
- The official harness: `scripts/simlab/dev_eval.sh --task T --policy P --layouts ~/dojo-data/simlab/layouts`.
  The eval firewall still applies. `Assets/Eval_Layout` and the eval-paired imitate `Traj` are not even
  downloaded here.
- Live LLM calls (the agent stream) need `.env` on the box. Rule 5 keeps it local and on compute3 only, so the lead
  decides. If yes: `scp .env HOST:controlr/.env && ssh HOST chmod 600 controlr/.env`, plus a row in
  [ENVIRONMENT.md](ENVIRONMENT.md).

## 8. Pins (and how to move them)

| Pin | Where | Source |
|---|---|---|
| RoboDojo + 3 submodule SHAs | `scripts/infra/pins/robodojo_commit.txt` | compute3 `~/dojo-robodojo/robodojo_commit.txt` (verbatim) |
| HF dataset revision | `HF_REV` in `box_steps.sh` | compute3 `hf_git` HEAD |
| Asset list | `scripts/infra/pins/robodojo_assets.txt` | compute3 `hf_git/.git/info/sparse-checkout` minus eval-paired Traj |
| controlr venv | `scripts/infra/pins/controlr-venv.txt` | `uv pip freeze` of compute3 `~/dojo-perception/.venv` |
| RoboDojo env (report only) | `scripts/infra/pins/robodojo-env.freeze.txt` | `pip freeze` of compute3's env (the robodojo step diffs against it) |
| uv, git-lfs | `UV_VERSION`, `GIT_LFS_VERSION` in `box_steps.sh` | compute3 |

A pin change changes the step fingerprint, so the next run redoes that step (and the steps that depend on it).

## 9. Deliberately not done

- No `apt`, no driver or sysctl changes. The preflight prints the commands.
- No Isaac Sim 6.0 render proxy. Without 5.1 rendering the box is physics-only.
- No pyrua / SAM3 env (gated weights, torch). Its env must also use uv-managed CPython: Ubuntu 22.04's
  python3.11 is 3.11.0rc1, which segfaults torch.jit.
- No catalogue build from Assets. The catalogue is copied, because its colour maps and annotations come from
  compute3 runs. `python -m dojo.perception.catalog build/annotate/lowres/precompute` is the way to extend it.
