# 2026-10-28: compute2 (isr-lab-4) bootstrapped as a role-full dojo box

Stream: Bootstrap. 0 LLM calls. Code: PR #203 (`--shared-home`, `XDG_RUNTIME_DIR`, smokes off the reserved
layouts, data manifests).

## 1. The box and its constraints

- Hardware: RTX 4090 24 GB, driver 580.126.20, i9-13900KF (32 threads), 62 GB RAM, Ubuntu 22.04. It had 138 GB free at the start (disk 93 % full).
- The machine is someone else's workstation:
  - a desktop session, a Unity editor (0.9 GB VRAM) and a uvicorn `vlm-api` on port 8000 (2.3 GB VRAM) run there;
  - `/root` is shared: other people's dirs, and root's own uv 0.12.22 in `~/.local/bin`.
- Access is tailscale SSH (root through `su`). It exports no `XDG_RUNTIME_DIR`, so `systemd-run --user` fails until that variable is set.
- Network:
  - The box's own internet is fast: pytorch 13 MB/s, pypi.nvidia 11 MB/s, HF 7-9 MB/s.
  - The tailnet reaches every peer only through the DERP relay, ~0.9 MB/s in total; parallel streams add nothing. compute4 -> compute2 measured 1.07 MB/s (nc).
  - compute2 and compute4 are both on 192.168.50.x but on different Wi-Fi networks, so there is no LAN path.

## 2. Data path

- From the internet, directly on compute2:
  - RoboDojo env: Isaac Sim 5.1 + IsaacLab + cuRobo, 28 min;
  - Assets from HF at the pin: 13 GB in 26 min;
  - the controlr venv: 8 s from the warm uv cache.
- From compute3, relayed through the local box (`tar | zstd -6`), small and critical files first:
  - Seeding first: 452 catalogue files (1.5 GB) were copied from compute2's own 24 Oct catalogue. They are md5-identical to compute3's, and their mtimes were set to compute3's so the bootstrap's relay counts them as present.
  - Compression: `visual_lo.glb` compresses 2.9x; PNG frames and `geom.npz` do not compress.
  - Main relay: 13,413 files (4.4 GB raw) in 63 min.
  - Then the rest of compute3's `~/dojo-perception-data` (`bg_*_scan`, `bg_synth`, `g1`, scripts) and the skills4-6 catalogues: 468 MB in 9 min.
- Verified with `box_steps.sh manifest` (sorted md5 manifests), recorded in `~/dojo-box.json` under `manifests`:
  - `~/dojo-perception-data` == compute3: 2491 files, 3.84 GB, md5 `9f82f54b0de5f23c4505202de7c82a74`. This includes the `scene/` backgrounds compute4 lacked: conveyor in `bg_folds`; conveyor, basket and cube_cushion in `bg_folds_51`.
  - `~/dojo-data/simlab/layouts` == compute3: 3459 files, md5 `009810e81d7e4ad95272439e0e69ac7a`.
- Disk after the install: 94 GB free (95 %). Ours: `dojo-robodojo` 38 GB, new data 2 GB, perception 3.6 GB, plus Kit / uv caches.

## 3. Old installs and the home (lead's decisions)

- `/root/controlr-robodojo` was our 4 Oct install: RoboDojo 266130a, HF 35efbc7 including `Assets/Eval_Layout` and all of `Traj`.
  - It is renamed to `/root/controlr-robodojo.old-20261004`, and `/root/controlr-robodojo` is now a symlink to `dojo-robodojo`.
  - The code's root-box default `RD_HOME` (rdpy.sh, dev_eval.sh, paths.py) therefore can no longer reach the old eval data (BACKLOG).
- Old `~/controlr` (7 Oct; its venv ran on the system 3.11.0rc1) is hardlink-snapshotted to `~/controlr-oct07`, without `.env`.
  - `~/controlr/.env` is now OMNIROUTE_* only, mode 600.
- Kept: `~/dojo-data/perception` (17 GB, the 24 Oct catalogue) and `~/dojo-pyrua` (11 GB).
- `--shared-home` left these untouched: root's uv and its receipt, `~/.config/uv`, and lingering (`/var/lib/systemd/linger` stays empty). Our uv 0.12.1 is in `~/dojo-bootstrap/tools/uv`.

## 4. Smokes and checks

No smoke made an LLM call. The genuine-5.1 render needed neither of compute4's fixes:
- Kit's two registry extensions (urdf importer 2.4.31, `omni.kit.pip_archive`) were already in root's `~/.local/share/ov` from the October runs;
- `~/.cache/ov` held 2.6 GB of materials.

| check | result |
|---|---|
| smoke_bridge (CPU, fake strict server) | end=ended, 172/550 |
| smoke_twin (layouts 100000 / 100001) | boot 4 s, stable, 35.1 env-steps/s (compute4 27.6), restore+40 1.16 s. G2 contact restore NOT checked: its record is on reserved 110007 |
| smoke_render (5.1 RTX, D0 stack_blocks 100000) | MAD vs isr-lab's frame 0: 0.4 / 0.5 / 0.5 (compute4: 0.7). Reset 29.7 s, stable, 106 s in total |
| frame 0 vs compute4's smoke_render | MAD 0.71 / 0.60 / 0.59 (head / left / right), p99 3-4, <= 0.03 % of pixels differ by > 16 |
| smoke_perception (EGL, push_T 100000 frame 0, genuine 5.1) | EGL on the NVIDIA 4090; t 0.5 mm, target_t 0.6 mm |
| G3 stack_blocks 100000, `--verify 4 --twins 4` (light lane) | success 1.0, 194/550 steps, plan 32 s. Verify: 6 segments, 24 candidates, 96 rollouts, all first-ok |
| `integ_eval.sh --renderer isaac51`, press_by_number 100000 settled, submit_v3, `{"agentic":"off","held_look":"b1"}` | success 1.00, 240/700 steps (compute4: 242/700). Episode 20.7 s, scope 106 s, memory peak 12.4 GB, 10.8 cores average, 0 agentic calls |
| smoke_dev_eval, smoke_proxy_eval | skipped: they replay G2 on reserved layout 110007 (D-C10a). No Isaac 6.0 here: 31 GB over this link would take ~9 h |

- RoboDojo env vs compute3's freeze: 8 unpinned packages are one minor version ahead (ConfigArgParse, grpcio, hydra-core, msgpack, narwhals, platformdirs, pytest, ruff).
- check_assets: ok=650, bad=10, the known upstream gaps.

## 5. Using compute2

- Before anything: `source ~/dojo-env.sh`. It sets `XDG_RUNTIME_DIR`, our uv, `RD_HOME` and the locks.
- Lingering is off. Start detached jobs as `setsid nohup bash -c 'source ~/dojo-env.sh; <cmd>; echo rc=$?' > LOG 2>&1 < /dev/null &`, so the bash stays alive as the scope's parent.
  - integ_eval.sh and proxy_eval.sh already wait for their scope.
  - g3.sh execs into its scope, so give it `G3_ABORT_MEM_GB=8`.
  - Tested: a scope launched this way survives an ssh disconnect. A scope whose launcher exits dies with the session.
- Lanes (the lead's caps for this box):
  - one rendered job on `~/dojo-gpu.lock`;
  - one other GPU job on `~/dojo-gpu-light.lock`;
  - `light2-4` are not used here.
- Calibration: genuine-5.1 (`d0_51`, `bg_folds_51`, `color_calib_51.json`). `integ_eval.sh --renderer isaac51` picks them by itself.
- `~/controlr` is at 5df689a (this PR's branch); redeploy main after the merge (`--force code`).
- Spend: 0 LLM calls. Downloads on compute2: ~13 GB from HF and ~12 GB from pip/conda. Over the tailnet: ~3.5 GB on the wire.
