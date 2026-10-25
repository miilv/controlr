# 2026-10-25: infra: one-command bootstrap for the new GPU boxes (4090, 5090)

Stream: infra (branch `feat/dojo-infra`). No LLM calls. Nothing was installed on compute3 or isr-lab. Both were only
read: the preflight, plus `find` / `cat` / `pip freeze` of compute3's install to take the pins.

**Done:**
- `scripts/bootstrap_gpu_box.sh HOST [--role full|physics]`: the local driver. `scripts/infra/box_steps.sh`: the
  remote half. `scripts/infra/pins/`: what the box gets.
- The steps:
  - preflight: read-only, the script is streamed;
  - conventions, code, venv;
  - robodojo: RoboDojo's own install functions, at compute3's pinned commits;
  - assets: sparse git-lfs at the pinned HF revision;
  - data / catalog: relayed `compute3 tar | ssh HOST tar` through this box, incremental by a size + mtime manifest,
    so nothing is stored here;
  - catalog_post: shape caches;
  - assets_post: `curobo.yml` and `check_assets`;
  - four smokes and READY.
- The remote steps run detached under `setsid nohup`, with a fingerprint marker per step. A re-run skips what is
  done, re-attaches to what is running, and re-runs what failed. `--dry-run` prints every local and remote command
  and does no ssh.
- An XPolicyLab policy `dojo_replay` (`scripts/infra/xpl_dojo_replay/`) gives `dev_eval.sh`, the official harness,
  a scripted policy: G2's `stack` trajectory on dev layout 110007.
- `docs/RUNBOOK-new-gpu-box.md`, tests in `tests/test_bootstrap_gpu_box.py` (14 tests), and a sandbox ssh shim
  `scripts/infra/dev/ssh`.

**Pins (from compute3):**
- RoboDojo f57e163. Submodules:
  - IsaacLab afca7b0;
  - cuRobo 895c651;
  - XPolicyLab e08ba78.
  - cuRobo and XPolicyLab are NOT the commits the superproject records (d17b54c / 5d0be14). compute3's install used
    `--remote`, so a plain `submodule update` would differ. The bootstrap fetches each SHA.
- HF dataset revision 269457a (= main today).
- Assets:
  - compute3's sparse list (449 paths, 13.5 GB, 2.75k LFS files) minus imitate_sorting_sequence's eval-paired Traj
    (0.6 GB, firewall);
  - `--assets all` = 40.6 GB / 7.1k files (everything except Eval_Layout + that Traj). Of the HF repo's 7.6 TB,
    Assets are 41 GB.
- The controlr venv is `uv pip freeze` of `~/dojo-perception/.venv` (49 packages). The RoboDojo env freeze
  (299 packages, python 3.11.17) is kept for a drift report only.

**Tested today:**

Preflight (read-only, nothing written on either host; checked afterwards):

| Host | Result | Failed checks | Warnings |
|---|---|---|---|
| compute3 | NO-GO, as expected | driver 595.91.07; disk 93-94 GB < 300 | GPU busy (5-8 processes), RAM available 13-28 GB across runs, inotify 65536 |
| isr-lab | NO-GO on disk only | 8 GB free on /home/anywherevla (its other mounts: / 144 GB, data2 8 GB) | GPU busy, inotify |

- isr-lab passes the driver check (580.178.04), and every other check: 4090 23 GB, 62 GB RAM, 32 threads,
  Ubuntu 22.04 glibc 2.35, Vulkan ICD, EGL, systemd user, network, tailscale. With `--min-disk-gb 5` it is GO.
- compute3 with `--role physics` is GO (the driver line drops to WARN).

Other tests:
- shellcheck 0.11.0: clean on both scripts and the shim.
- Dry runs:
  - role full, role physics (no render / dev_eval smokes), `--assets all` (Eval_Layout excluded), `--only preflight`;
  - asserted by the unit tests, which also test the preflight's driver gate (595 FAIL, 580 and 570 PASS, physics
    WARN) and disk gate against a stub `nvidia-smi`.
- End to end, for real, through the ssh shim (sandbox HOME under /tmp; compute3 as the real read-only source):
  - conventions; code (git archive, `.deployed_commit`); venv (uv 0.12.1, managed CPython 3.12.13,
    PyOpenGL 3.1.9, imports);
  - assets on a 3-path list through the static git-lfs fallback, then an incremental pull of one new path only;
  - data / catalog relays, and a second run says "up to date";
  - catalog_post regenerates a deleted shape cache;
  - the pinned RoboDojo clone (+ 3 submodules = compute3 SHAs, 32 s, 583 MB);
  - re-attach after the driver was killed mid-step; READY.
- The perception smoke ran for real on Mesa EGL (vendor override for the test only). Result: T 0.5 mm, pad 0.7 mm,
  identical to compute3's G1 reference for push_T 100000 frame 0.
- Full unit suite: 675 passed.

**Found along the way (all fixed; each would have broken tomorrow's run):**
- **The LFS pointer check aborted the assets step on the good path.** Under `set -e` + `pipefail`, `find | xargs grep -l`
  returns 123 as soon as one small file is NOT a pointer, so the step would have failed at the end of the 13.5 GB
  download.
- **`find -size -1k` matches only empty files**, so the pointer check saw nothing. It now uses `-size -400c` and was
  checked both ways: a fake pointer fails, small real files pass.
- **The frozen perception venv cannot be installed in one resolve.** pyrender pins PyOpenGL 3.1.0. It now installs
  the freeze minus pyopengl, then 3.1.9 with `--no-deps`, which is what compute3 did by hand.
- **`ldconfig -p | grep -q` gave a false "no EGL" on compute3**: grep exits early, ldconfig gets SIGPIPE, and pipefail
  fails the check.
- **Mesa llvmpipe passed the EGL check.** It now requires the NVIDIA vendor string.
- **Steps that re-run every time reported "never started"** (catalog_post, ready). The smoke result helper also lost
  its output on failure under `set -e`.
- **`ready` died where `nvidia-smi` is missing.**
- `pkill -f` matched my own shell once (the DEVELOPMENT pitfall). I killed by PID afterwards.

**Decisions:**
- **Data is relayed through the local box**, not copied box-to-compute3. The new boxes have no keys for compute3,
  and adding keys there would be a change on compute3.
- **No `.env` on the boxes.** Rule 5; the lead decides.
- **No apt.** The preflight prints the command. git-lfs, ffmpeg and rsync have no-root fallbacks.
- **uv is set to `python-preference = "only-managed"`** in `~/.config/uv/uv.toml` on the box, so stream deploys
  (`scripts/deploy.sh`) can never pick a system 3.11.0rc1.
- **`--prefix DIR`** covers boxes whose big disk is not `$HOME` (symlinks under the usual names).

**Not tested (needs a box):**
- the robodojo step (24 GB env), the 5.1 render, dev_eval with `dojo_replay`, the twin smoke;
- wall times on the new network.

Main risks:
- 5.1 RTX on the 5090: compute3's crash may be Blackwell + Kit 107.3, not the driver. The fallback is `--role physics`.
- The replay reaching success under `main.py`'s reset: the harness passes either way; success 0 is a WARN.

**Command for tomorrow:** `scripts/bootstrap_gpu_box.sh HOST --only preflight`, then `scripts/bootstrap_gpu_box.sh HOST`.
