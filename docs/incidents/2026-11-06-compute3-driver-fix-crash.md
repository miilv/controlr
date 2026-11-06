# 2026-11-06 — compute3 crashed when the lead restored its NVIDIA driver under a live session

**What happened.**
- Since its reboot on 4 Nov ~19:37 (local time), compute3 ran kernel 7.0.0-42 with no NVIDIA kernel module. `nvidia-smi` failed, and the TOY / CONV / SEG GPU work was blocked ([previous incident](2026-11-05-compute3-cuda-unavailable-toy.md)).
- The cause: unattended upgrades installed the kernel, but the matching prebuilt module package `linux-modules-nvidia-595-open-7.0.0-42-generic` was missing. The metapackage `linux-modules-nvidia-595-open-generic-hwe-24.04` was still at 7.0.0-38.
- With Ilia's explicit go-ahead, the lead installed that package and upgraded the metapackage with `apt-get install`. That also moved the driver userspace from 595.91.07 to 595.99.02, the version the module is built for.
- The module loaded during the install, and `nvidia-smi` showed the RTX 5090.
- About a minute later the box stopped answering and came back freshly booted. In wtmp, the previous boot's sessions end in `crash`, not a clean shutdown. Nobody issued a reboot, and `Unattended-Upgrade::Automatic-Reboot` is off.
- **The most likely cause** is the live load of `nvidia_drm` / `nvidia_modeset` under the running Xorg / GNOME session on the same GPU.

**Impact.**
- compute3's `physicalai` account is shared with a lab colleague. Their live desktop session (Xorg, GNOME, VS Code) and a `grasp_lab/cli_visual_force_control.py` robot / force-control script were killed by the crash.
- No controlr job was running there.
- After the reboot the GPU works: kernel 7.0.0-42, driver 595.99.02, all four NVIDIA modules loaded.

**Fix of the process.**
- Before any driver, kernel-module or apt change on a shared box:
  - check `who`, the running processes and the GPU users;
  - agree a time with the people affected;
  - prefer a scheduled reboot over a live module load when a display session is running.
- compute3 is off-limits to our streams until Ilia lifts it.
