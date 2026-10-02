# 2026-10-02 — compute3 lost its GPU after an automatic kernel upgrade

**Symptom:** `nvidia-smi` — "couldn't communicate with the NVIDIA driver"; Isaac Sim won't start.
The machine had rebooted on 2026-10-02 at 01:51 into kernel `7.0.0-34-generic`.

**Root cause:** the driver is installed as Ubuntu's prebuilt modules
`linux-modules-nvidia-595-open-<kernel>`; they were installed up to `7.0.0-31`, and the package
for the new `7.0.0-34` was never installed — the meta package
`linux-modules-nvidia-595-open-generic-hwe-24.04` lagged the kernel (7.0.0-31 installed, 7.0.0-38
available). Secure Boot is off, DKMS isn't used.

**How it was fixed** (with the owner's permission, no reboot, ~3 min):
`sudo apt-get install -y linux-modules-nvidia-595-open-7.0.0-34-generic && sudo modprobe nvidia nvidia_uvm`
→ RTX 5090, driver 595.91.07. The other user's CPU job and session were unaffected.

**What changed:** the procedure in [RUNBOOK §1](../RUNBOOK.md); a BACKLOG item to upgrade the
meta package together with the kernel, otherwise the next kernel update repeats this.

**Lessons:** on a shared machine kernel updates arrive without us; on "no GPU", first compare
`uname -r` with the installed `linux-modules-nvidia-*`.
