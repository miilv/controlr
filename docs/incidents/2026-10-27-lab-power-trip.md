# 2026-10-27 — sustained full GPU load on compute5-8 tripped the lab's power; the boxes were withdrawn

**Symptom:** at about 10:05 UTC, ssh and `tailscale ping` to compute5-8 stopped answering. All four boxes share one site and LAN. Headscale marked them offline a few minutes later. The DERP relay and compute3 stayed healthy. Ilia then learned that the lab's power had gone out, overloaded by our machines. The lab will not give us these boxes again.

**Root cause:** most likely physical plus load. All four boxes were plugged into one extension cord on a single
outlet (Ilia, after the fact). At full load a box draws roughly 0.7-0.9 kW: an RTX 5090 up to ~575-600 W, plus the CPU.
Four of them make ~3 kW sustained through one socket, at or above what a single outlet or power strip is rated for
(16 A / 230 V ≈ 3.7 kW; many strips are rated 10 A ≈ 2.3 kW). We never asked the lab's professor for permission
to run this load, and never asked how the boxes were powered. What pushed it over was the load we put on the site,
set by the lead's (the agent's) own decisions:
- From ~00:00Z the team rule was "fill the boxes to the brim".
- At ~09:15Z a third heavy (rendered) lock was added on compute7 and compute8.
- At ~08:00Z a slot-borrowing rule was added. Streams could take any free light slot, so idle capacity was filled too.
- The boxes ran at ~90 % GPU utilisation with 7-10 GPU jobs each (Fleet's measurement: VRAM 27-31 of 32 GB, peaks of 8-10 jobs), around the clock. One RTX 5090 is limited to 600 W, and there were four of them plus their CPUs on one site.
- Nobody asked what electrical budget the site had. No GPU power limit was set.

The extra load was not even productive. Fleet's analysis of 816 rendered and 876 G3 jobs (10:20Z, same day) showed that every extra GPU job on a box slowed all the others. Rendered wall time grew ×1.07 per extra G3 job and ×1.35 per extra rendered job, and at ≥ 9 jobs VRAM saturated. A per-box cap (≤ 3 rendered, ≤ 7 GPU jobs) had been introduced about 15 minutes before the outage.

**How it was fixed:** it wasn't on our side. Power was lost, and so were the boxes. Nothing of ours restarts on boot: there are no enabled systemd units and no `@reboot` cron entries, only `loginctl enable-linger`. Our processes there died with the power. Their run data (round 2b/3 partials, A/Bs, belt frames, part of the new layouts) stays on those disks and is unreachable.

**What changed so it won't repeat:**
- Work continues on compute3 and compute4 only, with conservative per-box caps: compute4 ≤ 2 rendered + ≤ 2 G3, compute3 ≤ 1 proxy + ≤ 3 G3.
- Rule: before scaling load on hardware we don't own outright, get the owner's permission (the professor, for a lab) and ask how it is powered (outlets, circuits). Agree a power budget with them. Ask for a GPU power limit (`nvidia-smi -pl`, admin) where possible. Never "max out" a site. Monitor `power.draw` as well as utilisation.
- The lead's memory carries this as a standing rule.

**Lessons:**
- "Use all the compute" is a throughput goal, not a load goal. The cap that would have kept the site safe (a few jobs per box) was also the faster one.
- A lab box is not a cloud VM. Power, heat and other people's experiments share the same circuit.
