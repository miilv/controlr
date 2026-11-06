# Compute6 dev diagnosis and tie dispatch recovery

The lead released a bounded dev-only investigation at 12:38:51Z, with a
14:08:51Z deadline and permission for user-level cache/configuration fixes.
One dev episode completed without infrastructure failure at 13:13:39Z.
Measured dispatch resumed at 13:22:31Z, using the same controller deployment
and an explicitly recorded infrastructure recipe. This entry follows
[the initial launch](2026-11-06-integ6-tie-preflight.md) and
[the watchdog incident](../incidents/2026-11-06-compute6-tie-watchdog.md).
PR #396 had already merged; this recovery is documented separately.

## Cache and driver comparison

The robot-harness [box recipe](https://github.com/msannikov03/robot-harness/blob/main/coord/BOXES.md)
describes offline Kit extensions and a populated `~/.cache/ov` for Wi-Fi 5090s.
Compute6 lacked these compute4 SDK extension-cache entries:

- `isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311`;
- `omni.kit.pip_archive-d38fa9ecd1fb6df4`;
- the accompanying `cache_db.json`, `index`, and `urls` metadata.

These entries were copied into the private overlay
`compute6:~/dojo-data/integ6-r3/extscache`, leaving the RoboDojo environment
intact. All **3,649 extension file SHA256 values** matched compute4.
The source hash-manifest SHA256 is
`ca77c6a30c149a0e032e5cf89af8faa5ea522249b97e1ab6682f73484c634eaa`.
The ten `~/.cache/ov/client` files were already byte-identical; their directory
was copied preserving timestamps. Compute4's texture cache was about 20 GB
versus 6.2 GB on compute6; it was not copied wholesale without causal evidence.

Both boxes use NVIDIA 580.178.04, RTX 5090 and NVIDIA EGL rendering.
PyOpenGL 3.1.9, pyrender 0.1.45, pyglet 2.1.19 and numpy 2.5.4 match;
neither has PyOpenGL-accelerate. A tiny 100-frame cube render took 0.043 s on
compute4 and 0.095 s on compute6. This limited probe does not establish full
render-performance parity. The checked GL/thread environment on compute4 had
no special thread-count or software-renderer override.

The pinned controlr deployment remains
`416a2a0374e06628350816a019862caf5156e730`, with launcher
`5b9337bc96cc65290cbe5007d2f3dd6368f31084`.
The catalogue gate still passes with MD5 `2cd7455cbdce2ab38e4e9d586e28db61`;
`candidate_parity --packages-ref` again returns `parity=true`, `diff={}`
(`~/dojo-data/wave-1106-c6/parity-r3.json`). The prior exact tie input hash
checks remain applicable; no measurement input was changed. Accepted simulation
Python **3.11.18 versus 3.11.17** and msgpack **1.2.4 versus 1.2.3** remain in
every new job's provenance.

## Dev evidence and bounded configuration

Dev runs used id **100002**, the same candidate, renderer, pinned deployment
and lane implementation, with isolated state/output under
`~/dojo-integ-c6-r3-state` and `~/dojo-data/integ6-r3/runs`.
Only the dev runs loaded a diagnostic `sitecustomize.py` through `PYTHONPATH`:
it recorded thread stacks every 45 seconds, without locals or image inspection.

`c6dev_imitate_offline_100002` started 12:52:52Z and hit the existing early
watchdog at 12:58:52Z. Kit was ready in about 4.1 seconds, with startup complete
at 5.877 seconds; both copied extension versions loaded. The driver waited on
the runtime command queue, policy waited for perception, and changing
perception stacks showed inverse-layout proposals, SciPy Nelder-Mead fitting
and EGL framebuffer reads. CPU usage reached about four cores and GPU usage
53%. This was active computation, not evidence of a blocked extension-registry
connection, a stuck demo replay or a driver crash. It does not prove that every
earlier tie attempt had the same cause.

The existing watchdog can terminate after 300 seconds without an environment
step when a zenity/display warning occurs, even while perception is busy.
The user-level recovery profile sets `INTEG_WATCHDOG_S=600`, equal to the
existing `INTEG_WATCHDOG_HARD_S=600`. The hard limit, 25-minute episode cap,
1980-second lane cap, controller and perception algorithms are unchanged.
Both `SIMLAB_KIT_ARGS` and `SIMLAB_PHYSICS_KIT_ARGS` contain:

```text
--ext-folder /home/robotics/dojo-data/integ6-r3/extscache --/app/extensions/registryEnabled=0
```

`c6dev_imitate_offline_grace_100002` started **13:06:09Z** and exited **rc=0 at
13:13:39Z**, occupying the lane for 451 seconds. It produced a normal terminal
policy result (`perception_gross`, success false, 669/1600 steps, two calls),
with clean teardown and no watchdog, infrastructure failure or time cap.
This qualifies infrastructure completion, not task success. The cache changes
alone did not prevent the first dev timeout, so recovery is not attributed
solely to the offline recipe. No treatment outcomes or scoring frames were
used to select this configuration.

The dev's automatic retry was archived without running it. Its owned services
and power logger were stopped through checked PID/unit ownership at 13:17:07Z.
Measured workers have no diagnostic `PYTHONPATH`. No system, driver, power-limit
or service configuration was changed, and other streams' light locks were
untouched.

## Disclosed release

The original four infrastructure attempts affected **one scheduled pair**:
100005/r1, arms A and B, each with its original attempt and automatic retry.
The lead's reference to two pairs was reconciled against those exact START
records; no second affected pair was invented. Before release, the permitted
additional attempt for each affected arm was booked as:

```text
c6b_imitatesg_tieAr1_100005__retry_r3infra
c6b_imitatesg_tieBr1_100005__retry_r3infra
```

The `__retry` tag prevents another automatic retry of either extra attempt.
At **13:18:45Z**, under compute6's queue lock, these two jobs were prepended to
the 106 held jobs (53 complete unstarted pairs). The receipt is
`~/dojo-integ-c6-state/r3-infra-rerun-release.json`, with a disclosure in
`~/dojo-data/wave-1106-c6/r3-rerun-disclosure.log`. Queue SHA256 changed from
`1fd828135e239045831c8b056dcb573bb36ce50f36529c206adc74469b7d4ab1` to
`52e89594cd8661416211b24b9303a14ad9a641ed890c82ce2cc9b438ba709908`.
The runtime recipe was added to deployment metadata before these jobs started
and is retained in each job's `provenance.json`.

Compute4's locked `lead-notes.log` records the release. All 54 pairs remain
assigned to compute6; none was restored to compute4 or split across boxes.
Two lane managers were restarted, with heavy locks `~/dojo-gpu.lock` and
`~/dojo-gpu-heavy2.lock`, the existing 16 GiB-used gate and 50 GB disk floor.
Shared side services are one set of seg/digit/kong processes. The five-minute
power logger remains configured to drain heavy2 after two consecutive readings
above 700 W; HTTP 429 stops further dispatch through the existing global STOP.

First recovery START: **13:22:31Z**, heavy1,
`c6b_imitatesg_tieAr1_100005__retry_r3infra`. Heavy2 is live and waits for
resource admission. The first recovery power sample is **105.27 W**.
The nominal two-lane budget is about **7.3 hours from release (20:42Z)**,
excluding admission waits and retries; it is not measured throughput.
Operational ownership: managers **247818 / 247820**, power **247816**, shared
seg **245933**, digit **245993**, kong **246056**. State/logs remain in
`~/dojo-integ-c6-state`, measured outputs in `~/dojo-data/integ-c6/runs2`.

One environment-inspection command initially used an incorrect NUL delimiter
and exposed transient localhost RPC authentication metadata in tool output.
It was corrected to a strict allowlist with a real NUL split; those values
were not saved, committed or reused. No provider API key was printed and no
source job was stopped or modified.
