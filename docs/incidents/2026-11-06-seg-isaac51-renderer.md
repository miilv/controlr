# compute3: Isaac 5.1 rendering still crashes after driver recovery

SEG round 2 ran one bounded genuine-render smoke check after the driver repair:
`seg_render --task arrange_largest_number --ids 100030 --variants 1 --poses 1`,
under `~/dojo-gpu.lock`, with the 10 GB disk guard. Kernel/driver changes were
not attempted. This was a simulator process, not a real-robot run.

Isaac Sim 5.1.0.0 exits 139 during startup. Its fatal stack contains
`librtx.scenedb.plugin.so` followed by `libcarb.scenerenderer-rtx.plugin.so`.
No layout completed and no training RGB/masks were produced. This reproduces
the compatibility failure already documented in `runner.physics_only_experience`;
working CUDA on driver 595.99.02 does not establish working Isaac 5.1 RTX.

Evidence: compute3 `~/dojo-data/s-v2e-seg/render_probe.log` and `render_probe.sh`.
The probe had a 360 s timeout, core-size limit zero and private cache/config/home
directories under the stream's data root. No retry, installation, driver/module
change, service operation or reboot was attempted. No compute4 access.

The machine remained responsive; the desktop sessions and the pre-existing
Python GPU service (PID 3613, 2418 MiB) were still present afterward. v2b's
corrected EGL bake could start normally immediately after the probe.

Genuine-data rehearsal and v2e training remain blocked on compute3. Resume after
an externally provided compatible genuine renderer or a separately approved and
pre-registered change of training domain; do not silently use 6.0 proxy images.
