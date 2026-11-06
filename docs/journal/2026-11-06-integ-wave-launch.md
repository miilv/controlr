# Integration wave preparation

The lead released compute4 after D-C19 and requested current main, with Claude
Sonnet 5.5 for all camera perceiver calls. These supersede the wave plan's code
pin and cx routes before collection. Main dfcdaf6f was deployed to a new directory;
the catalogue gate passed and all 316 RuntimeConfig rows validated. The candidate
SHA256 matches the prospective freeze. Active queue was empty; old deferred queue
is left untouched. Physics preparation is separate from camera dispatch.

The operational lane script had only a free-memory guard. Its versioned copy adds
used-memory admission at 16 GiB, a 50 GB disk floor, and serialized service startups.
It uses explicit merged-deployment paths and private locks; legacy reserve-file
reads are removed for this booking. No sealed artifacts or other team's directories
are required by the runner. The status tool reads launcher logs only.

Physics results, layout manifests and actual starts remain operational evidence
under `~/dojo-data/wave-1106` on compute4; no camera result is claimed here.

Lead amendments before collection: RoboDojo's standard environment device is CPU,
including historical physics runs; this startup line alone is not an invalid GPU
signal. CUDA-no-device errors remain infrastructure failures. Use one Integration
lane alongside the other team's one Isaac, and the explicitly authorized shared
`~/coord/lane_gate.stamp` boot protocol (120 s); all other coord paths stay forbidden.

For the paired tie comparison the lead amended existing M15 demo provenance before
collection: require status=ok / validation.ok, current settled layout mtime no newer
than demo pkl mtime, and freeze current SHA256 of both for identity across arms.
All 15 M15 demos pass these checks, with no exclusions. Nanosecond mtimes and SHA256s
are recorded in compute4 `~/dojo-data/wave-1106/m15-demo-provenance.json`. Historical
absence of a loaded-layout hash in demo_index does not block this paired comparison.
No scoring demos were regenerated.

## Toy physics head review

Both five-record jobs completed with rc=0, no unstable episodes, infrastructure
failures or CUDA errors. A: 3/5 successes, mean score 0.84. B: 4/5, mean 0.92;
one B-only success, zero A-only losses. The common failed id 100004 scores 0.6 in
both arms (A place_miss; B not_home). B records 63 held reads / 38 adoptions;
46 residual readings have lateral median 0.30 mm, maximum 0.62 mm, angular
maximum 1.98 degrees. Per-view attempts/adoptions: hover 50/34, look 2/1,
show 11/11. Original terminal object traces, held events and summaries are retained
under compute4 `~/dojo-data/ab-wave-1106/physics/` and summarized at
`~/dojo-data/wave-1106/physics-report.json`.

The lead approved the toy physics head: `peg_held_read` is eligible for its
pre-registered camera comparison. This is smoke validation, not camera adoption.
The interrupted first A attempt is preserved separately as
`toy-off.partial-pre-amendment`; it was stopped solely because the original plan
incorrectly rejected the normal CPU environment device. It is not pooled.
