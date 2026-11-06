# Segmenter dev gate integrity and rehearsal coverage

**Question:** do the repeated v2cn/v2d losses survive a dev-only cut, and is the
old-class rehearsal preserved?

**Setup:** archived frame-0 predictions, five source sets, ids 100000–04 and
100030–49 only; v2b vs v2cn and v2d. No new training/inference or episode A/B.
Provenance, weight MD5s, recipe evidence and a proposed resumption sequence are
in the [SEG journal](../journal/2026-11-05-dojo-seg-v2e.md).

**Budget/spend:** zero LLM calls/tokens, zero GPU jobs. CPU checkpoint/manifest
inspection only. No compute4 access.

**Results:** 162 dev frames / 975 objects per arm. Five frames in each arm have
OpenGL errors: the baseline is **invalid for adoption** and requires regeneration.
Diagnostic found totals are 888 / 887 / 887. Both candidates lose two found
objects in each of arrange_largest_number's number and cube_cushion categories.
Full class tables: [v2cn](data/seg-v2e-dev/v2cn.md),
[v2d](data/seg-v2e-dev/v2d.md). Archived cuts remain on compute3 under
`~/dojo-data/s-v2e-seg/bake` with source/config/output hashes in manifests.

**Diagnosis:** v2d has zero training-image support for 140 old classes, including
all 29 number assets. Class-id prefix, input resolution and stride are preserved.
Missing rehearsal is a supported hypothesis, not a proven causal explanation;
training ablations are blocked because compute3's NVIDIA driver is unavailable.

**Next:** restore GPU availability externally, establish a clean dev-only v2b
baseline, rebuild training support before any fine-tune, then gate v2e. No
candidate or camera A/B is qualified by this report. No success/failure videos
exist because no episodes were run.
