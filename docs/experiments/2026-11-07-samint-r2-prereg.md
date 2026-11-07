# SAMINT round 2: small instruction-only prompts and uniform phantom filters

Prospective registration, before any round-2 inference or scored filter results.
[Design](data/samint-r2/design.json), component/evaluator `68cc25e817f84c948cd460d3f7dd898083c0eac8`.
[Round 1](2026-11-07-samint-general.md) remains rejected; task-keyed SAM routes/banks
remain default-off, diagnostic-only and forbidden for board candidates.

Use the same 180 views / 60 recordings / 54 input layouts / all 11 recorded SAMBAKE
cells, immutable frame/instruction manifests from round 1. Same SAM3 BF16 checkpoint,
v2b weights, 0.2/0.25 confidence and historical >=30px / IoU>=0.5 / phantom metric.
No sealed data; IDs only 100000–09 / 100020–24. Task/GT enter scoring after prediction.
The [camera/GT integrity manifest](data/samint-r2/geometry-manifest.json) additionally
freezes camera calibration and precomputed scoring files, without giving GT to filters.

## Prompts and inference

The generator accepts instruction text only. Deterministic grammatical chunks,
compound tokens, adjective/noun phrases and generic number/function-word removal;
no task names, task configs, checkers, manually supplied object lists or aliases.
Add exactly **object, container, tool** for every episode. No catalogue prompt sweep.
[Generated prompts](data/samint-r2/generated-prompts.json) are an audit of the pure
function, not a bank loaded/selected by runtime. Prospective counts are **6–13**;
no result was used to change extraction. Size lookup alone reads the whole catalogue.

Reuse the existing SAM socket server, one owned process on compute3, two clients;
no new service protocol. Shared box admission/who/GPU checked first, one free light
lock, combined allocator caps 30% (SAM25% / v2b5%), 10G RAM / 600% CPU scope. Warm up
once on an already-registered RGB without scoring. Record all masks once, self-dedup
SAM at IoU>0.5, and evaluate **all 32 fixed combinations** without more model calls.
Every union preserves v2b first, adding SAM masks only at IoU<=0.5. The historical
v2b rerun must match the published reference; discrepancies prohibit qualification.

## Five uniform checks

Each filter is ablated alone, with all combinations and leave-one-out results also
retained. Constants and bit order are in the design JSON; no per-cell adjustments.

1. **Multiview:** all mask-centroid ray pairs, irrespective of label/identity. Positive
   triangulation, camera baseline>=5cm, ray-angle sine>=0.05, ray gap<=4cm. Backproject
   up to 128 mask pixels at centroid depth into the other view; require >=25% coverage
   in both directions. Keep the best coverage/gap pair, without task/label pruning.
2. **Support:** estimated bounds bottom within 2.5cm of the one public table plane
   z=0.765, or another proposal's top within 2.5cm with >=10% horizontal overlap.
3. **Size:** sorted estimated extents within [0.5*min,2*max] of all catalogue assets
   matching the proposed label. Label unknown/generic: use the entire catalogue's
   envelope. No task-conditioned asset/candidate subset or placement metadata.
4. **v2b agreement:** SAM mask overlaps any v2b mask at IoU>=0.1; ordinary union
   dedup remains unchanged. This may cost the SAM-only gains; report that cost.
5. **Score:** SAM confidence>=0.5. Base v2b detections are never score-filtered anew.

Only RGB and camera calibration are recorded, **no measured depth**. Centroid-depth
silhouette bounds are a triangulated **support/size proxy**, not observed physical
contact or true 3-D shape. Geometry-dependent filters fail closed when no valid pair
exists. Report missing-geometry rejection separately from support/size rejection.
Support and size share geometry estimation, but do not require the multiview flag to
be enabled; standalone ablations are independent flags on the same raw proposals.
No GT positions, task-config heights, role labels or checker tolerances enter them.

## Gates, timing, and the prospective decision

Gain >=5pp for **both** micro and macro task recall. Every task/camera/phase group
must avoid recall loss>=5pp, mean-IoU loss>=0.02, and phantom increase>=0.5/image
with >=2 added phantoms. Same round-1 practical-harm gates; small n is no waiver.
Report per-cell recalls/IoU/phantoms and recall costs relative to unfiltered round 2;
paired layout-cluster bootstrap 10,000 resamples, seed0 for the selected combination.

**Hard latency maxima: head<=2s, each wrist<=1s.** Count measured socket upload,
server queue/inference, base detector, self-dedup and selected filter CPU. For any
geometry check, charge the other views' detector/RPC cost and common geometry CPU;
peer masks are not free. Also require the full-prompt two-client RPC maxima within
each camera budget. Report p50/p90/max and three-view dependency cost, startup and
warmup separately. Controlled scaling: registered prompt prefixes 1,3,6,12,full on
one maximum-count instruction, each camera, three single and two simultaneous clients.
No changes to the configuration after timings or scoring are observed.

If no combination qualifies, choose the closest by: fewest failing gate families
(gains/harms/latency/baseline reference), then fewest harm groups, then largest minimum
micro/macro gain, lower worst latency/budget ratio, fewer filters, lowest bitmask.
Publish all 32 combinations so this ranking cannot hide the recall/phantom tradeoff.

Only a qualified combination with a demonstrated end-to-end task/checker-free path
can proceed to a **new single general** camera prereg, fresh salted owned IDs,
Sonnet5.5, strong plus offline-risk guards, and refined per-writer audit with #417.
Egg's bounded three-salt refill authorization remains conditional on that panel.
Freeze prereg/manifests before queueing; append under compute4 queue lock after all
existing rows (currently 50), never reorder/split pairs, never remove STOP or start
lanes. Otherwise queue nothing. This registration itself authorizes no camera run.
Zero LLM calls. No compute4 GPU work or compute6/7 work. Kill only the owned SAM child.
