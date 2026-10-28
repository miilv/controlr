# 2026-10-28 — E3 follow-up: where the candidates come from (B1 / learned-depth seeds, stream S's detector) and the verdict (stream V)

Follows [2026-10-28-e3-featmetric](2026-10-28-e3-featmetric.md).

**Question.** That report found that the feature metric prefers the GT pose over B1's for 39 of 40 gross objects
when GT is among the candidates. Can realistic candidate sources supply such a candidate?
- the B1 foreground blobs;
- the VGGT raised regions;
- swap positions;
- stream S's twin-trained detector (YOLO11s-seg v1, asset key + mask).

**Setup.** As in the report above:
- d0_51 + d0_ext frame 0, B1's dump records;
- gate: the doubt rule (d > 0.55 or |dh| > 15 mm);
- each seed is fitted by B1's render-and-compare at the seed (`mv3d._fit_at`, lifted onto the measured top);
- 3D NMS at 15 mm; the feature metric picks one;
- policies are applied offline (`scripts/perception/mv/e3_policy.py`).

Recall diagnostic (`e3_recall` in the run notes): for each gated B1-gross object, is any candidate within 30 mm of GT,
and does the metric then prefer it?

0 LLM calls. GPU: compute2's light slot.

## Results

| candidate sources, fit | scenes | gated gross objects with a candidate within 30 mm of GT | that candidate beats B1's pose | best policy: within / gross / broken |
|---|---|---|---|---|
| B1 foreground + raised + swaps; the other B1 objects masked out of the fit (v1) | 207 | 6 / 59 | 6 / 6 | 930 -> 934, 73 -> 68, 0 broken (margin 0.1) |
| same, fitted against the whole foreground (v2) | 207 | 15 / 59 | 12 / 15 | 930 -> 935, 73 -> 68, 2 broken (with "confident B1 objects keep their blob", 25 mm); without that protection, up to 12 right objects -> gross |
| + stream S's detections (key or group match), others masked | d0_51 92 | 8 / 37 (detector: 6) | 7 / 8 | 326 -> 329, 48 -> 45, 2 broken (margin 0.05, loss cap 0.6) |
| + stream S's detections (key, group or category match), others masked | d0_ext 115 | 6 / 19 (detector: 4) | 6 / 6 | 604 -> 610, 25 -> 19, **0 broken** (margin 0.1, loss cap 0.8; detector seeds alone: +4 / -4 / 0) |
| one policy for both sets (detector + foreground + raised, margin 0.1, loss cap 0.6) | 207 | | | +5 within, gross -5, 1 broken |

- **The selector works and generation is the limit.** When a candidate near the right pose exists, the feature metric
  prefers it over B1's pose in 31 of 35 cases across the runs. But realistic sources put one there for only 10-32 % of
  the gated gross objects.
- **Why detections are lost.** S reports 81 % detection recall on B1's gross objects, but B1's fit throws most of them
  away. It fits against B1's foreground with the other B1 objects masked out, and in a swap the right blob belongs to
  another (wrong) B1 object.
- **Fitting a detection inside its own mask** is stream S's `seg_propose`, measured end to end by S. It is not
  duplicated here.
- **More candidates also bring wrong ones.** Unmasked fits on other objects' blobs, and look-alikes, sometimes beat
  B1's right pose. A "confident objects keep their blob" rule is needed when candidates are dense.

## Verdict

- **Frame-0 candidate fusion with B1 / learned-depth / swap seeds tops out at +3..+5 within** (gross -3..-5, 0-2
  broken). It is not wired into B1.
- **The feature metric as a selector is ready.** The detector mask-fit path belongs to S's proposer.
- **Cheap next step:** `fm_refine` on top of `seg_propose`'s output, so the feature metric polishes its fits.
- **Shipped** (PRs #207 / #212 / #214 / #216, flags default off):
  - `height_source="mv3d"`;
  - `b1_doubt` (AUC 0.91; 67 / 95 at 7.7 %) with its gate and look consumers;
  - `fm_refine` (+ `fm_refine_w_sil`).
- Arm 10 in Integration's queue: b1_doubt + fm_refine (w_sil 1.0) + `AGENTIC_DOUBT_RISK=0.5`.
