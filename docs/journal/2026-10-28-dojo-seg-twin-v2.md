# 2026-10-28: segmenter v2b, the proposer's mask de-dup, and the Legality calibration renders (stream S)

Report: [../experiments/2026-10-28-seg-twin-v2.md](../experiments/2026-10-28-seg-twin-v2.md). Previous:
[2026-10-28-dojo-seg-twin.md](2026-10-28-dojo-seg-twin.md). 0 LLM calls.

**Done**
- **v2 renders and training.** v2 renders on compute2: 12,351 images, 511 classes. v2b: YOLO11s-seg from v1, 960 px,
  20 epochs, repeat-factor sampling.
- **v2b + de-dup** reaches 951 / 1013 within tolerance (50 fixed / 1 broken; v1 942, 40 / 0). The lead adopted it for
  the rendered arm. Weights: compute4:~/dojo-data/segtrain/models/v2b/{best.pt,classes.json} (sha256 d7b2f167...).
- **`_seg_dedup`** in `seg_propose` (`seg_dup_iou` 0.7). `scripts/perception/seg/eval_val.py` scores per task / view /
  arm pose on the validation draws.
- **`seg_identity_refit`** (#221) on v2b: wrong identities 235 -> 229 with the same within. It stays off by default.
- **Legality's frozen calibration set** (inserted before my eval, at the lead's request).
  - 6 D0 tasks × ids 100040-49, frames 0-3, genuine 5.1, made with their make_d0 @ 1aa7b7a (stops at an id not ours),
    generated records, scratch DOJO_DATA. In compute2:~/dojo-data/segtrain/d0cal_51, 209 MB.
  - 58 / 60 layouts: put_bottles 100040 and 100049 are unstable, and the loop stopped at 100050.
  - My held-out evaluation never uses 100040-49.
- `scripts/perception/seg/d0_render.py`: D0-format frames of an explicit id list, never walking past it. Written for
  that job, then not used: Legality wanted their make_d0.

**Decisions**
- 20 epochs instead of 40, so the numbers came 1.4 h earlier. Val mAP50 had flattened by epoch 15 (0.880 -> 0.888).
- The kong glyph class (0.27-0.44 on wrists) is too weak for tile matching. A dedicated wrist-crop classifier comes
  next, on compute3.

**Found along the way**
- Per-class NMS gives duplicate masks under two classes. Any consumer of the detector must de-dup class-agnostically
  (V's E3 seeds too).
- The chained-script pattern needs a separate name per run. A killed training run writes "v2: end rc=143" to
  train_jobs.txt, which a waiting post-script reads as finished: the retry ran as `v2b`.

**Next**
- The kong glyph crop classifier (compute3 light2).
- v3 data: post-contact states (held, toppled, stacked, in container, dense clutter, the full `_random` catalogue), on
  compute2 after stream C's replays.
