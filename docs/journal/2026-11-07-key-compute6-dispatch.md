# 2026-11-07 — KEY compute6 preparation and dispatch

The lead merged #412 (`6d1a4abb02d2529beff478bb50c4c61d759191cb`) and
released round-2 preparation on compute6, user robotics. Both camera arms
use that merged code at `~/dojo-integ-c6-key-6d1a4abb`. The isolated Python
3.12.13 venv reproduces the existing lane package freeze; candidate/package
parity is `diff={}`, catalogue passes, and every existing resolved candidate
field matches the earlier deployment. New fields retain inactive defaults.
The accepted offline Kit overlay, simulation Python/msgpack, 600-second
watchdog and box-local calibration mapping remain unchanged. No system,
shared service or RoboDojo environment changes were made.

Before generation, source/config/background/checkpoint/package and four-salt
pins were hashed. Existing score-blind salt/design registries had no collision.
All CPU generation, raw checks and physics settling used a free admitted
light lock, used VRAM <=20480 MiB / load <=26 at admission, six-core/20-GiB
scopes and a 50-GB disk floor. No compute4 or dc19 path was accessed.
The first setup command reached the deployment before the archive copy had
finished and stopped on a missing candidate file; rerunning after the complete
copy passed all gates. No incomplete deployment was used for generation.

The original two blocks/two refills retained **9/20** key settles and the
launch assertion correctly stopped. Before any camera work, the lead
explicitly authorized one additional five-id block: identical validation,
lowest valid id only, with a stated n=9 fallback if none passed, no further
draws. The [preregistration amendment](../experiments/2026-11-07-key-scoped-prereg.md)
and new salt/hash were committed as `6e3a397c` before generation.
The final block retained 100002/100004; only 100002 was selected. All 35 raw
checks passed. Final selection is **10 gain / 5 charger / 5 stack pairs**;
14 failed key settles and one unused valid key input are retained with hashes.
The n=9 fallback was not used. Preparation finished at **02:09:32Z**.

The [frozen manifest](../experiments/data/key-scoped/freeze.json) SHA256 is
`5bbaa131fc2813bc459537bcf088fcbe79d51fb17026adaabf92047370be2305`.
Its chained original/amendment freezes, all validation indices, pair identities,
raw/settled hashes, resolved A/B configs and queue hash are committed before
camera collection. Operational scripts/logs and complete manifests remain at
`compute6:~/dojo-data/key-1107/`.

The switches consume existing motion search/budget paths: handover search
breadth/orientation/vias and local joint-waypoint slicing. They do not change
any perception read, belief writer, acceptance predicate or write schema.
Existing motion execution and its downstream observations/writes can differ;
this is not an identical-episode claim. The preregistration records the source
paths and why the perception/belief-change write-audit gate does not apply.
No physics-audit pass is claimed. All prep made **zero model calls** and no
processes were killed. Every staged camera row pins `cx/gpt-6-luna` explicitly.
No adoption or camera verdict is authorized by preparation/dispatch.

## Dispatch receipt

At **02:11:25 UTC**, the 40 KEY rows were inserted at queue positions **1–40**
before 92 remaining tie rows. KONG had drained. Active tie 100008/r2 A and B
were both claimed, so no queued mate/retry needed protection. Admission then
queue locks prevented the pop-before-START race. Old rows retain exact order.
The [receipt](../experiments/data/key-scoped/dispatch.json) preserves the full
batch, active jobs, before/after hashes and manifest pin. `notes.log` records
these details. The complete input freeze was committed as `af11db4a` before
insertion. Every new row pins cx; unrelated rows/services/lanes are unchanged.

Queue SHA256 `37fad54195b73ce74b6f63958b04e04b03be8ab6af7d5973b3c4fd19968da1f8`
→ `1447f85702da8d3bec332a14b239efaacb3f6597b6cbca9c98aed96e9170c796`.
Manifest SHA256 remains
`5bbaa131fc2813bc459537bcf088fcbe79d51fb17026adaabf92047370be2305`.
Estimated finish **04:11–06:11 UTC**, subject to already claimed tie jobs,
admission/model speed and retries. The prior two-lane KONG block is only a
rough timing reference; the existing job caps allow about 11 two-lane hours
before waits/retries. No camera verdict is made by this dispatch handoff.
