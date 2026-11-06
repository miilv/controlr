# Compute3 camera lane preparation

**Dispatch blocked after startup; no valid comparison rows. ETA withdrawn.**

INTEG3 was explicitly released to use compute3 after the shared-session driver
incident. No driver, package-manager, service or VPN changes are authorized or
made. The colleague's desktop session remains present. Initial GPU check:
RTX 5090, driver 595.99.02, 3,233 MiB used, 6.13 W; disk 110 GB free.

Main `86ad1987` is staged as a new `~/dojo-integ-c3-86ad1987` deployment.
The candidate package reference is compute4's 416a2a03 snapshot retained on
compute6; compute4 itself is never contacted. Compute6 is read-only. Offline
Kit extscache, backgrounds and side-model assets are copied into the private
`~/dojo-data/integ3-1106` root. No secret is copied into the deployment.

The peg/waist registration and salts were committed as `ce58682a` before any
fresh layout generation. PEN's conditional #397 arm and the lead's common
Claude-route override were committed as `a1e1e24e` before collection. No camera
or implementation-agent model calls have been made at this checkpoint.

The c3 lane profile supports a private environment file, only heavy1 on the
existing `~/dojo-gpu.lock`, no coord access, admission at <=14 GiB used and a
`who`/GPU receipt before each camera job. Other profiles retain their defaults.
Preparation uses the same exclusive heavy lock, <=14 GiB admission and checked
raw/settled records. Five-minute power logging is required before GPU work.

The three blocks are conditional PEN (10 pairs), peg re-test (20 gain + 10 + 10
guard pairs), waist re-test (20 gain + 10 guard pairs). Historical timings
predict over 12 single-lane camera hours for the re-tests alone; the eight-hour
work period cannot imply all blocks finish. Dispatch remains stopped until
package/data parity, route audit and input manifests are recorded.


## Preparation receipts

PR #404 merged as `d871174d` with green CI; local verification was **1,892
passed / 191 skipped**. The runtime deployment remains the frozen original
main `86ad1987ac4750dad187a45bf7add1dbcd25b98a`; launcher helpers are `0df5943f`.
The stopped-profile dry run on compute3 passed catalogue validation and honored
STOP without consuming a queue or accessing coord.

The catalogue gate passes, index MD5 `2cd7455cbdce2ab38e4e9d586e28db61`.
All **4,540** copied extension/background/calibration/model files match the
read-only compute6 SHA256 inventory. Its offline extension overlay contains
4,439 files / 198 MiB. Sources remain intact. Compute3's RoboDojo Python
3.11.17, msgpack 1.2.3 and Isaac Sim 5.1.0 match the compute4 reference;
NVIDIA driver differs: **595.99.02 vs 580.178.04**. RoboDojo root `f57e1633`,
XPolicyLab `e08ba788`, IsaacLab `afca7b09`, cuRobo `895c6517` match the recorded
reference revisions. Driver/conda environments were not changed.

All **70/70** original fresh layouts passed: 40 two-pass peg records and 30
RAW conveyor checks. No drops, tolerance changes, exclusions or refills.
PEN's existing 100020–29 raw/settled/index records were copied intact into
`pen-inputs`. Public instruction/CPU routing audit found only fill_pen_holder
reaching fill_container; its JSON/source hashes remain in the metadata root.

The complete schedule is **80 pairs / 160 camera jobs**, ordered PEN, peg,
waist, with AB/BA alternation and both members on compute3. Frozen hashes:

- layout manifest: `f06c49ab2016a9dac47559baa5836e362c50468952117ccb7b98722653671a91`;
- schedule: `b7bbc26cb222d34a8fd419fd2a35326d91d063ca870f78da4c0717de15b4b031`;
- full runtime configs: `f87bea18907b93477287fd62c52ba88922daf9b3835dbcb5bd5c98799483dd9f`;
- relocated candidate: `b5bde8d98c7837c76535e03efd0c42c9d7af0d2d796490e5ae08ef6a74ad4df7`;
- committed salt file: `d468aea30f7d0917a941b78218a90c75b896d37fbf45dc874f0a14264cf5b0a7`.

Every effective A/B config constructs; each pair differs in exactly its one
registered switch. Hashes were verified before dispatch. Per-path receipts,
all raw/loaded/index files, logs and operational scripts remain under
`~/dojo-data/integ3-1106/meta` and `~/dojo-integ-c3-state`. Local artifacts stay
under 1 GB. No compute4/dc19/coord access or independent model calls occurred.

The pinned venv build initially timed out on pypi.nvidia.com. Checked owned
installer PIDs were stopped; the identical pinned packages were fetched
through PyPI plus the torch wheel index. The standard extras script completed,
including torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 5.21.0,
ultralytics 8.4.191 and timm 1.0.30. Final candidate parity is **true**, diff
**{}**, with **zero package differences** against the compute4 416a2a03 snapshot.
No CUDA/torch version substitution or system installation was used.

At **15:41:13Z**, the launch wrapper (PID **1673442**) was released after a final
input-hash and 160-row queue verification. Active queue SHA256 before dequeue:
`0a56eeec21f87401253ec3ad9438639fddcafbb46a975106e16261f0fdb60a4e`.
The wrapper starts one set of owned shared side services and a single heavy1
manager, and stops those services plus its checked power-logger PID on exit.
The colleague's desktop sessions were still present; pre-launch GPU used
3,340 MiB, free 28,749 MiB, power 8.04 W, disk 99 GB free. All subsequent
jobs retain the 14 GiB used admission gate and existing exclusive heavy lock.

First START: **15:41:19Z**, heavy1,
`c3b_fill_pen_holder_penAr1_penoriginal_100020`. At **15:41:34Z**, the lane
validated worker PID **1674392** against the private background path and all
three shared side-service addresses. Shared wrapper PIDs are seg **1673474**,
digit **1673718**, kong **1673840**. All model workers use the pinned deployment.
The first START admitted at **4,461 MiB used / 27,629 MiB free**.

Planning ETA is approximately **03:00–05:00Z on 7 November**, excluding retries,
resource waits and review barriers; it is not measured throughput. The original
single-lane estimate exceeds the eight-hour work window. No completed paired
block or comparative outcome is claimed at launch.

Read-only status command (uses only the registered schedule and lane-log lines):

```bash
ssh compute3 'python3 ~/dojo-integ-c3-state/status.py'
```

State: `~/dojo-integ-c3-state`; outputs: `~/dojo-data/integ3-1106/runs`;
power: `~/dojo-integ-c3-state/power.jsonl`. Create the state's `STOP` file to
finish the current job then drain; the launch wrapper cleans up its services.
HTTP 429 also stops dispatch. Missing registered pairs prevent adoption; the
status helper applies the pre-registered per-cell rule only after full n.


## Startup failed; dispatch stopped

The first Isaac process crashed during startup. The existing no-row retry
logic pulled five total attempts before STOP drained the manager at
**15:44:06Z**: A/100020 original + retry, B/100020 original + retry, and
B/100021 original. Every completion logged **rc=0 / no row**; none qualifies
as a comparison. The [startup incident](../incidents/2026-11-06-integ3-camera-startup.md)
records the current-attempt native crash evidence and the pre-existing same-day
renderer incident missed during preflight. Zero paired cells are complete.

**The earlier 03:00–05:00Z estimate is withdrawn.** A working CUDA GPU and
passing physics preparation do not establish genuine Isaac 5.1 RTX support.
No system-level repair or renderer/domain substitution is within this task's
scope. Do not resume under the present recipe or count the attempts as losses.

STOP is retained; **158 queue rows** remain. Both arms of 100020 exhausted their
one permitted retry; B/100021 has its one retry queued. The complete original
80-pair schedule is preserved, and no extra measured attempt is authorized by
this journal. Recovery needs a validated compatible genuine renderer and a
fresh prospective infrastructure-attempt disclosure before queue repair.

No `calls.jsonl` or agentic `transcript.jsonl` was created in any attempt;
recorded model spend is zero calls / tokens. Implementation made zero separate
model calls. No measurement frames or treatment outcomes were used for tuning.

## Final cleanup

The three shared server PIDs exited. Their user scopes drained, with the last
already-stopping owned glyphk scope explicitly SIGKILLed after its metadata
and scope ownership were checked. The launch wrapper and power logger exited;
all three owned scope cgroups are empty. `meta/cleanup-final.json` retains the
checked PID/scope receipt. No other process was signalled.

The five failed attempts occupied **167 lane-seconds**. Seven five-minute power
samples ranged **22.67–105.66 W**, with no reading above 700 W. STOP remains set,
all outputs/attempt logs and the original queue/manifests are preserved, and no
Isaac job or owned side service remains running. The colleague's desktop
sessions remain present. All six registered cells have **zero valid pairs**;
peg and waist camera collection never started. There is no completion ETA.
