# Compute3 camera lane preparation

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
installer PIDs were stopped; the identical pinned packages are being fetched
through PyPI plus the torch wheel index. No CUDA/torch version substitution or
system installation is used. Final package parity and camera START receipts
must be added before reporting this lane launched.
