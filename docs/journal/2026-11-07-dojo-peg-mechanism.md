# 2026-11-07 — TOY: what the peg re-test can say about mechanism

**Finding:** “22 harmful versus 11 helpful” is not evidence that the fitted held pose is twice as often worse as better.
All 22 harmful records are **show-pose writes with a stale world-frame before pose and a different tracked physical
ring**. That is a serious logging/association confound. The saved traces also show sub-2-mm corrections consumed by a
1-mm task, show/return motions, and upstream A/B perception differences. They do **not** isolate a single causal source
of the +8. The strongest next pose candidate is a **hover-only, transactional read**; a motion-only ablation would test
the side-effect hypothesis. Neither is implemented or run here.

The registered audit still **FAILS**, `peg_held_read` is **NOT ADOPTED**, and its one re-test remains exhausted.
This diagnosis does not retrospectively substitute another metric or reclassify the registered verdict.

## Sources and boundaries

Baseline main for this analysis: `71390a78`. Frozen experiment source: `86ad1987`. Inputs:

- [Camera verdict and all 80 rows/pairs](../experiments/data/peg-retest/verdict-camera.json).
- [Audit summary/fidelity](../experiments/data/peg-retest/audit-summary.json) and
  [required physics write-audit definition](../experiments/2026-11-06-peg-waist-retest-prereg.md#required-physics-write-audit-fail-overall-not-adoptable).
- Read-only copies of the 80 named `c4b_*_peg*` reports/actions/episode rows under
  `compute4:~/dojo-data/integ-c4/runs2`, selected **only** by that verdict's row manifest.
- Existing `compute4:~/dojo-data/peg-retest-1107/audit/writes.jsonl`, saved replays and analysis script.
  Writes file SHA256: `b1f57e841146084449a2c8dca9c85712c77a1a49c01d4f006807d0fae05f249b`.

Local working copies/calculations: ignored `runs/peg-mechanism` (about 9 MB transferred). No new LLM calls, simulator,
camera job, replay execution, remote write, queue/lane access, sealed-data read, candidate change or implementation edit.
The existing audit reports 80/80 terminal scores reproduced, exact frames for all 1,696 writes, and 532/532 grasp-close
GT probes agreeing within 2 mm. Those are inherited fidelity checks, not checks newly executed in this investigation.

All **160/160 copied reports/actions** match the original SHA256s in the camera verdict.

This is the **fresh re-test**, not the earlier wave. Equal numeric ids in different `peggainN` blocks are different
salted layouts. Its five rescued and two lost pairs contribute +180 summed points; the other thirteen contribute −20.
Dividing +160 by twenty gives **+8 pooled points**. The original paired CI [−6, +22.5] still includes zero.

## What the held writer count measures

Joining each held `belief_write` to the `held_pose` event at the same run/step gives exactly:

| read view | harmful | helpful | neutral | writes |
|---|---:|---:|---:|---:|
| hover | 0 | 7 | 114 | 121 |
| show | 22 | 4 | 0 | 26 |
| total | 22 | 11 | 114 | 147 |

All these writes are on the gain cell; the guards have none. A has no held-pose writes.

At show, **all 22 registered harmful writes** have logged world translations of **96.590–108.011 mm**, while the
read's reported **TCP-relative fit shift is only 0.127–3.059 mm**. The holding arm has moved between the two logged
world poses. In frozen `RuntimePolicy._held_pose`, `before` is copied from `prov.belief.objects[lab].T` **before**
`sync_robot(current_state)`; the new attachment is then synchronized to the arm's current show pose. The logged delta
therefore includes kinematic transport, not just the fitted attachment correction. The scorer chooses its tracked
object nearest this stale before pose, within the category group, which can select a different ring on the table/peg.

I checked the already saved show frames, without new rendering:

- For each of the 26 shows, associate the physical ring at its **first consumed pick**, using nearest initial GT of
  the same asset. Initial association distances are at most 4.46 mm. This identity agrees with the ring nearest the
  show-time after pose on **26/26** records; it is not chosen solely from the after estimate.
- On all **22/22 harmful** records, the registered audit's `gt_label` is **different** from that picked ring.
  The after pose's **root-position** error to the picked ring is median **0.270 mm**, maximum **0.435 mm**.
- On the four show-“helpful” records the tracked identity agrees. Their reported huge improvement also includes
  the roughly 10-cm transport, rather than proving an equally large attachment correction.
- A diagnostic transport check maps the cached before pose from the last hover's commanded FK to the saved actual
  show flange. The resulting prior root errors span 0.136–3.292 mm across the 26 shows; none acquires >2 mm root error
  from the read and six remove >2 mm. This uses commanded hover FK where exact hover GT is absent, **root position**
  rather than the registered body-centre metric, and a fixed picked identity. It is **not a replacement gate audit**.

Concrete rescued-pair example: gain1/100003, B block2, step 744 is scored harmful **71.0 → 167.5 mm**. It follows the
24-step show/return protocol; the fit shift is only **2.55 mm**. The apparent 96-mm worsening cannot be interpreted as
that fit moving the ring 96 mm in its jaws. Another example outside the flips, gain1/100000 B block0 at step 788:
registered GT is stationary `block5`, whereas the picked/held GT is `block0`; after-root error to `block0` is **0.225 mm**.

Two distinct cautions follow. First, a registered **neutral** change is not “no change”: this audit uses >2 mm of
added/removed 3-D body-centre error, while this task has a 1-mm pair/alignment test and a sub-mm threading clearance.
Second, the runtime reanchors **every ok read**, even if its sigma later causes the tail to reject it. Counts of belief
writes and counts of transforms actually used by the guarded tail are different populations.

## Per-pair trace attribution

Notation: `g1/100001` means `peg-gain1`, numeric id 100001. Full source stems are
`c4b_play_stacking_toy_peg{A|B}r1_peggain{1|3|4}_<id>`; each row's report is
`play_stacking_toy_perception_verify-bundle_<id>_report.json.gz`. Steps below refer to `events[].step`; place/peg
receipts are in `log`/`history`. H/P/N are the **unchanged registered** held-write verdicts, not causal labels.

| pair | A → B | episode steps A / B | B H/P/N | B reported extra show/return steps | mechanism supported by the trace |
|---|---:|---:|---:|---:|---|
| g1/100001 | 60 → 100 | 956 / 927 | 1/0/7 | 24 | neutral pose correction precedes avoided retry; later show is confounded |
| g1/100003 | 60 → 100 | 933 / 961 | 2/1/6 | 48 | strongest direct helpful-hover evidence, plus two show excursions |
| g1/100004 | 30 → 100 | 904 / 1026 | 1/1/4 | 48 | show/return and different base target; nominal helpful write includes transport |
| g3/100001 | 30 → 100 | 1005 / 1148 | 0/0/6 | 0 | early rescue precedes any held write; upstream/base confound, later neutral corrections |
| g4/100001 | 60 → 100 | 1051 / 1086 | 1/0/8 | 24 | neutral first-hover correction, no early show or extra logged dwell |
| g3/100000 | 100 → 60 | 961 / 1050 | 2/0/6 | 114 | early neutral-star read/base/initial-look divergence; loss begins before harmful show writes |
| g4/100000 | 100 → 60 | 1074 / 1116 | 1/0/7 | 24 | neutral-star read followed by show/re-pick; mixed pose/motion, not isolated |

**g1/100001, rescued:** A picks block2 at step 63, places it, judges it tilted and attempts an empty-jaws retry at
175. B's block2 hover write at **96** is neutral (**0.5 → 0.5 mm**, fit shift 0.58 mm, sigma 0.297 mm); its tail explicitly
adopts the read, finishes the same 29-step place, and skips that retry. Thus an early show/pause is not the explanation
for this branch difference. B's later block8 show at **290** is harmful in the registered audit, adopted, followed by
an empty-jaws re-pick at 337—**not** a clean rescue of that piece. B later reaches block9 whereas A reports no placement
for it. No held write is audit-helpful anywhere in this winning episode. Attribution: **pose-compatible sub-threshold
correction/trajectory branching**, with later motion and perception confounds; not proof that the first read caused
terminal success.

**g1/100003, rescued:** Both follow group order 3→2→1→0. A reports all four pieces of its last pole on-peg, yet the
episode checker gives 60; that does not by itself localize the failed pole, and belief receipts are not GT success. B's block1 hover at **679** removes
**2.9 → 0.7 mm** error, sigma **0.305 mm**, with an explicitly adopted transform and a 19-step place (same place length
as A). This is the best **pose-estimate** witness among the wins. B then shows block2 at **744** and block4 at **820**,
24 reported steps each; both are registered harmful under the stale-before association discussed above. Block7's hover
at **914** has sigma **0.505 mm**, so its neutral write is **not** used by the tail; it reaches a contact stop and a
“once more” receipt as terminal success is attained. The show motions may help or hurt; the saved pair cannot separate
those effects from the accepted transforms.

**g1/100004, rescued:** The first piece's read fails in B, and the adopted base refinements already differ: A step 118
base y **−0.1786**, B step 117 **−0.1770 m**. Before the sole helpful write, B's pole target differs from A's. A block5
place ends in a retry/slip and block9 has no pair. B performs a block5 show at **269** (24 steps), nominally helpful
**92.5 → 7.1 mm**, but the fit itself moves only **0.182 mm**; most of the audit improvement is show transport. B proceeds
to block9, also shows it at **420** (registered harmful), then retries it unsuccessfully. Later B adopts block3's neutral
hover read at **851** (0.22-mm fit shift) and avoids A's block4 retry. Attribution: **mixed show/motion and changed base/
trajectory**, with a neutral pose contribution plausible; the helpful label does not identify the physical rescue.

**g3/100001, rescued:** There is **no show excursion** and no helpful or harmful held write. B's block1 read at **79**
and block6 read at **264** both fail, with no write or executed look. Their guarded placements still differ from A:
block6 takes **59 vs 20** place steps, avoids A's slipped retry, and B can attempt block8. This improvement starts
**before B's first held write at 494**. The first place targets already differ by **0.843 mm**, and the retained base
beliefs are A x **−0.0032**, B **−0.0024 m**. A fixed extra “read pause” is not established: a failed hover read with
infeasible looks adds no logged motion/still actions. Later neutral reads on block2/3/4/5 are accepted and can affect the
last poles. Attribution: **early upstream perception/base/contact divergence, later pose-compatible corrections**;
neither a pure held-pose explanation nor a demonstrated settle-only rescue.

**g4/100001, rescued:** B adopts the first block1 hover write at **121**, neutral **0.3 → 0.4 mm**, fit shift **0.49 mm**,
sigma **0.310 mm**. Both first places last **59 steps**; A judges the ring tilted and attempts an empty-jaws retry at
203, while B skips it. There was no preceding show or extra logged placement dwell. B block0's neutral write at **208**
(fit shift 2.76 mm) is also adopted. The later block9 show at **314**, registered harmful, does not prevent B's terminal
success and still triggers a slipped retry. Attribution: **neutral/sub-2-mm pose/trajectory effect is plausible for the
early change**, not an early motion-only settle; base targets and later branches remain confounds.

**g3/100000, lost:** Initial perception looks differ **before** the first held read: A's right review uses `d0.14/e55`
(13 steps), B uses `plan` (14). B adopts block0's neutral hover at **215**, fit shift **2.75 mm**, body error **6.4 → 6.6 mm**;
the place receipt is 9.7-mm xy error, followed by the slipped retry at **281** and no block8 place. A places block0 and
attempts block8. B's two harmful writes occur only later, at **635** and **725**, on the four-ring pole; these cannot be
the initiating cause of the earlier lost branch. Three 38-step show protocols add **114 steps**, but B ends at 1050/1200,
so this is **not a demonstrated step-budget loss**. Attribution: **early pose/base/perception divergence**, with later
show overhead; do not blame the aggregate harmful count for this loss.

**g4/100000, lost:** First read at **84** fails; first place lengths match (26). B adopts block4's neutral hover at
**186** (2.85-mm fit shift), then shows block7 at **295** (24 steps, registered harmful) and fails its retry at **342**.
A also reports a block7 place miss but ultimately scores 100: again, belief receipts are not checker truth. B's later
neutral reads improve several belief receipts without restoring checker success. Block8 at **1076** writes a **6.46-mm**
correction but sigma **0.669 mm** rejects it for the tail; B still finishes at 1116/1200. Attribution: **mixed estimate/
show/regrasp dynamics**, not an identified harmful-fit or time-cap mechanism.

Across all seven, the first consumed place targets already differ by **0.301–0.886 mm**, before the first held write.
Initial perception realization is not held fixed by the paired-layout design. The planned peg-group order remains
3→2→1→0; missing/repeated pieces are consequences of branch outcomes, not evidence of a new deliberate stack order.
There are no time caps in these seven. Three of the five winning B rows have **zero audit-helpful held writes**.
Those facts preclude assigning all +8 points to either “better 3-D estimates” or “extra pauses” from observation alone.

## New lever proposal: hover-only transactional pose read

Propose **`peg_hover_read`**, default off, as a new lever, not another retest of `peg_held_read`:

1. One read at the existing hover; **no look/show fallback motion**.
2. Compute the candidate attachment without mutating provider belief. Apply the same lateral-sigma <=0.5 mm and
   axis-sigma <=7.5° gate **before both belief write and tail use**. Rejected reads change neither.
3. Compare/log before and after at the **same measured TCP pose/time**, and carry the grasp-associated identity for
   diagnostics. Log the candidate/accept reason and read view separately from kinematic transport. No GT in the controller.
4. Keep base refinement, guarded threading, release logic, retry rule, and ordering unchanged.

Static filtering of the **existing 147 records**, retaining their existing verdicts:

| hypothetical filter | harmful kept / removed | helpful kept / removed | neutral kept |
|---|---:|---:|---:|
| hover view only, existing ok verdict | 0 / 22 | 7 / 4 | 114 |
| hover only + existing 0.5-mm tail sigma gate | **0 / 22** | **4 / 7** | **60** |
| hover only + TCP-fit shift <=3 mm | 0 / 22 | 5 / 6 | 109 |
| TCP-fit shift <=2 mm, any view | **10 / 12** | **3 / 8** | 92 |
| TCP-fit shift <=3 mm, any view | **21 / 1** | **8 / 3** | 109 |

The four helpful records retained by the transactional hover/sigma proposal are: g1/100002 step **898**, g1/100003
**679**, g2/100004 **76**, g4/100003 **284**. All rejected show entries disappear because the **motion/read path is
absent**, not because we reinterpret their audit verdicts. A native “agrees within 2 mm” gate is a poor candidate here:
it retains ten harmful entries while removing eight helpful ones. Do not gate on the huge logged world displacement;
that would be selecting on stale-world transport rather than attachment agreement.

These are **in-sample retention counts**, not predicted success flips or a prospective harmful-write guarantee.
Changing writes/motion changes later observations and policy paths. Repeated consistent **hover** reads are mostly
absent, so an N-consistent-read gate cannot be estimated honestly from these records. Any implemented new lever needs
its own unit tests, healthy G3 screen, fresh pre-registration/camera A/B, and unchanged write gate. No remaining re-test
of the old lever is implied.

## Proposed non-perception ablation and G3 screen — DO NOT RUN

A fixed “settle pause” is **not yet the demonstrated gain mechanism**: two strong early branch changes have no extra
place steps, and the g3/100001 early improvement has no pose write or executed look. Nevertheless, show/return is a
real physical intervention worth isolating. Propose a separate **`peg_show_return_no_read`** lever: at each pre-place
hover, attempt one existing collision-checked **geometric** generic show/return plan, hold for the existing
`HELD_LOOK_STILL` two steps, then return. No image estimator, candidate pose, belief reattachment, object re-detection,
or read-driven decision. Use the existing deterministic candidate order and reserve guard; skip/log infeasible or
budget-exceeding plans. Keep the original attachment and guarded tail. Only ordinary kinematic propagation is allowed;
no semantic pose write. This reproduces the intervention class seen in the 24/38-step excursions, not their adaptive
selection, and makes no camera-benefit claim.

Prospective screen for the lead to approve, with a frozen generator/catalogue/commit and **20 training layouts
100030–49, paired across four arms**, one repeat, no LLM, normal shared lock/disk/resource limits:

- A: base refine retained, original blind guarded threading, old held read off.
- S: A + exactly **two still env steps** at the hover (a separate `peg_hover_settle2` candidate).
- M: A + geometric show/return-only above; no perception writes.
- P: A + transactional `peg_hover_read` above; for G3 use the fixed existing held-noise table/request provider, with
  the acceptance transaction before provider mutation. Do not let its simulated provider silently reanchor rejects.

Use mode B, camera-like retained base offset **[+1.6,+1.5] mm**, runtime-like held belief; same initial states and
per-layout noise seeds where used. Leave stuck-confirm off. Register all 80 outcomes, no early stopping or best-seed
selection. Log planned/skipped excursions, actual added steps, grasp-associated ring-to-TCP/hole error at hover/return/
release, true peg-relative error, re-picks/slips, no-pair failures, contact stops, and final checker scores. Compare
stage-aligned **same-physical-ring** metrics, not label-only receipts. Verify zero semantic pose writes in S/M and
same-time before/after diagnostics in P. Reject a screen on unhealthy physics/collider initialization, unstable layouts,
or lost replay fidelity; do not substitute software-fallback scores.

This is a discriminating screen: S beating A supports a small-settle mechanism; M beating A without writes supports
motion/re-settle; P beating A while S/M do not supports the read estimate. A mixture or wide spread remains inconclusive.
Do not tune candidate paths, hold counts or gates on those twenty outcomes. No G3 result alone authorizes adoption;
any chosen **new** lever would need a fresh camera A/B under the rule in force at its pre-registration. No run, code
change, new camera registration, or dispatch is authorized by this journal.
