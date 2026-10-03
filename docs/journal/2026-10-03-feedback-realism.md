# 2026-10-03 — which feedback messages could exist on the real robot

**Done:** audited every message the model can receive under the default `feedback.level: short`
(`protocol/feedback.py`, `robot/safety.py` briefs, `robot/isaac/client.py::stop_messages`) against
the real rig's sensors (PHANTOM `phantom/drivers/real/{ur,robotiq}.py`, `configs/hardware.yaml`:
RTDE pose/joints/`getActualTCPForce`/protective stop, Robotiq position + OBJ status, D435).

**Real-robot-safe (same source on hardware):**
- `STATE` tcp x y z / yaw (RTDE), grip width + open/closed (Robotiq position, last command);
- WARN from the harness's own rules: per-line limits, table clearance, workspace edge, TCP outside
  workspace, IK (edge of reach / singularity / joint swing / unreachable orientation), joint near
  limit, gripper width limited, reply not understood / wrong number of values;
- `WARN: the arm ended short of its target (blocked?)` (commanded vs measured TCP);
- `STOP: motion stopped early`.

**Simulator ground truth (not available on the real robot):**
- `STOP: the held packet pushed against the box wall (the box moved)`, `STOP: the arm/gripper
  pushed the blue box (it moved)`, `STOP: … pushed against the table/mat / the blue box / the
  packet` — object attribution and box motion come from PhysX contact views; a real arm only feels
  a wrist force (or a UR protective stop);
- box proximity / "would hit the box" WARNs — need the box's current pose (dynamic in sim);
- `WARN: task not complete yet` — uses the sim's goal check; no oracle on hardware;
- `WARN: gripper linkage abnormal` — PHANTOM's sim diagnostic (hardware: Robotiq fault code);
- `STOP: physics became unstable` — sim artefact (ends the episode; the model never sees it).

**Real-only stops the sim never produces:** UR protective stop, human e-stop, Robotiq fault, RTDE loss.

**Observed in the contacts-and-speed eval:** every wall contact was followed by "backing off …
lifting to clear the rim" — partly because the STOP told the model it was the box wall. Episode-
ending STOPs (instability, the 3rd STOP) are never seen by the model; one release episode received
a false "the gripper pushed against the packet" while opening.

**Decision (pending implementation, BACKLOG P1):** `feedback.sensing: real` as the default — one
contact detector emulating the wrist force/torque reading in the sim, generic STOP text (optionally
with the force direction), box WARNs off, no goal feedback; `privileged` keeps today's texts for
debugging. Blocked on knowing whether the UR3 is a CB3 or an e-Series (BACKLOG P3).
