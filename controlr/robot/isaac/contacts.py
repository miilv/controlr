"""Contact reduction and force-stop decisions of the Isaac server, as plain numpy.

WHY a separate module: the stop rules (which contact pair stops the arm at which force,
what counts as the held packet jamming) are the safety logic the model's feedback rests
on, so they are unit-tested on CPU (``tests/test_isaac_contacts.py``); the server only
feeds them PhysX force matrices.

Pair keys: ``<robot group>-<env group>`` for robot contacts (robot groups ``arm`` /
``gripper`` (housing, fingers and pads); env groups ``table`` (bench + mat), ``box``,
``object``) and ``packet-<part>`` for the manipulated object against the environment
(``table``, ``box_floor``, ``box_wall``). ``unstable`` marks a diverged solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORCE_STOP_RISE_N = 20.0    # a force stop needs this much more force than before the motion
UNSTABLE_FORCE_N = 5000.0   # no real contact on this rig gets near this: the solver blew up


def env_group(path: str, object_path: str) -> str:
    """Robot-view filter path -> env group."""
    if path == object_path:
        return "object"
    if path.startswith("/World/Bin"):
        return "box"
    return "table"


def packet_part(path: str) -> str:
    """Packet-view filter path -> part key (``packet-<part>``)."""
    if path.startswith("/World/Bin"):
        return "box_floor" if path.rsplit("/", 1)[-1] == "Bottom" else "box_wall"
    return "table"


PART_TEXT = {"box_wall": "the box wall", "box_floor": "the box floor", "table": "the table/mat"}


def pair_forces(matrix, row_groups, col_groups, prefix: str | None = None) -> dict[str, float]:
    """Max over bodies of the net contact force norm per (row group, column group).
    ``matrix``: (n_rows, n_cols, 3) newtons (PhysX ``get_contact_force_matrix(dt)``);
    ``prefix`` replaces the row group (``"packet"`` for the packet view). Non-finite
    data -> ``{"unstable": inf}``."""
    m = np.asarray(matrix, float)
    if m.size and not np.all(np.isfinite(m)):
        return {"unstable": float("inf")}
    f = np.linalg.norm(m.reshape(len(row_groups), len(col_groups), 3), axis=2) if m.size else \
        np.zeros((len(row_groups), len(col_groups)))
    out: dict[str, float] = {}
    for i, rg in enumerate(row_groups):
        for j, cg in enumerate(col_groups):
            v = float(f[i, j])
            if v <= 0.0:
                continue
            key = f"{prefix or rg}-{cg}"
            out[key] = max(out.get(key, 0.0), v)
    return out


@dataclass(frozen=True)
class StopLimits:
    """Thresholds (N); 0 disables a class of stops."""
    env_n: float = 80.0            # arm/gripper vs table/mat
    box_n: float = 80.0            # arm/gripper vs the box
    object_n: float = 40.0         # robot vs the object while it is NOT held
    held_object_n: float = 40.0    # the held object vs the environment
    # the box is light and slides at a few newtons: a push below box_n would go unnoticed, so
    # robot (or held packet) contact with the box while the box has moved this far since the
    # motion began also stops the arm. 0 = off.
    box_push_m: float = 0.005
    touch_n: float = 0.5           # "in contact" for the push rule

    def threshold(self, key: str) -> float:
        if key.startswith("packet-"):
            return self.held_object_n
        env = key.split("-", 1)[1] if "-" in key else ""
        return {"table": self.env_n, "box": self.box_n, "object": self.object_n}.get(env, 0.0)


def stop_floors(f0: dict[str, float], limits: StopLimits) -> dict[str, float]:
    """Per pair: the force that stops THIS motion — max(threshold, force before the motion
    + FORCE_STOP_RISE_N), capped at 2x threshold, so an arm already resting on something can
    still back away, but pressing harder stops it."""
    out = {}
    for key, f in f0.items():
        thr = limits.threshold(key)
        if thr > 0 and np.isfinite(f):
            out[key] = max(thr, min(f + FORCE_STOP_RISE_N, 2.0 * thr))
    return out


def check_stop(now: dict[str, float], floors: dict[str, float], limits: StopLimits, *,
               held: bool, gripping: bool, prev: dict[str, float] | None = None,
               predict: bool = False, box_moved_m: float = 0.0) -> dict | None:
    """The first pair whose force exceeds its floor, as ``{"kind", "pair", "force",
    "threshold", "predicted"}``; None if the motion may continue.

    * ``unstable`` (non-finite or > UNSTABLE_FORCE_N anywhere) always stops;
    * robot vs table / box: kind ``env``;
    * robot vs object: kind ``object`` — only while the object is NOT held and the gripper
      is not closing on it (then that contact IS the grasp);
    * held object vs table / box: kind ``held_object`` — only while held. The grip's own
      pad forces never count.
    ``predict``: with sparse sampling, also stop when the force extrapolated one sample
    ahead (``2*now - prev``) would exceed the floor (bounds the overshoot).
    ``box_moved_m``: how far the box has moved since the motion began; with
    ``limits.box_push_m`` > 0 a robot-box contact (or, while held, a packet-box contact)
    while the box has moved that far stops as a push (``"pushed": True``)."""
    if "unstable" in now or any(not np.isfinite(f) or f > UNSTABLE_FORCE_N for f in now.values()):
        return {"kind": "unstable", "pair": "unstable", "force": float("inf"), "threshold": UNSTABLE_FORCE_N,
                "predicted": False}
    for key, f in now.items():
        thr = limits.threshold(key)
        if thr <= 0:
            continue
        if key.startswith("packet-"):
            if not held or gripping:        # closing on a packet that stands on the mat loads that contact
                continue
            kind = "held_object"
        elif key.endswith("-object"):
            if held or gripping:
                continue
            kind = "object"
        elif key.endswith("-table") or key.endswith("-box"):
            kind = "env"
        else:
            continue
        floor = floors.get(key, thr)
        # extrapolate only robot contacts (the arm drives into them); the held packet's
        # contacts jitter with the grip and would stop early
        ahead = 2.0 * f - float((prev or {}).get(key, f)) if predict and kind != "held_object" else f
        if f > floor or ahead > floor:
            return {"kind": kind, "pair": key, "force": float(f), "threshold": float(floor),
                    "predicted": bool(f <= floor)}
    if limits.box_push_m > 0 and box_moved_m > limits.box_push_m:
        for key, f in now.items():
            if f <= limits.touch_n:
                continue
            if key.startswith(("arm-box", "gripper-box")) or (held and key.startswith("packet-box")):
                return {"kind": "held_object" if key.startswith("packet-") else "env", "pair": key,
                        "force": float(f), "threshold": float(limits.threshold(key)), "predicted": False,
                        "pushed": True, "moved_m": float(box_moved_m)}
    return None
