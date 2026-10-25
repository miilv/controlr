"""Perception routes for the bridge, behind the runtime's perception-client interface.

The runtime (``RuntimePolicy``) calls ``estimate / look_plan / look_poses / refine / check / search_regions``
(``dojo.runtime.perception_service.PerceptionClient``). Behind a robot server there are two sources:

* ``local`` (default): our perception service (``dojo.runtime.perception_service``, own venv, EGL) on the contract
  observations the backend built: the three frames + measured joints; cameras from the public config + FK. The
  same code path as the proxy runs, every op available.
* ``server``: ``ServerPerception``: estimates from the J1 server's ``perceive`` verb (our B1 adapter
  ``dojo.perception.j1_adapter`` running inside robot-harness's server, inputs and outputs logged there), converted
  to a layout record by ``record_from_perceived``; ``look_plan`` from the call's ``needs_look``; the local-only ops
  (refine, check, look_poses, search_regions) go to a local client when one is given, else they are no-ops (the
  runtime then keeps its prior: no refine correction, no divergence, no extra look poses, no search).

Either way the twin is loaded from what perception returned (``estimate_to_layout`` of the provider's checkpoint)
plus the measured proprio, inside our verifier (``dojo.runtime.verifier.BundleVerifier``).
"""

from __future__ import annotations

ARM_OF_CAM = {"cam_left_wrist": "left", "cam_right_wrist": "right"}


def record_from_perceived(perceived: list[dict], needs_look: list[dict] | None = None) -> dict:
    """J1 ``perceive`` objects -> perception's layout record (``Rigid/Geometry/Articulation -> category -> rows``).
    An object needs ``asset`` (``Type/category/NNNNN``, our catalogue key) to become a row; others are listed under
    ``_unusable``. ``needs_look`` entries with reason ``not_found`` become ``_not_found`` (no candidates known)."""
    rec: dict = {}
    unusable = []
    for o in perceived or []:
        key = str(o.get("asset") or "")
        parts = key.split("/")
        if len(parts) != 3 or not parts[2].isdigit():
            unusable.append({"label": o.get("label"), "why": f"asset {key!r}"})
            continue
        typ, cat, idx = parts
        rec.setdefault(typ, {}).setdefault(cat, []).append({
            "category": cat, "category_idx": int(idx), "label": o["label"],
            "default_pos": [float(v) for v in o["pos"]],
            "default_ori": [float(v) for v in (o.get("quat") or [1.0, 0.0, 0.0, 0.0])], "scale": [1.0, 1.0, 1.0],
            "_perception": {"confidence": float(o.get("confidence", 0.5)), "source": o.get("source"),
                            "via": "server_perceive"}})
    nf = [{"label": n["label"], "candidates": []} for n in needs_look or []
          if n.get("reason") == "not_found" and n.get("label")]
    if nf:
        rec["_not_found"] = nf
    if unusable:
        rec["_unusable"] = unusable
    return rec


class ServerPerception:
    """``estimate`` through the server's ``perceive`` verb (the server perceives ITS current step: the runtime calls
    estimate right after the observation it passes, so frames[-1] is that step); ``prior_n`` chains the calls of an
    episode (the adapter refines prior objects a moved wrist camera sees)."""

    def __init__(self, backend, local=None, want=None):
        self.backend, self.local, self.want = backend, local, want
        self.last: dict | None = None
        self.calls: list[dict] = []

    def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None) -> dict:
        prior_n = self.last["n"] if (self.last and len(frames) > 1) else None
        out = self.backend.perceive(want=self.want, prior_n=prior_n)
        self.last = out
        self.calls.append({"op": "perceive", "n": out.get("n"), "prior_n": prior_n,
                           "objects": len(out.get("perceived") or [])})
        return record_from_perceived(out.get("perceived"), out.get("needs_look"))

    def look_plan(self, frame, record, max_looks: int = 2) -> list[dict]:
        plan = []
        for n in (self.last or {}).get("needs_look") or []:
            arm = ARM_OF_CAM.get(n.get("cam"))
            if arm is None or n.get("flange") is None or n.get("target") is None:
                continue
            plan.append({"arm": arm, "purpose": n.get("reason"), "label": n.get("label"), "target": n["target"],
                         "flange_pose": n["flange"]})
        return plan[:max_looks]

    def look_poses(self, target, arm, variants=None) -> list[dict]:
        return self.local.look_poses(target, arm, variants) if self.local is not None else []

    def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None) -> dict:
        if self.local is None:
            return {}
        return self.local.refine(frame, objects, fold, max_shift_m, arm=arm, identity_vocabulary=identity_vocabulary,
                                 held=held)

    def check(self, frame, objects, fold) -> dict:
        return self.local.check(frame, objects, fold) if self.local is not None else {}

    def search_regions(self, frame, record, vocabulary) -> list[dict]:
        return self.local.search_regions(frame, record, vocabulary) if self.local is not None else []


__all__ = ["record_from_perceived", "ServerPerception"]
