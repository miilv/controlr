"""Perception routes for the bridge, behind the runtime's perception-client interface.

The runtime (``RuntimePolicy``) calls ``estimate / look_plan / look_poses / refine / check / search_regions``
(``dojo.runtime.perception_service.PerceptionClient``). Behind a robot server there are two sources:

* ``local`` (default): our perception service (``dojo.runtime.perception_service``, own venv, EGL) on the contract
  observations the backend built: the three frames + measured joints; cameras from the public config + FK. The
  same code path as the proxy runs, every op available.
* ``server``: ``ServerPerception``: estimates from the J1 server's ``perceive`` verb (our B1 adapter
  ``dojo.perception.j1_adapter`` running inside robot-harness's server, inputs and outputs logged there), converted
  to a layout record by ``record_from_perceived``; ``look_plan`` from the call's ``needs_look``; every other op
  (refine, check, look_poses, search_regions, and the ones only our service has: scan_looks, agentic_estimate,
  needs_agent, held_look, watch, ...) goes to a local client when one is given. Without one, refine / look_poses /
  search_regions are no-ops (the runtime keeps its prior), ``check`` says ``_unavailable`` (resync logs the check as
  unverified, never as "no divergence"), and any other op raises; the driver refuses ``--perception server`` with no
  local client when the config enables a feature that needs one (``LOCAL_ONLY``).

Either way the twin is loaded from what perception returned (``estimate_to_layout`` of the provider's checkpoint)
plus the measured proprio, inside our verifier (``dojo.runtime.verifier.BundleVerifier``).
"""

from __future__ import annotations

ARM_OF_CAM = {"cam_left_wrist": "left", "cam_right_wrist": "right"}
# RuntimeConfig fields (value when off) whose feature calls a perception op only our local service has
LOCAL_ONLY = {"scan": "off", "agentic": "off", "held_look": "off", "watch": False, "wrist_check": "off",
              "held_pose": "off", "slot_read": "off", "bg_perception": "off"}
UNAVAILABLE = "_unavailable"        # check()'s key when nothing can check (no local client)


def local_only_features(cfg) -> list[str]:
    """The config's features that need a local perception client (``LOCAL_ONLY`` fields not at their off value)."""
    return sorted(k for k, off in LOCAL_ONLY.items() if hasattr(cfg, k) and getattr(cfg, k) != off)
# categories whose pose is their position alone (a sphere looks and grasps the same at every turn): only these may
# take the identity orientation when the server's object has no quat
ROTATION_FREE_CATEGORIES = frozenset({"ball"})


def _measured_quat(q) -> list[float] | None:
    """A measured orientation (wxyz, 4 finite numbers of norm 0.9..1.1, normalised), else None."""
    import math

    try:
        v = [float(x) for x in q]
    except (TypeError, ValueError):
        return None
    n = math.sqrt(sum(x * x for x in v)) if len(v) == 4 and all(math.isfinite(x) for x in v) else 0.0
    return [x / n for x in v] if 0.9 <= n <= 1.1 else None


def record_from_perceived(perceived: list[dict], needs_look: list[dict] | None = None) -> dict:
    """J1 ``perceive`` objects -> perception's layout record (``Rigid/Geometry/Articulation -> category -> rows``).
    An object needs ``asset`` (``Type/category/NNNNN``, our catalogue key) to become a row; others are listed under
    ``_unusable``. ``needs_look`` entries with reason ``not_found`` become ``_not_found`` (no candidates known).
    Orientation is never invented (I1-5): an object without a measured ``quat`` becomes a row only when its category
    is rotation-free (``ROTATION_FREE_CATEGORIES``: identity, ``_perception.orientation`` "rotation_free"); any other
    is refused as a planning row until a later call measures it: listed in ``_orientation_unknown`` and
    ``_unusable`` (with its position), and in ``_not_found`` with its asset as the candidate, so the runtime treats
    it as missing (no plan on it, the twin check off, the search may re-measure it) instead of planning on a made-up
    turn. Every row carries ``_perception.orientation_known``. Nor is a digit: a number card's digit is the reply's
    ``value`` (the J1 schema's form of ``attributes.digit``; the asset index follows it), and a card without one is
    unread (``dojo.perception.digits``), never its asset's digit."""
    from dojo.perception import digits

    rec: dict = {}
    unusable, ori_unknown, nf = [], [], []
    for o in perceived or []:
        key = str(o.get("asset") or "")
        parts = key.split("/")
        if len(parts) != 3 or not parts[2].isdigit():
            unusable.append({"label": o.get("label"), "why": f"asset {key!r}"})
            continue
        typ, cat, idx = parts
        quat, ori = _measured_quat(o.get("quat")), "measured"
        if quat is None and cat in ROTATION_FREE_CATEGORIES:
            quat, ori = [1.0, 0.0, 0.0, 0.0], "rotation_free"
        if quat is None:
            ori_unknown.append(o["label"])
            unusable.append({"label": o["label"], "why": "orientation unknown (no measured quat)", "asset": key,
                             "pos": [float(v) for v in o["pos"]]})
            nf.append({"label": o["label"], "candidates": [key], "why": "orientation unknown"})
            continue
        p = {"confidence": float(o.get("confidence", 0.5)), "source": o.get("source"),
             "via": "server_perceive", "orientation_known": True, "orientation": ori}
        if cat in digits.DIGIT_OF:
            d = digits.as_digit(o.get("value"))
            if d is None:
                p.update(digit=None, digit_source="unread", digit_unread="no_value")
            else:
                p.update(digit=d, digit_source="j1")
                i_d = digits.idx_for_digit(cat, d)
                idx = idx if i_d is None else i_d
        rec.setdefault(typ, {}).setdefault(cat, []).append({
            "category": cat, "category_idx": int(idx), "label": o["label"],
            "default_pos": [float(v) for v in o["pos"]],
            "default_ori": quat, "scale": [1.0, 1.0, 1.0], "_perception": p})
    nf += [{"label": n["label"], "candidates": []} for n in needs_look or []
           if n.get("reason") == "not_found" and n.get("label") and n["label"] not in ori_unknown]
    if nf:
        rec["_not_found"] = nf
    if ori_unknown:
        rec["_orientation_unknown"] = ori_unknown
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
        if self.local is None:
            return {UNAVAILABLE: "server perception: no local checker"}
        return self.local.check(frame, objects, fold)

    def search_regions(self, frame, record, vocabulary) -> list[dict]:
        return self.local.search_regions(frame, record, vocabulary) if self.local is not None else []

    def __getattr__(self, name):
        """Ops only the local service has (scan_looks, agentic_estimate, needs_agent, held_look, watch, ...)."""
        if name.startswith("_") or name in ("backend", "local", "want", "last", "calls"):
            raise AttributeError(name)
        if self.local is None:
            raise AttributeError(f"server perception has no `{name}` and no local client to forward it to "
                                 "(run with --perception local, or set DOJO_PERCEPTION_PORT/AUTHKEY)")
        return getattr(self.local, name)


__all__ = ["record_from_perceived", "ServerPerception", "LOCAL_ONLY", "UNAVAILABLE", "local_only_features"]
