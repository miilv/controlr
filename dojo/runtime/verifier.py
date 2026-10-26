"""Verification backend for the skills layer: the bundle's twin process, loaded from OUR estimate.

``BundleVerifier`` implements ``dojo.skills.verify.TwinBackend`` (``rollout(candidates, k, held_labels)
-> [Outcome]``) over ``dojo.simlab.twin_service.TwinClient``. It never copies the episode env: the twin
starts from

* the provider's last at-rest checkpoint (estimate of every task object + proprio), through
  ``load_from_estimate`` (inside the service's ``verify``); or
* while an object is in the jaws (``load_from_estimate`` cannot put it there): the twin's own end state
  of the candidate that was executed last (``keep="nominal"`` snapshots, broadcast to every env).

Actions executed since that base state are replayed in front of every candidate (prefix), so the twin
always reaches the episode's current commanded state before the candidate starts. Sample 0 is nominal;
the others get pose noise on the at-rest objects only (``perturb["labels"]``).

``verify="off"`` in the runtime config means no verifier at all (single plan); results carry the flag.
"""

from __future__ import annotations

import time

import numpy as np

from dojo.skills.kinematics import ARMS
from dojo.skills.verify import Outcome

from . import estimate as _est


class BundleVerifier:
    def __init__(self, client, task_scene: str, provider, bridge, perturb_mm: float = 2.0, perturb_deg: float = 1.0,
                 layout_fn=None, log=None, max_prefix: int = 400, load_noise: tuple | None = None, seed: int = 0,
                 load: str = "raw"):
        self.tc = client
        self.scene = task_scene
        self.provider = provider
        self.bridge = bridge
        self.perturb_mm, self.perturb_deg = perturb_mm, perturb_deg
        self.layout_fn = layout_fn
        self.log = log or (lambda *a, **k: None)
        self.max_prefix = max_prefix
        # dev experiment (mode gt): a persistent per-object pose error on every load from the estimate, (pos_mm,
        # rot_deg) standard deviations per axis: does belief noise alone make the twin reject every candidate?
        self.load_noise = tuple(load_noise) if load_noise else None
        self._noise: dict = {}
        self._rng = np.random.default_rng(seed)
        self.load = load                        # raw | consistent (``twin_load``: level + seat before the layout)
        self.disabled: str | None = None        # reason verification is off for the rest of the episode
        self.kept: dict[int, int] = {}          # candidate -> snapshot id (last rollout, nominal samples)
        self.chain: dict | None = None          # {"sid", "n"}: twin end state of the executed candidate
        self._last_n0 = None
        self._last_cands: list = []
        self.stats = {"calls": 0, "rollouts": 0, "steps": 0, "wall_s": 0.0, "prefix_steps": 0,
                      "from_snapshot": 0, "from_estimate": 0, "errors": 0}

    # skills hook (wired by the policy next to the provider's)
    def on_verified(self, info: dict) -> None:
        if info.get("none_ok"):
            # every candidate failed in the twin: its end state is not a base to chain from (next load: the estimate)
            for s_ in self.kept.values():
                self._drop(s_)
            self.kept = {}
            if self.chain is not None and self.chain.get("sid") is not None:
                self._drop(self.chain["sid"])
            self.chain = None
            return
        idx = 0 if info.get("idx") is None else int(info["idx"])
        sid = self.kept.get(idx)
        for ci, s in self.kept.items():
            if ci != idx:
                self._drop(s)
        self.kept = {}
        if self.chain is not None and self.chain.get("sid") not in (None, sid):
            self._drop(self.chain["sid"])
        n_len = chosen_len(info, idx)
        self.chain = {"sid": sid, "n": self.bridge.n_executed + n_len} if sid is not None else None

    def _drop(self, sid) -> None:
        try:
            self.tc.drop_snapshot(int(sid))
        except Exception:  # noqa: BLE001
            pass

    def _base(self, held: tuple):
        """(world dict, executed-action index the base state corresponds to, kind)."""
        if held and self.chain is not None and self.chain.get("sid") is not None and self.chain["n"] <= self.bridge.n_executed:
            return {"snapshot_id": int(self.chain["sid"])}, self.chain["n"], "snapshot"
        cp = self.provider.checkpoint
        objs = self._noisy(cp["objects"]) if self.load_noise else cp["objects"]
        if self.load == "consistent":
            objs = self._consistent(objs)
        lay = (self.layout_fn or _est.estimate_to_layout)(self.scene, objs)
        return {"layout": lay, "robot": cp["robot"]}, cp["n"], "estimate"

    def _consistent(self, objects: list[dict]) -> list[dict]:
        from .twin_load import consistent
        bel = getattr(self.provider, "belief", None)
        objs = getattr(bel, "objects", None) or {}
        assets = {lab: o.asset for lab, o in objs.items() if not o.clutter}
        fixed = {lab for lab, o in objs.items() if o.obj_type == "Geometry"}
        try:
            fixed |= _est._fixed_labels(self.scene)
        except Exception:  # noqa: BLE001 (no generator config for this scene: every non-Geometry object is movable)
            pass
        out, st = consistent(objects, assets, fixed)
        self.stats["load_leveled"] = self.stats.get("load_leveled", 0) + len(st["leveled"])
        self.stats["load_seated"] = self.stats.get("load_seated", 0) + len(st["seated"])
        if st["seated"]:
            self.stats["load_seat_max_mm"] = max(self.stats.get("load_seat_max_mm", 0.0),
                                                 max(abs(d) for _, d in st["seated"]))
        return out

    def _noisy(self, objects: list[dict]) -> list[dict]:
        """``objects`` with each label's own fixed offset (drawn once per label and episode): position N(0, pos_mm)
        per axis (z included: a stacked piece may load into its support), rotation N(0, rot_deg) per axis."""
        pos_mm, rot_deg = self.load_noise
        out = []
        for o in objects:
            lab = o.get("label")
            if lab not in self._noise:
                self._noise[lab] = (self._rng.normal(0.0, pos_mm / 1000.0, 3), self._rng.normal(0.0, np.radians(rot_deg), 3))
            dp, rv = self._noise[lab]
            o2 = dict(o)
            o2["pos"] = (np.asarray(o["pos"], float) + dp).tolist()
            o2["quat"] = _quat_mul(_quat_rv(rv), np.asarray(o["quat"], float)).tolist()
            out.append(o2)
        return out

    def rollout(self, candidates: list[list[dict]], k: int = 2, held_labels: tuple = (), src: int = 0) -> list[Outcome]:
        """On a twin error: log it, disable verification for the rest of the episode and return no outcomes
        (skills then runs candidate 0, i.e. the single plan)."""
        if self.disabled:
            return []
        try:
            return self._rollout(candidates, k, held_labels)
        except Exception as e:  # noqa: BLE001 (the episode must go on)
            self.stats["errors"] += 1
            self.disabled = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            self.log(f"[verify] twin error, verification off for this episode: {self.disabled}")
            return []

    def _rollout(self, candidates: list[list[dict]], k: int = 2, held_labels: tuple = ()) -> list[Outcome]:
        t0 = time.perf_counter()
        held = tuple(held_labels) or self.provider.belief.held_labels()
        world, n_base, kind = self._base(held)
        prefix = self.bridge.actions_since(n_base)
        if len(prefix) > self.max_prefix:
            raise RuntimeError(f"verify prefix too long ({len(prefix)} steps)")
        self.stats["prefix_steps"] += len(prefix)
        self.stats["from_" + kind] += 1
        at_rest = [o.label for o in self.provider.belief.movable() if o.label not in held]
        perturb = {"pos_mm": self.perturb_mm, "yaw_deg": self.perturb_deg, "labels": at_rest}
        cands = [{"actions": [_jsonable(a) for a in prefix + list(c)], "meta": {"i": i, "prefix": len(prefix)}}
                 for i, c in enumerate(candidates)]
        outs = self.tc.verify(world, cands, perturb=perturb, k=k, keep="nominal")
        self.kept = {int(o["candidate"]): int(o["snapshot_id"]) for o in outs if o.get("snapshot_id") is not None}
        res = []
        for o in outs:
            rw = o.get("reward") or {}
            failed = bool((not rw.get("success_flag", True) and not rw.get("ended", False)) or
                          (rw.get("ended") and not rw.get("success") and rw.get("steps_used", 0) < rw.get("step_lim", 1)))
            objs = {lab: (np.asarray(v[:3], float), np.asarray(v[3:7], float)) for lab, v in o["final_objects"].items()}
            q = {a: np.asarray(o["final_robot"][f"{a}_q"], float) for a in ARMS}
            res.append(Outcome(int(o["candidate"]), int(o["sample"]), objs, q, bool(o.get("success")), failed,
                               float(o.get("score", 0.0)), bool(o.get("ended")),
                               extra={"held": o.get("held"), "dropped": o.get("dropped"), "fell": o.get("fell"),
                                      "moved_mm": o.get("moved_mm")}))
        self.stats["calls"] += 1
        self.stats["rollouts"] += len(outs)
        self.stats["steps"] += max((len(c["actions"]) for c in cands), default=0)
        self.stats["wall_s"] += time.perf_counter() - t0
        return res


def _quat_rv(rv) -> np.ndarray:
    """Quaternion (w, x, y, z) of the rotation vector ``rv`` (rad)."""
    a = float(np.linalg.norm(rv))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.r_[np.cos(a / 2), np.sin(a / 2) * np.asarray(rv, float) / a]


def _quat_mul(a, b) -> np.ndarray:
    """Hamilton product a * b, (w, x, y, z), normalised."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return q / np.linalg.norm(q)


def chosen_len(info: dict, idx: int) -> int:
    """Env steps of the executed candidate. skills passes ``actions`` = the chosen candidate's action list
    (one dict per step); a list of per-candidate lists is accepted too."""
    a = info.get("actions") or []
    if a and isinstance(a[0], dict):
        return len(a)
    return len(a[idx]) if idx < len(a) else 0


def _jsonable(a: dict) -> dict:
    return {k: np.asarray(v, np.float32) for k, v in a.items()}


__all__ = ["BundleVerifier"]
