"""Background perception in the runtime (``RuntimeConfig.bg_perception``, default off).

Every observation the episode produces (the runtime's own, plus one every ``bg_obs_every`` buffered steps of a long
segment: ``Bridge.obs_every``) goes to the perception service's background perceiver (op ``bg_step``,
``dojo.perception.bgperc``) together with the belief. Its moves (an untouched object's pose confirmed by >= 2 views, or
by one very good fit, off the belief) are written into the belief; held objects and objects at a closed jaw are never
touched (an observation in the middle of a segment comes before the provider attached what the jaws closed on).
``needs_look`` (required objects still in doubt) is logged; acting on it (a targeted look) is the caller's.
"""
from __future__ import annotations

import numpy as np

JAW_M = 0.08          # an object this near a closed jaw's TCP counts as held for the background
TCP_OFFSET = 0.1501   # flange -> grasp point along flange +x (= dojo.perception.tracker.TCP_OFFSET)


def tcp_from_flange(f) -> np.ndarray:
    """The grasp point from a flange pose (x, y, z, qw, qx, qy, qz): numpy only (the runtime's python)."""
    f = np.asarray(f, float).reshape(-1)
    w, x, y, z = f[3:7] / max(float(np.linalg.norm(f[3:7])), 1e-12)
    ax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])   # R[:, 0]
    return f[:3] + ax * TCP_OFFSET


class BgHook:
    def __init__(self, pc, prov, scene: str, fold: int, cfg: dict | None = None, events: list | None = None,
                 max_events: int = 60, log=None):
        self.pc, self.prov, self.scene, self.fold = pc, prov, scene, int(fold)
        self.cfg = dict(cfg or {})
        self.events = events if events is not None else []
        self.max_events = max_events
        self.log = log or (lambda *a, **k: None)
        self.calls, self.moves, self.errors, self.looks = 0, 0, 0, 0
        self.stats: dict = {}
        self._first = True

    def _held(self, obs: dict, belief) -> set:
        held = set(belief.held_labels())
        st = obs["state"]
        for arm in ("left", "right"):
            g = st.get(f"{arm}_ee_joint_state")
            f = st.get(f"{arm}_ee_pose")
            if g is None or f is None or float(np.asarray(g, float).reshape(-1)[0]) >= 0.5:
                continue
            tcp = tcp_from_flange(f)
            held |= {lab for lab, o in belief.objects.items() if float(np.linalg.norm(o.pos - tcp)) < JAW_M}
        return held

    def __call__(self, obs: dict) -> None:
        b = getattr(self.prov, "belief", None)
        if b is None or not getattr(self.prov, "mode", "perception") == "perception":
            return
        held = self._held(obs, b)
        objs = [{"label": lab, "key": o.asset.key, "pos": [float(v) for v in o.pos], "quat": [float(v) for v in o.quat],
                 "conf": float(getattr(o, "confidence", 0.5))} for lab, o in b.objects.items() if not o.clutter]
        params = (getattr(self.prov, "notes", None) or {}).get("params") or {}
        required = sorted({v for v in params.values() if isinstance(v, str) and v in b.objects})
        step = int(obs.get("steps_used", 0))
        try:
            r = self.pc.bg_step(obs, step, objs, sorted(held), required, self.scene, self.fold, self.cfg,
                                reset=self._first)
        except Exception as e:  # noqa: BLE001 (the background never stops an episode)
            self.errors += 1
            if self.errors <= 3:
                self.events.append({"kind": "bg_error", "step": step, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            return
        self._first = False
        self.calls += 1
        self.stats = r.get("stats") or {}
        for mv in r.get("moves") or []:
            lab = mv.get("label")
            o = b.objects.get(lab)
            if o is None or lab in held or lab in b.held_labels():
                continue
            to = np.asarray(mv["to"], float)
            ev = {"kind": "bg_move", "label": lab, "step": step, "from7": [round(float(v), 4) for v in np.r_[o.pos, o.quat]],
                  "to7": [round(float(v), 4) for v in to], "n_views": mv.get("n_views"), "reread": bool(mv.get("reread"))}
            o.pos, o.quat = to[:3].copy(), to[3:7].copy()
            self.moves += 1
            if len(self.events) < 10 ** 6 and self.moves <= self.max_events:
                self.events.append(ev)
            self.log(f"[bg] {lab} moved {1000 * float(np.hypot(*(to[:2] - np.asarray(ev['from7'][:2])))):.0f} mm "
                     f"({mv.get('n_views')} views)")
        nl = r.get("needs_look") or []
        if nl:
            self.looks += 1
            if self.looks <= 20:
                self.events.append({"kind": "bg_needs_look", "step": step, "labels": nl})

    def report(self) -> dict:
        return {"calls": self.calls, "moves": self.moves, "errors": self.errors, "needs_look": self.looks,
                "stats": self.stats}


__all__ = ["BgHook"]
