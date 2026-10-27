"""The WorldState the expert programs see, and how it is kept between observations.

``RuntimeProvider`` is the skills ``provider`` (``refresh(holding) -> WorldState``). Arms always come
from the official proprio. Objects come from one of two sources:

* mode ``gt`` (dev-only upper bound): the GT rows the driver captured with the latest observation;
* mode ``perception``: a BELIEF. It starts from perception's layout record (inverse layout + looks) and
  is carried forward between checkpoints by predictions, because perception only estimates objects
  resting on the table:
    - the twin's nominal outcome of the verified candidate that actually ran (``on_verified``),
    - otherwise ``KinematicBelief``: an object follows the jaws that closed on it and, when they open,
      drops onto the highest support under its footprint (the table, a movable object's top, or the mesh surface of
      a static fixture under it: a board, a bolt's top), and stays where it was let go when a fixture holds it (a
      nut on a bolt's thread, a mug on a hook: ``fixture_rest``);
  At a resync checkpoint (right after an object was released) the predicted state is rendered against
  the new frames (perception service ``check``: per object, the fraction of the pixels where it is predicted
  front-most that are observed foreground); an object whose prediction the frames do not show is a divergence: the table-level objects are re-estimated, and
  ``Divergence`` asks the policy to re-plan from the corrected state.

The provider also keeps the last at-rest checkpoint (estimate + robot + executed-action index) so the
verifier can load the twin from an estimate while an object is in the jaws (prefix replay), and it
measures the twin's prediction error against GT in mode gt (a dev metric, never fed back).
"""

from __future__ import annotations

import copy

import numpy as np

from dojo.skills.geometry import T_to_pose7, inv_T
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin
from dojo.skills.world import ObjectState, WorldState, arms_from_proprio, table_contains_xy

from .estimate import est_objects, objects_from_gt_rows, objects_from_record, robot_from_state

MAX_JOINT_STEP = 0.2
FIXTURE_CELL = 0.001        # m: cell of a fixture's top-surface height map (``fixture_height_map``)
FIXTURE_ERODE = 1           # cells: the fixture's surface is shrunk by this before the lookup, so a ring's 0.4 mm
                            # clearance on its peg reads as free and a nut's thread on its bolt (1.6 mm) as held
FIXTURE_HOLD = 0.001        # m: an object point this far below a fixture's surface is held by the fixture
STILL_RAD = 0.002           # rad: the arm counts as still at a release when its command moved less than this
_FIXTURE_HM: dict = {}


def fixture_height_map(o, cell: float = FIXTURE_CELL) -> dict | None:
    """Top surface of a static fixture's mesh in its own frame: the max z of dense surface samples per ``cell``
    (-inf: no surface), eroded by ``FIXTURE_ERODE`` cells. None without a mesh. Cached per asset."""
    if o.asset.mesh is None:
        return None
    key = (o.asset.key, cell)
    if key in _FIXTURE_HM:
        return _FIXTURE_HM[key]
    V, F = np.asarray(o.asset.mesh[0], float), np.asarray(o.asset.mesh[1], int)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    L = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)), np.linalg.norm(c - b, axis=1))
    nseg = np.clip(np.ceil(L / (0.7 * cell)), 1, 400).astype(int)
    x0, y0 = float(V[:, 0].min()) - cell, float(V[:, 1].min()) - cell
    nx = int(np.ceil((float(V[:, 0].max()) + cell - x0) / cell)) + 1
    ny = int(np.ceil((float(V[:, 1].max()) + cell - y0) / cell)) + 1
    Z = np.full((ny, nx), -np.inf)

    def put(P):
        np.maximum.at(Z, (((P[:, 1] - y0) / cell).astype(int), ((P[:, 0] - x0) / cell).astype(int)), P[:, 2])
    put(V)
    for n in np.unique(nseg):
        u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        keep = (u + v) <= 1.0
        u, v = u[keep], v[keep]
        idx = np.flatnonzero(nseg == n)
        for i in range(0, len(idx), max(1, 2_000_000 // len(u))):        # bounded memory per batch
            j = idx[i:i + max(1, 2_000_000 // len(u))]
            P = a[j][:, None, :] + u[None, :, None] * (b[j] - a[j])[:, None, :] + v[None, :, None] * (c[j] - a[j])[:, None, :]
            put(P.reshape(-1, 3))
    for _ in range(FIXTURE_ERODE):
        Zp = np.pad(Z, 1, constant_values=-np.inf)
        Z = np.min(np.stack([Zp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), 0)
    hm = {"x0": x0, "y0": y0, "cell": cell, "Z": Z}
    _FIXTURE_HM[key] = hm
    return hm


def drop_rest(o, movables, fixtures=()) -> tuple[np.ndarray | None, dict | None]:
    """Where ``o``, let go at its current pose, comes to rest: onto the highest support under its footprint, the table
    or a ``movables`` object's top (bbox, up to 10 mm into it); with ``fixtures`` (static Geometry objects, passed for a
    release: the jaws opened on it while the arm was still) also a fixture's mesh surface under it (``fixture_rest``: a
    piece on a board, a nut on a bolt's top), and a fixture that holds it (a nut threaded on a bolt, a mug on a hook)
    keeps it where it was let go. Returns (the new root position, None = stays; a fixture event or None). Without
    ``fixtures`` (as before) an object let go on one fell through it to the table (a nut 53 mm under its bolt's top, a
    piece 17 mm under the board)."""
    lo, hi = o.aabb()
    support = TABLE_Z if table_contains_xy(o.center_w) else 0.05
    for other in movables:
        if other.label == o.label:
            continue
        olo, ohi = other.aabb()
        if (olo[0] < hi[0] - 1e-3 and ohi[0] > lo[0] + 1e-3 and olo[1] < hi[1] - 1e-3 and ohi[1] > lo[1] + 1e-3
                and ohi[2] <= lo[2] + 0.01):
            support = max(support, float(ohi[2]))
    ev = None
    if fixtures:
        how, d = fixture_rest(o, fixtures)
        if how == "held":
            return None, {"how": "held by a fixture"}
        z = float(lo[2]) - d if how == "support" else -np.inf     # the bbox bottom after falling d onto the fixture
        if z > support:
            support = z
            ev = {"how": "on a fixture", "z": round(float(z), 4)}
    return o.pos + np.array([0.0, 0.0, support - lo[2]]), ev


def fixture_rest(o, fixtures) -> tuple[str, float | None]:
    """How static ``fixtures`` (Geometry objects) take an object let go at its pose ``o``, from their meshes' top
    surfaces under the object's mesh points: ("held", None) when some point is more than ``FIXTURE_HOLD`` below a
    fixture's surface (a nut on a bolt's thread, a handle around a hook's arm: it stays where it was let go);
    ("support", d) when a fixture's surface lies under it, d = how far the object falls until its mesh meets it (a piece
    on a board, a nut on a bolt's top; < 0: raised by up to ``FIXTURE_HOLD``); ("none", None) otherwise (no fixture under it, or no meshes: the table / movable supports
    decide, as before). Fixtures must stand upright; others are skipped."""
    if o.asset.mesh is None:
        return "none", None
    P = np.asarray(o.asset.mesh[0], float) @ o.R.T + o.pos
    lo, hi = P.min(0), P.max(0)
    best = None                                 # the smallest fall onto any fixture
    for g in fixtures:
        if g.label == o.label or g.asset.mesh is None or float(g.R[2, 2]) < 0.996:
            continue
        glo, ghi = g.aabb()
        if glo[0] > hi[0] or ghi[0] < lo[0] or glo[1] > hi[1] or ghi[1] < lo[1] or ghi[2] < TABLE_Z - 0.05:
            continue
        hm = fixture_height_map(g)
        if hm is None:
            continue
        loc = (P - g.pos) @ g.R
        ix = np.floor((loc[:, 0] - hm["x0"]) / hm["cell"]).astype(int)
        iy = np.floor((loc[:, 1] - hm["y0"]) / hm["cell"]).astype(int)
        inb = (ix >= 0) & (iy >= 0) & (ix < hm["Z"].shape[1]) & (iy < hm["Z"].shape[0])
        if not inb.any():
            continue
        zs = hm["Z"][iy[inb], ix[inb]] + float(g.pos[2])
        fin = np.isfinite(zs)
        if not fin.any():
            continue
        gap = P[inb][fin][:, 2] - zs[fin]
        if float(gap.min()) < -FIXTURE_HOLD:
            return "held", None
        d = float(gap.min())
        best = d if best is None else min(best, d)
    return ("support", best) if best is not None else ("none", None)


RELEASE_OPEN = 0.15             # a gripper command this far above its tightest since the grasp releases the object


class Divergence(Exception):
    """The frames disagree with the predicted state beyond the threshold: re-plan."""

    def __init__(self, labels: list[str], detail: dict):
        super().__init__(f"divergence on {labels}")
        self.labels = labels
        self.detail = detail


class LostObject(Exception):
    """A pick-hover refine could not find the object anywhere near its belief pose: search for it before picking."""

    def __init__(self, label: str, pos, quat, key: str):
        super().__init__(f"{label} not found near its belief pose")
        self.label, self.pos, self.quat, self.key = label, np.asarray(pos, float), np.asarray(quat, float), key


class KinematicBelief:
    def __init__(self, objects: dict[str, ObjectState], kin: X5Kin, state: dict, attach_margin: float = 0.015):
        self.kin = kin
        self.objects = {k: copy.deepcopy(v) for k, v in objects.items()}
        self.q = {a: np.asarray(state[f"{a}_arm_joint_state"], float).reshape(-1)[:6].copy() for a in ARMS}
        self.g = {a: float(np.asarray(state[f"{a}_ee_joint_state"]).reshape(-1)[0]) for a in ARMS}
        self.attached: dict[str, tuple[str, np.ndarray]] = {}
        self.margin = attach_margin
        self.released: list[str] = []          # labels released since the last ``pop_released``
        self.released_by: dict[str, str] = {}   # label -> arm that released it (last release)
        self.grasp_z: dict[str, tuple] = {}     # arm -> (TCP z at closing, grasp counter, object pose7 at closing)
        self.g_held: dict[str, float] = {}      # arm -> the lowest gripper command since its object was attached
        self._n_grasp = 0
        self.drop_events: list[dict] = []       # releases a fixture took (``fixture_rest``), for the provider's events
        self.cmd: dict[str, np.ndarray] = {}    # arm -> the last commanded joint target (a release counts when still)

    def movable(self):
        return [o for o in self.objects.values() if o.obj_type != "Geometry" and not o.clutter]

    def _object_at(self, p, exclude=()) -> str | None:
        best, bd = None, 1e9
        for o in self.movable():
            if o.label in exclude:
                continue
            loc = o.R.T @ (p - o.center_w)
            if np.all(np.abs(loc) <= o.asset.extents / 2 + self.margin):
                d = float(np.linalg.norm(loc))
                if d < bd:
                    best, bd = o.label, d
        return best

    def _set_T(self, label: str, T: np.ndarray) -> None:
        o = self.objects[label]
        p7 = T_to_pose7(T)
        o.pos, o.quat = np.asarray(p7[:3], float), np.asarray(p7[3:], float)
        o.source = "predicted"

    def fixtures(self):
        return [o for o in self.objects.values() if o.obj_type == "Geometry" and not o.clutter]

    def _drop(self, label: str, release: bool = False) -> None:
        o = self.objects[label]
        pos, ev = drop_rest(o, self.movable(), self.fixtures() if release else ())
        if pos is not None:
            o.pos = pos
        o.source = "predicted"
        if ev is not None:
            self.drop_events.append({"label": label, **ev})

    def step(self, action: dict) -> None:
        for a in ARMS:
            tgt = np.asarray(action[f"{a}_arm_joint_state"], float).reshape(-1)[:6]
            prev = self.cmd.get(a)
            still = prev is not None and float(np.abs(tgt - prev).max()) < STILL_RAD   # the command did not move
            self.cmd[a] = tgt.copy()
            self.q[a] = self.q[a] + np.clip(tgt - self.q[a], -MAX_JOINT_STEP, MAX_JOINT_STEP)
            g_new = float(np.clip(np.asarray(action[f"{a}_ee_joint_state"]).reshape(-1)[0], 0, 1))
            T_tcp = self.kin.tcp(a, self.q[a])
            # a closing command below half open, from any opening: narrow picks open to < 0.5 before they close (a bread
            # slice 0.37, a mallet handle 0.42-0.46, a 3-5 mm mug wall 0.42-0.44); requiring a crossing of 0.5 from above
            # never attached them, the belief stayed on the table and every such pick read "rose 0 mm" (grasp_slip)
            if g_new < 0.5 and g_new < self.g[a] - 1e-6 and a not in self.attached:
                # a free object at the pads first; else one the other hand holds: a handover, both hands hold it
                # until one opens (before, the taker's close was ignored and the giver's open dropped the object to
                # the table while the taker held it: a key, a coin)
                held = [v[0] for v in self.attached.values()]
                lab = self._object_at(T_tcp[:3, 3], exclude=held) or self._object_at(T_tcp[:3, 3])
                if lab is not None:
                    self.attached[a] = (lab, inv_T(T_tcp) @ self.objects[lab].T)
                    self.g_held[a] = g_new
                    self._n_grasp += 1
                    o = self.objects[lab]
                    self.grasp_z[a] = (float(T_tcp[2, 3]), self._n_grasp, np.r_[o.pos, o.quat].astype(float))
            if a in self.attached:
                self.g_held[a] = min(self.g_held.get(a, g_new), g_new)
            # released by an opening command: above 0.6 as before, or RELEASE_OPEN above the tightest command since the
            # grasp (a narrow release, open_g < 0.6, must not leave the object riding with the hand)
            if a in self.attached and g_new > min(0.6, self.g_held[a] + RELEASE_OPEN):
                lab, _ = self.attached.pop(a)
                self.g_held.pop(a, None)
                if lab not in [v[0] for v in self.attached.values()]:     # the other hand still holds it: no drop
                    self._drop(lab, release=still)
                    self.released.append(lab)
                    self.released_by[lab] = a
            self.g[a] = g_new
            if a in self.attached:
                lab, T_rel = self.attached[a]
                self._set_T(lab, T_tcp @ T_rel)

    def sync_robot(self, state: dict) -> None:
        for a in ARMS:
            self.q[a] = np.asarray(state[f"{a}_arm_joint_state"], float).reshape(-1)[:6].copy()
            if a in self.attached:
                lab, T_rel = self.attached[a]
                self._set_T(lab, self.kin.tcp(a, self.q[a]) @ T_rel)

    def adopt(self, objects: dict) -> list[str]:
        """Twin prediction ``{label: (pos, quat)}`` for every object it carries; attached objects keep
        following their arm with the twin's in-hand pose."""
        done = []
        for lab, (p, q) in objects.items():
            if lab in self.objects:
                o = self.objects[lab]
                o.pos, o.quat = np.asarray(p, float).copy(), np.asarray(q, float).copy()
                o.source = "predicted"
                done.append(lab)
        for a, (lab, _) in list(self.attached.items()):
            self.attached[a] = (lab, inv_T(self.kin.tcp(a, self.q[a])) @ self.objects[lab].T)
        return done

    def held_labels(self) -> tuple:
        return tuple(v[0] for v in self.attached.values())

    def pop_released(self) -> list[str]:
        r, self.released = self.released, []
        return r


class RuntimeProvider:
    def __init__(self, bridge, task: str, instruction: str, mode: str, assets=None, kin: X5Kin | None = None,
                 gt_source=None, checker=None, div_iou: float = 0.2, div_min_vis: float = 0.5,
                 div_prec: float = 0.4, div_min_px: int = 40, div_min_frac: float = 0.3, confirm_mm: float = 15.0,
                 replan_on_divergence: bool = True, log=None):
        self.bridge = bridge
        self.task, self.instruction, self.mode = task, instruction, mode
        self.assets, self.kin = assets, kin or X5Kin()
        self.gt_source = gt_source              # mode gt only: () -> GT rows of the latest observation
        self.checker = checker                  # mode perception: (obs, objects) -> {label: check}, (obs) -> record
        self.div_iou, self.div_min_vis = div_iou, div_min_vis
        self.div_prec, self.div_min_px, self.div_min_frac = div_prec, div_min_px, div_min_frac
        self.confirm_mm = confirm_mm
        self.max_jump_mm = 150.0                # a re-estimate farther than this from the belief is not believed
        self.post_place_refine = False          # set by the policy (RuntimeConfig.post_place)
        self.watcher = None                     # mode perception: (obs) -> None, may update at-rest objects (watch)
        self.tracker_on = False                 # mode perception: re-estimates pass perception's episode tracker
        self.touched: set[str] = set()          # labels the robot moved since their last good reading (tracker)
        self.held_checker = None                # mode perception: (obs, arm, label) -> {"held": bool|None, ...}
        self.held_min_lift_m = 0.04
        self.held_confirm = None                # second, independent held look (agent) required before a drop
        self._pp_doubt: set[str] = set()        # released, post-place refine not ok (``resync``: fallen if it then diverges)
        self.fallen_check = True                # policy: fallen_check
        self.confirm_wrists = True              # confirm_pose also scores the wrist views (policy: confirm_wrists)
        self.held_drop_conf = 0.9               # a "not held" this sure drops unless the head sees the object in the jaws
        self.motion_reader = None               # (obs_a, obs_b) -> {"moving": bool|None}: scene change outside our arms
        self.garment_reader = None              # (obs) -> a garment read (``dojo.perception.garment``) or None
        self.garment_updater = None             # (obs0, obs, read) -> {"keypoints": the hem corners again} or None
        self._garment = None
        self._garment_obs0 = None
        self.actor_window = 12                  # actors_moving compares with the frame this many steps back
        self._head_hist: list = []              # (step, obs) for actors_moving
        self.track_reader = None                # (obs) -> record: B1 re-read for moving objects (belt tracking)
        self.track_every = 4
        self.track_window = 40
        self.track_min_speed = 3e-4             # m / step: slower is "at rest"
        self.track_match_m = 0.08               # a read farther than this from every prediction is a new object
        self.track_cat_cost = 0.02              # m: matching cost of a category change between reads
        self.track_new_conf = 0.5               # a new object needs at least this B1 confidence
        self.track_max_speed = 0.006            # m / step: the fastest rider we expect (belts: 4 mm / step)
        self.track_outlier_m = 0.015            # a read this far off its track's line is dropped from the fit
        self.track_lost_reads = 6               # an unmatched track is dropped after this many reads (arms still)
        self._missed: dict = {}                 # label -> reads in a row without a match
        self._q_last = None                     # arm joints at the last read (is the robot still?)
        self.track_vel_reads = 4                # a track's velocity counts toward the belt's after this many reads
        self.track_vel_span = 16                # ... spanning this many steps
        self._tracks: dict = {}                 # label -> [(step, xy)] reads
        self._id_votes: dict = {}               # label -> [(asset key, confidence)] of its latest reads
        self.track_id_window = 4
        self._tentative: list = []              # unmatched reads of the last re-read: (step, xy, ObjectState)
        self._unconfirmed: dict = {}            # label -> re-reads that did not see a never-confirmed object
        self.track_confirm_reads = 2
        self._n_new = 0
        self._track_last = None
        self.velocity: dict = {}                # label -> np.array([vx, vy]) m / step (fitted)
        self._fit: dict = {}                    # label -> (step of the fit's origin, xy there)
        self._held_checked: dict = {}           # arm -> label already checked for the current grasp
        self.post_place_min_mm = 2.0
        self.post_place_confirm_mm = 20.0       # larger post-place corrections need the head view to agree
        self.replan = replan_on_divergence
        self.log = log or (lambda *a, **k: None)
        self.home_q = None
        self.notes: dict = {}
        self.belief: KinematicBelief | None = None
        self.pred = None
        self.n_seen = 0
        self.checkpoint = None
        self.events: list[dict] = []
        self.stats = {"refresh": 0, "adopted": 0, "resync": 0, "divergence": 0, "pred_err_mm": []}

    # ------------------------------------------------------------------ setup
    def init_objects(self, objects: dict[str, ObjectState]) -> None:
        st = self.bridge.observe()["state"]
        self.belief = KinematicBelief(objects, self.kin, st)
        self.touched = set()
        self._held_checked = {}
        self.n_seen = self.bridge.n_executed
        self._checkpoint(st)

    def init_from_record(self, record: dict) -> None:
        self.init_objects(objects_from_record(record, self.assets, self.bridge.steps_used))

    def init_from_gt(self) -> None:
        self.init_objects(self._gt_objects())

    def _gt_objects(self) -> dict[str, ObjectState]:
        return objects_from_gt_rows(self.gt_source(), self.assets, self.bridge.steps_used)

    # ------------------------------------------------------------------ predictions
    def on_verified(self, info: dict) -> None:
        """skills hook: the candidate that is about to run, with its twin outcomes."""
        if info.get("none_ok"):
            # the twin rejected every candidate: its prediction of what runs is a failure we do not believe (mode B:
            # the twin is seeded from our belief, and a wrong belief makes it wrong too); keep the kinematic prediction
            # and let a measurement (post-place, refine) correct it
            self.pred = None
            self.stats["pred_untrusted"] = self.stats.get("pred_untrusted", 0) + 1
            return
        idx = info.get("idx")
        idx = 0 if idx is None else int(idx)
        nom = next((o for o in info["outcomes"] if o.candidate == idx and o.sample == 0), None)
        if nom is None:
            return
        from .verifier import chosen_len
        self.pred = {"objects": nom.objects, "n_start": self.bridge.n_executed, "n_actions": chosen_len(info, idx),
                     "kind": info.get("kind"), "label": info.get("label"), "extra": getattr(nom, "extra", {})}

    def _advance(self, st: dict) -> None:
        allx = self.bridge.executed
        for a in allx[self.n_seen:]:
            self.belief.step(a)
            self.touched.update(self.belief.held_labels())
        if self.belief.drop_events:
            self.events.extend({"kind": "belief_drop", "step": self.bridge.steps_used, **e} for e in self.belief.drop_events)
            self.belief.drop_events = []
        self.n_seen = len(allx)
        self.belief.sync_robot(st)
        self.touched.update(self.belief.released)
        if self.pred is not None and self.n_seen - self.pred["n_start"] >= self.pred["n_actions"]:
            if self.n_seen - self.pred["n_start"] == self.pred["n_actions"] or self.pred["n_actions"] > 0:
                before = {l: o.pos.copy() for l, o in self.belief.objects.items()}
                self.belief.adopt(self.pred["objects"])
                self.touched.update(l for l, o in self.belief.objects.items()
                                    if l in before and float(np.linalg.norm(o.pos - before[l])) > 0.004)
                self.stats["adopted"] += 1
            self.pred = None

    def _checkpoint(self, st: dict) -> None:
        self.checkpoint = {"objects": est_objects(self.belief.objects), "robot": robot_from_state(st),
                           "n": self.bridge.n_executed, "step": self.bridge.steps_used}

    # ------------------------------------------------------------------ outcome of a non-grasp action
    def confirm_pose(self, label: str, pos, quat) -> dict:
        """A skill that moved ``label`` without grasping it (a topple, a push) proposes the pose it expects; nothing
        else updates the belief after such a motion (no release, so no post-place; no twin segment). The head view of
        the current frame decides (perception check, z-buffered among the others; ``_post_place_confirmed``): the
        expected pose replaces the belief only if the head view clearly prefers it. Returns {"adopted", "confirm"}."""
        if self.mode == "gt" or self.belief is None or label not in self.belief.objects:
            return {"adopted": False, "why": "gt" if self.mode == "gt" else "unknown label"}
        obs = self.bridge.observe()
        o = self.belief.objects[label]
        others = [x for x in self.belief.movable() if x.label != label]
        ev = {"kind": "confirm_pose", "label": label, "step": self.bridge.steps_used,
              "prior7": [round(float(v), 4) for v in np.r_[o.pos, o.quat]],
              "expected7": [round(float(v), 4) for v in np.r_[np.asarray(pos, float), np.asarray(quat, float)]]}
        ok = self._post_place_confirmed(obs, o, {"pos": pos, "quat": quat}, others, ev, wrists=self.confirm_wrists)
        if ok:
            o.pos, o.quat, o.source = np.asarray(pos, float).copy(), np.asarray(quat, float).copy(), "perception"
            self.touched.add(label)
            if not self.belief.held_labels():
                self._checkpoint(obs["state"])
            self.stats["confirm_pose_adopted"] = self.stats.get("confirm_pose_adopted", 0) + 1
        ev["adopted"] = ok
        self.events.append(ev)
        return {"adopted": ok, "confirm": ev.get("confirm")}

    # ------------------------------------------------------------------ resync
    def post_place(self, obs: dict, released: list[str]) -> list[dict]:
        """Right after a release the arm is at its retreat pose above the placed object: re-measure that object
        from the releasing arm's wrist view (perception refine from the predicted pose, stacked height kept) and
        write the measured pose into the belief, so the place check and the next twin load see where it is."""
        out = []
        if self.checker is None or not hasattr(self.checker, "refine"):
            return out
        for lab in released:
            o = self.belief.objects.get(lab)
            arm = self.belief.released_by.get(lab)
            if o is None or arm is None:
                continue
            others = [x for x in self.belief.movable() if x.label != lab]
            held = None
            for a2, (hl, _) in self.belief.attached.items():
                ho = self.belief.objects[hl]
                held = (ho.asset.key, [float(v) for v in np.r_[ho.pos, ho.quat]])
            r = self.checker.refine(obs, o, others, arm, held) or {}
            ev = {"kind": "post_place", "label": lab, "arm": arm, "step": self.bridge.steps_used,
                  **{k: r.get(k) for k in ("ok", "shift_mm", "iou", "visible_frac", "why")}}
            if r.get("ok"):
                self.touched.discard(lab)              # a good reading: the tracker's "last good reading"
            dz_mm = (abs(float(r["pos"][2]) - float(o.pos[2])) * 1000 if r.get("pos") is not None else 0.0)
            if r.get("z_dropped_mm") is not None:
                ev["z_dropped_mm"] = r.get("z_dropped_mm")
            big = float(r.get("shift_mm") or 0) > self.post_place_confirm_mm
            if r.get("ok") and big and not self._post_place_confirmed(obs, o, r, others, ev):
                # a released object 20+ mm from where the jaws let it go needs a second look (5.1 tower: a 41 mm slide of
                # a plank on a same-colour board, ok at iou 0.72-0.84, adopted, then a re-pick from a wrong belief)
                self.stats["post_place_unconfirmed"] = self.stats.get("post_place_unconfirmed", 0) + 1
            elif r.get("ok") and ((r.get("shift_mm") or 0) >= self.post_place_min_mm or dz_mm >= self.post_place_min_mm
                                  or r.get("z_dropped_mm") is not None):
                # an xy shift OR a height correction (perception re-fits a released object from its support: a belief
                # that floated 56 mm high after a no-candidate place comes back down)
                ev["from"] = np.round(o.pos, 4).tolist()
                o.pos, o.quat, o.source = np.asarray(r["pos"], float), np.asarray(r["quat"], float), "perception"
                ev["to"] = np.round(o.pos, 4).tolist()
                self.stats["post_place_moved"] = self.stats.get("post_place_moved", 0) + 1
            self.stats["post_place"] = self.stats.get("post_place", 0) + 1
            self.events.append(ev)
            if r.get("ok"):
                self._pp_doubt.discard(lab)
            else:
                self._pp_doubt.add(lab)                 # not measured where it was let go (a mug off its hook?)
            out.append(ev)
        return out

    # ------------------------------------------------------------------ watch reads (scripted actors, belts)
    def garment(self, label: str | None = None) -> dict | None:
        """A garment lying flat on the table: the camera's first read (named points, root pose), kept for the episode
        (``live`` False: it does not follow the cloth, the fold skill plans every fold from it). None without a reader
        or a read."""
        if self._garment is None and self.garment_reader is not None:
            try:
                self._garment_obs0 = self.bridge.observe()
                r = self.garment_reader(self._garment_obs0)
            except Exception as e:  # noqa: BLE001 (an optional read)
                self.events.append({"kind": "garment_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})
                r = None
            if r:
                self._garment = {**r, "label": label, "live": False}
                self.events.append({"kind": "garment_read", "asset": r.get("asset"), "iou": r.get("iou"),
                                    "pos": r.get("pos"), "quat": r.get("quat")})
        return self._garment

    def garment_update(self, label: str | None = None, names=()) -> dict | None:
        """The garment's hem corners re-read from the current frame (its first read does not follow the cloth): the
        corners' shift since that read's frame moves its hem points. None without an updater or a read."""
        if self.garment_updater is None or self._garment is None or self._garment_obs0 is None:
            return None
        try:
            r = self.garment_updater(self._garment_obs0, self.bridge.observe(), self._garment)
        except Exception as e:  # noqa: BLE001 (an optional read)
            self.events.append({"kind": "garment_error", "note": f"update: {type(e).__name__}: {str(e)[:120]}"})
            return None
        if r:
            self.events.append({"kind": "garment_update", "shift_mm": r.get("shift_mm")})
            kp = {k: v for k, v in (r.get("keypoints") or {}).items() if not names or k in names}
            return {"keypoints": kp, "live": False}
        return None

    def actors_moving(self) -> bool | None:
        """Did anything but our arms move: the current head frame vs the one ``actor_window`` steps back (a window
        longer than a scripted arm's pauses, e.g. ~10 steps while its gripper closes, so a pause reads as motion).
        None when this provider cannot tell (no reader, too little of the view visible); skills then hold a fixed
        time. Call once per env step while waiting."""
        if self.motion_reader is None:
            return None
        obs = self.bridge.observe()
        step = int(obs["steps_used"])
        self._head_hist = [(s_, o) for s_, o in self._head_hist if s_ >= step - self.actor_window] + [(step, obs)]
        ref = self._head_hist[0]
        if ref[0] >= step:
            return True                                 # nothing to compare with yet: not at rest
        try:
            r = self.motion_reader(ref[1], obs) or {}
        except Exception as e:  # noqa: BLE001 (an optional read)
            self.events.append({"kind": "motion_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})
            return None
        self.stats["motion_reads"] = self.stats.get("motion_reads", 0) + 1
        return r.get("moving")

    def belt_velocity(self) -> np.ndarray:
        """Median fitted velocity (m / step, world xy) of the objects tracked as moving, from tracks long enough to trust
        (>= ``track_vel_reads`` reads over >= ``track_vel_span`` steps) and not faster than ``track_max_speed`` (a track
        that swapped riders or took a phantom); zeros when none. Genuine-5.1 100005: an early 3-read fit gave
        (7.95, -10.69) mm / step for a 4.0 mm / step belt, and the intercept missed."""
        v = []
        for lab, vv in self.velocity.items():
            tr = self._tracks.get(lab) or []
            sp = float(np.linalg.norm(vv))
            if (self.track_min_speed < sp <= self.track_max_speed and len(tr) >= self.track_vel_reads
                    and tr[-1][0] - tr[0][0] >= self.track_vel_span):
                v.append(vv)
        return np.median(np.array(v), axis=0) if v else np.zeros(2)

    def _track(self, obs: dict) -> None:
        """Belt tracking: a B1 re-read every ``track_every`` steps. Reads are matched to the belief's movable, not-held
        objects by predicted position (a category change costs ``track_cat_cost``: B1 can name a rider differently
        from one read to the next); per object a line is fitted over the last ``track_window`` steps, and a moving
        object's belief pose is the fit at the current step (no read latency: the sim waits for the read). A read that
        matches nothing is an object that came into view (onto the belt): it joins the belief under a fresh label
        (B1 hands its interchangeable labels out in x order, so a newcomer's label is usually taken already: the
        2026-10-26 breadth conveyor never added one). Identity: every matched read votes for its asset (weighted by
        confidence); the belief takes the asset with the most votes, and the latest read's orientation."""
        step = int(obs["steps_used"])
        if self._track_last is not None and step - self._track_last < self.track_every:
            self._predict(step)
            return
        self._track_last = step
        try:
            rec = self.track_reader(obs)
        except Exception as e:  # noqa: BLE001
            self.events.append({"kind": "track_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})
            return
        est = objects_from_record(rec, self.assets, step)
        held = set(self.belief.held_labels())
        from .estimate import _assign
        el = [e for e in est.values() if e.obj_type == "Rigid" and not e.clutter]
        bl = [o for o in self.belief.movable() if o.label not in held and o.obj_type == "Rigid"]
        pairs = []
        if el and bl:
            pred = [self._predicted_xy(o, step) for o in bl]
            C = np.array([[float(np.linalg.norm(e.pos[:2] - p_)) + (0.0 if e.category == o.category else self.track_cat_cost)
                           for e in el] for o, p_ in zip(bl, pred)])
            # without a velocity fit the prediction is the last reading: allow for a rider's travel since then (the
            # first re-read comes after the scan and the looks, 15-40 steps after the frame-0 reading)
            gate = [self.track_match_m + (0.0 if o.label in self._fit else
                                          self.track_max_speed * max(0, step - self._last_seen(o))) for o in bl]
            pairs = [(i, j) for i, j in _assign(C) if C[i, j] < gate[i]]
        used = {j for _, j in pairs}
        for i, j in pairs:
            o, e = bl[i], el[j]
            self._tracks.setdefault(o.label, []).append((step, e.pos[:2].copy()))
            self._vote(o.label, e)
        # a belief object no re-read ever confirmed is a phantom of the episode-start estimates (the scan estimate
        # fuses frame 0 as if nothing moved, and fills a label from frame 0's reading of a rider that had moved on: a
        # ghost of the same asset is "the next one" for a matching skill): dropped after ``track_confirm_reads``
        matched = {bl[i].label for i, _ in pairs}
        # a track no read has matched for ``track_lost_reads`` reads while our arms held still is a rider we lost (its
        # re-acquired self is a newcomer by then): dropped, or it would be a ghost "next one" of its identity
        idle = self._q_last is not None and all(float(np.abs(self.belief.q[a] - self._q_last[a]).max()) < 0.01 for a in ARMS)
        self._q_last = {a: self.belief.q[a].copy() for a in ARMS}
        for o in bl:
            if o.label in matched:
                self._missed.pop(o.label, None)
            elif o.label in self._tracks:
                n = self._missed[o.label] = self._missed.get(o.label, 0) + 1
                if idle and n >= self.track_lost_reads and not self.belief.attached:
                    del self.belief.objects[o.label]
                    self._tracks.pop(o.label, None)
                    self.velocity.pop(o.label, None)
                    self._fit.pop(o.label, None)
                    self.events.append({"kind": "track_lost", "label": o.label, "step": step, "reads": n,
                                        "pos": [round(float(v), 3) for v in o.pos[:2]]})
        for o in bl:
            if o.label in matched or o.label in self._tracks or o.label not in self.belief.objects:
                continue
            n = self._unconfirmed[o.label] = self._unconfirmed.get(o.label, 0) + 1
            if n >= self.track_confirm_reads:
                del self.belief.objects[o.label]
                self.events.append({"kind": "track_drop", "label": o.label, "step": step, "reads": n,
                                    "pos": [round(float(v), 3) for v in o.pos[:2]]})
        # an object that came into view (e.g. onto the belt) joins after two reads in a row at consistent places (a
        # crowded belt makes B1 split blobs into extra objects, read once here and once there)
        prev_tent, self._tentative = self._tentative, []
        for j, e in enumerate(el):
            if j in used or e.confidence < self.track_new_conf:
                continue
            seen = None
            for k, (s0, p0, e0) in enumerate(prev_tent):
                if float(np.linalg.norm(e.pos[:2] - p0)) < self.track_match_m + self.track_max_speed * (step - s0):
                    seen = k
                    break
            if seen is None:
                self._tentative.append((step, e.pos[:2].copy(), e))
                continue
            s0, p0, e0 = prev_tent.pop(seen)
            lab = e.label
            if lab in self.belief.objects:
                self._n_new += 1
                lab = f"{e.category}_t{self._n_new}"
                e = copy.deepcopy(e)
                e.label = lab
            self.belief.objects[lab] = e
            self._tracks[lab] = [(s0, p0.copy()), (step, e.pos[:2].copy())]
            self._vote(lab, e0)
            self._vote(lab, e)
            self.events.append({"kind": "track_new", "label": lab, "asset": e.asset.key if e.asset else None,
                                "pos": [round(float(v), 3) for v in e.pos[:2]], "step": step})
        for lab, tr in self._tracks.items():
            tr[:] = [(s_, p) for s_, p in tr if s_ >= step - self.track_window]
            if len(tr) >= 3 and tr[-1][0] - tr[0][0] >= 8:
                t = np.array([s_ for s_, _ in tr], float)
                P = np.array([p for _, p in tr])
                A = np.c_[t - t[-1], np.ones(len(t))]
                coef, *_ = np.linalg.lstsq(A, P, rcond=None)
                res = np.linalg.norm(A @ coef - P, axis=1)
                keep = res <= max(self.track_outlier_m, 3.0 * float(np.median(res)))
                if keep.sum() >= 3 and not keep.all():     # a read the arm half hid: refit without it
                    coef, *_ = np.linalg.lstsq(A[keep], P[keep], rcond=None)
                self.velocity[lab] = coef[0]
                self._fit[lab] = (float(t[-1]), coef[1])
        self._predict(step)
        self.stats["track_reads"] = self.stats.get("track_reads", 0) + 1

    def _last_seen(self, o) -> int:
        tr = self._tracks.get(o.label)
        return int(tr[-1][0]) if tr else int(getattr(o, "t", 0) or 0)

    def _vote(self, label: str, e) -> None:
        """One identity vote of a matched read; the belief object takes the leading asset and the read's orientation."""
        o = self.belief.objects.get(label)
        if o is None or e.asset is None:
            return
        hist = self._id_votes.setdefault(label, [])
        hist.append((e.asset.key, max(float(e.confidence), 0.05)))
        del hist[:-self.track_id_window]               # the latest reads: a rider gets better views as it rides in
        v: dict = {}
        for k, c in hist:
            v[k] = v.get(k, 0.0) + c
        best = max(v, key=v.get)
        if o.asset is not None and best != o.asset.key and best == e.asset.key:
            self.events.append({"kind": "track_identity", "label": label, "from": o.asset.key, "to": best,
                                "votes": {k: round(x, 2) for k, x in v.items()}})
            o.asset, o.category, o.idx = e.asset, e.category, e.idx
        if o.asset is not None and o.asset.key == e.asset.key:
            o.quat = np.asarray(e.quat, float).copy()

    def _rider_velocity(self, lab: str, vb=None) -> np.ndarray | None:
        """The velocity a tracked object moves with: zero when its own long track says it is at rest, else the belt's
        (every rider moves with the belt, and one rider's short or swapped track is noisy), else its own fit."""
        vo = self.velocity.get(lab)
        tr = self._tracks.get(lab) or []
        if vo is not None and float(np.linalg.norm(vo)) <= self.track_min_speed and len(tr) >= self.track_vel_reads:
            return np.zeros(2)
        vb = self.belt_velocity() if vb is None else vb
        if float(np.linalg.norm(vb)) > self.track_min_speed:
            return vb
        return vo

    def _predicted_xy(self, o, step: int, vb=None) -> np.ndarray:
        v = self._rider_velocity(o.label, vb)
        f = self._fit.get(o.label)
        tr = self._tracks.get(o.label)
        if f is not None and v is not None:
            return f[1] + v * (step - f[0])                      # the track's smoothed position, moved on
        if tr:                                                   # no fit yet: the latest reading, moved on
            return tr[-1][1] + (v if v is not None else 0.0) * (step - tr[-1][0])
        return o.pos[:2].copy()

    def _predict(self, step: int) -> None:
        held = set(self.belief.held_labels())
        vb = self.belt_velocity()
        for lab in list(self._tracks):
            o = self.belief.objects.get(lab)
            v = self._rider_velocity(lab, vb)
            if o is None or lab in held or v is None or float(np.linalg.norm(v)) <= self.track_min_speed:
                continue
            o.pos = np.r_[self._predicted_xy(o, step, vb), o.pos[2]]
            o.source = "tracked"

    def _held_check(self, obs: dict) -> None:
        """Once per grasp, when the arm holding ``label`` has lifted ``held_min_lift_m`` above the closing pose:
        perception's held_look on that arm's wrist frame. A confident "not held" takes the object out of the jaws in
        the belief: it lies where it was grasped (dropped onto its support), so the pick's own check sees no rise."""
        for arm, (lab, _T) in list(self.belief.attached.items()):
            g = self.belief.grasp_z.get(arm)
            if g is None or self._held_checked.get(arm) == (lab, g[1]):
                continue
            z = float(self.kin.tcp(arm, self.belief.q[arm])[2, 3])
            if z - g[0] < self.held_min_lift_m:
                continue
            self._held_checked[arm] = (lab, g[1])
            r = self.held_checker(obs, arm, lab) or {}
            ev = {"kind": "held_look", "arm": arm, "label": lab, "step": self.bridge.steps_used,
                  **{k: r.get(k) for k in ("held", "confidence", "source", "calls", "dark_risk", "error", "evidence",
                                           "opening")}}
            self.events.append(ev)
            if r.get("held") is False and float(r.get("confidence") or 0) >= 0.5 and self.held_confirm is not None:
                # never take an object out of the jaws on one vision negative (5.1: B1 said "not held" at confidence
                # 1.0 on blocks lifted 87 mm in the jaws; skills then opened the jaws and dropped them): a second,
                # independent look must agree, else carry on and let the place / post-place checks decide
                r2 = self.held_confirm(obs, arm, lab) or {}
                ev["confirm"] = {k: r2.get(k) for k in ("held", "confidence", "source", "calls")}
                # a sure negative stands unless the head view sees the object in the jaws. The head check at the pick
                # pose is inconclusive whenever the belief's pick pose was wrong, which is when the jaws close on air:
                # replayed in physics, all 52 "not held" reads of breadth2 / step A (confidence 0.9-1.0) were empty
                # jaws, and 51 of them were kept (5.1 xylophone 100002: mallet believed 88 mm off, strikes in the air)
                sure = float(r.get("confidence") or 0) >= self.held_drop_conf
                held2 = r2.get("held") is True and float(r2.get("confidence") or 0) >= 0.5
                if sure and not held2:
                    ev["sure"] = True
                elif not (r2.get("held") is False and float(r2.get("confidence") or 0) >= 0.5):
                    ev["kept"] = True
                    self.stats["held_look_overruled"] = self.stats.get("held_look_overruled", 0) + 1
                    continue
            elif r.get("held") is False:
                ev["kept"] = True
                continue
            if r.get("held") is False and float(r.get("confidence") or 0) >= 0.5:
                self.belief.attached.pop(arm)
                getattr(self.belief, "g_held", {}).pop(arm, None)
                if lab in [v[0] for v in self.belief.attached.values()]:
                    # a handover (#101): this hand's jaws are empty, the other hand still holds the object
                    ev["dropped"] = "this hand only (the other hand holds it)"
                    self.stats["held_look_empty"] = self.stats.get("held_look_empty", 0) + 1
                    continue
                o = self.belief.objects[lab]
                o.pos, o.quat = g[2][:3].copy(), g[2][3:].copy()
                self.belief._drop(lab)
                self.touched.add(lab)
                ev["dropped"] = True
                self.stats["held_look_empty"] = self.stats.get("held_look_empty", 0) + 1

    def _post_place_confirmed(self, obs: dict, o, r: dict, others, ev: dict, min_px: int = 40, wrists: bool = False) -> bool:
        """The second look for a large post-place correction: the HEAD view of the same step, rendered with the object at
        the predicted (released) pose and at the corrected one (perception check, z-buffered among the others). The
        correction stands only if the head view clearly prefers it (precision >= 0.6 and >= the prediction's + 0.15).
        ``wrists`` (confirm_pose: nothing was released, so the wrist view is not the source of the proposal): a WRIST
        view whose silhouette IoU clearly prefers it (>= 0.6 and >= the prior's + 0.15, over >= ``min_px`` pixels) also
        confirms; the arm that toppled / pushed the object hangs over it and hides it from the head."""
        if self.checker is None or not hasattr(self.checker, "check"):
            ev["confirm"] = "no head check"
            return False
        corr = copy.deepcopy(o)
        corr.pos, corr.quat = np.asarray(r["pos"], float), np.asarray(r["quat"], float)
        try:
            cp = self.checker.check(obs, [o] + list(others)).get(o.label) or {}
            cc = self.checker.check(obs, [corr] + list(others)).get(o.label) or {}
        except Exception as e:  # noqa: BLE001 (no second look: keep the prediction)
            ev["confirm"] = f"head check error {type(e).__name__}"
            return False
        hp, hc = cp.get("head") or {}, cc.get("head") or {}
        pp = hp.get("prec") if (hp.get("n_vis") or 0) >= min_px else None
        pc = hc.get("prec") if (hc.get("n_vis") or 0) >= min_px else None
        ok = pc is not None and pc >= 0.6 and (pp is None or pc >= pp + 0.15)
        ev["confirm"] = {"head_prec_predicted": pp, "head_prec_corrected": pc, "confirmed": ok}
        if wrists and not ok:
            for cam in ("cam_left_wrist", "cam_right_wrist"):
                vp = (cp.get("per_view") or {}).get(cam) or {}
                vc = (cc.get("per_view") or {}).get(cam) or {}
                ip = vp.get("iou") if (vp.get("n_px") or 0) >= min_px else None
                ic = vc.get("iou") if (vc.get("n_px") or 0) >= min_px else None
                if ic is None:
                    continue
                ev["confirm"][f"{cam}_iou"] = [ip, ic]
                if ic >= 0.6 and (ip is None or ic >= ip + 0.15):
                    ok = True
                    ev["confirm"].update({"confirmed": True, "by": cam})
                    break
        return ok

    def resync(self, obs: dict, released: list[str] | None = None) -> dict:
        """Post-place wrist refine of what was just released, then render-compare the predicted state against
        the frames; on a confirmed divergence re-estimate and raise."""
        self.stats["resync"] += 1
        measured = set()
        if released and self.post_place_refine:
            measured = {e["label"] for e in self.post_place(obs, released) if e.get("ok")}
        # the wrist just measured these; the just-released ones are judged by their post-place check (and the place
        # check), never "fixed" from a low head check (5.1: a perfect 3-stack re-estimated from iou 0.09-0.32)
        objs = [o for o in self.belief.movable() if o.label not in measured and o.label not in set(released or ())]
        # nothing left to check (a one-object task right after its release: breadth3 deposit_coin / insert_key crashed
        # the perception service with an empty z-buffer stack)
        chk = self.checker.check(obs, objs) if objs else {}
        bad = {}
        for lab, c in chk.items():
            head = c.get("head") or {}
            if head.get("prec") is not None:            # z-buffered: predicted front-most pixels vs foreground
                need = max(self.div_min_px, self.div_min_frac * head.get("n_px", 0))   # judge only what is visible
                if head.get("n_vis", 0) >= need and head["prec"] < self.div_prec:
                    bad[lab] = c
            elif head and head.get("vis", 0) >= self.div_min_vis and head.get("iou", 1.0) < self.div_iou:
                bad[lab] = c
        ev = {"kind": "resync", "step": self.bridge.steps_used, "check": {k: c.get("head") for k, c in chk.items()},
              "diverged": sorted(bad)}
        self.events.append(ev)
        if not bad:
            return ev
        self.stats["checks_failed"] = self.stats.get("checks_failed", 0) + 1
        rec = self.checker.estimate(obs)
        est = objects_from_record(rec, self.assets, self.bridge.steps_used)
        # labels of same-category objects are interchangeable in a fresh estimate: match by position
        est = self._match_by_position(est)
        upd = {}
        cand = {lab: est[lab] for lab in bad if est.get(lab) is not None and est[lab].confidence >= 0.5
                and lab not in self.belief.held_labels()}
        if self.tracker_on and cand:
            from .track import filter_readings
            ok, tev = filter_readings(self.belief.objects, cand, self.touched)
            kept = sorted(lab for lab, a in ok.items() if not a)
            if kept:
                self.events.append({"kind": "tracker_kept", "step": self.bridge.steps_used, "labels": kept,
                                    "touched": sorted(self.touched & set(cand)),
                                    "events": [e for e in tev if e.get("event") not in ("init",)][:8]})
                self.stats["tracker_kept"] = self.stats.get("tracker_kept", 0) + len(kept)
            cand = {lab: e for lab, e in cand.items() if ok.get(lab)}
        for lab, e in cand.items():
            o = self.belief.objects[lab]
            d_mm = float(np.linalg.norm(e.center_w[:2] - o.center_w[:2])) * 1000
            if d_mm < self.confirm_mm or d_mm > self.max_jump_mm:
                continue
            upd[lab] = {"from": np.round(o.pos, 4).tolist(), "to": np.round(e.pos, 4).tolist(), "mm": round(d_mm, 1)}
            o.pos, o.quat, o.source, o.confidence = e.pos.copy(), e.quat.copy(), "perception", e.confidence
            self.touched.discard(lab)
        ev["updated"] = upd
        fallen = sorted(lab for lab in bad if lab in self._pp_doubt and lab not in upd and self.fallen_check
                        and lab not in self.belief.held_labels())
        if fallen:
            # released where the wrist could not confirm it (post-place not ok) and now missing from the head view: it
            # fell (5.1 hang: mugs believed "held by a fixture" on their hooks, every later divergence unconfirmed, no
            # retry). Off the fixture in the belief (the table / movable supports only), touched, and a re-plan.
            for lab in fallen:
                o = self.belief.objects[lab]
                pos, _ = drop_rest(o, [x for x in self.belief.movable() if x.label != lab], ())
                if pos is not None:
                    o.pos = pos
                o.source = "predicted"
                self.touched.add(lab)
                self._pp_doubt.discard(lab)
            self.stats["fallen"] = self.stats.get("fallen", 0) + len(fallen)
            self.events.append({"kind": "fallen", "step": self.bridge.steps_used, "labels": fallen})
            self.log(f"[runtime] fallen {fallen}: off its fixture in the belief, re-plan")
            ev["fallen"] = fallen
            if self.replan:
                raise Divergence(fallen, ev)
            return ev
        if not upd:
            # the frames disagree but the re-estimate does not place the object elsewhere: keep the belief and
            # go on (re-planning a half-built structure from scratch is worse than a doubtful check)
            self.events.append({"kind": "divergence_unconfirmed", "step": self.bridge.steps_used, "labels": sorted(bad)})
            self.log(f"[runtime] unconfirmed divergence {sorted(bad)}: continue")
            return ev
        self.stats["divergence"] += 1
        self.events.append({"kind": "divergence", "step": self.bridge.steps_used, "labels": sorted(bad), "updated": upd})
        self.log(f"[runtime] divergence {sorted(bad)} updated {sorted(upd)}")
        if self.replan:
            raise Divergence(sorted(bad), ev)
        return ev

    def _match_by_position(self, est: dict) -> dict:
        """Re-key a fresh estimate onto the belief's labels: per category, assign estimated objects to belief
        objects by centre distance (one to one)."""
        from .estimate import _assign
        out = {}
        cats = {o.category for o in self.belief.movable()}
        for cat in cats:
            bl = [o for o in self.belief.movable() if o.category == cat]
            el = [e for e in est.values() if e.category == cat]
            if not bl or not el:
                continue
            C = np.array([[float(np.linalg.norm(e.center_w[:2] - b.center_w[:2])) for e in el] for b in bl])
            for i, j in _assign(C):
                out[bl[i].label] = el[j]
        return out

    # ------------------------------------------------------------------ skills provider
    def refresh(self, holding: dict | None = None) -> WorldState:
        obs = self.bridge.observe()
        st = obs["state"]
        self.stats["refresh"] += 1
        arms = arms_from_proprio(st, holding)
        if self.home_q is None:
            self.home_q = {a: arms[a].q.copy() for a in ARMS}
        had_pred = self.pred is not None
        self._advance(st)
        released = self.belief.pop_released()
        if self.mode == "gt":
            objects = self._gt_objects()
            if had_pred and self.pred is None:          # dev metric: twin prediction vs GT after the segment
                errs = [float(np.linalg.norm(objects[l].pos - o.pos)) * 1000 for l, o in self.belief.objects.items()
                        if l in objects and not o.clutter]
                if errs:
                    self.stats["pred_err_mm"].append(round(max(errs), 2))
            for l, o in objects.items():                  # keep the belief on the truth (prefix checkpoints)
                if l in self.belief.objects and l not in self.belief.held_labels():
                    self.belief.objects[l].pos, self.belief.objects[l].quat = o.pos.copy(), o.quat.copy()
        else:
            if self.held_checker is not None:
                self._held_check(obs)
            if self.track_reader is not None:
                self._track(obs)
            if released and self.checker is not None and not self.belief.held_labels():
                self.resync(obs, released)
            if self.watcher is not None and not self.belief.held_labels():
                self.watcher(obs)
            objects = {k: copy.deepcopy(v) for k, v in self.belief.objects.items()}
        if not self.belief.held_labels() and not any((holding or {}).values()):
            self._checkpoint(st)
        return WorldState(task=self.task, instruction=self.instruction, objects=objects, arms=arms,
                          home_q=self.home_q, step=int(obs["steps_used"]), step_lim=int(obs["step_lim"]),
                          notes=self.notes)


__all__ = ["RuntimeProvider", "KinematicBelief", "Divergence", "LostObject"]
