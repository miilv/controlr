"""dojo.perception.wrist_check (R1): object selection by corridor distance, existence verdicts, neighbour fits,
phantom drops, relocation of a misplaced belief onto an unexplained blob, MUSE identity / flip margins, the in-process
SAM adapter. A synthetic top-down world (1 mm per pixel, square objects); no GPU, no catalogue."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from dojo.perception import wrist_check as W  # noqa: E402

H, Wd = 240, 320
SIDE = 0.04                          # every object: a 40 mm square


def _px(x, y):
    return int(round(Wd / 2 + 1000 * x)), int(round(H / 2 + 1000 * y))


def _sq(x, y, side=SIDE):
    m = np.zeros((H, Wd), np.uint8)
    u, v = _px(x, y)
    h = int(round(500 * side))
    m[max(v - h, 0):max(v + h, 0), max(u - h, 0):max(u + h, 0)] = 1
    return m


class _Cam:
    name = "cam_left_wrist"
    w, h = Wd, H

    def ray_plane(self, uv, z):
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.c_[(uv[:, 0] - Wd / 2) / 1000.0, (uv[:, 1] - H / 2) / 1000.0, np.full(len(uv), z)]


class _Mask:
    def __init__(self, mask, score=0.9):
        self.mask, self.score = mask, score


class _Sam:
    """Clicks on a true object return its mask; clicks on the table return a big table patch."""

    def __init__(self, truth):
        self.truth = truth

    def points(self, img, uv, model="", multimask=True):
        u, v = (int(round(c)) for c in np.asarray(uv, float).reshape(-1, 2)[0])
        for x, y in self.truth.values():
            m = _sq(x, y)
            if m[v, u]:
                return [_Mask(m)]
        t = np.zeros((H, Wd), np.uint8)
        t[:120] = 1
        return [_Mask(t)]


class _Check(W.WristCheck):
    """The wrist check over a synthetic world: ``truth`` = {label: (x, y)} of the real objects."""

    def __init__(self, truth, cfg=None, **kw):
        self.truth = truth
        super().__init__(cfg or W.CheckConfig(), refine_fn=self._fake_refine, **kw)

    @staticmethod
    def camera(arm, state):
        return _Cam()

    @staticmethod
    def silhouette(cam, key, pose):
        return _sq(pose[0], pose[1])

    @staticmethod
    def truncated(cam, key, pose):
        return 0.0

    @staticmethod
    def robot(cam, state, closed_jaws):
        return np.zeros((H, Wd), np.uint8)

    def foreground(self, img, rob):
        fg = np.zeros((H, Wd), np.uint8)
        for x, y in self.truth.values():
            fg |= _sq(x, y)
        return fg

    def fg_frac(self, img, cam, sil, occ, others, belief_key=None, pose=None):
        fg = self.foreground(img, occ)
        return float((fg & sil).sum()) / max(int(sil.sum()), 1)

    def _fake_refine(self, key, img, state, prior, arm, **kw):
        """Snaps to a true object within 25 mm of the prior (``search_radius``: within that) unless a known other object
        sits there (refine's "neighbour:" rejection); iou 0.95. Else not ok, foreground share at the prior."""
        others = [p for _, p in (kw.get("others") or [])]
        free = [t for t in self.truth.values() if all(np.hypot(t[0] - p[0], t[1] - p[1]) > 0.7 * SIDE for p in others)]
        sil = _sq(prior[0], prior[1])
        fg = self.foreground(img, None)
        op = {"fg_px": int((fg & sil).sum()), "sil_px": int(sil.sum())}
        best = min(free, key=lambda t: np.hypot(t[0] - prior[0], t[1] - prior[1]), default=(9.0, 9.0))
        if np.hypot(best[0] - prior[0], best[1] - prior[1]) < max(0.025, kw.get("search_radius") or 0):
            return {"ok": True, "iou": 0.95, "pose": [best[0], best[1]] + list(prior[2:]), "obs_prior": op,
                    "turned_deg": 0.0}
        return {"ok": False, "iou": 0.1, "pose": list(prior), "obs_prior": op, "turned_deg": 0.0}


Q = [1.0, 0.0, 0.0, 0.0]


def _bel(**xy):
    return {lab: {"key": f"Rigid/block/0000{i}", "pos": [x, y, 0.78], "quat": list(Q)}
            for i, (lab, (x, y)) in enumerate(xy.items())}


@pytest.fixture
def square_footprints(monkeypatch):
    def fp(key, pos, quat):
        x, y = pos[0], pos[1]
        h = SIDE / 2
        return np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])

    monkeypatch.setattr(W, "footprint", fp)


def test_corridor_dist():
    sq = np.array([[0, 0], [0.04, 0], [0.04, 0.04], [0, 0.04]], float)
    assert W.corridor_dist(sq, [[0.02, 0.02, 0.8]]) == 0.0
    assert W.corridor_dist(sq, [[0.10, 0.02, 0.8]]) == pytest.approx(0.06, abs=1e-6)
    # a path passing over the polygon touches it even if neither end does
    assert W.corridor_dist(sq, [[-0.05, 0.02, 0.9], [0.09, 0.02, 0.8]]) == 0.0


def test_yaw_flip_is_an_involution_about_the_centre():
    from dojo.perception.cameras import quat_wxyz_to_R

    c = [0.1, -0.2, 0.8]
    p, q = [0.13, -0.2, 0.78], [np.cos(0.2), 0.0, 0.0, np.sin(0.2)]
    p2, q2 = W.yaw_flip(p, q, c)
    assert np.allclose(p2[:2], [0.07, -0.2]) and p2[2] == pytest.approx(0.78)
    assert np.allclose(quat_wxyz_to_R(q2), np.diag([-1, -1, 1.0]) @ quat_wxyz_to_R(q))
    p3, q3 = W.yaw_flip(p2, q2, c)
    assert np.allclose(p3, p) and np.allclose(quat_wxyz_to_R(q3), quat_wxyz_to_R(q))


def test_required_selects_target_reference_and_corridor_neighbours(square_footprints):
    wc = _Check({})
    bel = _bel(a=(0.0, 0.0), b=(0.05, 0.0), c=(0.2, 0.0), d=(0.0, 0.09))
    objs = wc.required(bel, "a", None, [[0.0, 0.0, 0.8]])
    assert [(o.label, o.role) for o in objs] == [("a", "target"), ("b", "neighbour"), ("d", "neighbour")]
    objs = wc.required(bel, None, "c", [[0.0, 0.0, 0.8]], exclude={"a"})
    assert [o.label for o in objs] == ["c", "b", "d"] and objs[0].role == "reference"


def test_check_fixes_neighbour_pose_drops_phantom_keeps_target(square_footprints):
    # a: the target, where believed; b: a neighbour believed 12 mm off; p: a phantom on empty table
    truth = {"a": (0.0, 0.0), "b": (0.062, 0.0)}
    wc = _Check(truth, sam=_Sam(truth), cfg=W.CheckConfig(relocate=False))
    bel = _bel(a=(0.0, 0.0), b=(0.05, 0.0), p=(-0.06, 0.0))
    out = wc.check({"images": {"cam_left_wrist": np.zeros((H, Wd, 3), np.uint8)}, "state": {}}, bel, "left", "pick",
                   target="a", corridor=[[0.0, 0.0, 0.8]])
    o = out["objects"]
    assert o["a"]["verdict"] == "present" and "applied" not in o["a"]
    assert o["b"]["verdict"] == "ok" and o["b"]["applied"]
    assert out["belief"]["b"]["pos"][:2] == pytest.approx([0.062, 0.0])
    assert o["p"]["verdict"] == "phantom" and o["p"]["dropped"] and "p" not in out["belief"]
    assert out["belief"]["a"] == bel["a"]                   # the target's pose is the runtime's
    # the input belief is never mutated
    assert bel["b"]["pos"][:2] == [0.05, 0.0] and "p" in bel


def test_target_phantom_is_flagged_not_dropped(square_footprints):
    wc = _Check({}, sam=_Sam({}), cfg=W.CheckConfig(relocate=False))
    bel = _bel(a=(0.0, 0.0))
    out = wc.check({"images": {"cam_left_wrist": np.zeros((H, Wd, 3), np.uint8)}, "state": {}}, bel, "left", "pick",
                   target="a", corridor=[[0.0, 0.0, 0.8]])
    assert out["objects"]["a"]["verdict"] == "phantom" and "a" in out["belief"]


def test_relocation_moves_a_phantom_onto_the_unexplained_blob(square_footprints):
    # k is believed at (-0.06, 0) (empty table) but really sits at (0.0, 0.07): in view, in the corridor, unexplained
    truth = {"a": (0.0, 0.0), "k": (0.0, 0.07)}
    wc = _Check(truth, sam=_Sam(truth), cfg=W.CheckConfig(relocate_min_px=300))
    bel = _bel(a=(0.0, 0.0), k=(-0.06, 0.0))
    out = wc.check({"images": {"cam_left_wrist": np.zeros((H, Wd, 3), np.uint8)}, "state": {}}, bel, "left", "pick",
                   target="a", corridor=[[0.0, 0.0, 0.8]])
    assert out["objects"]["k"]["verdict"] == "phantom"
    f = [x for x in out["found"] if x.get("applied")]
    assert len(f) == 1 and f[0]["label"] == "k"
    assert out["belief"]["k"]["pos"][:2] == pytest.approx([0.0, 0.07])   # relocated, not dropped
    assert not out["objects"]["k"].get("dropped")


def test_radius_search_relocates_a_phantom_found_nearby_but_not_onto_a_known_neighbour(square_footprints):
    # m is believed 54 mm from where it is (radius search finds it); p sits on empty table 60 mm from the target a,
    # which is known: the search must not grab a, so p stays a phantom and is dropped
    truth = {"a": (0.0, 0.0), "m": (0.0, 0.07)}
    wc = _Check(truth, sam=_Sam(truth), cfg=W.CheckConfig(relocate_min_px=10 ** 6))
    bel = _bel(a=(0.0, 0.0), m=(0.05, 0.05), p=(-0.06, 0.0))
    out = wc.check({"images": {"cam_left_wrist": np.zeros((H, Wd, 3), np.uint8)}, "state": {}}, bel, "left", "pick",
                   target="a", corridor=[[0.0, 0.0, 0.8]])
    o = out["objects"]
    assert o["m"]["verdict"] == "relocated" and out["belief"]["m"]["pos"][:2] == pytest.approx([0.0, 0.07])
    assert [f["label"] for f in out["found"] if f.get("via") == "radius"] == ["m"]
    assert o["p"]["verdict"] == "phantom" and "p" not in out["belief"]


def test_unseen_objects_get_no_verdict_without_the_head(square_footprints):
    class _Occluded(_Check):
        @staticmethod
        def robot(cam, state, closed_jaws):
            return np.ones((H, Wd), np.uint8)               # the gripper hides the whole view

    wc = _Occluded({"a": (0.0, 0.0)}, cfg=W.CheckConfig(relocate=False))
    out = wc.check({"images": {"cam_left_wrist": np.zeros((H, Wd, 3), np.uint8)}, "state": {}}, _bel(a=(0.0, 0.0)),
                   "left", "pick", target="a", corridor=[[0.0, 0.0, 0.8]])
    assert out["objects"]["a"]["verdict"] == "unseen" and out["belief"] == _bel(a=(0.0, 0.0))


class _FakeEmb:
    """tokens(): CLS = the crop's normalised mean colour, patches = per-patch colours (4 x 4 grid) weighted by their
    position (as ViT tokens carry their position: a pooled patch vector tells a mirrored crop apart)."""

    def tokens(self, imgs):
        cls, pat = [], []
        for im in imgs:
            f = np.asarray(im, float)
            c = f.reshape(-1, 3).mean(0)
            cls.append(c / max(np.linalg.norm(c), 1e-9))
            g = f.reshape(4, f.shape[0] // 4, 4, f.shape[1] // 4, 3).mean((1, 3))
            pos = (np.arange(16).reshape(4, 4, 1) + 1) / 16.0
            pat.append(np.concatenate([g, g * pos], axis=-1))
        cls = [np.r_[c, c] / np.sqrt(2) for c in cls]
        return np.asarray(cls), np.asarray(pat)


def test_muse_scorer_prefers_the_matching_render():
    m = W.MuseScorer(_FakeEmb(), cls_weight=0.5, gem_p=1.0)
    obs = np.zeros((32, 32, 3), np.uint8)
    obs[:, :16] = (200, 30, 30)
    obs[:, 16:] = (30, 30, 200)
    mask = np.ones((32, 32), np.uint8)
    same, swapped = obs.copy(), obs[:, ::-1].copy()
    green = np.full_like(obs, (30, 200, 30))
    s = m.scores(obs, mask, [(same, mask), (swapped, mask), (green, mask)])
    assert s[0] > s[1] > s[2]                               # the patch term separates the flip, the CLS the colour


def test_local_sam_adapter_unpacks_the_models_answer():
    from dojo.perception import sam_masks

    class _M:
        def handle(self, req, img):
            assert req["op"] == "points" and req["key"] == sam_masks.image_key(img)
            m = np.zeros(img.shape[:2], bool)
            m[2:5, 3:7] = True
            return {"masks": [(*sam_masks.pack(m), 0.8, 0)]}

    s = sam_masks.LocalSam([], M=_M())
    out = s.points(np.zeros((10, 12, 3), np.uint8), [[4, 3]])
    assert len(out) == 1 and out[0].mask.sum() == 12 and out[0].score == 0.8
