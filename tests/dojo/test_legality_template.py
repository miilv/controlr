"""Legality audit 2026-10-28, P1 finding 3 (robot-harness docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md): the
twin's template was our generator's record for seed 100000 (``paths.SEED_BASE``), itself one of the evaluated dev
layouts. Observed objects kept its sampled fields (``relative_plane``), and an unobserved fixed-XY label was copied
whole, sampled identity included. Now the twin's record is the estimate plus what the public configs fix
(``estimate.public_constants``): unknown sampled identities and relations stay unknown. The regression tests vary only
the sampled values behind the template (the draws) and require the same twin input and the same actions."""

import copy

import numpy as np

from dojo.runtime.estimate import constants_from, estimate_to_layout

FIXTURES = {"Room": {"default": "r"}, "Table": {"default": "t"}, "Ground": {}, "Background": {}}
GROUPS = {
    "Rigid": [{"select_mode": {"label": ["cube"]}, "common": {"relative_plane": ["mat0", "mat1"], "place_tag": "up"},
               "category": [{"name": "cube", "index": [0, 1, 2]}]}],
    "Geometry": [
        {"select_mode": {"label": ["mat0", "mat1"], "mode": "unique"}, "category": [{"name": "mat", "index": [0, 1]}]},
        # fixed XY, but two candidate assets: the identity is sampled, so the twin must not get it
        {"select_mode": {"label": ["hidden_fixed_card"]}, "common": {"xlim": [0.1, 0.1], "ylim": [0.2, 0.2]},
         "category": [{"name": "card", "index": [5, 7]}]},
        # fixed XY and one asset: a public constant
        {"select_mode": {"label": ["camera_stand"]}, "common": {"xlim": [0, 0], "ylim": [0.4, 0.4]},
         "category": [{"name": "camera_stand", "index": [0]}]},
        # fixed XY, one asset, but a random yaw (draws disagree): not a constant record
        {"select_mode": {"label": ["turned_bin"]}, "common": {"xlim": [-0.3, -0.3], "ylim": [0.3, 0.3],
                                                              "rotate_rand": True},
         "category": [{"name": "bin", "index": [3]}]},
    ],
}
STAND = {"category": "camera_stand", "category_idx": 0, "label": "camera_stand", "default_pos": [0.0, 0.4, 0.765],
         "default_ori": [1.0, 0.0, 0.0, 0.0]}


def _draw(card_idx, cube_on, bin_yaw):
    """One template draw: the sampled values (card identity, the cube's mat, the bin's yaw) are this draw's."""
    q = [float(np.cos(bin_yaw / 2)), 0.0, 0.0, float(np.sin(bin_yaw / 2))]
    return {"Rigid": {"cube": [{"category": "cube", "category_idx": 2, "label": "cube", "relative_plane": cube_on,
                                "default_pos": [0.3, 0.2, 0.8], "default_ori": [1, 0, 0, 0]}]},
            "Geometry": {"card": [{"category": "card", "category_idx": card_idx, "label": "hidden_fixed_card",
                                   "default_pos": [0.1, 0.2, 0.8], "default_ori": [1, 0, 0, 0]}],
                         "camera_stand": [dict(STAND)],
                         "bin": [{"category": "bin", "category_idx": 3, "label": "turned_bin",
                                  "default_pos": [-0.3, 0.3, 0.8], "default_ori": q}]}}


def _const(card_idx, cube_on, yaws):
    return constants_from(copy.deepcopy(GROUPS), FIXTURES, [_draw(card_idx, cube_on, y) for y in yaws],
                          lambda t, c: [])


OBJS = [{"type": "Rigid", "category": "cube", "category_idx": 2, "label": "cube", "pos": [0.3, 0.2, 0.8],
         "quat": [1, 0, 0, 0]},
        {"type": "Geometry", "category": "mat", "category_idx": 0, "label": "mat0", "pos": [0.31, 0.21, 0.766],
         "quat": [1, 0, 0, 0]},
        {"type": "Geometry", "category": "mat", "category_idx": 1, "label": "mat1", "pos": [-0.2, 0.0, 0.766],
         "quat": [1, 0, 0, 0]}]


def _layout(const, objs=OBJS):
    return estimate_to_layout("t", copy.deepcopy(objs), constants=const, meta_fn=lambda t, c, i: {})


def test_sampled_identity_and_support_behind_the_template_never_reach_the_twin():
    a = _layout(_const(7, "mat_secret", [0.1, 0.9, 1.7, 2.3]))
    b = _layout(_const(5, "mat1", [0.4, 0.4, 1.0, 2.0]))
    assert a == b
    labels = {inst["label"] for t in ("Rigid", "Geometry") for lst in (a.get(t) or {}).values() for inst in lst}
    assert "hidden_fixed_card" not in labels                    # sampled identity: unknown, not copied
    assert "turned_bin" not in labels                           # sampled yaw (draws disagree): not copied
    assert "camera_stand" in labels                             # fixed by the config: a constant
    cube = a["Rigid"]["cube"][0]
    assert cube["relative_plane"] == "mat0"                     # the believed support (nearest observed mat)
    assert cube["place_tag"] == "up" and cube["category_idx"] == 2 and cube["default_pos"] == [0.3, 0.2, 0.8]
    assert {"Room", "Table", "Ground", "Background"} <= set(a)


def test_an_unobserved_support_stays_unknown():
    lay = _layout(_const(7, "mat_secret", [0.1, 0.9]), objs=OBJS[:1])
    assert "relative_plane" not in lay["Rigid"]["cube"][0]


def test_a_single_support_named_by_the_config_is_a_constant():
    g = copy.deepcopy(GROUPS)
    g["Rigid"][0]["common"]["relative_plane"] = "mat1/top"
    const = constants_from(g, FIXTURES, [_draw(5, "mat1", 0.0)], lambda t, c: [])
    assert _layout(const, objs=OBJS[:1])["Rigid"]["cube"][0]["relative_plane"] == "mat1"


def test_the_twin_input_and_the_actions_do_not_follow_the_draws(monkeypatch):
    """Policy level (the audit's 'neither the twin input nor the selected action may change'): a whole episode with
    bundle verification through the real ``estimate_to_layout``; only the template draws behind the constants
    differ (the hidden card's identity, a support relation)."""
    import dojo.runtime.estimate as E
    from dojo.runtime.firewall import DEV_GT_KEY
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.env import KinematicEnv
    from dojo.skills.world import AssetInfo

    stack = "Stack the three blocks with different textures."
    pos = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]
    groups = {"Rigid": [{"select_mode": {"label": ["block_0", "block_1", "block_2"], "mode": "unique"},
                         "category": [{"name": "block", "index": [4, 5, 6]}]}],
              "Geometry": copy.deepcopy(GROUPS["Geometry"][1:3])}

    class Assets:
        def get(self, typ, cat, idx, clutter=False, mesh=False):
            return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015),
                             bbox_max=np.full(3, 0.015))

    class Twin:
        def __init__(self):
            self.loads, self.snaps, self.base = [], {}, None

        def info(self):                 # main before the P1-4 fix still asks; the answer is the instruction's scene
            return {"task": "stack_blocks"}

        def _env(self, layout, robot):
            rows = [{"label": r["label"], "category": c, "category_idx": r["category_idx"], "pos": r["default_pos"],
                     "quat": r["default_ori"], "half": [0.015] * 3} for c, lst in (layout.get("Rigid") or {}).items()
                    for r in lst]
            env = KinematicEnv(rows, step_lim=10_000)
            for a in ("left", "right"):
                env.q[a] = np.asarray(robot[f"{a}_q"], float)
                env.g[a] = float(robot[f"{a}_grip"])
            return env

        def load_from_estimate(self, layout, robot=None, settle_steps=50):
            self.loads.append(copy.deepcopy(layout))
            self.base = self._env(layout, robot)
            return {"fast": True, "stable": True}

        def verify(self, world, candidates, perturb=None, k=1, record_traj=False, seed=0, keep=None):
            if "layout" in world:
                self.load_from_estimate(world["layout"], world.get("robot"))
            base = self.snaps[world["snapshot_id"]] if "snapshot_id" in world else self.base
            out = []
            for ci, c in enumerate(candidates):
                for si in range(k):
                    e = copy.deepcopy(base)
                    for a in c["actions"]:
                        e.step(a)
                    o = {"candidate": ci, "sample": si, "success": False, "score": 0.0, "ended": False,
                         "reward": {"success": False, "success_flag": True, "ended": False, "steps_used": 0,
                                    "step_lim": 1000},
                         "final_objects": {r["label"]: list(r["pos"]) + list(r["quat"]) for r in e.gt_objects()},
                         "final_robot": {f"{a}_q": e.q[a].tolist() for a in ("left", "right")}}
                    if keep == "nominal" and si == 0:
                        sid = max(self.snaps, default=0) + 1
                        self.snaps[sid] = e
                        o["snapshot_id"] = sid
                    out.append(o)
            return out

        def drop_snapshot(self, sid):
            self.snaps.pop(int(sid), None)

    def episode(card_idx, cube_on):
        draws = [{"Geometry": {"card": [{"category": "card", "category_idx": card_idx, "label": "hidden_fixed_card",
                                         "default_pos": [0.1, 0.2, 0.8], "default_ori": [1, 0, 0, 0],
                                         "relative_plane": cube_on}],
                               "camera_stand": [dict(STAND)]}}] * 3
        const = constants_from(copy.deepcopy(groups), FIXTURES, draws, lambda t, c: [])
        monkeypatch.setattr(E, "public_constants", lambda task: const)
        monkeypatch.setattr("dojo.simlab.gen_layouts.metadata", lambda t, c, i: {})
        rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
                 "half": [0.015] * 3} for i, p in enumerate(pos)]
        env = KinematicEnv(rows, step_lim=550, instruction=stack)
        tw = Twin()
        pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="bundle", n_cands=2, verify_k=1), assets=Assets(),
                            twin_client=tw)
        pol.reset({"instruction": stack, "step_lim": 550})
        acts = []
        for _ in range(2000):
            obs = {"images": {c: np.zeros((48, 64, 3), np.uint8) for c in ("cam_head", "cam_left_wrist",
                                                                          "cam_right_wrist")},
                   "state": env.proprio(), "instruction": stack, "steps_used": env.steps_used, "step_lim": 550,
                   DEV_GT_KEY: env.gt_objects()}
            d = pol.next(obs)
            if d.get("done"):
                break
            ended = False
            for a in d["actions"]:
                acts.append({k: np.round(np.asarray(v, float), 9).tolist() for k, v in a.items()})
                if env.step(a):
                    ended = True
                    break
            if ended:
                break
        return pol.episode_report(), tw.loads, acts

    rep_a, loads_a, acts_a = episode(7, "mat_secret")
    rep_b, loads_b, acts_b = episode(5, "mat1")
    assert rep_a["error"] is None and rep_b["error"] is None
    assert loads_a and loads_a == loads_b
    assert all("card" not in (ld.get("Geometry") or {}) for ld in loads_a)       # the sampled identity never loads
    assert all("camera_stand" in (ld.get("Geometry") or {}) for ld in loads_a)   # the config's constant does
    assert acts_a == acts_b and rep_a["decisions"] == rep_b["decisions"]


def test_template_draws_are_never_a_layout_of_ours():
    """The draws' sub-seeds sit below every layout record's sub-seed block (seed * 1000 + ..., seed >= SEED_BASE)."""
    from dojo.simlab import gen_layouts as G
    from dojo.simlab import paths
    assert G.TEMPLATE_SUBSEED_MAX < paths.SEED_BASE * 1000 <= G.variant_subseed(paths.SEED_BASE, 0, 0)
