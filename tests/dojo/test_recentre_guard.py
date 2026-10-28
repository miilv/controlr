"""``RuntimeConfig.refine_recentre_guard`` = "vis": a pick refine that would move the object far enough to recentre, from
a view showing less than ``place_min_vis`` of it, keeps the prior (no recentre). Cases: stream M's arm-8 write audit, the
numbers of the two harmful recentres it names (a bottle believed within 0.2 / 3.6 mm slid 30 / 47 mm along itself at
iou 0.85 / 0.90 from views 0.53 / 0.63 visible). CPU only: a fake perception client."""

import numpy as np
import pytest

from dojo.runtime.belief import RuntimeProvider
from dojo.runtime.estimate import objects_from_gt_rows
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

# (case, prior pose7 (the belief, right), the refine's fit pose7, shift mm, iou, visible_frac)
AUDIT = {
    "gcoff_100004@183 bottle1": ([-0.1452, -0.0202, 0.86, 0.9889, 0.0, 0.0, -0.1488],
                                 [-0.1364, 0.0084, 0.86, 0.9867, 0.0, 0.0, -0.1628], 30.0, 0.8544, 0.526),
    "gcon_100004@56 bottle0": ([-0.3425, -0.0493, 0.7925, 0.3899, -0.59, 0.3897, 0.5899],
                               [-0.3512, -0.0031, 0.7925, 0.3894, -0.5903, 0.3904, 0.5894], 47.0, 0.9032, 0.63),
}


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.array([-0.028, -0.028, -0.095]),
                         bbox_max=np.array([0.028, 0.028, 0.095]))


class _Refine:
    def __init__(self, fit7, shift, iou, vis):
        self.fit7, self.shift, self.iou, self.vis = fit7, shift, iou, vis

    def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None, **kw):
        o = objects[0]
        return {o["label"]: {"ok": True, "pos": self.fit7[:3], "quat": self.fit7[3:], "shift_mm": self.shift,
                             "iou": self.iou, "visible_frac": self.vis, "turned_deg": 1.0}}


def _setup(guard, case, vis=None):
    prior7, fit7, shift, iou, v = AUDIT[case]
    rows = [{"label": "bottle", "category": "bottle", "category_idx": 60, "pos": prior7[:3], "quat": prior7[3:],
             "half": [0.028, 0.028, 0.095]},
            {"label": "other", "category": "bottle", "category_idx": 60, "pos": [0.3, -0.2, 0.86], "quat": [1, 0, 0, 0],
             "half": [0.028, 0.028, 0.095]}]
    env = KinematicEnv(rows, step_lim=600, instruction="Put the bottles into the dustbin.")
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07, refine_recentre_guard=guard),
                        assets=_Assets(), perception_client=_Refine(fit7, shift, iou, v if vis is None else vis))
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 600, "layout_id": 1})
    b = pol.bridge
    img = np.zeros((48, 64, 3), np.uint8)
    b.obs, b.fresh = {"images": {c: img for c in CAMS}, "state": env.proprio(), "instruction": env.instruction,
                      "steps_used": 0, "step_lim": 600}, True
    prov = RuntimeProvider(b, "x", env.instruction, "perception", _Assets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    return pol, prov, np.asarray(prior7, float)


@pytest.mark.parametrize("case", sorted(AUDIT))
def test_vis_keeps_the_prior_against_a_slide_seen_from_a_partial_view(case):
    pol, prov, prior7 = _setup("vis", case)
    hint = pol._refine_hook(prov, None, "bottle", "left", "pick")
    o = prov.belief.objects["bottle"]
    assert hint is None and np.allclose(o.pos, prior7[:3]) and np.allclose(o.quat, prior7[3:])
    ev = pol.report["events"][-1]
    assert ev["kind"] == "refine" and ev["applied"] is False and "prior kept, no recentre" in ev["why"]
    assert prov.stats["recentre_partial_view"] == 1


@pytest.mark.parametrize("case", sorted(AUDIT))
def test_centre_today_applies_the_slide_and_recentres(case):
    pol, prov, prior7 = _setup("centre", case)
    hint = pol._refine_hook(prov, None, "bottle", "left", "pick")
    o = prov.belief.objects["bottle"]
    assert hint is not None and hint.get("rehover") is not None            # the audit's harmful chain starts here
    assert float(np.linalg.norm(o.pos[:2] - prior7[:2])) * 1000 > 25


def test_vis_lets_a_full_view_move_the_object():
    pol, prov, prior7 = _setup("vis", "gcon_100004@56 bottle0", vis=1.0)
    hint = pol._refine_hook(prov, None, "bottle", "left", "pick")
    assert hint is not None and hint.get("rehover") is not None
    assert float(np.linalg.norm(prov.belief.objects["bottle"].pos[:2] - prior7[:2])) * 1000 > 25


def test_default_is_today_and_the_field_is_validated():
    assert RuntimeConfig().refine_recentre_guard == "centre"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", refine_recentre_guard="footprint"), assets=_Assets())
