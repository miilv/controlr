"""RuntimeConfig.track_expire (Misha's G5 review #8): a belt track no read matched for ``track_window`` steps expires
whatever the arms do, is not extrapolated past that, and a same-asset ghost near a newcomer retires."""
import pytest
from test_runtime import STACK, FakeAssets, FakeEpisode, _env, objects_from_gt_rows

from dojo.runtime.bridge import Bridge


def _run(expire: str, gap=(24, 64), steps=90, behind=0.10):
    """One rider (block_0) at 4 mm / step; the reads lose it over ``gap`` (the arm hides it) while our arms keep moving,
    then read it again where it really is: ``behind`` m short of where its track predicts (the arm held it back), more
    than the match gate, so it comes back as a newcomer."""
    from dojo.runtime.belief import RuntimeProvider
    env = _env(positions=[(0.05, 0.04, 0.78)])
    b = Bridge()
    ep = FakeEpisode(env)
    b.obs, b.fresh = ep.observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.track_expire = expire
    for lab in [lab for lab in prov.belief.objects if lab != "block_0"]:
        del prov.belief.objects[lab]

    def reader(obs):
        s_ = obs["steps_used"]
        if gap[0] < s_ < gap[1]:
            return {"Rigid": {}}
        x = 0.05 + 0.004 * s_ - (behind if s_ >= gap[1] else 0.0)
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": "block_0",
                                     "default_pos": [x, 0.04, 0.78], "default_ori": [1, 0, 0, 0],
                                     "_perception": {"confidence": 0.8}}]}}
    prov.track_reader = reader
    for step in range(steps):
        prov._q_last = None                                # the arms moved since the last read (never "idle")
        b.obs, b.fresh = dict(ep.observe(), steps_used=step), True
        prov._track(b.obs)
    return prov


def test_today_a_rider_lost_while_the_arms_move_leaves_a_ghost():
    prov = _run("off")
    blocks = [o for o in prov.belief.objects.values() if o.category == "block"]
    assert len(blocks) == 2                                # the stale track of block_0 + the newcomer: the old behaviour


def test_track_expire_leaves_exactly_one_object_after_the_rider_is_read_again():
    prov = _run("on")
    blocks = [o for o in prov.belief.objects.values() if o.category == "block"]
    assert len(blocks) == 1
    assert abs(float(blocks[0].pos[0]) - (0.05 + 0.004 * 89 - 0.10)) < 0.03      # where the rider really is
    kinds = {e["kind"] for e in prov.events}
    assert "track_expired" in kinds or "track_ghost_retired" in kinds


def test_track_expire_retires_a_ghost_near_a_newcomer_before_the_window():
    # a short gap: the track has not expired yet when the rider is read again near its prediction offset by more than
    # the match gate, so it joins as a newcomer and the same-asset ghost next to it retires
    prov = _run("on", gap=(24, 40), steps=60)
    blocks = [o for o in prov.belief.objects.values() if o.category == "block"]
    assert len(blocks) == 1


def test_an_expired_track_is_not_extrapolated():
    prov = _run("on", gap=(24, 200), steps=80)          # lost for good: no re-read
    assert not [o for o in prov.belief.objects.values() if o.category == "block"]


def test_track_expire_default_off_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().track_expire == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(track_expire="maybe"))
