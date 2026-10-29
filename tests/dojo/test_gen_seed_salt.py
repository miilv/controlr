"""The generator's --seed-salt (robot-harness D-C19): default off = today's records byte for byte; a salt moves every
random stream to H(id ‖ salt). Never run on ids >= 100100 here: the unit test stubs the draw, the regression test
regenerates a few DEV records (100000-04) and compares them with the stored ones (skipped without RoboDojo)."""

import json

import pytest


def _stub(monkeypatch):
    from dojo.simlab import gen_layouts as G
    subs = []

    def draw(tcfg, scfg, groups, sub, semantics=False):
        subs.append(sub)
        return {"Rigid": {}}, 0, 0, None

    monkeypatch.setattr(G, "task_config", lambda t: {})
    monkeypatch.setattr(G, "scene_config", lambda t: {})
    monkeypatch.setattr(G, "merged_object_groups", lambda t: {})
    monkeypatch.setattr(G, "_draw", draw)
    monkeypatch.setattr(G, "validate", lambda lay: [])
    return G, subs


def test_no_salt_keeps_today_s_sub_seeds_and_a_salt_moves_them(monkeypatch):
    G, subs = _stub(monkeypatch)
    lay = G.generate("t", 100003)
    assert subs == [100003 * 1000] and "seed_salt" not in lay["_simlab"] and lay["_simlab"]["seed"] == 100003
    assert G.salted_seed(100003) == 100003 and G.salted_seed(100003, None) == 100003
    subs.clear()
    lay = G.generate("t", 100003, seed_salt="salt-A")
    a = G.salted_seed(100003, "salt-A")
    assert subs == [a * 1000] and a >= G.SALT_BASE and lay["_simlab"]["id"] == 100003
    assert lay["_simlab"]["seed_salt"].startswith("sha256:") and "salt-A" not in json.dumps(lay)
    assert G.salted_seed(100003, "salt-A") == a                                  # deterministic
    assert len({G.salted_seed(i, s) for i in (100003, 100004) for s in ("salt-A", "salt-B")}) == 4
    subs.clear()
    G.generate("t", 100003, variant=2, seed_salt="salt-A")
    assert subs == [G.variant_subseed(a, 2, 0)]
    with pytest.raises(ValueError):
        G.salted_seed(100003, "")
    with pytest.raises(SystemExit, match="empty salt"):
        G.main(["--task", "t", "--n", "1", "--seed-salt", ""])


# md5 of json.dumps(generate(task, id), sort_keys=True) at origin/main 9fbf5556 (simlab/v15, RoboDojo f57e1633,
# compute4, 29 Oct): the salt-less output must stay exactly this (DEV ids only)
TODAY = {("stack_blocks", 100000): "8ef6bd33ab4cd5290a4662c32c9e9089",
         ("stack_blocks", 100002): "f1dc753237487937ee19c07c2b2c67ea",
         ("arrange_largest_number", 100001): "f5e0c6a9c2d9b7ae519a843fae909d87",
         ("classify_objects", 100000): "7b7d4b8e28a56884e556e146184b7f04",
         ("pour_liquid_into_cup", 100002): "88e235c65a6e0e11430c72fffdb9479c",
         ("make_toast", 100001): "8f113eb337297abd5a7f8996f91afadf"}


def _have_robodojo() -> bool:
    from dojo.simlab import paths
    return (paths.TASK_CONFIGS / "stack_blocks.yml").is_file()


@pytest.mark.skipif(not _have_robodojo(), reason="needs RoboDojo's task configs (a dojo box)")
def test_no_salt_output_equals_today_s_for_dev_ids_and_a_salt_changes_it():
    import hashlib

    from dojo.simlab import gen_layouts as G
    md5 = lambda lay: hashlib.md5(json.dumps(lay, sort_keys=True).encode()).hexdigest()   # noqa: E731
    for (task, i), want in TODAY.items():
        assert md5(G.generate(task, i)) == want, (task, i)
    salted = G.generate("stack_blocks", 100000, seed_salt="test-salt")
    assert md5(salted) != TODAY[("stack_blocks", 100000)] and salted["_simlab"]["id"] == 100000
    assert md5(G.generate("stack_blocks", 100000, seed_salt="test-salt")) == md5(salted)
