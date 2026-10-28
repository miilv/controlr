"""AssetDB says when a mesh is missing (an incomplete catalogue changed G3 episodes silently) (CPU only)."""

from __future__ import annotations

import json

import numpy as np


def _asset(root, key):
    d = root / key
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": [[-0.01, -0.01, 0],
                                                                                          [0.01, 0.01, 0.02]]}}}))


def test_missing_mesh_is_recorded_and_flagged_unless_listed(tmp_path, capsys):
    from dojo.skills import world
    objs, cat = tmp_path / "objs", tmp_path / "cat"
    for key in ("Rigid/cube/00000", "Rigid/cube/00001", "Rigid/never/00000"):
        _asset(objs, key)
    (cat / "Rigid/cube/00000").mkdir(parents=True)
    V = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], float)
    np.savez(cat / "Rigid/cube/00000/geom.npz", vertices=V, faces=np.array([[0, 1, 2]]))
    world._MESH_LESS = {"Rigid/never/00000"}                      # the full catalogue has no mesh for this one
    try:
        db = world.AssetDB(objs, catalog_root=[cat])
        assert db.get("Rigid", "cube", 0, mesh=True).mesh is not None
        assert db.get("Rigid", "cube", 1, mesh=True).mesh is None
        assert db.get("Rigid", "never", 0, mesh=True).mesh is None
        db.get("Rigid", "cube", 1, mesh=True)                       # warned once
        assert db.missing_meshes == {"Rigid/cube/00001", "Rigid/never/00000"}
        assert db.unexpected_missing() == ["Rigid/cube/00001"]
        err = capsys.readouterr().err
        assert err.count("no mesh for Rigid/cube/00001") == 1 and "incomplete catalogue" in err
        assert "no mesh for Rigid/never/00000" in err
        # no mesh asked for: nothing recorded
        db2 = world.AssetDB(objs, catalog_root=[cat])
        db2.get("Rigid", "cube", 1)
        assert db2.missing_meshes == set()
    finally:
        world._MESH_LESS = None


def test_mesh_less_list_loads():
    from dojo.skills import world
    world._MESH_LESS = None
    keys = world.mesh_less()
    # the matryoshka dolls had no mesh until the catalogue fix of 2026-10-28 (stream W; mesh_less regenerated)
    assert "Rigid/audio/00000" in keys and "Rigid/key/00000" not in keys
    assert "Rigid/matryoshka_dolls/00000" not in keys and "Rigid/number/00007" not in keys
