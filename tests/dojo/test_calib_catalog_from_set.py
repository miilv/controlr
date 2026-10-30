"""scripts/perception/calib_catalog_from_set.py: the frozen set's M, unchanged, under "all" and the set's name;
deterministic bytes; the J1 adapter's _color_m and identity.load_color_M read it."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("ccfs", ROOT / "scripts/perception/calib_catalog_from_set.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = [[1.0123456789, 0.01, -0.02], [0.03, 0.98, 0.0], [0.0, 0.01, 1.05], [0.1, -0.05, 0.02]]


def test_writes_all_and_name_with_the_same_m_deterministically(tmp_path):
    src = tmp_path / "color_M.json"
    src.write_text(json.dumps({"name": "frozen_x", "M": M, "n": 239}))
    m = _mod()
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    m.main([str(src), str(out1)])
    m.main([str(src), str(out2)])
    assert out1.read_bytes() == out2.read_bytes()
    d = json.loads(out1.read_text())
    assert d == {"all": M, "frozen_x": M}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "M": [[1, 2]], "n": 1}))
    with pytest.raises(ValueError):
        m.catalog_from_set(bad)


def test_identity_reads_it_under_all(tmp_path, monkeypatch):
    src = tmp_path / "color_M.json"
    src.write_text(json.dumps({"name": "frozen_x", "M": M, "n": 239}))
    out = tmp_path / "cat.json"
    _mod().main([str(src), str(out)])
    from dojo.perception import identity
    identity._M_CACHE.clear()
    monkeypatch.delenv("DOJO_CALIB_SET", raising=False)
    monkeypatch.setenv("DOJO_COLOR_CALIB", str(out))
    got = identity.load_color_M()
    assert got is not None and got.tolist() == M
