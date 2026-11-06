"""Dev gate integrity: split isolation, paired keys, errors and metric semantics."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('seg_dev_gate', Path(__file__).resolve().parents[2] /
                                              'scripts/perception/seg/dev_gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def bake(root, episodes=('100000',), *, error=None, found=True, identity=True, distance=0):
    root.mkdir()
    objects, frames = [], []
    for ep in episodes:
        meta = dict(set='w51', task='arrange', episode=ep, frame=0)
        objects.append(dict(meta, label='rigid0', category='number', gt_pos=[0, 0, 0],
                            found=found, id_ok=identity, t_err_mm=distance))
        frames.append(dict(meta, error=error, phantoms=0))
    for name, rows in (('objects', objects), ('frames', frames)):
        (root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return root


def test_cut_excludes_all_measurement_ranges(tmp_path):
    ids = [str(i) for i in range(99999, 100151)]
    src = bake(tmp_path / 'source', ids)
    gate.cut(src, tmp_path / 'cut')
    _, frames = gate.load(tmp_path / 'cut')
    assert {int(r['episode']) for r in frames} == set(range(100000, 100005)) | set(range(100030, 100050))
    with pytest.raises(FileExistsError):
        gate.cut(src, tmp_path / 'cut')


def test_error_rows_can_be_archived_but_never_qualify(tmp_path):
    src = bake(tmp_path / 'source', error='OpenGL failure')
    out = tmp_path / 'cut'
    gate.cut(src, out)
    with pytest.raises(ValueError, match='frame errors'):
        gate.compare(out, out)
    assert 'INVALID GATE' in gate.compare(out, out, diagnostic=True)


def test_comparison_requires_same_objects_and_frames(tmp_path):
    a = bake(tmp_path / 'a')
    b = bake(tmp_path / 'b', ('100001',))
    with pytest.raises(ValueError, match='unpaired'):
        gate.compare(a, b)
    p = a / 'objects.jsonl'
    p.write_text(p.read_text() * 2)
    with pytest.raises(ValueError, match='duplicate'):
        gate.load(a)


def test_exact_zero_error_counts_and_identity_loss_is_separate(tmp_path):
    a = bake(tmp_path / 'a')
    b = bake(tmp_path / 'b', identity=False)
    assert gate.metrics(gate.load(a)[0]) == (1, 1, 1, 1)
    assert gate.metrics(gate.load(b)[0]) == (1, 0, 1, 0)
    assert '1 class losers' in gate.compare(a, b)


def test_live_bake_selects_only_dev_ids():
    spec = importlib.util.spec_from_file_location('seg_bake', Path(__file__).resolve().parents[2] /
                                                 'scripts/perception/seg/bake_seg.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {i for i in range(99999, 100151) if module.held_out(i)} == set(range(100000, 100005)) | set(range(100030, 100050))


def test_gl_worker_supports_ctypes_output_pointer(monkeypatch):
    # No context or GPU needed: glGenTextures uses this output-pointer conversion.
    import ctypes

    monkeypatch.setenv('PYOPENGL_PLATFORM', 'egl')
    pytest.importorskip('OpenGL.arrays.ctypesparameters')
    from OpenGL.arrays.arraydatatype import ArrayDatatype

    spec = importlib.util.spec_from_file_location('seg_bake_gl', Path(__file__).resolve().parents[2] /
                                                 'scripts/perception/seg/bake_seg.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.init_gl_arrays()
    ptr = ctypes.byref(ctypes.c_uint())
    handler = ArrayDatatype.getHandler(ptr)
    assert handler.from_param(ptr) is ptr


def test_asset_report_does_not_hide_within_category_cancellation(tmp_path):
    a = bake(tmp_path / 'a', ('100000', '100001'))
    b = bake(tmp_path / 'b', ('100000', '100001'))
    for root, good in ((a, 0), (b, 1)):
        p = root / 'objects.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        for i, r in enumerate(rows):
            r.update(gt_idx=i, id_ok=i == good)
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert '0 class losers' in gate.compare(a, b)
    assert '1 class losers' in gate.compare(a, b, by_asset=True)
