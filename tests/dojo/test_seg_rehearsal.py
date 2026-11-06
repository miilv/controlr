"""The rehearsal gate counts train images, not labels, validation or RFS repeats."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('rehearsal', Path(__file__).resolve().parents[2] /
                                              'scripts/perception/seg/audit_rehearsal.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def frame(img='a', layout='100030_v1', keys=('old',)):
    return dict(img=img, layout=layout, keys=list(keys))


def test_validation_support_cannot_replace_rehearsal():
    result = mod.audit(['old'], ['old', 'new'], [frame(layout='100049_v1')])
    assert result['missing_old'] == ['old']
    assert result['classes']['old']['val_images'] == 1


def test_unique_images_and_exact_prefix():
    f = frame(keys=('old', 'old', 'new'))
    assert mod.audit(['old'], ['old', 'new'], [f])['classes']['old']['train_images'] == 1
    with pytest.raises(ValueError, match='duplicate image'):
        mod.audit(['old'], ['old'], [frame(), frame()])
    with pytest.raises(ValueError, match='prefix'):
        mod.audit(['old'], ['new', 'old'], [f])


@pytest.mark.parametrize('layout', ['100000_v1', '100030', '100030_v0', '100050_v1'])
def test_rejects_nontraining_draws(layout):
    with pytest.raises(ValueError, match='draw'):
        mod.audit(['old'], ['old'], [frame(layout=layout)])


def test_auto_batch_budget_preserves_shared_headroom():
    spec = importlib.util.spec_from_file_location('seg_training', Path(__file__).resolve().parents[2] /
                                                 'scripts/perception/seg/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    gib = 2**30
    fraction, cap = train.auto_batch_budget(28 * gib, 32 * gib, 8)
    assert fraction * 28 == pytest.approx(20)
    assert cap * 32 == pytest.approx(20)
    with pytest.raises(RuntimeError, match='insufficient'):
        train.auto_batch_budget(6 * gib, 32 * gib, 6)
    with pytest.raises(ValueError, match='at least 6'):
        train.auto_batch_budget(28 * gib, 32 * gib, 5)


def test_proxy_capture_waits_for_paired_rgb_and_instance_buffers(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location('proxy_capture', Path(__file__).resolve().parents[2] /
                                                 'scripts/perception/seg/proxy_capture.py')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    images = {'cam_head': np.zeros((480, 640, 3), dtype=np.uint8)}
    monkeypatch.setattr(capture, '_original_render', lambda *args: (images, {}, []))
    updates = []
    fake = SimpleNamespace(inst={'cam_head': True}, app=SimpleNamespace(update=lambda: updates.append(1)),
                           instance_ids=lambda name: (None, {}))
    with pytest.raises(RuntimeError, match='remained empty'):
        capture.ready_render(fake, {})
    assert len(updates) == 400
    fake.instance_ids = lambda name: (np.zeros((480, 640), dtype=np.uint16), {})
    assert capture.ready_render(fake, {})[0] is images


def test_narrow_control_shares_new_images_labels_and_repetitions(tmp_path, monkeypatch):
    import json

    spec = importlib.util.spec_from_file_location('narrow_rehearsal', Path(__file__).resolve().parents[2] /
                                                 'scripts/perception/seg/narrow_rehearsal.py')
    narrow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(narrow)
    full = tmp_path / 'full'
    full.mkdir()
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'names': ['old']}))
    (full / 'classes.json').write_text(json.dumps({'names': ['old', 'new'], 'val_ids': ['100049']}))
    frames = []
    for i, (task, keys) in enumerate([('general_pickup', ['old']), ('general_pickup', ['new']),
                                     ('make_toast_random', ['old'])]):
        frames.append(dict(img=f'/render0/{task}/100030_v1/frame_{i}_cam_head.jpg', task=task,
                           layout='100030_v1', frame=i, cam='cam_head', keys=keys, source='isaac'))
    paths = []
    for f in frames:
        stem = narrow.frame_stem(f)
        for kind, suffix in [('images', '.jpg'), ('labels', '.txt')]:
            p = full / kind / 'train' / (stem + suffix)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('exact original bytes')
        paths.append(str(full / 'images/train' / (stem + '.jpg')))
    (full / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    (full / 'train_rfs.txt').write_text('\n'.join([paths[0], paths[1], paths[1], paths[2]]) + '\n')
    out = tmp_path / 'control'
    monkeypatch.setattr('sys.argv', ['narrow', '--full', str(full), '--base-classes', str(base), '--out', str(out)])
    narrow.main()
    kept = [json.loads(s) for s in (out / 'frames.jsonl').read_text().splitlines()]
    assert kept == frames[1:]
    rfs = (out / 'train_rfs.txt').read_text().splitlines()
    assert len(rfs) == 3 and Path(rfs[0]).name == Path(rfs[1]).name
    assert Path(rfs[0]).read_text() == 'exact original bytes'
    assert (out / 'labels/train' / (narrow.frame_stem(frames[1]) + '.txt')).read_text() == 'exact original bytes'
