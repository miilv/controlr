"""Add-only model composition must preserve every base detection and the default wire protocol."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from dojo.perception import seg_twin as seg
from dojo.perception.sam_masks import pack


class Base:
    classes = ['Rigid/old/00000']

    def __init__(self, mask):
        self.result = {'dets': [(*pack(mask), self.classes[0], .6, [0., 0., 2., 2.])]}

    def handle(self, req, img):
        assert not req.get('additive')
        return self.result

    def info(self):
        return {'base': True}


class Extra:
    classes = ['Rigid/new/00000']

    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def detect(self, *args):
        self.calls += 1
        return self.detections

    def info(self):
        return {'extra': True}


def test_prefix_unchanged_no_combined_cap_and_overlap_boundary():
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    lower = ~mask
    half = np.zeros_like(mask)
    half[0] = True  # IoU exactly .5
    base = Base(mask)
    extra = Extra([(mask, Extra.classes[0], .99, []), (lower, Extra.classes[0], .9, []),
                   (half, Extra.classes[0], .8, []), (lower, Base.classes[0], .99, []),
                   (lower, Extra.classes[0], .4, [])])
    model = seg._AdditiveSegModel(base, extra)
    request = dict(op='detect', additive=True, extra_iou=.5, extra_conf=.5, max_det=1)
    result = model.handle(request, None)
    assert result['dets'][0] is base.result['dets'][0]
    assert len(result['dets']) == 3  # base cap never applied to merged output
    assert result['sources'] == ['base', 'extra', 'extra']
    assert model.handle(dict(op='detect'), None) is base.result
    model.handle({**request, 'extra_allowed': Base.classes}, None)
    assert extra.calls == 1  # old-only task does not run extra inference
    with pytest.raises(ValueError, match='disjoint'):
        seg._AdditiveSegModel(base, base)


def test_default_client_packet_and_additive_provenance():
    client = object.__new__(seg.SegClient)
    calls = []
    result = Base(np.ones((2, 2), bool)).result

    def call(req, image):
        calls.append(req)
        return result

    client._c = SimpleNamespace(_call=call)
    assert client.detect(None)[0].source == 'base'
    assert calls[-1] == dict(op='detect', conf=.25, max_det=100)
    result['sources'] = ['extra']
    assert client.detect(None, additive=True, extra_allowed=['new'])[0].source == 'extra'
    assert calls[-1]['extra_allowed'] == ['new']


def test_additive_socket_roundtrip_and_missing_model_is_loud(tmp_path):
    from dojo.perception import sam_masks

    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    model = seg._AdditiveSegModel(Base(mask), Extra([(~mask, Extra.classes[0], .9, [])]))
    ready = threading.Event()
    sock = str(tmp_path / 'additive.sock')
    threading.Thread(target=sam_masks.serve, args=(sock, ['seg'], .01),
                     kwargs=dict(M=model, ready=ready), daemon=True).start()
    assert ready.wait(5)
    client = seg.SegClient(sock)
    image = np.zeros((4, 4, 3), np.uint8)
    assert len(client.detect(image)) == 1
    result = client.detect(image, additive=True)
    assert [d.source for d in result] == ['base', 'extra']
    assert np.array_equal(result[0].mask, mask) and np.array_equal(result[1].mask, ~mask)
    unloaded = object.__new__(seg._SegModel)
    with pytest.raises(ValueError, match='without a new-class model'):
        unloaded.handle(dict(op='detect', additive=True), image)


def test_consumer_base_priority_and_exact_candidate_request():
    pytest.importorskip('scipy')
    pytest.importorskip('cv2')
    from dojo.perception import inverse_layout as il

    cfg = il.EstimatorConfig()
    assert cfg.seg_additive is False
    mask = np.ones((5, 5), bool)
    base = seg.Det(mask, 'Rigid/old/00000', .6)
    extra = seg.Det(mask, 'Rigid/new/00000', .99, source='extra')
    kept, dropped = il._seg_dedup([extra, base], .7)
    assert kept == [base] and dropped == 1 and base.alts == []
    seen = []
    client = SimpleNamespace(detect=lambda image, **kwargs: seen.append(kwargs) or [])
    il._seg_detect(client, None, cfg, [])
    assert seen[-1] == {'conf': cfg.seg_min_score}
    cfg.seg_additive = True
    il._seg_detect(client, None, cfg, [{'candidates': ['Rigid/new/00000']}])
    assert seen[-1]['extra_allowed'] == ['Rigid/new/00000']


def test_new_only_export_keeps_negative_images_and_remaps(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/perception/seg/new_only_export.py'
    spec = importlib.util.spec_from_file_location('new_only_export_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    full, out = tmp_path / 'full', tmp_path / 'out'
    full.mkdir()
    keep = tmp_path / 'keep.txt'
    keep.write_text('Rigid/new/00000\n')
    meta = {'names': ['Rigid/old/00000', 'Rigid/new/00000'], 'val_ids': ['100049']}
    (full / 'classes.json').write_text(json.dumps(meta))
    frames = []
    for seed, keys in [('100030', ['Rigid/old/00000']), ('100031', ['Rigid/new/00000']),
                       ('100049', ['Rigid/new/00000'])]:
        f = dict(img=f'/render/task/{seed}_v1/frame_0_cam_head.png', task='task', layout=f'{seed}_v1',
                 frame=0, cam='cam_head', keys=keys)
        split = 'val' if seed == '100049' else 'train'
        for kind in ('images', 'labels'):
            (full / kind / split).mkdir(parents=True, exist_ok=True)
        stem = mod.frame_stem(f)
        (full / 'images' / split / (stem + '.png')).write_bytes(b'image')
        cls = 0 if seed == '100030' else 1
        (full / 'labels' / split / (stem + '.txt')).write_text(f'{cls} 0 0 1 0 1 1\n')
        frames.append(f)
    (full / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    monkeypatch.setattr('sys.argv', ['export', '--full', str(full), '--keep', str(keep), '--out', str(out)])
    mod.main()
    result = json.loads((out / 'classes.json').read_text())
    assert result['names'] == ['Rigid/new/00000'] and result['negative_images']['train'] == 1
    assert len(list((out / 'images/train').iterdir())) == 2
    assert sorted(p.read_text() for p in (out / 'labels/train').iterdir()) == ['', '0 0 0 1 0 1 1\n']
    assert json.loads((full / 'classes.json').read_text()) == meta


def test_select_head_preserves_selected_classifier_rows():
    torch = pytest.importorskip('torch')
    path = Path(__file__).resolve().parents[2] / 'scripts/perception/seg/select_head.py'
    spec = importlib.util.spec_from_file_location('select_head_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    conv = torch.nn.Conv2d(4, 3, 1)
    weights, bias = conv.weight.detach().clone(), conv.bias.detach().clone()
    head = SimpleNamespace(cv3=[torch.nn.Sequential(conv)], nc=3, reg_max=16)
    yolo = SimpleNamespace(model=SimpleNamespace(names={0: 'old', 1: 'new1', 2: 'new2'},
                                                model=[head], yaml={'nc': 3}))
    mod.select(yolo, ['new2', 'new1'])
    assert torch.equal(head.cv3[0][-1].weight, weights[[2, 1]])
    assert torch.equal(head.cv3[0][-1].bias, bias[[2, 1]])
    assert head.nc == 2 and head.no == 66 and yolo.model.names == {0: 'new2', 1: 'new1'}


def test_gate_rejects_phantom_concentration_and_old_asset_cancellation(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'scripts/perception/seg'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('additive_summary_test', scripts / 'summarize_additive.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base, candidate = tmp_path / 'base', tmp_path / 'candidate'
    objects = [dict(set='dev', task='task', episode='100000', frame=0, label=str(i), category='old',
                    gt_idx=i, gt_pos=[i, 0, 0], found=True, id_ok=i == 0, t_err_mm=0) for i in range(2)]
    frames = [dict(set='dev', task='task', episode='100000', frame=0, phantoms=0)]

    def save(path, objects, frames):
        path.mkdir(exist_ok=True)
        for name, rows in [('objects', objects), ('frames', frames)]:
            (path / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    save(base, objects, frames)
    save(candidate, objects, frames)
    recipe = {'gate': dict(objects=2, frames=1, phantom_total_increase_max=2,
                           phantom_increase_per_set_task_max=1, naming_denominator=49, new_naming_target=18)}
    names = ['Rigid/old/00000', 'Rigid/old/00001']
    naming = dict(hit=20, merged=0, missed=29)
    assert mod.evaluate(base, candidate, recipe, naming, names)['pass_gate']
    save(candidate, objects, [{**frames[0], 'phantoms': 2}])
    result = mod.evaluate(base, candidate, recipe, naming, names)
    assert not result['pass_gate'] and not result['phantoms']['pass_gate']
    changed = [{**row, 'id_ok': i == 1} for i, row in enumerate(objects)]
    save(candidate, changed, frames)
    result = mod.evaluate(base, candidate, recipe, naming, names)
    assert result['old_category']['losers'] == 0 and result['old_asset']['losers'] == 1
    assert not result['pass_gate']
