"""CPU-only contracts: default equivalence, scoped masks, identity and socket contention."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from dojo.perception import sam_routing as sr
from dojo.perception.sam_masks import SamMask, SideServerDown


@pytest.fixture
def il():
    pytest.importorskip('cv2')
    pytest.importorskip('scipy')
    from dojo.perception import inverse_layout
    return inverse_layout


def test_defaults_exact_old_detector_call(il):
    cfg = il.EstimatorConfig()
    assert cfg.sam_task_routes == {}
    c = Mock()
    expected = c.detect.return_value
    image = np.zeros((4, 4, 3), np.uint8)
    assert il._seg_detect(c, image, cfg, []) is expected
    c.detect.assert_called_once_with(image, conf=cfg.seg_min_score)
    assert sr.resolve({'pour_by_language': {'mode': 'sam'}}, 'deposit_coin') == {'mode': 'v2b'}


@pytest.mark.parametrize('mode', ['sam', 'union'])
def test_routing_no_invented_identity(monkeypatch, mode):
    image = np.zeros((10, 10, 3), np.uint8)
    mask = np.zeros((10, 10), np.uint8); mask[:3, :3] = 1
    base = SimpleNamespace(mask=mask.copy(), key='Rigid/watch/00001', score=.9)
    client = Mock()
    client.text.return_value = [SamMask(mask, .8, 'watch'), SamMask(1-mask, .7, 'pepper')]
    monkeypatch.setenv('DOJO_SAM_ADDR', '/mock')
    monkeypatch.setattr(sr, 'SamClient', Mock(return_value=client))
    groups = [{'candidates': ['Rigid/watch/00001', 'Rigid/watch/00002']}]
    out = sr.detect([base], image, groups, 'classify_objects_by_language', {'mode': mode, 'model': 'sam3'})
    assert len(out) == 1
    if mode == 'union':
        assert out[0] is base
    else:
        assert out[0].key == 'SAM/category/proposal'
        assert out[0].candidates == tuple(groups[0]['candidates'])
    assert client.text.call_args.kwargs['thr'] == .2
    client.conn.close.assert_called_once()


def test_missing_server_loud(monkeypatch):
    monkeypatch.delenv('DOJO_SAM_ADDR', raising=False)
    with pytest.raises(SideServerDown):
        sr.detect([], np.zeros((4, 4, 3)), [], 'pour_by_language', {'mode': 'sam'})


def test_invalid_route():
    with pytest.raises(ValueError):
        sr.resolve({'sealed': {'mode': 'sam'}}, '')


@pytest.mark.parametrize('task', list(sr.banks()))
def test_every_task_defaults_to_v2b(task, il):
    assert sr.resolve(il.EstimatorConfig().sam_task_routes, task) == {'mode': 'v2b'}


def test_sam_series_counts_without_an_asset_identity(il):
    m = np.ones((8, 8), np.uint8)
    p = sr.Proposal(m, 'SAM/category/proposal', .8, ('Rigid/watch/00001',))
    assert il._seg_count({'labels': ['x'], 'candidates': ['Rigid/watch/00001']}, [p], .5) == 1
    assert il._seg_count({'labels': ['x'], 'candidates': ['Rigid/pepper/00001']}, [p], .5) == 0


def test_two_clients_share_serialized_model(tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from dojo.perception.sam_masks import serve, SamClient
    active = 0

    class Fake:
        def info(self):
            return {}

        def handle(self, req, image):
            nonlocal active
            active += 1
            assert active == 1
            time.sleep(.01)
            active -= 1
            return {'masks': []}

    sock = str(tmp_path / 'sam.sock')
    ready = threading.Event()
    threading.Thread(target=serve, args=(sock, ['sam3']), kwargs={'M': Fake(), 'ready': ready}, daemon=True).start()
    assert ready.wait(3)

    def run(_):
        c = SamClient(sock, loud=True)
        try:
            return c.text(np.zeros((8, 8, 3), np.uint8), ['watch'], model='sam3')
        finally:
            c.conn.close()
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(run, range(2))) == [[], []]


def test_sam_fit_is_candidate_gated_and_never_gets_exact_hint(monkeypatch, il):
    from dojo.perception import seg_twin
    mask = np.ones((8, 8), np.uint8)
    allowed = 'Rigid/watch/00001'
    wrong = 'Rigid/pepper/00001'
    proposal = sr.Proposal(mask, 'SAM/category/proposal', .8, (allowed,), ('x',))
    group = {'labels': ['x'], 'candidates': [allowed, wrong]}
    cfg = il.EstimatorConfig()
    monkeypatch.setattr(seg_twin, 'client', Mock(return_value=object()))
    monkeypatch.setattr(il, '_seg_detect', Mock(return_value=[proposal]))
    monkeypatch.setattr(il, '_union_masks', Mock(return_value={}))
    shape = SimpleNamespace(key=allowed)
    other = SimpleNamespace(key=wrong)
    fit = SimpleNamespace(per_view={}, x=0., y=0., key=allowed,
                          sm=lambda: SimpleNamespace(sil=lambda *_: mask), T=lambda: np.eye(4))
    fitter = Mock(return_value=(.9, fit))
    monkeypatch.setattr(il, '_seg_fit', fitter)
    chosen = []
    il._seg_propose(chosen, {'cam_head': SimpleNamespace(cam=None, known=mask)},
                    np.zeros((8, 8, 3), np.uint8), cfg, [group], [(0, group, {})],
                    {0: {'shared_geometry': [shape, other]}})
    assert chosen == [fit]
    assert fitter.call_args.args[5] == {'shared_geometry': [shape]}
    assert 'seg_key' not in fit.per_view


def test_scoped_route_does_not_call_sam_on_parity_cell(monkeypatch, il):
    cfg = il.EstimatorConfig(sam_task_routes={'pour_by_language': {'mode': 'sam'}})
    cfg._sam_route_task = 'classify_objects'
    c = Mock()
    sam = Mock(side_effect=AssertionError('inactive SAM called'))
    monkeypatch.setattr(sr, 'detect', sam)
    assert il._seg_detect(c, np.zeros((8, 8, 3)), cfg, []) is c.detect.return_value
    sam.assert_not_called()
