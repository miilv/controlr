"""CPU-only contracts: default equivalence, scoped masks, identity and socket contention."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from dojo.perception import inverse_layout as il, sam_routing as sr
from dojo.perception.sam_masks import SamMask, SideServerDown


def test_defaults_exact_old_detector_call():
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
def test_every_task_defaults_to_v2b(task):
    assert sr.resolve(il.EstimatorConfig().sam_task_routes, task) == {'mode': 'v2b'}


def test_sam_series_counts_without_an_asset_identity():
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
