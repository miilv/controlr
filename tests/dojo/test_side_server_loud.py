"""Side servers (seg_twin / glyph_read over sam_masks.serve): one server can serve several lanes (the model and the
image cache under locks: concurrent requests give the serial answers), and a SET address that is unreachable, gone or
erroring raises SideServerDown (an infra failure of the job), never a silent run without the server."""
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import pytest

from dojo.perception import glyph_read, sam_masks, seg_twin
from dojo.perception.sam_masks import SamClient, SideServerDown


class _OneAtATime:
    """A fake model that fails if two requests are in it at once (ultralytics / the glyph ResNet are not thread-safe)."""

    def __init__(self):
        self.inside, self.max_inside, self.n = 0, 0, 0
        self.lock = threading.Lock()

    def info(self):
        return {}

    def handle(self, req, img):
        with self.lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.002)
        out = {"sum": int(img.astype(np.int64).sum()), "op": req["op"], "x": int(req.get("x", 0))}
        with self.lock:
            self.inside -= 1
            self.n += 1
        return out


def _serve(tmp_path, M, name="s.sock"):
    sock = str(tmp_path / name)
    ready = threading.Event()
    threading.Thread(target=sam_masks.serve, args=(sock, ["fake"], 0.02), kwargs={"M": M, "ready": ready},
                     daemon=True).start()
    assert ready.wait(5)
    return sock


def test_concurrent_lanes_get_the_serial_answers_one_request_at_a_time(tmp_path):
    M = _OneAtATime()
    sock = _serve(tmp_path, M)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(12)]
    serial = [SamClient(sock, loud=True)._call({"op": "x", "x": i}, im) for i, im in enumerate(imgs)]
    results, errors = {}, []

    def lane(k):
        try:
            c = SamClient(sock, loud=True)         # one client per lane process
            for r in range(3):
                for i, im in enumerate(imgs):
                    results[(k, r, i)] = c._call({"op": "x", "x": i}, im)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    th = [threading.Thread(target=lane, args=(k,)) for k in range(6)]
    [t.start() for t in th]
    [t.join(30) for t in th]
    assert not errors
    assert len(results) == 6 * 3 * len(imgs)
    assert all(results[(k, r, i)] == serial[i] for (k, r, i) in results)
    assert M.max_inside == 1


def test_set_but_unreachable_addresses_raise(monkeypatch, tmp_path):
    with pytest.raises(SideServerDown):
        SamClient(str(tmp_path / "nobody.sock"), loud=True)
    with pytest.raises(OSError):                   # SAM (not loud) keeps its plain error
        SamClient(str(tmp_path / "nobody.sock"))
    monkeypatch.setenv(seg_twin.ENV_ADDR, str(tmp_path / "nobody.sock"))
    seg_twin._CLIENT.clear()
    with pytest.raises(SideServerDown):
        seg_twin.client()
    monkeypatch.delenv(seg_twin.ENV_ADDR)
    assert seg_twin.client() is None
    for var in ("DOJO_GLYPH_ADDR", "DOJO_GLYPH_ADDR_DIGIT", "DOJO_GLYPH_ADDR_KONG"):
        monkeypatch.delenv(var, raising=False)
    glyph_read._CLIENT.clear()
    assert glyph_read.client("number") is None and glyph_read.client("mahjong") is None
    monkeypatch.setenv("DOJO_GLYPH_ADDR_DIGIT", str(tmp_path / "nobody.sock"))
    with pytest.raises(SideServerDown):
        glyph_read.client("number")
    glyph_read._CLIENT.clear()
    seg_twin._CLIENT.clear()


class _Broken(_OneAtATime):
    def handle(self, req, img):
        raise ValueError("boom")


def test_server_errors_are_loud_for_seg_and_glyph_and_plain_for_sam(tmp_path):
    sock = _serve(tmp_path, _Broken(), "b.sock")
    img = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(SideServerDown, match="boom"):
        SamClient(sock, loud=True)._call({"op": "x"}, img)
    with pytest.raises(RuntimeError, match="boom") as ei:
        SamClient(sock)._call({"op": "x"}, img)
    assert not isinstance(ei.value, SideServerDown)


def test_a_server_gone_mid_run_raises(tmp_path):
    sock = str(tmp_path / "gone.sock")
    lst = Listener(sock, family="AF_UNIX")

    def accept_then_die():
        conn = lst.accept()
        conn.recv()
        conn.close()                               # the server process died mid-request

    threading.Thread(target=accept_then_die, daemon=True).start()
    c = SamClient(sock, loud=True)
    with pytest.raises(SideServerDown, match="gone"):
        c._call({"op": "x"}, np.zeros((4, 4, 3), np.uint8))
    lst.close()


def test_the_service_records_it_and_eval_marks_the_row_infra(monkeypatch, tmp_path):
    from dojo.runtime import eval as ev
    from dojo.runtime import perception_service as ps

    ready = tmp_path / "perception.ready"
    monkeypatch.setenv("DOJO_PERCEPTION_READY", str(ready))
    assert ev.side_server_down_why(str(ready)) is None
    path = ps.side_server_down("side server at /x/seg.sock gone mid-run: EOFError")
    assert path == str(tmp_path / ps.SIDE_SERVER_DOWN_FILE)
    assert "gone mid-run" in ev.side_server_down_why(str(ready))
    assert ev.side_server_down_why("") is None


def test_numeral_reader_reraises_a_dead_server_but_keeps_other_errors_quiet(monkeypatch):
    import inspect

    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as il

    src = inspect.getsource(il._glyph_read_numeral)
    assert src.index("except SideServerDown") < src.index("except Exception")
