"""glyph_read routes by asset category: numerals / operators -> the digit model (DOJO_GLYPH_ADDR_DIGIT), mahjong ->
the kong model (DOJO_GLYPH_ADDR_KONG), any other category -> no reader. The old DOJO_GLYPH_ADDR is only a fallback,
and a reply of the wrong model gives no read (a digit model never reads a mahjong tile)."""
import numpy as np
import pytest

from dojo.perception import glyph_read as G

DIGIT = {"probs": [[0.0] * 3 + [1.0] + [0.0] * 11], "n_glyph": 15, "margins": {"head": 0.1, "wrist": 0.1},
         "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus", "minus", "multiplication", "division", "equal"]}
KONG = {"probs": [[0.0] * 5 + [1.0] + [0.0] * 37], "n_glyph": 42, "margins": {"head": 0.1, "wrist": 0.1}, "names": None}


@pytest.fixture
def servers(monkeypatch):
    from dojo.perception import sam_masks
    replies = {"/s/digit": DIGIT, "/s/kong": KONG}

    class Fake:
        def __init__(self, addr, **kw):
            self.addr = addr

        def _call(self, req, img):
            return replies[self.addr]

    monkeypatch.setattr(sam_masks, "SamClient", Fake)
    G._CLIENT.clear()
    for k in ("DOJO_GLYPH_ADDR", "DOJO_GLYPH_ADDR_DIGIT", "DOJO_GLYPH_ADDR_KONG"):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch
    G._CLIENT.clear()


def _mask():
    m = np.zeros((40, 40), np.uint8)
    m[10:30, 10:30] = 1
    return m


def test_each_category_gets_its_model(servers):
    servers.setenv("DOJO_GLYPH_ADDR_DIGIT", "/s/digit")
    servers.setenv("DOJO_GLYPH_ADDR_KONG", "/s/kong")
    img = np.zeros((40, 40, 3), np.uint8)
    r = G.classify_mask(img, _mask(), category="number")
    assert r["read"] and r["name"] == "3"
    assert G.classify_mask(img, _mask(), category="plus")["name"] == "3"          # operators: the digit model too
    assert G.client("mahjong").model == "kong" and G.client("number").model == "digit"
    assert G.client("block") is None and G.classify_mask(img, _mask(), category="block")["why"] == "no glyph server"


def test_a_digit_only_env_never_reads_a_mahjong_tile(servers):
    servers.setenv("DOJO_GLYPH_ADDR", "/s/digit")                                  # Integration's old single address
    img = np.zeros((40, 40, 3), np.uint8)
    assert G.classify_mask(img, _mask(), category="number")["name"] == "3"         # the fallback fits the digit model
    assert G.classify_mask(img, _mask(), category="mahjong") == {"read": False, "why": "wrong model"}
    servers.setenv("DOJO_GLYPH_ADDR_KONG", "/s/digit")                              # misconfigured: digit on KONG
    G._CLIENT.clear()
    assert G.classify_mask(img, _mask(), category="mahjong")["why"] == "wrong model"
