"""dojo.perception.watch: the tic-tac-toe cell masks (no catalogue, no GPU)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dojo.perception import watch as W  # noqa: E402
from dojo.perception.cameras import head_camera  # noqa: E402


def test_piece_mask_takes_gold_in_shade_and_leaves_the_arm_and_the_board():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (24, 24, 24)                              # the glossy black board: neutral
    img[100:140, 100:140] = (70, 58, 30)               # a cross in shade: dark (L ~25) but gold
    img[100:140, 300:340] = (235, 205, 120)            # a lit ring: bright gold
    img[300:380, 200:400] = (225, 225, 228)            # the opponent's white arm over the board
    rob = np.zeros((480, 640), np.uint8)
    m = W._piece_mask(img, head_camera(), [], rob)
    assert m[120, 120] and m[120, 320]                 # both pieces, the dark one included
    assert not m[340, 300] and not m[10, 10]           # not the arm, not the board
    occ = W._occluder_mask(img)
    assert occ[340, 300] and not occ[120, 120] and not occ[10, 10]
