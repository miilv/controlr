"""EstimatorConfig.thin_slot_keep: a thin slot child (deposit_coin's coin) is never refuted by its own few pixels."""
import inspect

import pytest


def test_thin_slot_keep_default_off_and_wired():
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import THIN_CHILD_M, EstimatorConfig, _place_support_slots
    assert EstimatorConfig().thin_slot_keep is False
    assert inspect.signature(_place_support_slots).parameters["thin_keep"].default is False
    assert THIN_CHILD_M >= 0.0019                 # the coin (1.9 mm) is a thin child
