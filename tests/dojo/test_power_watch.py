"""A transient spike must not drain a lane; sustained high readings must."""
import runpy
from pathlib import Path


reduce_lanes = runpy.run_path(str(Path(__file__).parents[2] / "scripts/dojo/queues/power_watch.py"))["reduce_lanes"]


def test_sustained_power_drains_second_lane(tmp_path):
    high = reduce_lanes(tmp_path, 701, False)
    assert not (tmp_path / "STOP.heavy2").exists()
    high = reduce_lanes(tmp_path, 700, high)
    assert not high
    high = reduce_lanes(tmp_path, 705, high)
    reduce_lanes(tmp_path, 710, high)
    assert (tmp_path / "STOP.heavy2").exists()
    assert (tmp_path / "lane2.caps").read_text() == "1 100\n"
