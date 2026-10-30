"""D-C19 parity of the venvs: the official side's package versions vs a compute4 lane venv's (pkg_versions.py JSON).
A mismatch (numpy / pillow / opencv / torch / ultralytics / transformers / ... or the python version) is a parity diff."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dojo"))

import pkg_versions as PV  # noqa: E402
import candidate_parity as CP  # noqa: E402


def test_versions_lists_every_package_and_the_python():
    v = PV.versions()
    assert set(v["packages"]) == set(PV.PACKAGES) and v["python"] and v["packages"]["numpy"]
    for p in ("numpy", "pillow", "opencv-python-headless", "torch", "ultralytics", "transformers", "scipy", "trimesh",
              "pyrender", "PyOpenGL"):
        assert p in PV.PACKAGES


def test_diff_names_each_mismatch_and_none_for_the_same_set():
    a = {"python": "3.11.9", "packages": {"numpy": "2.5.4", "pillow": "13.0.0", "PyOpenGL": "3.1.9", "torch": None}}
    assert PV.diff(a, json.loads(json.dumps(a))) == {}
    b = {"python": "3.11.8", "packages": {"numpy": "2.4.6", "pillow": "13.0.0", "PyOpenGL": "3.1.9", "torch": "2.11.0"}}
    assert PV.diff(a, b) == {"numpy": ["2.5.4", "2.4.6"], "torch": [None, "2.11.0"], "python": ["3.11.9", "3.11.8"]}


def test_packages_check_reads_the_lane_reference(tmp_path):
    here = PV.versions()
    ref = tmp_path / "lane.json"
    ref.write_text(json.dumps(here))
    pk = CP.packages_check(str(ref), sys.executable)
    assert pk["diff"] == {} and pk["ref_path"] == str(ref)
    here["packages"]["numpy"] = "0.0.1"
    ref.write_text(json.dumps(here))
    assert CP.packages_check(str(ref), sys.executable)["diff"]["numpy"][0] == "0.0.1"
    assert CP.packages_check(None, sys.executable)["diff"] == {}            # no reference: informational only
