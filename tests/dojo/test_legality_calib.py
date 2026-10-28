"""Legality audit 2026-10-28, P1 finding 1 (robot-harness docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md): the
layout id chose the perception calibration (``dev_calib_fold`` = crc32(layout id) % 2 -> backgrounds / colour map).
Now the policy sees the instruction and the step limit only, every perception call names the same constant set, the
perception service maps every request to its ONE frozen set (``DOJO_CALIB_SET``), and the dev drivers refuse a layout
the set was built from. The regression test is the audit's: identical observations + instruction, only the hidden
reset id changed -> the same calibration and the same decisions."""

import json
import zlib

import numpy as np
import pytest

from dojo.runtime.firewall import sanitize_info
from dojo.runtime.policy import CALIB_FOLD, RuntimeConfig, RuntimePolicy
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]


def _env():
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [0.015] * 3} for i, p in enumerate(POS)]
    return KinematicEnv(rows, step_lim=550, instruction=STACK)


class FakeAssets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


class Episode:
    """proxy_eval's episode stand-in; ``info`` carries what the old dev driver handed over (task, layout id, the
    crc32 fold it derived), so a policy that read any of it would diverge between two ids."""

    def __init__(self, env, layout_id, task):
        self.env, self.layout_id, self.task = env, layout_id, task
        self.actions = []

    def info(self):
        return {"stable": True, "task": self.task, "layout_id": self.layout_id, "instruction": self.env.instruction,
                "step_lim": self.env.step_lim, "reset_s": 1.0, "render_stage_s": 0.5,
                "dev_calib_fold": zlib.crc32(str(self.layout_id).encode()) % 2,
                "layout_file": f"/dev/null/{self.layout_id}.settled.json"}

    def observe(self):
        img = np.zeros((48, 64, 3), np.uint8)
        return {"images": {c: img for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "state": self.env.proprio(), "instruction": self.env.instruction, "steps_used": self.env.steps_used,
                "step_lim": self.env.step_lim, "renderer": "fake"}


class FoldSensitivePerception:
    """A perception service whose answer depends on the calibration set it is asked for: set 1 sees every block 4 cm
    further +x (a different background / colour map would change the estimate in a real service)."""

    def __init__(self):
        self.folds = []

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        self.folds.append(("estimate", fold))
        dx = 0.04 if int(fold) == 1 else 0.0
        recs = [{"category": "block", "category_idx": 5, "label": f"block_{i}",
                 "default_pos": [p[0] + dx, p[1], p[2]], "default_ori": [1, 0, 0, 0],
                 "_perception": {"confidence": 0.9}} for i, p in enumerate(POS)]
        return {"Rigid": {"block": recs}}

    def check(self, frame, objects, fold):
        self.folds.append(("check", fold))
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []


def _run(layout_id, task):
    env = _env()
    ep = Episode(env, layout_id, task)
    pc = FoldSensitivePerception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"), assets=FakeAssets(), perception_client=pc)
    pol.reset(ep.info())
    stored = dict(pol.info)
    for _ in range(2000):
        d = pol.next(ep.observe())
        if d.get("done"):
            break
        for a in d["actions"]:
            ep.actions.append({k: np.round(np.asarray(v, float), 9).tolist() for k, v in a.items()})
            if env.step(a):
                break
        else:
            continue
        break
    return pol.episode_report(), pc.folds, ep.actions, stored


def test_sanitized_info_is_the_instruction_and_the_step_limit_only():
    keep, dropped = sanitize_info({"task": "stack_blocks", "layout_id": 100003, "dev_calib_fold": 1, "stable": True,
                                   "reset_s": 1.0, "render_stage_s": 0.2, "instruction": "i", "step_lim": 5,
                                   "layout_file": "/x.json", "env_seeds": [0]})
    assert keep == {"instruction": "i", "step_lim": 5}
    assert {"task", "layout_id", "dev_calib_fold", "layout_file", "env_seeds"} <= set(dropped)


def test_the_layout_id_never_changes_the_calibration_or_the_decisions():
    """The audit's probe ids: 100000 -> old fold 1, 100004 -> old fold 0. Same frames, proprio and instruction; only
    the reset's hidden fields differ (layout id, launcher task, the crc32 fold the old driver derived)."""
    assert zlib.crc32(b"100000") % 2 != zlib.crc32(b"100004") % 2
    rep_a, folds_a, acts_a, info_a = _run(100000, "stack_blocks")
    rep_b, folds_b, acts_b, info_b = _run(100004, "push_T")
    assert info_a == info_b == {"instruction": STACK, "step_lim": 550}
    assert folds_a and {f for _, f in folds_a} == {f for _, f in folds_b} == {CALIB_FOLD}
    assert rep_a["error"] is None and rep_b["error"] is None
    assert rep_a["decisions"] == rep_b["decisions"] and rep_a["skill"] == rep_b["skill"]
    assert acts_a == acts_b and len(acts_a) > 0
    assert rep_a["records"]["initial"] == rep_b["records"]["initial"]


def test_the_runtime_sources_read_no_task_layout_id_or_fold_from_info():
    from pathlib import Path
    import re
    root = Path(__file__).resolve().parents[2] / "dojo"
    pat = re.compile(r"""info(\.get\(|\[)["'](task|layout_id|dev_calib_fold|env_seeds)["']""")
    for f in [*(root / "runtime").glob("*.py"), *(root / "agent").glob("*.py"), *(root / "skills").glob("*.py")]:
        if f.name in ("eval.py",):           # the dev driver (scorer side): logs the id, never hands it over
            continue
        for ln in f.read_text().splitlines():
            assert not pat.search(ln), (f.name, ln)


def test_the_bridge_driver_resets_the_policy_with_the_instruction_and_step_limit_only():
    from dojo.bridge.driver import run_episode

    class Backend:
        steps_used, ended, stats, chunks, executed = 0, False, {}, [], []

        class c:
            n_calls, wall_s = 0, 0.0

        def reset(self, timeout_s):
            return {"instruction": STACK, "step_lim": 550}

        def observe(self):
            return {}

    class Pol:
        cfg = RuntimeConfig(mode="perception", verify="off")
        bridge = None

        def reset(self, info):
            self.info = dict(info)

        def next(self, obs):
            return {"done": True}

        def episode_report(self):
            return {}

    p = Pol()
    run_episode(p, Backend(), log=lambda *_: None)
    assert p.info == {"instruction": STACK, "step_lim": 550}


# ------------------------------------------------------------------ the service: one frozen set
def test_the_service_maps_every_request_to_its_one_frozen_set(tmp_path, monkeypatch):
    from dojo.runtime import perception_service as PS
    s = tmp_path / "frozen_x"
    s.mkdir()
    (s / "color_M.json").write_text(json.dumps({"name": "frozen_x", "M": np.eye(4, 3).tolist(), "n": 9}))
    (s / "manifest.json").write_text(json.dumps({"name": "frozen_x", "layout_ids": list(range(100010, 100020))}))
    monkeypatch.setenv("DOJO_BG_FOLDS", str(tmp_path))
    monkeypatch.setenv("DOJO_CALIB_SET", "frozen_x")
    monkeypatch.delenv("DOJO_COLOR_CALIB", raising=False)
    p = PS._Perception()
    assert p.set_name(0) == p.set_name(1) == "frozen_x"
    assert list(p._cal) == ["frozen_x"] and np.allclose(p._cal["frozen_x"], np.eye(4, 3))
    info = PS.calib_info()
    assert info["set"] == "frozen_x" and info["color_key"] == "color_M.json"
    assert info["manifest"]["layout_ids"][0] == 100010
    monkeypatch.delenv("DOJO_CALIB_SET")
    legacy = PS._Perception()
    assert legacy.set_name(1) == "fold1"            # offline dev tools still name a D0 half explicitly


def test_the_default_colour_map_follows_the_frozen_set(tmp_path, monkeypatch):
    from dojo.perception import identity
    s = tmp_path / "frozen_x"
    s.mkdir()
    M = (np.eye(4, 3) * 0.5).tolist()
    (s / "color_M.json").write_text(json.dumps({"name": "frozen_x", "M": M, "n": 9}))
    cal = tmp_path / "color_calib.json"
    cal.write_text(json.dumps({"all": np.eye(4, 3).tolist(), "fold0": np.eye(4, 3).tolist()}))
    monkeypatch.setattr(identity, "_M_CACHE", {})
    monkeypatch.setenv("DOJO_COLOR_CALIB", str(cal))
    monkeypatch.setenv("DOJO_BG_FOLDS", str(tmp_path))
    monkeypatch.setenv("DOJO_CALIB_SET", "frozen_x")
    assert np.allclose(identity.load_color_M(), M)
    assert np.allclose(identity.load_color_M(which="fold0"), np.eye(4, 3))       # explicit dev choice unchanged
    monkeypatch.delenv("DOJO_CALIB_SET")
    assert np.allclose(identity.load_color_M(), np.eye(4, 3))                     # legacy: "all"


# ------------------------------------------------------------------ the frozen set and the drivers' check
def test_a_frozen_set_is_built_from_the_calibration_only_slice_and_nothing_else():
    from dojo.perception.calib_set import DEFAULT_IDS, check_buildable, parse_ids
    from dojo.simlab import paths
    assert parse_ids("100040-100042,100045") == [100040, 100041, 100042, 100045]
    assert parse_ids(DEFAULT_IDS) == list(range(100040, 100050))
    check_buildable(parse_ids(DEFAULT_IDS))                       # D-C10b: calibration-only, never evaluated
    # our measurement slices, Mikhail's slices (D-C10a), reserved ids, a mix, nothing
    for bad in ("100000-100009", "100005", "100010-100019", "100020-100039", "100039-100040", "100050", "100100", ""):
        with pytest.raises(ValueError):
            check_buildable(parse_ids(bad))
    assert all(paths.is_calibration_id(i) and not paths.is_our_measurement_id(i) and not paths.is_offslice_rendered(i)
               for i in range(100040, 100050))
    assert not paths.is_calibration_id(100039) and not paths.is_calibration_id(100050)


def test_the_dev_drivers_refuse_a_layout_the_set_was_built_from():
    from dojo.perception.calib_set import overlap
    from dojo.runtime.eval import calib_overlap
    man = {"name": "frozen_x", "layout_ids": [100040, 100041]}
    assert overlap(man, "100040") is True and overlap(man, 100003) is False and overlap(None, 100003) is None
    assert calib_overlap({"set": "frozen_x", "manifest": man}, "100041") is True
    assert calib_overlap({"set": "frozen_x", "manifest": man}, "100002") is False
    assert calib_overlap({"set": "fold0", "manifest": None}, "100002") is None      # legacy dir: unknown
    assert calib_overlap({"set": None, "manifest": None}, "100002") is None


def test_build_writes_the_set_and_its_manifest_from_the_named_ids_only(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from dojo.perception import calib_set, eval_g1
    d0 = tmp_path / "d0"
    for task in ("stack_blocks", "push_T"):
        for i in (100003, 100040, 100041):
            e = d0 / task / str(i)
            e.mkdir(parents=True)
            (e / "frame_0.json").write_text("{}")
    seen = {}

    def fake_bgs(eps, rng, max_n=40, n_clean=80):
        seen["eps"] = sorted(f"{e.parent.name}/{e.name}" for e in eps)
        img = np.zeros((4, 4, 3), np.uint8)
        return {"cam_head": img, "cam_left_wrist": img, "cam_right_wrist": img, "cam_head_clean": img,
                "cam_head_clean_unseen": np.zeros((4, 4), np.uint8)}

    monkeypatch.setattr(eval_g1, "backgrounds_from", fake_bgs)
    man = calib_set.build(d0, tmp_path / "bg", "frozen_t", [100040, 100041], colour=False)
    assert seen["eps"] == ["push_T/100040", "push_T/100041", "stack_blocks/100040", "stack_blocks/100041"]
    assert man["layout_ids"] == [100040, 100041] and man["episodes"] == 4
    assert (tmp_path / "bg" / "frozen_t" / "cam_head.png").exists()
    assert calib_set.load_manifest(tmp_path / "bg" / "frozen_t")["layout_ids"] == [100040, 100041]
    with pytest.raises(FileExistsError):                          # never rebuilt in place
        calib_set.build(d0, tmp_path / "bg", "frozen_t", [100040], colour=False)
    with pytest.raises(ValueError):
        calib_set.build(d0, tmp_path / "bg", "frozen_bad", [100003, 100040], colour=False)


def test_the_drivers_refuse_to_evaluate_a_calibration_only_layout(tmp_path):
    """D-C10b: 100040-49 calibrate perception and are never evaluated (integ, official and proxy drivers)."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for script in ("scripts/dojo/integ_eval.sh", "scripts/dojo/official_eval.sh"):
        assert "100040 && id <= 100049" in (root / script).read_text() or \
            "lid >= 100040 && lid <= 100049" in (root / script).read_text(), script
    for mod in ("dojo/runtime/eval.py", "dojo/simlab/proxy_eval.py"):
        assert "is_calibration_id" in (root / mod).read_text(), mod
    r = subprocess.run(["bash", str(root / "scripts/dojo/integ_eval.sh"), "--task", "stack_blocks",
                        "--out", str(tmp_path / "o"), "--layouts", "/nowhere/100043.settled.json"],
                       capture_output=True, text=True, timeout=60, env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 2 and "calibration-only" in r.stderr, r.stderr[-300:]


def test_every_box_gets_the_pinned_set_or_the_install_step_fails(tmp_path):
    """box_steps.sh calib_sets: the set under ~/dojo-perception-data must match its pinned md5; it is copied into the
    integration roots (~/dojo-data/<dir>_ilia) and verified there; a different set fails the step."""
    import hashlib
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    perc, data = tmp_path / "perc", tmp_path / "data"
    s = perc / "bg_folds_51" / "frozen_x"
    s.mkdir(parents=True)
    (s / "cam_head.png").write_bytes(b"png")
    (s / "manifest.json").write_text('{"layout_ids": [100040]}')
    (data / "bg_folds_51_ilia").mkdir(parents=True)
    listing = "".join(f"{hashlib.md5((s / n).read_bytes()).hexdigest()}  {n}\n"
                      for n in sorted(p.name for p in s.iterdir()))
    pins = tmp_path / "pins.txt"
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "DOJO_PERCEPTION_DATA": str(perc), "DOJO_DATA": str(data),
           "CALIB_SETS_PINS": str(pins)}

    def run():
        return subprocess.run(["bash", str(root / "scripts/infra/box_steps.sh"), "fn", "step_calib_sets"],
                              capture_output=True, text=True, timeout=60, env=env)
    pins.write_text(f"# test\nbg_folds_51 frozen_x {hashlib.md5(listing.encode()).hexdigest()}\n")
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert (data / "bg_folds_51_ilia" / "frozen_x" / "cam_head.png").read_bytes() == b"png"
    (data / "bg_folds_51_ilia" / "frozen_x" / "cam_head.png").write_bytes(b"drifted")   # a root that drifted is fixed
    assert run().returncode == 0 and (data / "bg_folds_51_ilia" / "frozen_x" / "cam_head.png").read_bytes() == b"png"
    pins.write_text("bg_folds_51 frozen_x 00000000000000000000000000000000\n")
    r = run()
    assert r.returncode != 0 and "pinned" in (r.stdout + r.stderr)
    # a box whose ~/dojo-perception-data stays untouched: the set staged under ~/dojo-bootstrap/calib_stage, installed
    # into a per-flavour extra root only
    stage = tmp_path / "dojo-bootstrap" / "calib_stage" / "bg_folds_51"
    stage.mkdir(parents=True)
    s.rename(stage / "frozen_x")
    extra = tmp_path / "pdata_51"
    extra.mkdir()
    pins.write_text(f"bg_folds_51 frozen_x {hashlib.md5(listing.encode()).hexdigest()}\n")
    env["CALIB_EXTRA_ROOTS"] = f"bg_folds={tmp_path / 'nope'} bg_folds_51={extra}"
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert (extra / "frozen_x" / "cam_head.png").read_bytes() == b"png" and not (perc / "bg_folds_51" / "frozen_x").exists()
