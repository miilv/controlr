"""COIN2 under the one-general-system rule: the thin-object contact options name no task and read no task config or
checker, and the wall probe recovers a believed plane error from proprioceptive readings alone (kinematic env)."""
import ast
import inspect
import math
import re
from pathlib import Path

import numpy as np

from dojo.skills import coin_contact as CC

ROOT = Path(__file__).resolve().parents[2]


def _task_names():
    from dojo.agent.registry import TASKS
    return list(TASKS) + ["deposit_coin"]


def test_contact_module_and_switch_comments_name_no_task():
    src = inspect.getsource(CC)
    pol = (ROOT / "dojo/runtime/policy.py").read_text()
    a = pol.index("# the thin-object-in-holder contact sequence")
    b = pol.index('disc_reserve: str = "off"')
    block = pol[a:b]
    for t in _task_names():
        for form in (t, t.replace("_", " ")):
            assert form not in src, form
            assert form not in block, form


def test_contact_module_reads_no_task_config_layout_or_checker():
    tree = ast.parse(inspect.getsource(CC))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert all(m in ("math", "numpy", "__future__") or m.startswith("dojo.skills") for m in mods), mods
    code = re.sub(r'""".*?"""', "", inspect.getsource(CC), flags=re.S)        # code, not docstrings / comments
    code = re.sub(r"#[^\n]*", "", code)
    for word in ("task_info", "checker", "reward", "success_flag", "layout", "step_lim", "env.end"):
        assert word not in code, word


def _scene(true_yaw_deg: float):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.geometry import R_to_quat, rot_z
    Rz = rot_z(math.radians(true_yaw_deg))
    Rx = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])        # the disc's thin local z turned horizontal
    c = np.array([0.25, -0.15, 0.7805])
    rows = [{"label": "disc", "category": "thing", "pos": c.tolist(), "quat": R_to_quat(Rz @ Rx),
             "half": [0.01425, 0.01425, 0.00097]},
            {"label": "holder", "category": "stand", "type": "Geometry", "pos": [c[0], c[1], 0.7736],
             "quat": R_to_quat(Rz), "half": [0.016, 0.0055, 0.00855]}]
    return KinematicEnv(rows, step_lim=600, attach_radius=0.003), Rz @ Rx


def test_wall_probe_recovers_a_believed_yaw_error_from_jaw_readings_only():
    from dojo.skills.g3 import OffsetBeliefProvider
    from dojo.skills.skills import GTProvider, Skills
    true_yaw, err = 20.0, 8.0
    env, R_true = _scene(true_yaw)
    n_true = R_true[:, 2]
    prov = OffsetBeliefProvider(GTProvider(env, None), {"disc": [0.0, 0.0, err], "holder": [0.0, 0.0, err]})
    sk = Skills(env, prov)
    base = env.proprio

    def proprio():                                  # a synthetic strict-v2 official value: the twin's V model
        st = base()
        for arm in ("left", "right"):
            if env.g[arm] < 0.5:
                T = sk.kin.tcp(arm, sk.ex.q[arm])       # the commanded hand (the twin tracks it at the squeeze)
                psi = math.degrees(math.acos(min(1.0, abs(float(T[:3, 1] @ n_true)))))
                # yaw term only: the first squeeze's descent line leaves the commanded TCP ~0.9 mm off the centre line
                # (IK tolerance) and the fit assumes one offset for all squeezes (BACKLOG, COIN2 follow-up)
                st[f"{arm}_ee_joint_state_official"] = np.array([CC.WALL_A + 0.0019 * psi])
            else:
                st[f"{arm}_ee_joint_state_official"] = np.array([env.g[arm]])
        return st
    env.proprio = proprio
    sk.refresh()
    arm = next(a for a in ("right", "left")
               if sk.kin.ik(a, CC.wall_frame(sk, sk.world.obj("disc"), sk.world.obj("holder")), sk.ex.q[a]).ok)
    out = CC.wall_probe(sk, "disc", "holder", arm)
    assert out["ok"], out
    assert len(out["reads"]) >= 3 and abs(out["fit"]["apex"] + err) <= 0.5, out["fit"]
    w = sk.refresh()
    n_bel = w.obj("disc").R[:, 2]
    assert math.degrees(math.acos(min(1.0, abs(float(n_bel @ n_true))))) <= 0.5
    assert prov.writes and {x["writer"] for x in prov.writes} == {"wall_probe"}


def test_wall_probe_without_an_official_reading_writes_nothing():
    from dojo.skills.g3 import OffsetBeliefProvider
    from dojo.skills.skills import GTProvider, Skills
    env, _ = _scene(20.0)
    prov = OffsetBeliefProvider(GTProvider(env, None), {"disc": [0.0, 0.0, 8.0], "holder": [0.0, 0.0, 8.0]})
    sk = Skills(env, prov)
    sk.refresh()
    arm = next(a for a in ("right", "left")
               if sk.kin.ik(a, CC.wall_frame(sk, sk.world.obj("disc"), sk.world.obj("holder")), sk.ex.q[a]).ok)
    out = CC.wall_probe(sk, "disc", "holder", arm)
    assert not out["ok"] and "official" in out.get("why", "") and not getattr(prov, "writes", [])
