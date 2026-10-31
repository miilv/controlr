"""Headless RoboDojo runner: one task env in-process, loaded from OUR layout records.

Runs inside RoboDojo's conda python (Isaac Sim 5.1 + IsaacLab) with ``RD_ROOT`` and
``RD_ROOT/XPolicyLab`` importable (``scripts/simlab/rdpy.sh`` sets that up). The env is
RoboDojo's own ``EvalEnv`` (``src/eval_client/eval_env.py``) built the way
``src/eval_client/main.py`` builds it, with exactly two substitutions:

* ``SeedManager`` -> ``DictSeedManager``: layouts come from a dict we fill, never from
  ``Assets/Eval_Layout`` (the directory is not listed or opened);
* ``WsModelClient`` -> a no-op: there is no policy server; we drive ``take_action`` ourselves.

Everything else (reset sequence, settle + stability check, action interpolation, step
accounting, reward latching, is_episode_end) is RoboDojo's code path, so episodes here behave
like eval episodes.

Two strictly separated read paths:

* ``obs()``: the official policy contract, what ``TASK_ENV.get_obs()`` gives a policy:
  3 RGB 480x640 (cam_head, cam_left_wrist, cam_right_wrist), ``{left,right}_arm_joint_state``
  (6), ``{left,right}_ee_pose`` (flange link, 7: xyz + wxyz, env frame),
  ``{left,right}_ee_joint_state`` (commanded gripper opening 0..1), instruction.
* ``gt()``: DEV-ONLY ground truth: every object's pose/velocity, articulation joints, camera
  K and extrinsics, measured gripper joints, finger contacts, reward/score state.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dojo.simlab import paths
from dojo.simlab.layout import OBJECT_TYPES, iter_objects, quat_angle_deg

ARMS = ("left", "right")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])      # USD camera (-z forward, y up) -> OpenCV
KIT_ENABLE_EXTS = ("isaacsim.replicator.behavior", "isaacsim.sensors.camera")

_APP = None


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# Kit keeps only its last 30 log files (``/log/fileMaxCount``, ~40 min of busy boxes): keep more for after-the-fact greps
# (the per-episode check reads the process's own log as it goes: ``dojo.simlab.kitlog``)
KIT_LOG_SETTINGS = ("--/log/fileMaxCount=200",)
PHYSICS_ONLY_SETTINGS = ("--/app/extensions/fsWatcherEnabled=0",) + KIT_LOG_SETTINGS


def physics_only_experience(out_dir: Path) -> Path:
    """IsaacLab's ``isaaclab.python.headless.kit`` without the RTX renderer, MDL and UI deps, with
    ``${app}`` paths made absolute. Kit then never loads ``librtx.scenedb`` (which segfaults at
    startup with Isaac Sim 5.1 on driver 595 + RTX 5090, compute3)."""
    il = paths.RD_ROOT / "third_party" / "IsaacLab"
    src = (il / "apps" / "isaaclab.python.headless.kit").read_text()
    out = []
    for line in src.splitlines():
        t = line.strip()
        if t.startswith('"omni.kit.usd.mdl"') or t.startswith('"omni.usd.metrics.assembler.ui"'):
            continue
        if t.startswith("renderer.enabled"):
            line = 'renderer.enabled = ""'
        line = line.replace('"${app}/../source"', f'"{il / "source"}"').replace('"${app}",', f'"{il / "apps"}",')
        out.append(line)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "simlab.physics.kit"
    p.write_text("\n".join(out) + "\n")
    return p


def install_camera_stubs() -> None:
    """Physics-only mode: replace RoboDojo's camera / capture / render-sync modules and the one
    replicator helper it uses (MDL material creation) before anything imports them. Every RoboDojo camera is a bare ``pinhole`` Xform without collision, so the physics
    scene is unchanged; observations simply have no ``vision`` entries."""
    import types

    class CameraManager:
        def __init__(self, num_envs, camera_config, device, seeds_per_env=None):
            self.num_envs, self.num_cams = num_envs, 0
            self.camera_names = [[] for _ in range(num_envs)]
            self.cameras_xform = [[] for _ in range(num_envs)]

        def update_env_seeds(self, seeds):
            pass

        def initialize(self, sim):
            self.sim = sim

        def post_init(self):
            pass

        def reset(self):
            pass

        def destroy(self):
            pass

    class TiledCaptureManager:
        def __init__(self, *a, **k):
            pass

        def initialize(self, sim):
            pass

        def step(self, env_ids=None, cam_ids=None):
            return []

        def reset(self, *a, **k):
            pass

        def destroy(self):
            pass

    def create_mdl_material(mtl_url: str, mtl_name: str, mtl_path: str):
        """USD-only stand-in for isaacsim.replicator.behavior's helper (which needs the MDL/RTX
        stack): a Material with an MDL shader prim, as CreateMdlMaterialPrim authors it."""
        import omni.usd
        from pxr import UsdShade
        stage = omni.usd.get_context().get_stage()
        mtl = UsdShade.Material.Define(stage, mtl_path)
        sh = UsdShade.Shader.Define(stage, mtl_path + "/Shader")
        sh.SetSourceAsset(mtl_url, "mdl")
        sh.SetSourceAssetSubIdentifier(mtl_name, "mdl")
        mtl.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return mtl

    for pkg in ("isaacsim.replicator", "isaacsim.replicator.behavior", "isaacsim.replicator.behavior.utils"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    su = types.ModuleType("isaacsim.replicator.behavior.utils.scene_utils")
    su.create_mdl_material = create_mdl_material
    sys.modules[su.__name__] = su

    cm = types.ModuleType("env.camera_manager.camera_manager")
    cm.CameraManager = CameraManager
    tc = types.ModuleType("env.camera_manager.capture.tiled_capture_manager")
    tc.TiledCaptureManager = TiledCaptureManager
    rs = types.ModuleType("env.camera_manager.capture.render_sync")
    rs.wait_for_latest_cameras = lambda *a, **k: None
    rs.add_zero_delay_kit_args = lambda args: None
    rs.zero_delay_active = lambda: False
    sys.modules.update({cm.__name__: cm, tc.__name__: tc, rs.__name__: rs})


def config_spawns_garment(task: str) -> bool:
    """Does the task's config spawn a Garment (particle cloth)? Keyed by the object type in the config."""
    try:
        txt = (paths.TASK_CONFIGS / f"{task}.yml").read_text()
    except OSError:
        return False
    return any(line.startswith("Garment:") for line in txt.splitlines())


# The physics-only kit writes no simulation state back to USD. RoboDojo reads a CPU-device garment's particles from
# its USD ``points`` (``GarmentObject.sample_mesh_vertices``), so in physics-only mode the cloth read as its spawn mesh
# forever (fold_clothes: every checker point frozen 6-11 cm above the table, a pinched sleeve "never moved"). These
# turn the write-back on; the rendered kit (the official eval) writes USD anyway.
GARMENT_KIT_SETTINGS = ("--/physics/updateToUsd=true", "--/physics/updateParticlesToUsd=true")


def physics_only_kit_settings(extra_settings: tuple = ()) -> tuple:
    """The physics-only kit's settings: ours, then ``extra_settings``, then ``SIMLAB_PHYSICS_KIT_ARGS`` (dev, space
    separated; unset: none)."""
    return PHYSICS_ONLY_SETTINGS + tuple(extra_settings) + tuple(os.environ.get("SIMLAB_PHYSICS_KIT_ARGS", "").split())


def launch_app(headless: bool = True, device_id: int = 0, physics_only: bool = False, extra_settings: tuple = ()):
    """Start Kit once per process. Default: the eval client's flags (headless, cameras, the two
    sensor extensions, zero-delay render sync). ``physics_only``: no renderer at all (see
    ``physics_only_experience``); observations then carry no images."""
    global _APP
    if _APP is not None:
        return _APP
    for p in (str(paths.RD_ROOT), str(paths.RD_ROOT / "XPolicyLab")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    if physics_only:
        install_camera_stubs()
        kit = physics_only_experience(paths.DATA / "simlab" / "kit")
        argv = ["--headless", "--experience", str(kit), "--kit_args=" + " ".join(physics_only_kit_settings(extra_settings))]
        args = ap.parse_args(argv)
        # RoboDojo's IsaacLab fork defaults enable_cameras=True (offscreen rendering) and its
        # ENABLE_CAMERAS env check never fires (int compared with `is False`): set the arg.
        args.enable_cameras = False
    else:
        kit = " ".join(f"--enable {e}" for e in KIT_ENABLE_EXTS) + " " + " ".join(KIT_LOG_SETTINGS)
        if os.environ.get("SIMLAB_KIT_ARGS"):          # e.g. --portable-root <dir>: keep Kit's data out of a shared install
            kit += " " + os.environ["SIMLAB_KIT_ARGS"]
        argv = ["--enable_cameras", "--kit_args=" + kit]
        if headless:
            argv.append("--headless")
        args = ap.parse_args(argv)
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        add_zero_delay_kit_args(args)
    _APP = AppLauncher(args).app
    return _APP


class _NoPolicyClient:
    def __init__(self, *a, **k):
        pass

    def call(self, *a, **k):
        return None

    def close(self):
        pass


class DictSeedManager:
    """SeedManager stand-in: layout id -> record, from memory. Never touches Eval_Layout."""

    def __init__(self, config):
        self.config = config
        self.num_envs = int(config["num_envs"])
        self.task_name = str(config["task_name"])
        self.config_name = str(config["config_name"])
        self.layouts: dict[int, dict] = {}
        self.seed_info: dict[int, dict] = {}
        self.seed_list: list[int] = []
        self.idx = self.st_idx = self.ed_idx = 0
        self.type = "dev"

    def init_eval(self, completed_layout_ids=None, abandoned_layout_ids=None):
        return None

    def get_seeds(self, max_count=None):
        return None

    def get_seed_scene_info(self, seed: int) -> dict:
        if int(seed) not in self.layouts:
            raise KeyError(f"layout {seed} not loaded into DictSeedManager")
        return copy.deepcopy(self.layouts[int(seed)])

    def eval_step(self):
        pass


def physics_link_pose(art, link_name: str):
    """[x, y, z, qw, qx, qy, qz] (world) of an articulation link from its physics view, or None."""
    av = getattr(art, "_articulation_view", None)
    pv = getattr(av, "_physics_view", None)
    if av is None or pv is None:
        return None
    names = list(getattr(av, "body_names", None) or [])
    if link_name not in names:
        return None
    T = pv.get_link_transforms()
    if hasattr(T, "numpy") and not isinstance(T, np.ndarray):
        try:
            T = T.numpy()
        except Exception:  # noqa: BLE001 (a torch tensor on cuda)
            T = T.cpu().numpy()
    T = np.asarray(T, dtype=np.float64).reshape(-1, len(names), 7)[0, names.index(link_name)]
    return np.array([T[0], T[1], T[2], T[6], T[3], T[4], T[5]], dtype=np.float32)   # xyz + quat xyzw -> wxyz


def install_physics_link_poses() -> None:
    """``ArticulationObject.get_link_pose`` from physics (``usd_sync_links``); the USD read stays the fallback."""
    from env.scene_manager.objects import articulation as A
    if getattr(A.ArticulationObject.get_link_pose, "_simlab_physics", False):
        return
    usd = A.ArticulationObject.get_link_pose

    def get_link_pose(self, link_name: str):
        try:
            p = physics_link_pose(self, link_name)
        except Exception:  # noqa: BLE001
            p = None
        return p if p is not None else usd(self, link_name)
    get_link_pose._simlab_physics = True
    A.ArticulationObject.get_link_pose = get_link_pose


def build_env_cfg(task: str, num_envs: int = 1, device_id: int = 0, eval_cfg_type: str = "arx_x5",
                  need_planner: bool = True, support_planner: bool = False):
    """``src/eval_client/main.py::main`` config assembly, minus the policy server. ``need_planner=False``
    skips RoboDojo's cuRobo planners (replay-only runs; physics is unaffected). ``support_planner`` also
    builds one for the support arm (RoboDojo ships it planner-less; ``demo_gen`` plans its demonstrations)."""
    from omegaconf import OmegaConf

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs

    task_registry = __import__(f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"])
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, eval_cfg_type + ".yml"))
    eval_cfg.update({"task_name": task, "num_envs": num_envs, "device_id": device_id, "eval_batch": num_envs > 1,
                     "policy_name": "simlab", "additional_info": "simlab", "seed": 0,
                     "physx_monitor_enabled": False})
    deploy_cfg = {"policy_name": "simlab", "port": 0, "host": "localhost", "protocol": "ws",
                  "policy_server_url": "ws://localhost:0", "evaluation_id": "simlab", "trial_id": "simlab",
                  "action_case_id": "simlab", "repeat_index": None}
    cfgp = lambda kind: os.path.join(ENV_CONFIG_PATH, kind, eval_cfg["config"][kind] + ".yml")  # noqa: E731
    env_cfg = OmegaConf.create({
        "sim": load_yaml(cfgp("sim")), "scene": load_yaml(cfgp("scene")), "camera": load_yaml(cfgp("camera")),
        "robot": load_yaml(cfgp("robot")),
        "task_env": load_yaml(task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"),
                                                             task)),
        "eval_cfg": eval_cfg, "deploy_cfg": deploy_cfg})
    n = resolve_random_task_num_envs(task, num_envs, env_cfg.sim)
    eval_cfg["num_envs"] = n
    OmegaConf.update(env_cfg, "sim.scene.num_envs", n, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", n, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task)
    OmegaConf.update(env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0),
                     force_add=True)
    env_cfg.sim.seed = [0 for _ in range(n)]
    if not need_planner:
        for rc in env_cfg.robot.robots:
            rc["need_planner"] = False
    elif support_planner:
        for rc in env_cfg.robot.robots:
            if rc.get("type") == "support":
                rc["need_planner"] = True
    return env_cfg


DEMO_TASKS = ("imitate_sorting_sequence",)
# RoboDojo keys a task's random draws (make_kong's discard group, tic-tac-toe's opponent cell, the instruction's
# wording) and imitate's demo file by the ENV seed. Upstream SeedManager (and dev_eval) seeds with the layout's
# POSITION in the eval call (0..N-1); this runner seeds with our layout key unless SIMLAB_ENV_SEED=index.
ENV_SEED_MODES = ("layout_id", "index")
# env seed -> our layout key, where they differ (dev_eval; SIMLAB_ENV_SEED=index): the demo loader translates
DEMO_KEY: dict[int, int] = {}
# our layout id -> a staged demo file that replaces DATA/simlab/traj (the sealed block; md5-checked when registered)
DEMO_FILE: dict[int, Path] = {}


def env_seed_mode() -> str:
    """``SIMLAB_ENV_SEED``: ``layout_id`` (default: the env seed is our layout key) or ``index`` (upstream: the
    layout's position within the ``load`` call, 0..num_envs-1)."""
    mode = os.environ.get("SIMLAB_ENV_SEED", "layout_id") or "layout_id"
    if mode not in ENV_SEED_MODES:
        raise ValueError(f"SIMLAB_ENV_SEED={mode!r}: one of {ENV_SEED_MODES}")
    return mode


def dev_demo(task: str, key: int) -> dict:
    """The support-arm demonstration for OUR layout ``key`` (seed, or seed * 100 + env when envs share one).

    RoboDojo reads it from ``Assets/Traj/<bench>/<task>/<eval_seed>/<seed>.pkl``: files paired with the EVAL
    layouts (``target_label`` is the order the policy must reproduce). We never read those. A dev demo
    comes from ``DATA/simlab/traj/<task>/<seed>.pkl`` (same format), else a no-demo stub: labels in a
    seed-fixed order and an empty trajectory. The support arm then stays home, so RoboDojo's post-demo check
    flags the env unstable as soon as an episode steps. That is fine for pre-settling.
    """
    import pickle
    seed = key // 100 if key >= 100 * paths.SEED_BASE else key
    if seed < paths.SEED_BASE:
        raise RuntimeError(f"{task}: seed {seed} is not a dev seed; eval-paired demos are off limits")
    if seed in DEMO_FILE:                             # a staged (sealed) demo: it exists, or this fails loudly
        with open(DEMO_FILE[seed], "rb") as fh:
            return pickle.load(fh)
    f = paths.DATA / "simlab" / "traj" / task / f"{seed}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    order = demo_order(seed)
    return {"joint_path": [], "target_label": [f"t{i}" for i in order], "aim_label": [f"aim{i}" for i in order],
            "target_place_tag": [None] * 5, "_simlab_no_demo": True}


def demo_order(seed: int, n: int = 5) -> list[int]:
    """The sorting order of a dev layout's demonstration (seed-fixed; ``demo_gen`` plans this order)."""
    import random as _random
    order = list(range(n))
    _random.Random(seed).shuffle(order)
    return order


def env_seed_keys(layout_keys: list[int]) -> list[int]:
    """The env seeds ``load`` resets with (``SIMLAB_ENV_SEED``): our layout keys, or their positions 0..n-1 with
    ``DEMO_KEY`` mapping each position back to its layout key (imitate's demo)."""
    DEMO_KEY.clear()
    if env_seed_mode() == "layout_id":
        return list(layout_keys)
    DEMO_KEY.update(enumerate(layout_keys))
    return list(range(len(layout_keys)))


def salted_demo_keys(keys: list[int], layouts: list[dict]) -> None:
    """A SALTED record (``_simlab.seed_salt``) resets with its salted ``_simlab.seed`` (~1e9, ``gen_layouts.env_seed``),
    not its layout id, while its demo is ``traj/<task>/<id>.pkl`` (demo_gen keys it by the id). Without this map
    ``dev_demo`` read the salted seed as ``seed * 100 + env`` and served the no-demo stub, so demo_gen's validation of a
    salted imitate record always failed (D-C19 generation, 30 Oct: 20 / 20 dropped). In both env-seed modes the env
    seed then maps to ``_simlab.id``. Unsalted records (seed == id) are left exactly as before."""
    for k, lay in zip(keys, layouts):
        meta = lay.get("_simlab") or {}
        if meta.get("seed_salt") and meta.get("id") is not None and DEMO_KEY.get(k, k) != int(meta["id"]):
            DEMO_KEY[int(k)] = int(meta["id"])


def _routed_load_pkl(orig):
    def load_pkl(path):
        p = Path(str(path))
        if paths.eval_asset_violation(p) and "Traj" in p.parts:
            task = next((t for t in DEMO_TASKS if t in p.parts), None)
            if task is None:
                raise PermissionError(f"eval-paired demo with no dev route: {p}")
            key = int(p.stem)                         # the ENV seed: our layout key unless DEMO_KEY maps it
            return dev_demo(task, DEMO_KEY.get(key, key))
        return orig(path)
    load_pkl._simlab_routed = True
    return load_pkl


def install_dev_demo_loader() -> int:
    """Route RoboDojo's ``load_pkl`` of eval-paired Traj demos to ``dev_demo``: the shared
    ``utils.load_file.load_pkl`` (task modules imported later bind it) and any task module already imported."""
    import utils.load_file as lf
    n = 0
    for m in [lf] + [m for name, m in list(sys.modules.items()) if ".tasks." in name or name.startswith("tasks.")]:
        f = getattr(m, "load_pkl", None)
        if f is not None and not getattr(f, "_simlab_routed", False):
            m.load_pkl = _routed_load_pkl(f)
            n += 1
    return n


def resolve_usd_sync_links(flag: bool | None, physics_only: bool) -> bool:
    """An explicit ``flag`` wins, then ``SIMLAB_USD_SYNC_LINKS=0/1``, then ``physics_only`` (see RoboDojoRunner)."""
    if flag is not None:
        return bool(flag)
    env = os.environ.get("SIMLAB_USD_SYNC_LINKS")
    return env == "1" if env in ("0", "1") else bool(physics_only)


class RoboDojoRunner:
    """One RoboDojo task env (``num_envs`` parallel copies) driven from our layout records."""

    def __init__(self, task: str, num_envs: int = 1, device_id: int = 0, headless: bool = True,
                 video_dir: str | None = None, workdir: str | None = None, need_planner: bool = True,
                 physics_only: bool = False, support_planner: bool = False, usd_sync_links: bool | None = None):
        self.task = task
        self.physics_only = physics_only
        # usd_sync_links (default: ON when physics_only; env SIMLAB_USD_SYNC_LINKS=0/1 overrides): RoboDojo's checker
        # reads articulation link poses from USD (ArticulationObject.get_link_pose). The physics-only kit never writes
        # physics back to USD (updateToUsd = false, omni.physx.fabric loaded): moved articulations (a laptop) keep
        # their spawn pose and joint-rooted ones (toasters 00002/00004) read identity. The official rendered kit has
        # updateToUsd = true, USD = physics on every link (probed on isr-lab). ON: get_link_pose answers from the
        # articulation's physics view, as the official env effectively does.
        self.usd_sync_links = resolve_usd_sync_links(usd_sync_links, physics_only)
        paths.install_asset_firewall()                 # no Eval_Layout, no eval-paired Traj, from here on
        # a garment's particles are read from USD: physics-only writes them back for configs that spawn one
        self.garment_usd = bool(physics_only and config_spawns_garment(task))
        self.app = launch_app(headless=headless, device_id=device_id, physics_only=physics_only,
                              extra_settings=GARMENT_KIT_SETTINGS if self.garment_usd else ())
        # EvalEnv writes eval_result/... relative to cwd: keep that out of the shared install.
        wd = Path(workdir or (paths.DATA / "simlab" / "work"))
        paths.assert_not_eval(wd)
        wd.mkdir(parents=True, exist_ok=True)
        os.chdir(wd)
        os.environ.setdefault("ROBODOJO_RUN_ID", time.strftime("%Y-%m-%d_%H-%M-%S"))
        import src.eval_client.eval_env as ee
        ee.SeedManager = DictSeedManager
        ee.WsModelClient = _NoPolicyClient
        self.env_cfg = build_env_cfg(task, num_envs=num_envs, device_id=device_id, need_planner=need_planner,
                                     support_planner=support_planner)
        self.num_envs = int(self.env_cfg.sim.scene.num_envs)
        install_dev_demo_loader()                      # imitate_sorting_sequence: our demos, never the eval ones
        self.env = ee.create_eval_env(self.env_cfg, self.app)
        install_dev_demo_loader()                      # task modules imported by create_eval_env
        if self.usd_sync_links:
            install_physics_link_poses()
        if not video_dir:
            self.env._stream_vision = lambda env_idx, frame: None     # no mp4 streaming on get_obs
        self.layouts: list[dict] = [{} for _ in range(self.num_envs)]
        self.seeds: list[int] = [0] * self.num_envs
        self.timing: dict[str, float] = {}
        self._contact_sub = None
        self._contacts: dict[str, set] = {}

    # ------------------------------------------------------------------ loading
    def load(self, layouts: dict | list[dict], seeds: list[int] | None = None, fresh: bool = True) -> dict:
        """Reset the env into ``layouts`` (one record per env). ``fresh`` closes the sim first, as
        RoboDojo's eval loop does between batches (needed when the object SET changes on GPU
        physics; RoboDojo's sim config runs PhysX on CPU, where respawning is enough)."""
        from utils.cluttered_generator import UnStableError

        if isinstance(layouts, dict):
            layouts = [layouts] * self.num_envs
        if len(layouts) != self.num_envs:
            raise ValueError(f"{len(layouts)} layouts for {self.num_envs} envs")
        if seeds is None:
            seeds = [int((l.get("_simlab") or {}).get("seed", paths.SEED_BASE)) for l in layouts]
        # distinct keys per env even if two envs share a seed
        keys = [s * 100 + i for i, s in enumerate(seeds)] if len(set(seeds)) < len(seeds) else list(seeds)
        keys = env_seed_keys(keys)
        salted_demo_keys(keys, layouts)
        sm = self.env.seed_manager
        sm.layouts.clear()
        for k, lay in zip(keys, layouts):
            sm.layouts[k] = {kk: v for kk, v in lay.items() if not kk.startswith("_")}
        self.layouts = [copy.deepcopy(l) for l in layouts]
        self.seeds = list(seeds)
        t0 = time.perf_counter()
        if fresh and self.env.sim is not None:
            self.env.close()
        t1 = time.perf_counter()
        stable = True
        try:
            self.env.reset(seed=keys)
        except UnStableError:
            stable = False
        t2 = time.perf_counter()
        if stable:
            self.env.run_reward()
            if hasattr(self.env, "get_score"):
                self.env.get_score()
            if getattr(self.env, "interact", False) and hasattr(self.env, "query_support_arm_traj"):
                for i in range(self.num_envs):          # as EvalEnv.run_eval: the support arm starts at step 1
                    self.env.query_support_arm_traj(env_idx=i)
        self.timing.update({"close_s": t1 - t0, "reset_s": t2 - t1})
        self._contact_sub = None
        return {"stable": stable and all(self.env.success), "per_env_ok": list(map(bool, self.env.success)),
                "close_s": t1 - t0, "reset_s": t2 - t1,
                "instruction": [self.env.obs_manager.instruction[i] for i in range(self.num_envs)]}

    # ------------------------------------------------------------------ official observation
    def obs(self, env_idx: int = 0) -> dict:
        """Exactly the policy contract (``EvalEnv.get_obs`` minus mp4 streaming)."""
        return self.obs_batch([env_idx])[0]

    def obs_batch(self, env_idx_list: list[int] | None = None) -> list[dict]:
        env = self.env
        env_idx_list = list(range(self.num_envs)) if env_idx_list is None else env_idx_list
        env.obs_manager.render_for_capture()
        data = env.obs_manager.get_obs(env_idx_list=env_idx_list)
        out = []
        for i in env_idx_list:
            raw = data[i]
            st = raw.get("state") or {}
            out.append({
                "images": {n: np.ascontiguousarray(np.asarray(v["color"])[..., :3]).astype(np.uint8)
                           for n, v in (raw.get("vision") or {}).items()},
                "state": {k: _np(v).reshape(-1) for k, v in st.items()},
                "instruction": raw.get("instruction"),
            })
        return out

    # ------------------------------------------------------------------ actions
    def hold_action(self, env_idx: int = 0) -> dict:
        """The action that keeps every arm and gripper at its current commanded target."""
        prev = self.env.robot_manager.control_manager.prev_control[env_idx]
        st = self.obs_state(env_idx)
        act = {}
        for arm in ARMS:
            act[f"{arm}_arm_joint_state"] = np.asarray(st[f"{arm}_arm_joint_state"], np.float32)
            act[f"{arm}_ee_joint_state"] = np.asarray(st[f"{arm}_ee_joint_state"][:1], np.float32)
        del prev
        return act

    def obs_state(self, env_idx: int = 0) -> dict:
        """Proprio part of the official obs without rendering cameras (cheap)."""
        rm = self.env.robot_manager
        out = {}
        for robot in rm.robot_list:
            if robot.type != "target":
                continue
            arm = robot.arm_name.split("_")[0]
            out[f"{arm}_arm_joint_state"] = _np(rm.get_joint(robot, [env_idx])[env_idx])
            out[f"{arm}_ee_pose"] = _np(rm.get_real_endpose(robot, [env_idx])[env_idx])
            key = rm.process_name(robot.gripper_name)
            pos = rm.control_manager.prev_control[env_idx][key]["position"]
            val = float(pos[0])
            sc = robot.gripper_scale
            val = (val - sc[0]) / (sc[1] - sc[0]) if robot.gripper_move["sign"] == 1 else (sc[1] - val) / (sc[1] - sc[0])
            out[f"{arm}_ee_joint_state"] = np.array([val])
        return out

    def step(self, action: dict | list[dict]) -> list[bool]:
        """One counted env step (RoboDojo ``take_action``: 4 physics substeps of 4 ms at 25 Hz,
        joint-space interpolation). ``action`` is one dict (env 0 / all envs) or one per env.
        Returns the per-env end flags."""
        env = self.env
        if isinstance(action, dict):
            action = [action] * self.num_envs
        if self.num_envs == 1:
            env.take_action(action[0])
        else:
            # like RoboDojo's own batch policies: only envs still running (ended / step-limited ones
            # are skipped inside take_action_batch, and its control queue wants matching lists)
            idx = [i for i in range(self.num_envs) if not env.end_flag[i] and env.take_action_cnt[i] < env.step_lim]
            if idx:
                env.take_action_batch([action[i] for i in idx], env_idx_list=idx)
        if self._contact_sub is not None:
            pass
        return list(map(bool, env.end_flag))

    # ------------------------------------------------------------------ dev-only ground truth
    def reward(self, env_idx: int = 0) -> dict:
        env, rm = self.env, self.env.reward_manager
        return {"success": bool(env.success[env_idx]) and bool(env.end_flag[env_idx]),
                "success_flag": bool(env.success[env_idx]), "ended": bool(env.end_flag[env_idx]),
                "stages_left": len(rm.check_list[env_idx]), "final_left": len(rm.final_check_list[env_idx]),
                "queries": [[int(q[1]), int(q[2])] for q in rm.query_list[env_idx]],
                "steps_used": int(env.take_action_cnt[env_idx]), "step_lim": int(env.step_lim),
                # the task marked the episode invalid (make_kong's scripted discard failed, play_tic_tac_toe's opponent
                # did not place, imitate's demonstrator failed: mark_env_unstable). RoboDojo's eval client then drops
                # it from the eval (no result row, ``unstable_nums``): it is not a sample.
                "unstable": int(env_idx) in (getattr(env, "unstable_envs", None) or set())}

    def score(self, env_idx: int = 0) -> float:
        """Episode score as RoboDojo would book it now: 1.0 on success, else process score/100."""
        env = self.env
        if env.success[env_idx] and env.end_flag[env_idx]:
            return 1.0
        if not hasattr(env, "get_score"):
            return 0.0
        return float(env.reward_manager.get_score()[env_idx]) / 100.0

    def objects_gt(self, env_idx: int = 0) -> list[dict]:
        lm = self.env.scene_manager.layout_manager
        rows = []
        for otype in OBJECT_TYPES:
            for rec in lm.object_records_by_type[otype].layout_records_by_env[env_idx]:
                name = rec["inst_name"]
                pq = lm.get_instance_pose(env_idx=env_idx, inst_name=name)
                if pq is None:                     # RoboDojo only resolves rigid/articulation/garment/geometry
                    try:
                        pq = lm.get_scene_object(env_idx, name).get_local_pose()
                    except Exception:  # noqa: BLE001 (fluids: particles, no single pose)
                        pq = (None, None)
                pos, quat = pq
                row = {"inst_name": name, "prim_path": rec.get("prim_path"), "type": otype,
                       "category": lm.decode_inst_name(name)[0],
                       "category_idx": int(rec.get("category_idx", -1)), "label": rec.get("label"),
                       "clutter": rec.get("type") == "cluttered",
                       "pos": _np(pos).reshape(-1)[:3].tolist() if pos is not None else None,
                       "quat": _np(quat).reshape(-1)[:4].tolist() if quat is not None else None,
                       "layout_pos": list(rec["default_pos"]), "layout_quat": list(rec["default_ori"])}
                obj = lm.get_scene_object(env_idx, name)
                if otype in ("Rigid", "Dynamic"):
                    try:
                        row["lin_vel"] = _np(obj.get_linear_velocity()).reshape(-1).tolist()
                        row["ang_vel"] = _np(obj.get_angular_velocity()).reshape(-1).tolist()
                    except Exception:  # noqa: BLE001
                        pass
                if otype == "Articulation":
                    try:
                        row["joint_pos"] = _np(obj.get_current_joint_positions()).reshape(-1).tolist()
                    except Exception as e:  # noqa: BLE001
                        row["joint_error"] = str(e)
                rows.append(row)
        return rows

    def cameras_gt(self, env_idx: int = 0) -> dict:
        cm = self.env.camera_manager
        out = {}
        for i, name in enumerate(cm.camera_names[env_idx]):
            T = _np(cm.get_camera_extrinsics(i, env_idx))
            T_rel = T.copy()
            T_rel[:3, 3] -= _np(self.env.scene_manager.env_origins[env_idx]).reshape(-1)[:3]
            out[name] = {"K": _np(cm.get_camera_intrinsics(i, env_idx)).tolist(),
                         "T_world_cam_usd": T_rel.tolist(), "T_world_cam_cv": (T_rel @ CV_FROM_USD).tolist(),
                         "width": 640, "height": 480}
        return out

    def garment_gt(self, env_idx: int = 0, label: str | None = None) -> dict | None:
        """Dev-only GT of a garment: its particle positions (world, env-relative), the metadata's named points
        (``passive.functional`` particle ids -> mean position) and the root pose the checker takes its axes from."""
        lm = self.env.scene_manager.layout_manager
        for rec in lm.object_records_by_type["Garment"].layout_records_by_env[env_idx]:
            if label is not None and rec.get("label") != label:
                continue
            name = rec["inst_name"]
            g = lm.get_scene_object(env_idx, name)
            meta = lm.get_instance_metadata(inst_name=name, env_idx=env_idx) or {}
            P, _, pw, ow = g.sample_mesh_vertices()
            org = _np(self.env.scene_manager.env_origins[env_idx]).reshape(-1)[:3]
            P = _np(P).reshape(-1, 3) - org
            kp = {}
            for k, v in ((meta.get("passive") or {}).get("functional") or {}).items():
                ids = [int(i) for i in (v or {}).get("id", []) if 0 <= int(i) < len(P)]
                if ids:
                    kp[k] = P[ids].mean(0).tolist()
            pos, quat = lm.get_instance_pose(env_idx=env_idx, inst_name=name)
            return {"label": rec.get("label"), "inst_name": name, "points": P, "keypoints": kp,
                    "pos": _np(pos).reshape(-1)[:3].tolist() if pos is not None else None,
                    "quat": _np(quat).reshape(-1)[:4].tolist() if quat is not None else None}
        return None

    def robot_gt(self, env_idx: int = 0) -> dict:
        rm = self.env.robot_manager
        out = {}
        for robot in rm.robot_list:
            key = rm.robot_key[rm.robot_list.index(robot)]
            arm = robot.arm_name.split("_")[0] if robot.type == "target" else robot.robot_name
            jp = _np(key.data.joint_pos[env_idx])
            jv = _np(key.data.joint_vel[env_idx])
            rec = {"type": robot.type, "joint_names": list(key.joint_names), "joint_pos": jp.tolist(),
                   "joint_vel": jv.tolist(), "flange": _np(rm.get_real_endpose(robot, [env_idx])[env_idx]).tolist()}
            if getattr(robot, "ee_type", None) == "gripper":
                gi = list(robot.gripper_joint_indices)
                rec["gripper_joint_pos"] = jp[gi].tolist()
                rec["gripper_scale"] = list(map(float, robot.gripper_scale))
                g = jp[gi][0]
                sc = robot.gripper_scale
                rec["gripper_open_measured"] = float((g - sc[0]) / (sc[1] - sc[0]) if robot.gripper_move["sign"] == 1
                                                     else (sc[1] - g) / (sc[1] - sc[0]))
                pk = rm.process_name(robot.gripper_name)
                prev = rm.control_manager.prev_control[env_idx].get(pk)
                if prev is not None:
                    c = float(prev["position"][0])
                    rec["gripper_open_cmd"] = float((c - sc[0]) / (sc[1] - sc[0]) if robot.gripper_move["sign"] == 1
                                                    else (sc[1] - c) / (sc[1] - sc[0]))
            links = {}
            pose_w = _np(key.data.body_link_pose_w[env_idx])
            org = _np(self.env.scene_manager.env_origins[env_idx]).reshape(-1)[:3]
            for li, ln in enumerate(key.body_names):
                p = pose_w[li].copy()
                p[:3] -= org
                links[ln] = p.tolist()
            rec["links"] = links
            try:
                paths_ = list(key.root_physx_view.link_paths[env_idx])
                rec["link_prims"] = {p.rsplit("/", 1)[-1]: p for p in paths_}
            except Exception:  # noqa: BLE001
                pass
            out[arm] = rec
        return out

    # contacts: PhysX contact report on the gripper finger links (dev only)
    def enable_contacts(self) -> list[str]:
        """Subscribe to PhysX contact reports for every robot link whose name looks like a finger
        or gripper pad. Returns the prim paths watched."""
        import omni.physx
        from pxr import PhysxSchema
        rm = self.env.robot_manager
        stage = self.env.scene_manager.stage
        watched = []
        for robot in rm.robot_list:
            key = rm.robot_key[rm.robot_list.index(robot)]
            try:
                link_paths = list(key.root_physx_view.link_paths[0])
            except Exception:  # noqa: BLE001
                link_paths = []
            for lp in link_paths:
                leaf = lp.rsplit("/", 1)[-1].lower()
                if any(t in leaf for t in ("finger", "gripper", "link7", "link8", "pad")):
                    prim = stage.GetPrimAtPath(lp)
                    if prim and prim.IsValid():
                        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                        api.CreateThresholdAttr().Set(0.0)
                        watched.append(lp)
        self._contacts = {}

        def on_report(headers, data):
            from pxr import PhysicsSchemaTools
            for h in headers:
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                self._contacts.setdefault(a0, set()).add(a1)
                self._contacts.setdefault(a1, set()).add(a0)

        self._contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_report)
        self._watched = watched
        return watched

    def contacts_gt(self, clear: bool = True) -> dict:
        out = {}
        for lp in getattr(self, "_watched", []):
            out[lp] = sorted(self._contacts.get(lp, set()))
        if clear:
            self._contacts = {}
        return out

    def gt(self, env_idx: int = 0) -> dict:
        """DEV-ONLY ground truth. Never feed to an eval-time policy."""
        g = {"objects": self.objects_gt(env_idx), "cameras": self.cameras_gt(env_idx),
             "robot": self.robot_gt(env_idx), "reward": self.reward(env_idx), "score": self.score(env_idx),
             "layout_seed": int(self.seeds[env_idx]),
             "env_origin": _np(self.env.scene_manager.env_origins[env_idx]).reshape(-1)[:3].tolist(),
             "sim_time_steps": int(self.env.take_action_cnt[env_idx])}
        if self._contact_sub is not None:
            g["contacts"] = self.contacts_gt()
        return g

    def camera_spec(self, env_idx: int = 0) -> dict:
        """The three RoboDojo cameras as built by ``CameraManager``: an Xform (at ``mount`` link, or the
        env root) with local pose ``pos`` + ``ori_euler_deg`` (RoboDojo ``utils.rotations.euler_angles_to_quat``,
        intrinsic XYZ, i.e. R = Rx Ry Rz) holding a pinhole USD camera at identity, with the
        template's focal length / apertures / clipping. Works without a renderer."""
        from omegaconf import OmegaConf

        from env_cfg.camera import template as T
        # = camera_manager.REAL_MAP (that module is stubbed in physics-only mode)
        REAL_MAP = {"Gemini_345Lg": T.GEMINI_345LG, "third_view": T.THIRD_VIEW, "d435": T.D435,
                    "large_d435": T.LARGE_D435}
        cfg = OmegaConf.to_container(self.env.camera_config, resolve=True)
        root = f"/World/envs/env_{env_idx}"
        out = {}
        for name, c in cfg.items():
            if not isinstance(c, dict) or "camera" not in c:
                continue
            cam = c["camera"]
            tpl = dict(REAL_MAP.get(cam.get("type"), {}))
            out[name] = {"type": cam.get("type"), "mesh": cam.get("mesh"),
                         "mount": f"{root}/{cam['mount_link']}" if cam.get("mount_link") else None,
                         "pos": list(cam.get("pos") or [0, 0, 1]), "ori_euler_deg": list(cam.get("ori") or [0, 0, 0]),
                         "resolution": list(tpl.get("resolution", (640, 480))),
                         "focal_length": tpl.get("focal_length"), "horizontal_aperture": tpl.get("horizontal_aperture"),
                         "vertical_aperture": tpl.get("vertical_aperture"),
                         "clipping_range": list(tpl.get("clipping_range", (0.01, 10.0)))}
        return out

    def export_stage(self, path: str | Path) -> Path:
        """Write the current USD stage (room, table, ground, dome light, robots, objects; asset
        references stay absolute) for an external renderer. Poses on prims are the authored ones;
        live physics poses are in ``gt()``."""
        import omni.usd
        path = Path(path)
        paths.assert_not_eval(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        omni.usd.get_context().get_stage().Export(str(path))
        return path

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            if self.app is not None:
                self.app.close()


def settle_report(layout: dict, gt_objects: list[dict]) -> dict:
    """How far each object moved from its record pose during RoboDojo's settle (mm, deg)."""
    rows = []
    for o in gt_objects:
        if o["pos"] is None:
            continue
        d = float(np.linalg.norm(np.subtract(o["pos"], o["layout_pos"]))) * 1000
        a = quat_angle_deg(o["quat"], o["layout_quat"])
        rows.append({"inst": o["inst_name"], "label": o["label"], "clutter": o["clutter"], "type": o["type"],
                     "drift_mm": round(d, 2), "drift_deg": round(a, 2)})
    task_rows = [r for r in rows if not r["clutter"] and r["type"] != "Geometry"]
    return {"objects": rows,
            "max_task_drift_mm": max((r["drift_mm"] for r in task_rows), default=0.0),
            "max_task_drift_deg": max((r["drift_deg"] for r in task_rows), default=0.0),
            "max_clutter_drift_mm": max((r["drift_mm"] for r in rows if r["clutter"]), default=0.0),
            "fell_off": [r["inst"] for r in rows if r["drift_mm"] > 100]}


def save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(np.asarray(img, np.uint8)).save(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Load our layouts into RoboDojo; dump obs + GT; check settle.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--layouts", nargs="+", required=True, help="layout JSON files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", action="store_true", help="also save the 3 camera PNGs per layout")
    ap.add_argument("--physics-only", action="store_true", help="no renderer (no images)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    paths.assert_not_eval(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run = RoboDojoRunner(a.task, workdir=str(out / "work"), physics_only=a.physics_only)
    t_boot = time.perf_counter() - t0
    summary = []
    for lp in a.layouts:
        paths.assert_not_eval(lp)
        lay = json.loads(Path(lp).read_text())
        info = run.load(lay)
        rec = {"layout": lp, **{k: v for k, v in info.items()}}
        if info["stable"]:
            g = run.gt()
            rec["settle"] = settle_report(lay, g["objects"])
            rec["reward"] = g["reward"]
            rec["score"] = g["score"]
            if a.images and not a.physics_only:
                o = run.obs()
                stem = Path(lp).stem
                for cam, img in o["images"].items():
                    save_png(img, out / f"{stem}_{cam}.png")
                rec["shapes"] = {c: list(i.shape) for c, i in o["images"].items()}
        summary.append(rec)
        s = rec.get("settle", {})
        print(f"[simlab] {Path(lp).name}: stable={info['stable']} reset={info['reset_s']:.1f}s "
              f"close={info['close_s']:.1f}s task_drift={s.get('max_task_drift_mm')}mm/"
              f"{s.get('max_task_drift_deg')}deg clutter_drift={s.get('max_clutter_drift_mm')}mm "
              f"fell={s.get('fell_off')}", flush=True)
    (out / "settle_summary.json").write_text(json.dumps(jsonable({"task": a.task, "boot_s": t_boot,
                                                                   "layouts": summary}), indent=1))
    run.close()


if __name__ == "__main__":
    main()
