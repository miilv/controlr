"""Dev-eval: RoboDojo's OWN eval client on OUR dev layouts (needs a 5.1-capable RTX: compute2 / rented).

``src/eval_client/main.py`` runs unchanged: policy server over XPolicyLab's ws protocol, step limits,
success latching, ``_result.json``, mp4s. The only substitution is where layouts come from:
``SeedManager.init_eval`` / ``get_seed_scene_info`` read ``$SIMLAB_DEV_LAYOUTS/<task>/<id>.json``
(``<id>.settled.json`` with ``SIMLAB_DEV_LAYOUT_SUFFIX=.settled``)
(our records, ids sorted; ``--seed`` picks nothing) instead of ``Assets/Eval_Layout``, and our ``_simlab``
key is stripped. So a dev-eval score is computed exactly like a leaderboard score, on layouts we own.

    SIMLAB_DEV_LAYOUTS=/root/dojo-data/simlab/layouts python dojo/simlab/dev_eval.py <main.py args...>
    (scripts/simlab/dev_eval.sh wraps it with the policy server, like scripts/robodojo/run.sh)
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def patch_seed_manager(layout_root: str | os.PathLike) -> None:
    """Point RoboDojo's SeedManager at our dev layouts (class-level patch, before main.py imports it)."""
    from env.seed_manager import seed_manager as sm
    from utils.load_file import load_json

    from dojo.simlab import paths

    root = Path(layout_root)
    if "Eval_Layout" in root.resolve().parts:
        raise SystemExit("dev-eval must not read eval layouts")

    def init_eval(self, completed_layout_ids=None, abandoned_layout_ids=None):
        d = root / self.task_name
        sealed_m = os.environ.get("SIMLAB_SEALED_MANIFEST")   # a sealed joint block (D-C19): its manifest decides
        if sealed_m:
            from dojo.simlab import sealed
            man = sealed.load_manifest(sealed_m)
            suffix = ".settled" if sealed.task_entry(man, self.task_name)["settled"] else ""
        else:
            suffix = os.environ.get("SIMLAB_DEV_LAYOUT_SUFFIX", "")      # ".settled": pre-settled records
        files = sorted((p for p in d.glob(f"*{suffix}.json") if p.name[: -len(f"{suffix}.json")].isdigit()),
                       key=lambda p: int(p.name.split(".")[0]))
        if sealed_m:                                 # every staged record listed and md5-checked again in this process
            sealed.check_staged(d, man, self.task_name, files)
        reserved = [] if sealed_m else [p.name for p in files if paths.is_offslice_rendered(p.name.split(".")[0])]
        if reserved:                                 # D-C10: ids >= 100100 belong to the joint paired runs
            raise SystemExit(f"dev-eval: layout ids not ours to run rendered in {d} (D-C10a: 100010-19, 100050-99, "
                             f">= {paths.RESERVED_MIN_ID}): {reserved[:5]}")
        training = [p.name for p in files if paths.is_training_id(p.name.split(".")[0])]
        if training:                                 # D-C10c: the training / priors / calibration block is never evaluated
            raise SystemExit(f"dev-eval: training-block layout ids (D-C10c: {paths.TRAINING_BLOCK[0]}-"
                             f"{paths.TRAINING_BLOCK[1]}) in {d}: {training[:5]}")
        if not files:
            raise FileNotFoundError(f"no dev layouts in {d}")
        self.eval_seed = self.config.get("seed", 0)
        self.seed_info = {i: {"scene_layout": str(p), "layout_id": int(p.name.split(".")[0])} for i, p in enumerate(files)}
        excluded = {int(s) for s in (completed_layout_ids or [])} | {int(s) for s in (abandoned_layout_ids or [])}
        self.seed_list = [i for i in range(len(files)) if i not in excluded]
        self.st_idx, self.ed_idx, self.type, self.idx = 0, len(self.seed_list), "eval", 0
        self._current_batch_seeds = None
        print(f"[dev_eval] {len(files)} dev layouts{suffix} from {d} ({files[0].name}..{files[-1].name})", flush=True)

    def get_seed_scene_info(self, seed):
        info = self.seed_info.get(seed)
        if info is None:
            raise ValueError(f"Seed {seed} not found in seed list.")
        data = load_json(info["scene_layout"])
        return {k: v for k, v in data.items() if not k.startswith("_")}

    sm.SeedManager.init_eval = init_eval
    sm.SeedManager.get_seed_scene_info = get_seed_scene_info


def main() -> None:
    rd = Path(os.environ.get("RD_ROOT", "/root/controlr-robodojo/RoboDojo"))
    for p in (str(rd), str(rd / "XPolicyLab")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dojo.simlab import paths
    from dojo.simlab.runner import install_dev_demo_loader
    paths.install_asset_firewall()
    patch_seed_manager(os.environ["SIMLAB_DEV_LAYOUTS"])
    install_dev_demo_loader()                     # before main.py imports the task modules
    main_py = rd / "src" / "eval_client" / "main.py"
    sys.argv = [str(main_py)] + sys.argv[1:]
    runpy.run_path(str(main_py), run_name="__main__")


if __name__ == "__main__":
    main()
