#!/usr/bin/env bash
# PyRUA-Lean on compute2, everything under /root/dojo-pyrua (run once; steps are idempotent).
#
#   scp -r baselines/pyrua compute2:/root/dojo-pyrua/glue && ssh compute2 'bash /root/dojo-pyrua/glue/setup.sh'
#
# - clone of DAGroup-PKU/PyRUA-Lean pinned at the revision its adapter names as evaluated
#   (install.sh: c686120; src/ is identical to the official-evaluation commit 077941b);
# - venv (python 3.11, uv): torch 2.8.0 + torchvision 0.23.0 (cu128), then what upstream's
#   install.sh installs, with its constraints (numpy<2, opencv-headless<4.12, torch kept), EXCEPT
#   `pip install -e XPolicyLab`: that would write egg-info into the shared RoboDojo install, so
#   XPolicyLab's dependencies are installed instead and the server runs with PYTHONPATH=RoboDojo
#   (as upstream's setup_eval_policy_server.sh does anyway);
# - SAM3 checkpoint: facebook/sam3 on HF is gated and our token has no access (HTTP 403), so it
#   comes from upstream's documented fallback, the ModelScope mirror (download_assets.sh modelscope);
# - the one write into the shared install: symlink XPolicyLab/policy/PyRUA_Lean -> the clone's
#   integrations/xpolicylab/PyRUA_Lean (upstream's install.sh expects exactly this link).
set -euxo pipefail
ROOT=/root/dojo-pyrua
REV=c6861200bfb190169953d1827260ccf71320eea1
SAM3_REV=c1f6b1e83967c912f675d8975d437ccdc41e17d8
XPL=/root/controlr-robodojo/RoboDojo/XPolicyLab
UV=/root/.local/bin/uv
export UV_CACHE_DIR=$ROOT/.uvcache
mkdir -p "$ROOT" && cd "$ROOT"

[[ -d PyRUA-Lean ]] || git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git
git -C PyRUA-Lean checkout -q "$REV"

if [[ ! -x venv/bin/python ]]; then
  $UV venv --python-preference only-managed -p 3.11 venv   # NOT the system 3.11.0rc1: torch.jit segfaults on it
  $UV pip install -p venv/bin/python pip
  $UV pip install -p venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
fi
PY=$ROOT/venv/bin/python
cat > constraints.txt <<EOF
torch==2.8.0
torchvision==0.23.0
numpy>=1.26,<2
opencv-python-headless<4.12
EOF
pin() { $UV pip install -p "$PY" -c "$ROOT/constraints.txt" "$@"; }
# XPolicyLab's declared dependencies (its pyproject), without installing it
pin "numpy" "pyyaml>=6" "opencv-python-headless>=4.8" "h5py>=3.8" "websockets>=14.0" "msgpack>=1.0.8" \
    "msgpack-numpy>=0.4.8" "pydantic>=2.5"
pin -e "$ROOT/PyRUA-Lean"
pin "numpy" "pillow" "pyyaml>=6" "httpx>=0.27" "mcp>=1.8,<2" "uvicorn"
pin "sam3 @ git+https://github.com/RLinf/sam3.git@$SAM3_REV" "einops" "pycocotools" "psutil"

if [[ ! -s checkpoints/sam3.pt ]]; then
  $UV venv --python-preference only-managed -p 3.11 msvenv && $UV pip install -p msvenv/bin/python modelscope
  msvenv/bin/modelscope download --model facebook/sam3 sam3.pt --local_dir "$ROOT/checkpoints"
fi
sha256sum checkpoints/sam3.pt

[[ -e "$XPL/policy/PyRUA_Lean" ]] || ln -s "$ROOT/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean" "$XPL/policy/PyRUA_Lean"

# upstream install.sh's import check
cd "$ROOT" && PYTHONPATH=/root/controlr-robodojo/RoboDojo "$PY" - <<'PY'
import torch
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
from sam3.model_builder import build_sam3_image_model  # noqa: F401
import pyrualean.robots
pyrualean.robots.get_adapter("robodojo_xpl_rgb").host_module()
from XPolicyLab.policy.PyRUA_Lean.model import Model  # noqa: F401
print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()}, SAM3, PyRUA-Lean and the adapter import")
PY
