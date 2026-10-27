#!/usr/bin/env bash
# SAM mask server environment on a GPU box (dojo.perception.sam_masks serve): a deploy dir's .venv + torch (cu128) +
# Meta's sam3 / sam2 code (shallow clones, no-deps installs) + weights under ~/dojo-data/models.
#   scripts/perception/sam/setup_venv.sh [DEPLOY_DIR]      (default ~/controlr-sam)
# Weights: SAM 3.1 = Comfy-Org/sam3.1 (fp16 repack of facebook/sam3.1, SAM License; Ilia approved), SAM 2.1 hiera
# small / large (facebook, Apache-2.0). ~2.8 GB in total. Then:
#   flock -n ~/dojo-gpu-light.lock .venv/bin/python -m dojo.perception.sam_masks serve --sock ~/dojo-data/sam/sam.sock
set -euo pipefail
D="${1:-$HOME/controlr-sam}"
EXT="$HOME/controlr-sam-ext"
M="$HOME/dojo-data/models"
SAM3_SHA=f1647e8          # facebookresearch/sam3, 2026-10-23 (SAM 3.1 multiplex builders)
SAM2_SHA=2b90b9f          # facebookresearch/sam2, 2024-12-15
export PATH="$HOME/.local/bin:$PATH"
cd "$D"
PY=.venv/bin/python
uv pip install --python $PY -q -e ".[dev,perception]"
uv pip install --python $PY -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pyrender pins PyOpenGL 3.1.0, whose array handlers fail on numpy 2 (textured renders: press_by_number errors)
uv pip install --python $PY -q timm ftfy regex iopath einops psutil hydra-core omegaconf safetensors "PyOpenGL==3.1.9"
mkdir -p "$EXT" "$M/sam3.1" "$M/sam2.1/small" "$M/sam2.1/large"
clone() {  # clone URL DIR SHA: shallow, at the pinned commit
  [ -d "$2/.git" ] || git clone -q --depth 1 "$1" "$2"
  git -C "$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "$2" checkout -q "$3" 2>/dev/null || echo "WARN: $2 not at $3 (shallow head kept)"
}
clone https://github.com/facebookresearch/sam3.git "$EXT/sam3" "$SAM3_SHA"
clone https://github.com/facebookresearch/sam2.git "$EXT/sam2" "$SAM2_SHA"
uv pip install --python $PY -q --no-deps -e "$EXT/sam3"
SAM2_BUILD_CUDA=0 uv pip install --python $PY -q --no-deps --no-build-isolation -e "$EXT/sam2"
get() { [ -s "$2" ] || curl -sfL -o "$2" "$1"; }
get https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors "$M/sam3.1/sam3.1_multiplex_fp16.safetensors"
get https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt "$M/sam2.1/small/sam2.1_hiera_small.pt"
get https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt "$M/sam2.1/large/sam2.1_hiera_large.pt"
$PY -c "import torch, sam3, sam2; print('ok', torch.__version__, torch.cuda.is_available())"
