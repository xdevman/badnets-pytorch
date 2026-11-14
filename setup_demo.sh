#!/usr/bin/env bash
# Create the isolated demo environment for badnets-pytorch using uv.
#
#   ./setup_demo.sh            # CUDA 12.1 build (needs an NVIDIA driver >= 525)
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu ./setup_demo.sh   # CPU only
#
# The environment is created in ./.venv (gitignored); the system Python is never touched.
set -euo pipefail
cd "$(dirname "$0")"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "==> uv not found, installing it into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

echo "==> creating .venv (Python 3.12)"
uv venv --python 3.12 --clear .venv

echo "==> installing torch from $TORCH_INDEX"
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" \
  "torch==2.5.1" "torchvision==0.20.1"

echo "==> installing the remaining dependencies"
uv pip install --python .venv/bin/python -r requirements-demo.txt

echo "==> downloading MNIST and CIFAR10 into ./data"
.venv/bin/python data_downloader.py

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then try:             python demo.py --dataset MNIST"
