#!/usr/bin/env bash
# EoWorld setup: clone the EoMT repo and install dependencies.
#
# Designed for a local CUDA + Anaconda environment (tested target: single
# RTX 5090, 32 GB). Run from the EoWorld repo root:
#
#   bash scripts/00_setup.sh
#
# What it does:
#   1. Clones tue-mps/eomt into third_party/eomt (the upstream code we build on).
#   2. Installs EoMT's requirements + EoWorld's extra deps into the ACTIVE env.
#
# It does NOT create the conda env for you — do that first so you control the
# CUDA/PyTorch build for your GPU:
#
#   conda create -n eoworld python=3.11 -y
#   conda activate eoworld
#   # Install a CUDA-matched PyTorch for the RTX 5090 (Blackwell / sm_120).
#   # A recent CUDA 12.4+ wheel is required; check https://pytorch.org for the
#   # current command, e.g.:
#   #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   bash scripts/00_setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"
EOMT_DIR="$THIRD_PARTY/eomt"
EOMT_URL="https://github.com/tue-mps/eomt.git"

mkdir -p "$THIRD_PARTY"

if [ -d "$EOMT_DIR/.git" ]; then
  echo "[setup] eomt already cloned at $EOMT_DIR — pulling latest."
  git -C "$EOMT_DIR" pull --ff-only || echo "[setup] (pull skipped)"
else
  echo "[setup] Cloning EoMT into $EOMT_DIR"
  git clone "$EOMT_URL" "$EOMT_DIR"
fi

echo "[setup] Installing EoMT requirements (minus the pinned torch/torchvision,"
echo "        which you should install with a CUDA build matched to your GPU)."
grep -viE '^(torch|torchvision)==' "$EOMT_DIR/requirements.txt" > /tmp/eomt_reqs_noTorch.txt
python3 -m pip install -r /tmp/eomt_reqs_noTorch.txt

echo "[setup] Installing EoWorld extra requirements."
python3 -m pip install -r "$REPO_ROOT/requirements-eoworld.txt"

cat <<EOF

[setup] Done.
  EoMT clone:   $EOMT_DIR
  Next steps:
    1. python scripts/01_download_data.py            # download CholecSeg8k
    2. python scripts/02_inspect_masks.py --data-path <root>   # verify encoding
    3. python scripts/03_visualize_dataset.py --data-path <root>   # figures
    4. python scripts/04_quick_smoke_test.py --data-path <root>    # sanity gate
    5. bash   scripts/05_train_segmentation.sh <config> <data_root> [ckpt]
EOF
