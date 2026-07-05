#!/usr/bin/env bash
# EoWorld VIDEO setup: clone VidEoMT and install its dependencies.
#
# IMPORTANT: VidEoMT is Detectron2-based and needs a SEPARATE conda env from the
# image (EoMT) pipeline. Do NOT reuse the `eoworld` env. Create `videomt` first:
#
#   conda create -n videomt python=3.12.3 -y
#   conda activate videomt
#   # CUDA-matched torch for the RTX 5090 (Blackwell / sm_120). VidEoMT pins
#   # torch 2.7.0 / cu126; verify the current 5090-compatible build at pytorch.org:
#   pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
#   bash scripts/video/00_setup_videomt.sh
#
# Detectron2 must be built against the SAME torch/CUDA — this is the #1 source of
# setup pain. If the source build fails, see docs/RUNNING.md troubleshooting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"
VIDEOMT_DIR="$THIRD_PARTY/videomt"
VIDEOMT_URL="https://github.com/tue-mps/videomt.git"

mkdir -p "$THIRD_PARTY"

if [ -d "$VIDEOMT_DIR/.git" ]; then
  echo "[setup] videomt already cloned at $VIDEOMT_DIR — pulling latest."
  git -C "$VIDEOMT_DIR" pull --ff-only || echo "[setup] (pull skipped)"
else
  echo "[setup] Cloning VidEoMT into $VIDEOMT_DIR"
  git clone "$VIDEOMT_URL" "$VIDEOMT_DIR"
fi

echo "[setup] Installing Detectron2 (no build isolation, uses your installed torch)."
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

echo "[setup] Installing panopticapi."
python -m pip install 'git+https://github.com/cocodataset/panopticapi.git'

echo "[setup] Installing VidEoMT requirements."
python -m pip install -r "$VIDEOMT_DIR/requirements.txt"

cat <<EOF

[setup] Done.
  VidEoMT clone: $VIDEOMT_DIR
  Next:
    1. python scripts/video/10_convert_cholecseg8k_to_vspw.py \\
         --data-path <cholecseg8k> --out <datasets>/CholecSeg8k_VSPW
    2. export DETECTRON2_DATASETS=<datasets>
    3. python scripts/video/download_videomt_checkpoints.py --which vspw_segmenter
    4. python scripts/video/train_videomt.py --num-gpus 1 \\
         --config-file configs/cholecseg8k/video_vss/videomt_vitl_cholecseg8k.yaml \\
         MODEL.WEIGHTS checkpoints/videomt/vspw_segmenter.pth
  Full walkthrough: docs/RUNNING.md
EOF
