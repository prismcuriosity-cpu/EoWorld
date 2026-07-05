#!/usr/bin/env bash
# Fine-tune EoMT on CholecSeg8k (Experiment 8.1 — segmentation fidelity).
#
# Usage:
#   bash scripts/05_train_segmentation.sh <config> <data_root> [pretrained_ckpt]
#
# Examples:
#   # From scratch-ish (ImageNet/DINOv2 backbone only):
#   bash scripts/05_train_segmentation.sh \
#     configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml /data/cholecseg8k
#
#   # Recommended: fine-tune from a COCO-panoptic EoMT checkpoint (class head skipped):
#   bash scripts/05_train_segmentation.sh \
#     configs/cholecseg8k/semantic/eomt_base_640_dinov2.yaml /data/cholecseg8k \
#     checkpoints/coco_panoptic_eomt_base_640_2x/pytorch_model.bin
#
# Everything after the 3rd arg is forwarded verbatim to `main.py` (e.g. override
# --data.batch_size 4, --trainer.devices 1, --trainer.default_root_dir runs/foo).
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <config> <data_root> [pretrained_ckpt] [extra main.py args...]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EOMT_DIR="${EOMT_DIR:-$REPO_ROOT/third_party/eomt}"

CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"   # absolutise
DATA="$2"
shift 2

CKPT=""
if [ "$#" -ge 1 ] && [[ "$1" != --* ]]; then
  CKPT="$1"; shift
fi

if [ ! -d "$EOMT_DIR" ]; then
  echo "EoMT clone not found at $EOMT_DIR — run scripts/00_setup.sh (or set EOMT_DIR)."
  exit 1
fi

EXTRA=()
if [ -n "$CKPT" ]; then
  EXTRA+=(--model.ckpt_path "$CKPT" --model.load_ckpt_class_head False)
fi

echo "[train] config : $CONFIG"
echo "[train] data   : $DATA"
echo "[train] ckpt   : ${CKPT:-<none, backbone-pretrained only>}"
echo "[train] eomt   : $EOMT_DIR"

cd "$EOMT_DIR"
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python main.py fit \
  -c "$CONFIG" \
  --data.path "$DATA" \
  "${EXTRA[@]}" \
  "$@"
