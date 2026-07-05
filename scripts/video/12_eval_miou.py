#!/usr/bin/env python3
"""Compute VSS mIoU for CholecSeg8k from dumped VidEoMT predictions.

VidEoMT's VSSEvaluator only *saves* predictions (in VSPW format); metrics are
computed offline. Upstream's utils/eval_miou_vspw.py hardcodes 124 classes, so
this wrapper reuses its Evaluator with num_class = 13 and our split file.

    python scripts/video/12_eval_miou.py \
      --vspw-root $DETECTRON2_DATASETS/CholecSeg8k_VSPW \
      --pred output/videomt_vitl_cholecseg8k/eval/inference \
      --split val

GT masks (1-indexed on disk) and predictions (contiguous ids) are reconciled by
the same VSPW preprocessing (0->ignore, value-1) the upstream Evaluator applies.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEOMT_DIR = Path(os.environ.get("VIDEOMT_DIR", REPO_ROOT / "third_party" / "videomt"))

from eoworld.data.class_info import NUM_CLASSES, CLASS_NAMES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vspw-root", required=True, help="…/CholecSeg8k_VSPW")
    p.add_argument("--pred", required=True, help="dir with per-clip predicted PNGs")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = p.parse_args()

    sys.path.insert(0, str(VIDEOMT_DIR / "utils"))
    try:
        from eval_miou_vspw import Evaluator
    except ImportError:
        sys.exit(f"Could not import upstream eval_miou_vspw from {VIDEOMT_DIR}/utils. "
                 f"Run scripts/video/00_setup_videomt.sh or set VIDEOMT_DIR.")

    ev = Evaluator(NUM_CLASSES)
    ev.reset()

    root = Path(args.vspw_root)
    clips = [l.strip() for l in open(root / f"{args.split}.txt") if l.strip()]
    n = 0
    for clip in clips:
        mask_dir = root / "data" / clip / "mask"
        for tar in sorted(os.listdir(mask_dir)):
            pred_path = Path(args.pred) / clip / tar
            if not pred_path.exists():
                continue
            gt = np.array(Image.open(mask_dir / tar))[np.newaxis, :]
            pred = np.array(Image.open(pred_path))[np.newaxis, :]
            ev.add_batch(gt, pred)
            n += 1

    if n == 0:
        sys.exit("No matching (GT, prediction) pairs found — check --pred path/split.")

    miou = ev.Mean_Intersection_over_Union()
    acc = ev.Pixel_Accuracy()
    acc_cls = ev.Pixel_Accuracy_Class()
    fwiou = ev.Frequency_Weighted_Intersection_over_Union()
    print(f"\nEvaluated {n} frames across {len(clips)} clips ({args.split}).")
    print(f"  mIoU        : {miou:.4f}")
    print(f"  Pixel Acc   : {acc:.4f}")
    print(f"  Mean Acc    : {acc_cls:.4f}")
    print(f"  FW IoU      : {fwiou:.4f}")

    # Per-class IoU from the confusion matrix.
    cm = ev.confusion_matrix
    iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-9)
    print("\n  Per-class IoU:")
    for i, name in enumerate(CLASS_NAMES):
        seen = cm[i].sum() > 0
        print(f"    {name:<22} {iou[i]:.4f}" + ("" if seen else "  (absent in split)"))


if __name__ == "__main__":
    main()
