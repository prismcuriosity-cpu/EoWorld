#!/usr/bin/env python3
"""Convert CholecSeg8k to VSPW format for VidEoMT (video semantic segmentation).

    python scripts/video/10_convert_cholecseg8k_to_vspw.py \
      --data-path /path/to/cholecseg8k \
      --out $DETECTRON2_DATASETS/CholecSeg8k_VSPW

By default frames are symlinked (no duplication). Use --copy to write real files
(e.g. if the training box can't follow symlinks into the dataset dir).

Output layout (VSPW):
    <out>/data/<clip>/origin/*.png   <out>/data/<clip>/mask/*.png
    <out>/train.txt  <out>/val.txt  <out>/test.txt

Point VidEoMT at it with:  export DETECTRON2_DATASETS=<parent-of-CholecSeg8k_VSPW>
(the registration expects <out> to be named 'CholecSeg8k_VSPW').
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eoworld.video.vspw import convert_to_vspw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="CholecSeg8k root")
    p.add_argument("--out", required=True, help="destination VSPW root (…/CholecSeg8k_VSPW)")
    p.add_argument("--copy", action="store_true", help="copy frames instead of symlinking")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Converting {args.data_path} -> {args.out} (VSPW format) ...")
    summary = convert_to_vspw(
        args.data_path, args.out, link=not args.copy,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    print("\nDone:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nNext: set DETECTRON2_DATASETS to the PARENT of this folder, e.g.")
    print(f"  export DETECTRON2_DATASETS={Path(args.out).resolve().parent}")


if __name__ == "__main__":
    main()
