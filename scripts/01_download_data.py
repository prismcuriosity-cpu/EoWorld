#!/usr/bin/env python3
"""Download the CholecSeg8k dataset from Kaggle.

Two ways to get the data:

A) kagglehub (recommended). Requires a Kaggle account + API token
   (~/.kaggle/kaggle.json, see https://www.kaggle.com/docs/api):

       python scripts/01_download_data.py

   Prints the local path kagglehub extracted to; pass that as --data.path.

B) Manual: download from
   https://www.kaggle.com/datasets/newslab/cholecseg8k , unzip anywhere, and
   skip this script — just use that folder as the dataset root.

Either way, the expected layout is:
   <root>/video01/video01_00080/frame_80_endo.png (+ _endo_watershed_mask.png)
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug", default="newslab/cholecseg8k", help="Kaggle dataset slug")
    args = p.parse_args()

    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub not installed. Run `pip install kagglehub`, or download "
            "manually from https://www.kaggle.com/datasets/newslab/cholecseg8k"
        )

    print(f"Downloading {args.slug} via kagglehub ...")
    path = kagglehub.dataset_download(args.slug)
    print("\nDataset ready at:\n  " + path)
    print("\nUse it as the dataset root, e.g.:")
    print(f"  python scripts/02_inspect_masks.py --data-path {path}")
    print(f"  python scripts/03_visualize_dataset.py --data-path {path}")


if __name__ == "__main__":
    main()
