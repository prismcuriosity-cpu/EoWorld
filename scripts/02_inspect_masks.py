#!/usr/bin/env python3
"""Verify the CholecSeg8k watershed encoding against `eoworld.data.class_info`.

Reads a sample of watershed masks and reports which pixel values appear. If any
UNKNOWN value shows up, the canonical table in class_info.py is out of date for
your copy of the dataset — fix it there and everything downstream follows.

    python scripts/02_inspect_masks.py --data-path /path/to/cholecseg8k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoworld.data.discovery import find_samples, inspect_watershed_values


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="CholecSeg8k root dir")
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    n = len(find_samples(args.data_path))
    print(f"Discovered {n} (frame, watershed) pairs under {args.data_path}")
    if n == 0:
        sys.exit("No frames found — check the path and folder layout.")

    report = inspect_watershed_values(args.data_path, max_frames=args.max_frames)
    print(f"\nChecked {report['n_frames_checked']} frames.")
    print("Known watershed values -> class:")
    for v, name in report["known_values"].items():
        print(f"  {v:>3} -> {name}")

    if report["unknown_values"]:
        print("\n!! UNKNOWN watershed values (not in class_info):")
        print("  ", report["unknown_values"])
        print("  Update CHOLECSEG8K_CLASSES in eoworld/data/class_info.py.")
        sys.exit(1)
    print("\nAll watershed values are known. Encoding verified ✓")


if __name__ == "__main__":
    main()
