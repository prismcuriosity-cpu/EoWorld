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
    frac = report["pixel_fraction"]
    print(f"\nChecked {report['n_frames_checked']} frames.")
    print("Known watershed values -> class (share of pixels):")
    for v, name in report["known_values"].items():
        print(f"  {v:>3} -> {name:<22} {frac[v] * 100:6.2f}%")

    if report["ignore_values"]:
        print("\nExpected unlabeled/void values (mapped to ignore, harmless):")
        for v in report["ignore_values"]:
            print(f"  {v:>3} -> IGNORE            {frac[v] * 100:6.2f}%")

    if report["unknown_values"]:
        print("\n!! UNKNOWN watershed values (neither a class nor expected ignore):")
        for v in report["unknown_values"]:
            print(f"  {v:>3}  {frac[v] * 100:6.2f}% of pixels")
        print("  If one carries a large share, it may be a real class missing from")
        print("  CHOLECSEG8K_CLASSES in eoworld/data/class_info.py; if tiny, add it")
        print("  to IGNORE_WATERSHED_VALUES there.")
        sys.exit(1)

    n_classes_seen = len(report["known_values"])
    print(f"\nAll watershed values accounted for ({n_classes_seen} classes present, "
          f"unlabeled handled). Encoding verified ✓")


if __name__ == "__main__":
    main()
