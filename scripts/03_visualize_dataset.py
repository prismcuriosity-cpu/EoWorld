#!/usr/bin/env python3
"""Generate all journal-ready CholecSeg8k dataset figures.

Examples
--------
# Real dataset (scans every mask once, then caches stats to figures/stats.pkl):
python scripts/03_visualize_dataset.py --data-path /path/to/cholecseg8k

# Offline preview with synthetic stats (no dataset needed) — verifies the
# figure code and shows what the outputs look like:
python scripts/03_visualize_dataset.py --demo

Outputs land in --out (default: figures/), each as 300-dpi PNG + vector PDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import eoworld` work when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoworld.viz import dataset_report as dr
from eoworld.viz.style import set_journal_style, save_figure


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=str, default=None, help="CholecSeg8k root dir")
    p.add_argument("--out", type=str, default="figures", help="output directory")
    p.add_argument("--demo", action="store_true", help="use synthetic stats (no dataset)")
    p.add_argument("--limit", type=int, default=None, help="only scan first N frames")
    p.add_argument("--no-samples", action="store_true", help="skip the qualitative sample grid")
    args = p.parse_args()

    set_journal_style()
    out = Path(args.out)

    if args.demo or not args.data_path:
        if not args.demo:
            print("No --data-path given; falling back to --demo (synthetic stats).")
        stats = dr.make_synthetic_stats()
    else:
        cache = out / "stats.pkl"
        print(f"Scanning dataset at {args.data_path} (cache: {cache}) ...")
        stats = dr.compute_dataset_stats(args.data_path, cache_path=cache, limit=args.limit)
        print(f"  {stats['n_frames']:,} frames across {len(stats['videos'])} videos")

    written: list[Path] = []
    written += dr.fig_palette_card(out, save_figure)
    written += dr.fig_class_frequency(stats, out, save_figure)
    written += dr.fig_per_video_composition(stats, out, save_figure)
    written += dr.fig_cooccurrence(stats, out, save_figure)
    written += dr.fig_frames_per_video(stats, out, save_figure)
    written += dr.fig_instrument_timeline(stats, out, save_figure)
    if args.data_path and not args.demo and not args.no_samples:
        written += dr.fig_sample_grid(args.data_path, out, save_figure)

    print("\nWrote:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
