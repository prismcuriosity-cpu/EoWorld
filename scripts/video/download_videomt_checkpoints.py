#!/usr/bin/env python3
"""Download VidEoMT checkpoints (segmenter init weights + trained models).

For CholecSeg8k video semantic segmentation you want the VSPW **segmenter** weight
as training initialisation:

    python scripts/video/download_videomt_checkpoints.py --which vspw_segmenter

Files land in checkpoints/videomt/<name>.pth — pass as MODEL.WEIGHTS.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/tue-mps/VidEoMT/resolve/main"

CHECKPOINTS = {
    # Segmenter (per-frame) init weights — use these to START training.
    "vspw_segmenter": f"{BASE}/dinov2_segmenter/vspw_segmenter.pth",
    "yt2019_large_segmenter": f"{BASE}/dinov2_segmenter/yt_2019_large_segmenter.pth",
    "vipseg_segmenter": f"{BASE}/dinov2_segmenter/vipseg_segmenter.pth",
    # Full trained VidEoMT-L models (for reference / eval).
    "vspw_vitl": f"{BASE}/vspw_vit_large_95.0_64.9.pth",
}

GROUPS = {"all": list(CHECKPOINTS)}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  exists, skipping: {dest}")
        return
    print(f"  {url}\n    -> {dest}")

    def _hook(block, bsize, total):
        if total > 0:
            sys.stdout.write(f"\r    {min(100, block * bsize * 100 // total):3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write("\r    done \n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", nargs="+", default=["vspw_segmenter"],
                   help=f"names {list(CHECKPOINTS)} or 'all'")
    p.add_argument("--out", default="checkpoints/videomt")
    args = p.parse_args()

    names: list[str] = []
    for w in args.which:
        names.extend(GROUPS.get(w, [w]))
    unknown = [n for n in names if n not in CHECKPOINTS]
    if unknown:
        sys.exit(f"Unknown: {unknown}. Choose from {list(CHECKPOINTS)} or 'all'.")

    out = Path(args.out)
    for name in dict.fromkeys(names):
        print(f"[{name}]")
        download(CHECKPOINTS[name], out / f"{name}.pth")
    print("\nDone. Use e.g.:  MODEL.WEIGHTS", str(out / "vspw_segmenter.pth"))


if __name__ == "__main__":
    main()
