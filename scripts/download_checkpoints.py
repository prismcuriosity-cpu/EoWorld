#!/usr/bin/env python3
"""Download EoMT pretrained checkpoints to fine-tune from.

We default to the DINOv2 checkpoints because they are NOT gated (unlike the
DINOv3 deltas, which need Meta access — see docs/CHECKPOINTS.md). Fine-tuning
CholecSeg8k from a COCO-panoptic EoMT checkpoint (with the class head skipped) is
the recommended, fastest-to-competitive path.

Examples:
    # The three backbones used by the S/B/L configs:
    python scripts/download_checkpoints.py --which small base large

    # Everything (incl. semantic-pretrained large variants):
    python scripts/download_checkpoints.py --which all

Files land in checkpoints/<name>/pytorch_model.bin — pass that path as the 3rd
arg to scripts/05_train_segmentation.sh.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/tue-mps/{repo}/resolve/main/pytorch_model.bin"

# name -> HF repo. DINOv2 (ungated). Recommended starting weights per backbone.
CHECKPOINTS = {
    "small": "coco_panoptic_eomt_small_640_2x",
    "base": "coco_panoptic_eomt_base_640_2x",
    "large": "coco_panoptic_eomt_large_640",
    "large_ade_semantic": "ade20k_semantic_eomt_large_512",
    "large_cityscapes_semantic": "cityscapes_semantic_eomt_large_1024",
}

GROUPS = {
    "all": list(CHECKPOINTS),
    "sbl": ["small", "base", "large"],
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  exists, skipping: {dest}")
        return
    print(f"  {url}\n    -> {dest}")

    def _hook(block, bsize, total):
        if total > 0:
            pct = min(100, block * bsize * 100 // total)
            sys.stdout.write(f"\r    {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write("\r    done \n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", nargs="+", default=["sbl"],
                   help=f"names {list(CHECKPOINTS)} or groups {list(GROUPS)}")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    names: list[str] = []
    for w in args.which:
        names.extend(GROUPS.get(w, [w]))
    unknown = [n for n in names if n not in CHECKPOINTS]
    if unknown:
        sys.exit(f"Unknown checkpoint(s): {unknown}. Choose from {list(CHECKPOINTS)} or {list(GROUPS)}.")

    out = Path(args.out)
    for name in dict.fromkeys(names):  # dedupe, keep order
        repo = CHECKPOINTS[name]
        print(f"[{name}] {repo}")
        download(HF.format(repo=repo), out / repo / "pytorch_model.bin")
    print("\nDone. Example fine-tune:")
    print(f"  bash scripts/05_train_segmentation.sh \\")
    print(f"    configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml <data_root> \\")
    print(f"    {out}/coco_panoptic_eomt_small_640_2x/pytorch_model.bin")


if __name__ == "__main__":
    main()
