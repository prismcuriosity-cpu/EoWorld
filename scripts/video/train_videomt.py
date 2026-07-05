#!/usr/bin/env python3
"""Train / evaluate VidEoMT on CholecSeg8k (video semantic segmentation).

Thin wrapper around VidEoMT's ``train_net_video.py`` that FIRST registers the
CholecSeg8k VSS dataset, then hands off to VidEoMT's own ``main`` via Detectron2's
``launch``. This keeps the upstream repo unmodified.

Prerequisites:
  * VidEoMT env active (detectron2 installed) — see scripts/video/00_setup_videomt.sh
  * Dataset converted to VSPW and DETECTRON2_DATASETS pointing at its parent:
        export DETECTRON2_DATASETS=/parent/of/CholecSeg8k_VSPW

Train (single GPU):
    python scripts/video/train_videomt.py --num-gpus 1 \
      --config-file configs/cholecseg8k/video_vss/videomt_vitl_cholecseg8k.yaml \
      MODEL.WEIGHTS checkpoints/videomt/vspw_segmenter.pth \
      OUTPUT_DIR output/videomt_vitl_cholecseg8k

Evaluate (dumps VSPW-format predictions; compute mIoU with 12_eval_miou.py):
    python scripts/video/train_videomt.py --num-gpus 1 \
      --config-file configs/cholecseg8k/video_vss/videomt_vitl_cholecseg8k.yaml \
      --eval-only MODEL.WEIGHTS output/videomt_vitl_cholecseg8k/model_final.pth \
      MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
      OUTPUT_DIR output/videomt_vitl_cholecseg8k/eval

Only single-node runs are wired here (the CholecSeg8k target is one GPU). For
multi-GPU spawn you'd need the registration imported inside each worker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEOMT_DIR = Path(os.environ.get("VIDEOMT_DIR", REPO_ROOT / "third_party" / "videomt"))


def main() -> None:
    if not VIDEOMT_DIR.exists():
        sys.exit(f"VidEoMT clone not found at {VIDEOMT_DIR}. Run scripts/video/00_setup_videomt.sh "
                 f"(or set VIDEOMT_DIR).")
    # Make both VidEoMT and EoWorld importable.
    sys.path.insert(0, str(VIDEOMT_DIR))
    sys.path.insert(0, str(REPO_ROOT))

    # Registering the dataset must happen before the config (which names it) is used.
    import eoworld.video.register_cholec_vss  # noqa: F401  (registers on import)

    from detectron2.engine import default_argument_parser, launch
    import train_net_video as tnv  # VidEoMT entrypoint

    args = default_argument_parser().parse_args()
    if not args.dist_url:
        args.dist_url = "tcp://127.0.0.1:50263"
    if args.num_gpus > 1:
        print("[warn] multi-GPU: dataset registration runs only in the launcher process; "
              "single GPU (--num-gpus 1) is the supported path for CholecSeg8k.")
    launch(
        tnv.main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
