#!/usr/bin/env python3
"""Extract EoMT query-token states (z_t) for CholecSeg8k — the world-model state.

Run AFTER you have a fine-tuned segmentation checkpoint (from step 5). Example:

    python scripts/06_extract_query_tokens.py \
      --config configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml \
      --ckpt   runs/cholecseg8k_small/last.ckpt \
      --data-path /path/to/cholecseg8k \
      --out cache/query_tokens_small

Outputs one <video>.npz per video containing the temporally-ordered query-token
sequence (T, num_q, D) plus per-query class confidence — the input the latent
dynamics model (proposal Exp. 8.2) forecasts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eoworld.data.class_info import NUM_CLASSES
from eoworld.query_tokens.extract import (
    build_model,
    ensure_eomt_on_path,
    extract_dataset,
    load_config_fields,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="EoMT config used for training")
    p.add_argument("--ckpt", required=True, help="fine-tuned checkpoint (.ckpt or .bin)")
    p.add_argument("--data-path", required=True, help="CholecSeg8k root dir")
    p.add_argument("--out", required=True, help="output dir for per-video .npz")
    p.add_argument("--eomt-dir", default=os.environ.get("EOMT_DIR", str(REPO_ROOT / "third_party" / "eomt")))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    p.add_argument("--videos", nargs="*", default=None, help="subset of video ids")
    p.add_argument("--no-logits", action="store_true", help="don't save full class logits")
    p.add_argument("--no-masked-attn", action="store_true",
                   help="disable masked attention (matches EoMT validation mode)")
    args = p.parse_args()

    ensure_eomt_on_path(args.eomt_dir)

    net, fields = build_model(
        config_path=args.config,
        ckpt_path=args.ckpt,
        num_classes=args.num_classes,
        masked_attn_enabled=not args.no_masked_attn,
        device=args.device,
    )
    print(f"[extract] backbone={fields['backbone_name']} num_q={fields['num_q']} "
          f"img_size={fields['img_size']}")

    extract_dataset(
        net=net,
        data_root=args.data_path,
        out_dir=args.out,
        img_size=fields["img_size"],
        num_classes=args.num_classes,
        device=args.device,
        batch_size=args.batch_size,
        videos=args.videos,
        save_logits=not args.no_logits,
    )
    print("[extract] done.")


if __name__ == "__main__":
    main()
