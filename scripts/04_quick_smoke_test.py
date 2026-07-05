#!/usr/bin/env python3
"""Quick end-to-end sanity gate — run this BEFORE committing to a full run.

Checks, in order (each independent, so partial environments still get signal):

  1. EoMT clone + imports resolve.
  2. Data module: discovery, video-level split, one training sample has the
     expected (img, target) shapes.   [needs --data-path]
  3. Model: build EoMT-S, forward a batch, verify mask/class-logit shapes.
  4. Query-token extraction hook returns z_t of shape (B, num_q, D).
  5. (optional, --run-fit) launch a 2-step `main.py fit` to prove the training
     loop starts.                     [needs --data-path]

Usage:
  python scripts/04_quick_smoke_test.py                       # checks 1,3,4
  python scripts/04_quick_smoke_test.py --data-path <root>    # + checks 2
  python scripts/04_quick_smoke_test.py --data-path <root> --run-fit   # all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GREEN, RED, RESET = "\033[1;32m", "\033[1;31m", "\033[0m"


def ok(msg):
    print(f"{GREEN}✓{RESET} {msg}")


def fail(msg):
    print(f"{RED}✗ {msg}{RESET}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=None)
    p.add_argument("--eomt-dir", default=os.environ.get("EOMT_DIR", str(REPO_ROOT / "third_party" / "eomt")))
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", default=str(REPO_ROOT / "configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml"))
    p.add_argument("--run-fit", action="store_true")
    args = p.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"(CUDA unavailable — running checks on {device})")

    # 1. EoMT imports ------------------------------------------------------- #
    from eoworld.query_tokens.extract import ensure_eomt_on_path, load_config_fields
    ensure_eomt_on_path(args.eomt_dir)
    try:
        from models.eomt import EoMT
        from models.vit import ViT
        ok("EoMT clone found and imports resolve")
    except ImportError as e:
        fail(f"Could not import EoMT from {args.eomt_dir}: {e}")
        sys.exit("Run scripts/00_setup.sh first.")

    fields = load_config_fields(args.config)
    img_size = fields["img_size"]

    # 2. Data module -------------------------------------------------------- #
    if args.data_path:
        from eoworld.data.cholecseg8k import CholecSeg8kSemantic
        dm = CholecSeg8kSemantic(path=args.data_path, batch_size=2, num_workers=0,
                                 img_size=img_size)
        dm.setup()
        n = {k: len(getattr(dm, f"{k}_dataset")) for k in ("train", "val", "test")}
        assert sum(n.values()) > 0, "no frames discovered"
        img, target = dm.train_dataset[0]
        assert img.shape[-2:] == tuple(img_size), img.shape
        assert set(target) >= {"masks", "labels", "is_crowd"}
        assert target["masks"].shape[0] == target["labels"].shape[0]
        ok(f"Data module OK — frames train/val/test = {n['train']}/{n['val']}/{n['test']}, "
           f"sample masks {tuple(target['masks'].shape)}, labels {target['labels'].tolist()[:6]}...")
    else:
        print("(skipping data-module check — no --data-path)")

    # 3 + 4. Model forward + query extraction ------------------------------- #
    from eoworld.data.class_info import NUM_CLASSES
    from eoworld.query_tokens.extract import extract_frame_states

    encoder = ViT(img_size=img_size, patch_size=fields["patch_size"],
                  backbone_name=fields["backbone_name"])
    net = EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=fields["num_q"]).eval().to(device)
    x = torch.rand(2, 3, *img_size, device=device)
    with torch.no_grad():
        mask_logits, class_logits = net(x)
    assert class_logits[-1].shape == (2, fields["num_q"], NUM_CLASSES + 1), class_logits[-1].shape
    ok(f"Model forward OK — {len(class_logits)} prediction layers, "
       f"final class logits {tuple(class_logits[-1].shape)}, "
       f"mask logits {tuple(mask_logits[-1].shape)}")

    with torch.no_grad():
        z, cl = extract_frame_states(net, x)
    D = encoder.backbone.embed_dim
    assert z.shape == (2, fields["num_q"], D), z.shape
    ok(f"Query-token extraction OK — z_t shape {tuple(z.shape)} (num_q x D={D})")

    # 5. Optional tiny fit -------------------------------------------------- #
    if args.run_fit:
        if not args.data_path:
            fail("--run-fit needs --data-path")
            sys.exit(1)
        env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT}:{os.environ.get('PYTHONPATH','')}")
        cmd = [
            sys.executable, "main.py", "fit", "-c", str(Path(args.config).resolve()),
            "--data.path", args.data_path, "--data.num_workers", "0",
            "--trainer.limit_train_batches", "2", "--trainer.limit_val_batches", "1",
            "--trainer.max_epochs", "1", "--trainer.logger", "False",
            "--trainer.enable_checkpointing", "False", "--compile_disabled",
        ]
        print(f"\n[smoke] launching 2-step fit in {args.eomt_dir} ...")
        r = subprocess.run(cmd, cwd=args.eomt_dir, env=env)
        if r.returncode == 0:
            ok("2-step training loop ran end-to-end")
        else:
            fail(f"training loop exited with code {r.returncode}")
            sys.exit(r.returncode)

    print(f"\n{GREEN}Smoke test passed.{RESET} Ready for a full run (scripts/05_train_segmentation.sh).")


if __name__ == "__main__":
    main()
