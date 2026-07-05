#!/usr/bin/env python3
"""Recompute EoMT's attention-mask-annealing schedule for a given dataset size.

EoMT stages the disabling of masked attention across four blocks over training.
The step numbers in the configs assume the documented (frames, batch_size,
epochs). If you change any of those, regenerate the schedule:

    python scripts/compute_annealing.py --frames 5650 --batch-size 8 --epochs 40

then paste the printed ``start``/``end`` lists into your config's
``attn_mask_annealing_start_steps`` / ``attn_mask_annealing_end_steps``.
"""

import argparse
import math

# Proportions of total training steps, taken from the official ADE20K schedule.
START_FRAC = (0.0, 0.323, 0.484, 0.645)
END_FRAC = (0.148, 0.484, 0.645, 0.806)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, required=True, help="number of TRAIN frames")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    args = p.parse_args()

    steps_per_epoch = math.ceil(args.frames / args.batch_size)
    total = steps_per_epoch * args.epochs
    start = [round(f * total) for f in START_FRAC]
    end = [round(f * total) for f in END_FRAC]

    print(f"steps_per_epoch = {steps_per_epoch}")
    print(f"total_steps     = {total}")
    print(f"attn_mask_annealing_start_steps: {start}")
    print(f"attn_mask_annealing_end_steps:   {end}")


if __name__ == "__main__":
    main()
