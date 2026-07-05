"""Register CholecSeg8k as a VidEoMT video-semantic-segmentation dataset.

Mirrors VidEoMT's own ``data_video/datasets/vss.py`` (the VSPW registration) but
with our 13 CholecSeg8k classes and our split files. Importing this module
registers three datasets in Detectron2's ``DatasetCatalog``:

    cholecseg8k_vss_video_train / _val / _test

The data is expected at ``$DETECTRON2_DATASETS/CholecSeg8k_VSPW`` in VSPW layout
(produced by ``scripts/video/10_convert_cholecseg8k_to_vspw.py``).

Requires detectron2 (VidEoMT env). Our train/eval wrappers import this module
before launching so the datasets exist by the time the config references them.
"""

from __future__ import annotations

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from eoworld.video.vspw import get_cholec_vss_metadata

DATASET_DIRNAME = "CholecSeg8k_VSPW"

_SPLITS = {
    "cholecseg8k_vss_video_train": "train.txt",
    "cholecseg8k_vss_video_val": "val.txt",
    "cholecseg8k_vss_video_test": "test.txt",
}


def _gen_video_lists(image_root, split_txt):
    with open(split_txt, "r") as f:
        clips = [line.strip() for line in f if line.strip()]
    ret = []
    for clip in clips:
        clip_dir = os.path.join(image_root, clip)
        origin = sorted(os.listdir(os.path.join(clip_dir, "origin")))
        img_files = [os.path.join(clip_dir, "origin", x) for x in origin]
        mask_dir = os.path.join(clip_dir, "mask")
        if os.path.exists(mask_dir):
            masks = sorted(os.listdir(mask_dir))
            mask_files = [os.path.join(mask_dir, x) for x in masks]
        else:
            mask_files = [None] * len(img_files)
        ret.append({"video_id": clip, "file_names": img_files, "sem_mask_names": mask_files})
    assert ret, f"No clips found for split {split_txt}"
    return ret


def register_cholec_vss(root: str | None = None) -> None:
    root = root or os.getenv("DETECTRON2_DATASETS", "datasets")
    base = os.path.join(root, DATASET_DIRNAME)
    image_root = os.path.join(base, "data")
    metadata = get_cholec_vss_metadata()
    for name, split_txt in _SPLITS.items():
        if name in DatasetCatalog.list():
            continue  # already registered (idempotent)
        split_path = os.path.join(base, split_txt)
        DatasetCatalog.register(
            name, lambda ir=image_root, sp=split_path: _gen_video_lists(ir, sp)
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            evaluator_type=None,
            ignore_label=255,
            **metadata,
        )


# Register on import (matches VidEoMT's builtin convention).
register_cholec_vss()
