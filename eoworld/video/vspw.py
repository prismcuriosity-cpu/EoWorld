"""Convert CholecSeg8k into VSPW format for VidEoMT (video semantic segmentation).

VidEoMT is Detectron2-based and consumes video-semantic data in **VSPW** layout:

    <VSPW_ROOT>/
      data/
        <clip>/
          origin/00000.png 00001.png ...   # frames, temporally ordered
          mask/00000.png   00001.png ...    # 1-indexed class ids, 0 = ignore
      train.txt / val.txt / test.txt        # one clip name per line

CholecSeg8k is a natural fit: each ``videoNN_XXXXX`` folder is 80 consecutive
annotated frames — i.e. one short clip. We treat each such folder as a VSPW
"video" and split **by parent procedure** (``videoNN``) so no procedure leaks
across train/val/test.

Mask encoding — this is the subtle part and must match VidEoMT exactly:
VidEoMT's mapper (`_vspw_preprocess`) and the offline mIoU script both do
``0 -> ignore; value - 1``. So on disk a class with contiguous train id ``c`` is
stored as ``c + 1`` (range 1..13) and ignore is ``0``. After VidEoMT's ``-1`` it
becomes ``0..12`` again.

Only depends on numpy + PIL (no torch / detectron2), so it runs in the image env.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from eoworld.data.class_info import CHOLECSEG8K_CLASSES, NUM_CLASSES, watershed_to_train_ids
from eoworld.data.discovery import find_samples, make_video_splits


def _clip_of(img_path: Path) -> str:
    # e.g. .../video01/video01_00080/frame_80_endo.png -> "video01_00080"
    return img_path.parent.name


def _frame_num(img_path: Path) -> int:
    digits = "".join(ch for ch in img_path.stem.split("_")[1] if ch.isdigit()) \
        if "_" in img_path.stem else ""
    return int(digits) if digits else 0


def convert_to_vspw(
    root,
    out_root,
    link: bool = True,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict:
    """Materialise a VSPW-format copy of CholecSeg8k under ``out_root``.

    Args:
        root: CholecSeg8k dataset root.
        out_root: destination (the ``VSPW_ROOT`` above).
        link: symlink frames instead of copying (saves ~can't-afford duplication).
    Returns a summary dict.
    """
    root = Path(root)
    out_root = Path(out_root)
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = find_samples(root)
    if not samples:
        raise FileNotFoundError(f"No CholecSeg8k frames found under {root}")

    # Group frames by clip, ordered temporally.
    clips: dict[str, list] = {}
    clip_parent: dict[str, str] = {}
    for img_path, ws_path, video_id in samples:
        clip = _clip_of(img_path)
        clips.setdefault(clip, []).append((img_path, ws_path))
        clip_parent[clip] = video_id
    for clip in clips:
        clips[clip].sort(key=lambda pair: _frame_num(pair[0]))

    # Split by PARENT procedure so clips of the same video stay together.
    parents = sorted(set(clip_parent.values()))
    parent_split = make_video_splits(parents, val_frac=val_frac, test_frac=test_frac, seed=seed)
    parent_to_split = {p: s for s, ps in parent_split.items() for p in ps}

    split_clips = {"train": [], "val": [], "test": []}
    n_frames = 0
    for clip, frames in sorted(clips.items()):
        split = parent_to_split[clip_parent[clip]]
        split_clips[split].append(clip)
        origin_dir = data_dir / clip / "origin"
        mask_dir = data_dir / clip / "mask"
        origin_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, ws_path) in enumerate(frames):
            stem = f"{idx:05d}"
            # origin frame
            dst_img = origin_dir / f"{stem}.png"
            if not dst_img.exists():
                if link:
                    _symlink(img_path.resolve(), dst_img)
                else:
                    Image.open(img_path).convert("RGB").save(dst_img)
            # mask: contiguous train id + 1, ignore -> 0
            dst_mask = mask_dir / f"{stem}.png"
            if not dst_mask.exists():
                train_ids = watershed_to_train_ids(np.asarray(Image.open(ws_path)))
                vspw = np.where(train_ids < NUM_CLASSES, train_ids + 1, 0).astype(np.uint8)
                Image.fromarray(vspw).save(dst_mask)
            n_frames += 1

    for split, clip_list in split_clips.items():
        with open(out_root / f"{split}.txt", "w") as f:
            f.write("\n".join(sorted(clip_list)) + ("\n" if clip_list else ""))

    return {
        "n_clips": len(clips),
        "n_frames": n_frames,
        "clips_per_split": {k: len(v) for k, v in split_clips.items()},
        "parents_per_split": {k: len(v) for k, v in parent_split.items()},
        "out_root": str(out_root),
    }


def _symlink(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass
    except OSError:
        # Fall back to copy if the filesystem forbids symlinks (e.g. some Windows).
        Image.open(src).convert("RGB").save(dst)


def get_cholec_vss_metadata() -> dict:
    """Detectron2 metadata for the 13-class CholecSeg8k VSS dataset.

    Contiguous train ids are 0..12, so ``stuff_dataset_id_to_contiguous_id`` is the
    identity — matching how masks are stored (train_id+1 on disk).
    """
    classes = [c.name for c in CHOLECSEG8K_CLASSES]
    colors = [list(c.color) for c in CHOLECSEG8K_CLASSES]
    return {
        "stuff_classes": classes,
        "stuff_colors": colors,
        "thing_classes": None,
        "thing_colors": None,
        "stuff_classes_id": list(range(NUM_CLASSES)),
        "thing_classes_id": None,
        "stuff_dataset_id_to_contiguous_id": {i: i for i in range(NUM_CLASSES)},
        "thing_dataset_id_to_contiguous_id": None,
    }
