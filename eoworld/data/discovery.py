"""Dataset discovery & splitting — deliberately torch-free.

Kept separate from ``cholecseg8k.py`` so the visualisation scripts (which need
frame discovery and the video-level split) do not drag in torch / the EoMT
runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from eoworld.data.class_info import NUM_CLASSES, watershed_to_train_ids

IMG_SUFFIX = "_endo.png"
WATERSHED_SUFFIX = "_endo_watershed_mask.png"


def find_samples(root: Union[str, Path]) -> list[tuple[Path, Path, str]]:
    """Discover ``(image_path, watershed_path, video_id)`` triples under ``root``.

    Only frames with a matching watershed mask are kept.
    """
    root = Path(root)
    samples: list[tuple[Path, Path, str]] = []
    for img_path in sorted(root.rglob(f"*{IMG_SUFFIX}")):
        name = img_path.name
        if not name.endswith(IMG_SUFFIX) or "_mask" in name:
            continue
        watershed = img_path.with_name(name.replace(IMG_SUFFIX, WATERSHED_SUFFIX))
        if not watershed.exists():
            continue
        samples.append((img_path, watershed, _video_id_of(img_path, root)))
    return samples


def _video_id_of(img_path: Path, root: Path) -> str:
    return img_path.relative_to(root).parts[0]


def make_video_splits(
    video_ids: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, set[str]]:
    """Deterministically split *videos* (not frames) into train/val/test."""
    uniq = sorted(set(video_ids))
    order = sorted(uniq, key=lambda v: hashlib.md5(f"{seed}:{v}".encode()).hexdigest())
    n = len(order)
    n_test = max(1, round(n * test_frac)) if n > 2 else 0
    n_val = max(1, round(n * val_frac)) if n > 2 else 0
    test = set(order[:n_test])
    val = set(order[n_test : n_test + n_val])
    train = set(order[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


def inspect_watershed_values(root: Union[str, Path], max_frames: int = 200) -> dict:
    """Report the unique watershed pixel values present and classify each one.

    Run once after download to verify the canonical encoding in ``class_info``
    matches this copy of the dataset. Values are split into three buckets:

    * ``known_values``   — a real class (in CHOLECSEG8K_CLASSES)
    * ``ignore_values``  — expected unlabeled/void (IGNORE_WATERSHED_VALUES),
                           mapped to IGNORE_INDEX; harmless
    * ``unknown_values`` — neither: a genuine encoding mismatch to investigate
    """
    from eoworld.data.class_info import CHOLECSEG8K_CLASSES, IGNORE_WATERSHED_VALUES

    known = {c.watershed_id: c.name for c in CHOLECSEG8K_CLASSES}
    samples = find_samples(root)[:max_frames]
    seen: dict[int, int] = {}
    for _, ws_path, _ in samples:
        arr = np.asarray(Image.open(ws_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        vals, counts = np.unique(arr, return_counts=True)
        for v, ct in zip(vals.tolist(), counts.tolist()):
            seen[v] = seen.get(v, 0) + int(ct)
    total = sum(seen.values()) or 1
    unknown = sorted(v for v in seen if v not in known and v not in IGNORE_WATERSHED_VALUES)
    return {
        "n_frames_checked": len(samples),
        "known_values": {v: known[v] for v in sorted(seen) if v in known},
        "ignore_values": sorted(v for v in seen if v in IGNORE_WATERSHED_VALUES),
        "unknown_values": unknown,
        "pixel_counts": dict(sorted(seen.items())),
        "pixel_fraction": {v: seen[v] / total for v in sorted(seen)},
    }
