"""CholecSeg8k data module for EoMT (semantic segmentation).

This plugs the raw Kaggle CholecSeg8k folder tree straight into the EoMT training
CLI. It deliberately does NOT use EoMT's zip-based ``datasets.dataset.Dataset``
(that is built for COCO/ADE zips); instead it reads the folder tree directly and
returns the exact ``(img, target)`` format EoMT's semantic path expects, reusing
EoMT's own ``Transforms``.

Folder layout produced by the Kaggle download::

    <root>/
      video01/
        video01_00080/
          frame_80_endo.png
          frame_80_endo_watershed_mask.png
          ...
      video09/ ...

Usage (from inside the eomt clone, with the EoWorld repo on PYTHONPATH)::

    python main.py fit \
      -c /path/to/EoWorld/configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml \
      --data.path /path/to/cholecseg8k

The config's ``class_path`` is ``eoworld.data.cholecseg8k.CholecSeg8kSemantic``.

Requires (only when actually training/extracting): the EoMT repo importable
(``datasets.*``, ``models.*``) and this repo importable (``eoworld.*``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

# EoMT-side imports. Available whenever this module is used for training/eval
# (i.e. running `main.py` from the eomt clone). Not needed by the viz scripts,
# which import `eoworld.data.class_info` / `eoworld.data.discovery` instead.
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

from eoworld.data.class_info import (
    IGNORE_INDEX,
    NUM_CLASSES,
    watershed_to_train_ids,
)
from eoworld.data.discovery import (  # re-exported for convenience
    IMG_SUFFIX,
    WATERSHED_SUFFIX,
    find_samples,
    make_video_splits,
)


class CholecSeg8kDataset(torch.utils.data.Dataset):
    """Returns ``(img, target)`` matching EoMT's semantic format.

    ``target = {"masks": (K,H,W) bool, "labels": (K,) long, "is_crowd": (K,) bool}``
    where each of the K present classes contributes one binary mask.
    """

    def __init__(
        self,
        samples: list[tuple[Path, Path, str]],
        transforms: Optional[Transforms] = None,
        ignore_background: bool = False,
    ):
        self.samples = samples
        self.transforms = transforms
        self.ignore_background = ignore_background

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, ws_path, _ = self.samples[index]

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        ws = np.asarray(Image.open(ws_path))
        train_ids = watershed_to_train_ids(ws)  # (H, W) uint8, 255 = ignore

        masks, labels = [], []
        present = np.unique(train_ids)
        for cls in present.tolist():
            if cls == IGNORE_INDEX or cls >= NUM_CLASSES:
                continue
            if self.ignore_background and cls == 0:
                continue
            masks.append(torch.from_numpy(train_ids == cls))
            labels.append(cls)

        if not masks:  # degenerate frame: fall back to a single ignore mask
            h, w = train_ids.shape
            masks = [torch.zeros((h, w), dtype=torch.bool)]
            labels = [0]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.zeros(len(labels), dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CholecSeg8kSemantic(LightningDataModule):
    """LightningDataModule wiring CholecSeg8k into EoMT's semantic CLI."""

    def __init__(
        self,
        path: str,
        num_workers: int = 8,
        batch_size: int = 8,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = NUM_CLASSES,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        split_seed: int = 0,
        ignore_background: bool = False,
        check_empty_targets: bool = True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=IGNORE_INDEX,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed
        self.ignore_background = ignore_background

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None) -> "CholecSeg8kSemantic":
        all_samples = find_samples(self.path)
        if not all_samples:
            raise FileNotFoundError(
                f"No CholecSeg8k frames found under {self.path!r}. Expected files "
                f"named '*{IMG_SUFFIX}' with matching '*{WATERSHED_SUFFIX}'."
            )
        splits = make_video_splits(
            [vid for *_, vid in all_samples],
            val_frac=self.val_frac,
            test_frac=self.test_frac,
            seed=self.split_seed,
        )

        def subset(name: str) -> list[tuple[Path, Path, str]]:
            return [s for s in all_samples if s[2] in splits[name]]

        self.train_dataset = CholecSeg8kDataset(
            subset("train"), self.transforms, self.ignore_background
        )
        self.val_dataset = CholecSeg8kDataset(
            subset("val"), transforms=None, ignore_background=self.ignore_background
        )
        self.test_dataset = CholecSeg8kDataset(
            subset("test"), transforms=None, ignore_background=self.ignore_background
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
