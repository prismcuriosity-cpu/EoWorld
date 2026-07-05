"""Canonical CholecSeg8k class metadata — the single source of truth.

CholecSeg8k ships four PNGs per frame:

    frame_<n>_endo.png                 # RGB endoscopic frame
    frame_<n>_endo_mask.png            # raw annotation-tool mask (colour-coded)
    frame_<n>_endo_color_mask.png      # colour mask for visualisation
    frame_<n>_endo_watershed_mask.png  # watershed mask — SAME integer value in all
                                       # three channels, equal to the annotation-tool
                                       # class id. This is what we train on.

The watershed mask stores a small set of *non-contiguous* integer ids (e.g. Liver
is 21, Fat is 12). EoMT/semantic segmentation wants *contiguous* train ids
``0..num_classes-1`` (plus an ``ignore_index``). This module owns that mapping and
the official colour for every class so that every figure and every model uses one
consistent legend.

The numbers below are the widely-used canonical CholecSeg8k encoding. They are
verified against the actual dataset by ``scripts/02_inspect_masks.py`` /
``eoworld.data.discovery.inspect_watershed_values`` — run it once after download;
if any value disagrees, fix it *here* and everything else follows.

This file has no heavy dependencies (numpy only) so it can be imported both by the
plain visualisation scripts and by the EoMT training data module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE_INDEX = 255


@dataclass(frozen=True)
class ClassInfo:
    train_id: int          # contiguous id used by the model (0..12)
    name: str              # human-readable name
    short: str             # short label for dense figures
    watershed_id: int      # raw pixel value in *_watershed_mask.png
    color: tuple[int, int, int]  # official CholecSeg8k colour (RGB, 0-255)
    group: str             # coarse grouping: "anatomy" | "instrument" | "misc"


# Ordered by contiguous train id (0..12). Colours are the official CholecSeg8k
# palette; we reuse them for *every* figure so a class has one colour everywhere.
CHOLECSEG8K_CLASSES: list[ClassInfo] = [
    ClassInfo(0,  "Black Background",       "Background",  50, (127, 127, 127), "misc"),
    ClassInfo(1,  "Abdominal Wall",         "Abd. Wall",   11, (210, 140, 140), "anatomy"),
    ClassInfo(2,  "Liver",                  "Liver",       21, (255, 114, 114), "anatomy"),
    ClassInfo(3,  "Gastrointestinal Tract", "GI Tract",    13, (231,  70, 156), "anatomy"),
    ClassInfo(4,  "Fat",                    "Fat",         12, (186, 183,  75), "anatomy"),
    ClassInfo(5,  "Grasper",                "Grasper",     31, (170, 255,   0), "instrument"),
    ClassInfo(6,  "Connective Tissue",      "Conn. Tissue",23, (255,  85,   0), "anatomy"),
    ClassInfo(7,  "Blood",                  "Blood",       24, (255,   0,   0), "misc"),
    ClassInfo(8,  "Cystic Duct",            "Cystic Duct", 25, (255, 255,   0), "anatomy"),
    ClassInfo(9,  "L-hook Electrocautery",  "L-hook",      32, (169, 255, 184), "instrument"),
    ClassInfo(10, "Gallbladder",            "Gallbladder", 22, (255, 160, 165), "anatomy"),
    ClassInfo(11, "Hepatic Vein",           "Hep. Vein",   33, (  0,  50, 128), "anatomy"),
    ClassInfo(12, "Liver Ligament",         "Ligament",     5, (111,  74,   0), "anatomy"),
]

NUM_CLASSES = len(CHOLECSEG8K_CLASSES)

CLASS_NAMES = [c.name for c in CHOLECSEG8K_CLASSES]
SHORT_NAMES = [c.short for c in CHOLECSEG8K_CLASSES]
INSTRUMENT_TRAIN_IDS = [c.train_id for c in CHOLECSEG8K_CLASSES if c.group == "instrument"]
ANATOMY_TRAIN_IDS = [c.train_id for c in CHOLECSEG8K_CLASSES if c.group == "anatomy"]

# RGB colours as a (num_classes, 3) float array in [0, 1] for matplotlib.
COLORS_01 = np.array([c.color for c in CHOLECSEG8K_CLASSES], dtype=np.float32) / 255.0
# Same, 0-255 uint8, for painting label maps.
COLORS_255 = np.array([c.color for c in CHOLECSEG8K_CLASSES], dtype=np.uint8)


def _build_watershed_lut() -> np.ndarray:
    """256-entry lookup: watershed pixel value -> contiguous train id.

    Unknown values map to ``IGNORE_INDEX`` so unexpected pixels never silently
    become a real class.
    """
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for c in CHOLECSEG8K_CLASSES:
        lut[c.watershed_id] = c.train_id
    return lut


WATERSHED_TO_TRAIN = _build_watershed_lut()


def watershed_to_train_ids(mask: np.ndarray) -> np.ndarray:
    """Map a watershed mask (H, W) or (H, W, 3) to a (H, W) train-id map.

    The watershed mask holds the same value in all three channels, so we read
    channel 0 when given an RGB array.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    return WATERSHED_TO_TRAIN[mask.astype(np.uint8)]


def colorize_train_ids(train_ids: np.ndarray) -> np.ndarray:
    """Map a (H, W) train-id map to an (H, W, 3) uint8 RGB image using the
    official palette. ``IGNORE_INDEX`` is rendered black.
    """
    out = np.zeros((*train_ids.shape, 3), dtype=np.uint8)
    valid = train_ids < NUM_CLASSES
    out[valid] = COLORS_255[train_ids[valid]]
    return out
