"""Journal-ready CholecSeg8k dataset figures.

Two layers:

* ``compute_dataset_stats`` scans the raw dataset once (reads every watershed
  mask), returns a stats dict, and caches it to disk.
* ``fig_*`` functions turn that stats dict into publication figures. Every figure
  reuses the official class palette so a class has one colour across the paper.

``make_synthetic_stats`` produces a plausible stats dict without the dataset, so
the figure code can be previewed / smoke-tested offline.

Run via ``scripts/03_visualize_dataset.py`` (see ``--help``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from eoworld.data.class_info import (
    ANATOMY_TRAIN_IDS,
    CHOLECSEG8K_CLASSES,
    CLASS_NAMES,
    COLORS_01,
    INSTRUMENT_TRAIN_IDS,
    NUM_CLASSES,
    SHORT_NAMES,
    colorize_train_ids,
    watershed_to_train_ids,
)


# --------------------------------------------------------------------------- #
# Stats collection
# --------------------------------------------------------------------------- #
def _frame_sort_key(sample) -> tuple:
    # sample = (img_path, ws_path, video_id); order frames temporally within video
    stem = sample[0].parent.name  # e.g. video01_00080
    digits = "".join(ch for ch in stem.split("_")[-1] if ch.isdigit())
    return (sample[2], int(digits) if digits else 0, sample[0].name)


def compute_dataset_stats(
    root, cache_path: Optional[Path] = None, limit: Optional[int] = None
) -> dict:
    """Scan every watershed mask and aggregate per-class / per-video statistics."""
    from eoworld.data.discovery import find_samples

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    samples = find_samples(root)
    samples.sort(key=_frame_sort_key)
    if limit:
        samples = samples[:limit]
    videos = sorted({vid for *_, vid in samples})
    vidx = {v: i for i, v in enumerate(videos)}
    V, C = len(videos), NUM_CLASSES

    class_pixel_counts = np.zeros(C, dtype=np.int64)
    class_frame_counts = np.zeros(C, dtype=np.int64)
    cooccur = np.zeros((C, C), dtype=np.int64)
    per_video_frames = np.zeros(V, dtype=np.int64)
    per_video_class_pixels = np.zeros((V, C), dtype=np.int64)
    per_video_class_frames = np.zeros((V, C), dtype=np.int64)
    timeline: dict[str, list] = {v: [] for v in videos}

    for img_path, ws_path, vid in samples:
        arr = np.asarray(Image.open(ws_path))
        ids = watershed_to_train_ids(arr)
        vals, counts = np.unique(ids, return_counts=True)
        present = [int(v) for v in vals if v < NUM_CLASSES]
        vi = vidx[vid]
        per_video_frames[vi] += 1
        for v, ct in zip(vals.tolist(), counts.tolist()):
            if v < NUM_CLASSES:
                class_pixel_counts[v] += ct
                per_video_class_pixels[vi, v] += ct
        for a in present:
            class_frame_counts[a] += 1
            per_video_class_frames[vi, a] += 1
            for b in present:
                cooccur[a, b] += 1
        row = np.zeros(len(INSTRUMENT_TRAIN_IDS), dtype=bool)
        for k, cid in enumerate(INSTRUMENT_TRAIN_IDS):
            row[k] = cid in present
        timeline[vid].append(row)

    stats = {
        "n_frames": len(samples),
        "videos": videos,
        "class_pixel_counts": class_pixel_counts,
        "class_frame_counts": class_frame_counts,
        "cooccur": cooccur,
        "per_video_frames": per_video_frames,
        "per_video_class_pixels": per_video_class_pixels,
        "per_video_class_frames": per_video_class_frames,
        "instrument_timeline": {
            v: (np.stack(rows) if rows else np.zeros((0, len(INSTRUMENT_TRAIN_IDS)), bool))
            for v, rows in timeline.items()
        },
    }
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(stats, f)
    return stats


def make_synthetic_stats(n_videos: int = 17, seed: int = 0) -> dict:
    """A plausible stats dict for offline figure previews (no dataset needed)."""
    rng = np.random.default_rng(seed)
    C = NUM_CLASSES
    videos = [f"video{ i+1:02d}" for i in range(n_videos)]
    # Anatomy dominates pixels; instruments are rare -> strong class imbalance.
    base = np.array([8, 12, 30, 3, 22, 2, 6, 1, 0.6, 1.5, 9, 0.8, 0.9])[:C]
    per_video_frames = rng.integers(300, 600, size=n_videos)
    per_video_class_pixels = np.zeros((n_videos, C), dtype=np.int64)
    per_video_class_frames = np.zeros((n_videos, C), dtype=np.int64)
    cooccur = np.zeros((C, C), dtype=np.int64)
    timeline = {}
    for vi, v in enumerate(videos):
        nf = int(per_video_frames[vi])
        weights = base * rng.uniform(0.5, 1.5, size=C)
        weights = weights / weights.sum()
        per_video_class_pixels[vi] = (weights * nf * 854 * 480).astype(np.int64)
        pres_prob = np.clip(weights * 6, 0.05, 0.98)
        pres_prob[0:5] = np.clip(pres_prob[0:5] + 0.5, 0, 1)  # anatomy usually present
        frame_pres = rng.random((nf, C)) < pres_prob
        per_video_class_frames[vi] = frame_pres.sum(0)
        for f in range(nf):
            present = np.where(frame_pres[f])[0]
            for a in present:
                for b in present:
                    cooccur[a, b] += 1
        timeline[v] = frame_pres[:, INSTRUMENT_TRAIN_IDS].copy()
    return {
        "n_frames": int(per_video_frames.sum()),
        "videos": videos,
        "class_pixel_counts": per_video_class_pixels.sum(0),
        "class_frame_counts": per_video_class_frames.sum(0),
        "cooccur": cooccur,
        "per_video_frames": per_video_frames,
        "per_video_class_pixels": per_video_class_pixels,
        "per_video_class_frames": per_video_class_frames,
        "instrument_timeline": timeline,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_palette_card(save_dir, save):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_axis_off()
    ax.set_title("CholecSeg8k — class legend & watershed encoding", loc="left")
    for c in CHOLECSEG8K_CLASSES:
        y = NUM_CLASSES - c.train_id
        ax.add_patch(plt.Rectangle((0, y - 0.4), 0.7, 0.8, color=COLORS_01[c.train_id]))
        ax.text(0.9, y, f"{c.train_id:>2}  {c.name}", va="center", fontsize=9)
        ax.text(5.3, y, f"ws={c.watershed_id}", va="center", fontsize=8, color="#6b6b6b")
        ax.text(6.4, y, c.group, va="center", fontsize=8, color="#6b6b6b")
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, NUM_CLASSES + 1)
    return save(fig, save_dir, "fig_palette_card")


def fig_class_frequency(stats, save_dir, save):
    """Two-panel: pixel share (log) + frame presence rate. The class-imbalance
    figure reviewers look for first."""
    import matplotlib.pyplot as plt

    order = np.argsort(stats["class_pixel_counts"])[::-1]
    px = stats["class_pixel_counts"][order]
    px_share = px / px.sum() * 100
    frame_rate = stats["class_frame_counts"][order] / stats["n_frames"] * 100
    names = [SHORT_NAMES[i] for i in order]
    colors = COLORS_01[order]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    y = np.arange(len(order))

    a0.barh(y, px_share, color=colors, edgecolor="white", linewidth=0.6)
    a0.set_xscale("log")
    a0.set_yticks(y, names)
    a0.invert_yaxis()
    a0.set_xlabel("Share of labelled pixels (%, log scale)")
    a0.set_title("Pixel-level class imbalance", loc="left")
    for yi, val in zip(y, px_share):
        a0.text(val * 1.1, yi, f"{val:.2g}", va="center", fontsize=7, color="#1a1a1a")

    a1.barh(y, frame_rate, color=colors, edgecolor="white", linewidth=0.6)
    a1.set_yticks(y, names)
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("Frames containing the class (%)")
    a1.set_title("Per-frame class presence", loc="left")
    for yi, val in zip(y, frame_rate):
        a1.text(val + 1, yi, f"{val:.0f}", va="center", fontsize=7, color="#1a1a1a")

    fig.suptitle(
        f"CholecSeg8k class statistics  ·  {stats['n_frames']:,} frames, "
        f"{len(stats['videos'])} videos",
        x=0.01, ha="left", fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, save_dir, "fig_class_frequency")


def fig_per_video_composition(stats, save_dir, save):
    """Stacked per-video pixel composition — shows cross-procedure domain shift."""
    import matplotlib.pyplot as plt

    px = stats["per_video_class_pixels"].astype(float)
    frac = px / px.sum(1, keepdims=True).clip(min=1) * 100
    videos = stats["videos"]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(videos))
    bottom = np.zeros(len(videos))
    for c in range(NUM_CLASSES):
        ax.bar(x, frac[:, c], bottom=bottom, color=COLORS_01[c],
               width=0.82, edgecolor="white", linewidth=0.4, label=CLASS_NAMES[c])
        bottom += frac[:, c]
    ax.set_xticks(x, videos, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pixel composition (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-video class composition — domain shift across procedures", loc="left")
    ax.legend(ncol=1, bbox_to_anchor=(1.005, 1.0), loc="upper left", fontsize=7)
    fig.tight_layout()
    return save(fig, save_dir, "fig_per_video_composition")


def fig_cooccurrence(stats, save_dir, save):
    """Class co-occurrence (conditional P(row present | col present)) heatmap.
    Motivates the object-centric world-model state: which structures coexist."""
    import matplotlib.pyplot as plt

    co = stats["cooccur"].astype(float)
    diag = np.diag(co).clip(min=1)
    cond = co / diag[None, :]  # P(row | col)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cond, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), SHORT_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(NUM_CLASSES), SHORT_NAMES, fontsize=7)
    ax.set_title("Class co-occurrence  P(row present | col present)", loc="left")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cond[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if v < 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("conditional presence probability")
    fig.tight_layout()
    return save(fig, save_dir, "fig_cooccurrence")


def fig_frames_per_video(stats, save_dir, save, splits=None):
    """Frames per video, coloured by train/val/test assignment."""
    import matplotlib.pyplot as plt

    videos = stats["videos"]
    nf = stats["per_video_frames"]
    split_color = {"train": "#3b6fb0", "val": "#e0a500", "test": "#c0504d"}
    if splits is None:
        from eoworld.data.discovery import make_video_splits
        splits = make_video_splits(videos)
    assign = {v: s for s, vs in splits.items() for v in vs}
    colors = [split_color.get(assign.get(v, "train")) for v in videos]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(videos))
    ax.bar(x, nf, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, videos, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Annotated frames")
    ax.set_title("Frames per video and split assignment (video-level split)", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f"{s}  ({sum(1 for v in videos if assign.get(v)==s)} vids)")
                       for s, c in split_color.items()], loc="upper right")
    for xi, val in zip(x, nf):
        ax.text(xi, val + max(nf) * 0.01, f"{int(val)}", ha="center", fontsize=6)
    fig.tight_layout()
    return save(fig, save_dir, "fig_frames_per_video")


def fig_instrument_timeline(stats, save_dir, save, videos=None):
    """Instrument presence over time for a few videos — the temporal structure
    the latent-dynamics model is meant to forecast."""
    import matplotlib.pyplot as plt

    tl = stats["instrument_timeline"]
    videos = videos or stats["videos"][:6]
    inst_names = [CLASS_NAMES[i] for i in INSTRUMENT_TRAIN_IDS]
    inst_colors = [COLORS_01[i] for i in INSTRUMENT_TRAIN_IDS]

    fig, axes = plt.subplots(len(videos), 1, figsize=(9.5, 1.0 + 0.75 * len(videos)),
                             sharex=False)
    if len(videos) == 1:
        axes = [axes]
    for ax, v in zip(axes, videos):
        seq = tl.get(v, np.zeros((0, len(INSTRUMENT_TRAIN_IDS)), bool))
        for k in range(len(INSTRUMENT_TRAIN_IDS)):
            present = seq[:, k] if seq.size else np.array([])
            ax.fill_between(np.arange(len(present)), k, k + 0.8,
                            where=present, color=inst_colors[k], step="pre", linewidth=0)
        ax.set_yticks(np.arange(len(inst_names)) + 0.4, inst_names, fontsize=7)
        ax.set_ylabel(v, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xlim(0, max(1, seq.shape[0]))
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Instrument presence over time (motivates temporal forecasting)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, save_dir, "fig_instrument_timeline")


def fig_sample_grid(root, save_dir, save, n=4, seed=0):
    """Qualitative grid: RGB | ground-truth colour | overlay, for n random frames.
    Reads the actual dataset (skipped in synthetic/preview mode)."""
    import matplotlib.pyplot as plt
    from eoworld.data.discovery import find_samples

    samples = find_samples(root)
    if not samples:
        return []
    rng = np.random.default_rng(seed)
    picks = [samples[i] for i in rng.choice(len(samples), size=min(n, len(samples)), replace=False)]

    fig, axes = plt.subplots(len(picks), 3, figsize=(8.5, 2.7 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    col_titles = ["Endoscopic frame", "Ground-truth labels", "Overlay"]
    for r, (img_path, ws_path, vid) in enumerate(picks):
        img = np.asarray(Image.open(img_path).convert("RGB"))
        ids = watershed_to_train_ids(np.asarray(Image.open(ws_path)))
        lab = colorize_train_ids(ids)
        overlay = (0.55 * img + 0.45 * lab).astype(np.uint8)
        for c, im in enumerate([img, lab, overlay]):
            axes[r, c].imshow(im)
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=9)
        axes[r, 0].text(0.02, 0.95, vid, transform=axes[r, 0].transAxes,
                        color="white", fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    fig.tight_layout()
    return save(fig, save_dir, "fig_sample_grid")
