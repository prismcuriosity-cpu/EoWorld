# Project structure

```
EoWorld/
├── README.md
├── requirements-eoworld.txt          # extra deps on top of EoMT's
├── configs/
│   └── cholecseg8k/semantic/
│       ├── eomt_small_640_dinov2.yaml    # EoMT-S (ungated)  — smoke / runtime
│       ├── eomt_base_640_dinov2.yaml     # EoMT-B (ungated)  — fast-track default
│       ├── eomt_large_640_dinov2.yaml    # EoMT-L (ungated)  — accuracy
│       └── eomt_large_640_dinov3.yaml    # EoMT-L (GATED, optional)
├── eoworld/
│   ├── data/
│   │   ├── class_info.py             # ★ canonical 13-class table (names, watershed
│   │   │                             #   ids, official colours) — single source of truth
│   │   ├── discovery.py              # frame discovery + video-level splits (torch-free)
│   │   └── cholecseg8k.py            # EoMT LightningDataModule for CholecSeg8k
│   ├── viz/
│   │   ├── style.py                  # journal figure styling + palette
│   │   └── dataset_report.py         # stats collector + all fig_* functions
│   ├── query_tokens/
│   │   └── extract.py                # build EoMT, hook query tokens → z_t cache
│   ├── video/                        # VidEoMT (Detectron2) integration
│   │   ├── vspw.py                   # CholecSeg8k → VSPW converter + VSS metadata
│   │   └── register_cholec_vss.py    # Detectron2 dataset registration (13-class)
│   └── utils/
├── scripts/
│   ├── 00_setup.sh                   # clone EoMT + install deps
│   ├── 01_download_data.py           # CholecSeg8k via kagglehub
│   ├── 02_inspect_masks.py           # verify watershed encoding vs class_info
│   ├── 03_visualize_dataset.py       # → journal-ready figures
│   ├── 04_quick_smoke_test.py        # end-to-end sanity gate
│   ├── 05_train_segmentation.sh      # fine-tune EoMT (wraps main.py fit)
│   ├── 06_extract_query_tokens.py    # extract z_t from a fine-tuned model
│   ├── compute_annealing.py          # recompute annealing steps if bs/epochs change
│   ├── download_checkpoints.py       # fetch EoMT pretrained weights (ungated)
│   └── video/                        # VIDEO pipeline (VidEoMT, separate env)
│       ├── 00_setup_videomt.sh       # clone VidEoMT + detectron2
│       ├── 10_convert_cholecseg8k_to_vspw.py
│       ├── train_videomt.py          # register dataset + launch VidEoMT training
│       ├── 12_eval_miou.py           # offline VSS mIoU (13 classes)
│       └── download_videomt_checkpoints.py
├── configs/cholecseg8k/
│   ├── semantic/                     # EoMT image configs (S/B/L)
│   └── video_vss/                    # VidEoMT-L video config
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── RUNNING.md                    # ★ step-by-step run guide (both pipelines)
│   ├── CHECKPOINTS.md
│   └── ROADMAP.md
├── assets/preview/                   # committed example figures
├── figures/                          # runtime figure output (git-ignored)
└── third_party/eomt/                 # upstream EoMT (cloned by setup, git-ignored)
```

## How the pieces connect to EoMT

EoMT is a PyTorch-Lightning CLI (`main.py fit -c <config>`). We integrate by:

1. **Data** — `eoworld/data/cholecseg8k.py` subclasses EoMT's
   `datasets.lightning_data_module.LightningDataModule` and reuses its
   `Transforms`. Configs point `data.class_path` at
   `eoworld.data.cholecseg8k.CholecSeg8kSemantic`. No upstream files change.

2. **Import resolution** — training runs from inside `third_party/eomt` (so EoMT's
   `datasets.*` / `models.*` resolve) with the EoWorld repo on `PYTHONPATH` (so
   `eoworld.*` resolves). `scripts/05_train_segmentation.sh` sets both up for you.

3. **Class count** — the CLI links `data.num_classes → model.num_classes`
   automatically; our data module reports 13.

4. **Query tokens** — `eoworld/query_tokens/extract.py` rebuilds the EoMT network,
   loads a fine-tuned checkpoint, and captures the final-layer query embeddings via
   a forward pre-hook on `class_head` (whose input *is* the query slice
   `backbone.norm(x)[:, :num_q, :]`).

## How VidEoMT (video) integrates

VidEoMT is Detectron2-based and consumes **VSPW**-format video-semantic data. We
integrate without touching upstream:

1. **Convert** — `eoworld/video/vspw.py` turns each CholecSeg8k 80-frame clip into
   a VSPW "video" (`origin/*.png` + `mask/*.png`). Masks are **1-indexed**
   (`train_id + 1`, `0` = ignore) to match VidEoMT's `_vspw_preprocess` and the
   offline mIoU script, which both do `0→ignore, value−1`.
2. **Register** — `eoworld/video/register_cholec_vss.py` registers
   `cholecseg8k_vss_video_{train,val,test}` with 13-class metadata and an identity
   `stuff_dataset_id_to_contiguous_id`.
3. **Launch** — `scripts/video/train_videomt.py` imports that registration, then
   hands off to VidEoMT's own `train_net_video.main` via Detectron2 `launch`, so
   the upstream trainer/model/evaluator run unmodified.
4. **Evaluate** — VidEoMT's `VSSEvaluator` dumps predictions; `scripts/video/12_eval_miou.py`
   reuses upstream's `Evaluator` with `num_class=13` to compute mIoU/per-class IoU.

The class names/colours come from the same `eoworld/data/class_info.py` table used
by the image pipeline, so both pipelines share one legend.

## The watershed encoding (why `class_info.py` matters)

CholecSeg8k's `*_watershed_mask.png` stores the annotation-tool class id as the
pixel value (same in all 3 channels) — a small, **non-contiguous** set
(e.g. Liver = 21, Fat = 12). `class_info.py` maps those to contiguous train ids
`0..12`, holds the official colour per class, and builds a 256-entry LUT with
unknown values → `IGNORE_INDEX (255)`. Verify it against your download with
`scripts/02_inspect_masks.py`.
