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
│   └── download_checkpoints.py       # fetch EoMT pretrained weights (ungated)
├── docs/
│   ├── PROJECT_STRUCTURE.md
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

## The watershed encoding (why `class_info.py` matters)

CholecSeg8k's `*_watershed_mask.png` stores the annotation-tool class id as the
pixel value (same in all 3 channels) — a small, **non-contiguous** set
(e.g. Liver = 21, Fat = 12). `class_info.py` maps those to contiguous train ids
`0..12`, holds the official colour per class, and builds a 256-entry LUT with
unknown values → `IGNORE_INDEX (255)`. Verify it against your download with
`scripts/02_inspect_masks.py`.
