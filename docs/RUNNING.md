# EoWorld — Step-by-step running guide

This document walks every command for both pipelines, what to expect, and how to
fix the common errors. Two **separate conda environments** are used because the
two upstream repos have incompatible stacks:

| Pipeline | Repo | Stack | Conda env |
|---|---|---|---|
| **A. Image** perception (per-frame) | [EoMT](https://github.com/tue-mps/eomt) | PyTorch Lightning | `eoworld` |
| **B. Video** perception (temporal) | [VidEoMT](https://github.com/tue-mps/videomt) | Detectron2 | `videomt` |

Do them independently. The proposal (§4.1) uses EoMT for image-level perception
and VidEoMT for temporal perception; you can validate either first.

Conventions: `<root>` = your CholecSeg8k folder; `$REPO` = this repo's root.
Run every command from `$REPO` unless stated otherwise.

---

# Part A — Image pipeline (EoMT)

### A0. Environment + EoMT
```bash
conda create -n eoworld python=3.11 -y
conda activate eoworld
# RTX 5090 (Blackwell/sm_120) needs a recent CUDA build — check pytorch.org:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
bash scripts/00_setup.sh
```
**Expect:** `third_party/eomt/` created; deps installed; a "Done" banner.
**Verify:** `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

### A1. Download CholecSeg8k
```bash
python scripts/01_download_data.py          # needs ~/.kaggle/kaggle.json
```
**Expect:** a printed path like `/home/you/.cache/kagglehub/datasets/newslab/cholecseg8k/versions/X`.
Use that as `<root>`. (Manual download from Kaggle also fine.)

### A2. Verify the mask encoding
```bash
python scripts/02_inspect_masks.py --data-path <root>
```
**Expect:** `All watershed values are known. Encoding verified ✓`.
**If it flags unknown values:** edit `CHOLECSEG8K_CLASSES` in
`eoworld/data/class_info.py` to add them, then re-run.

### A3. Dataset figures (journal-ready)
```bash
python scripts/03_visualize_dataset.py --data-path <root>
```
**Expect:** `figures/fig_*.png` + `.pdf` (class frequency, per-video composition,
co-occurrence, split assignment, instrument timeline, sample overlays), plus a
cached `figures/stats.pkl`.

### A4. Smoke gate — confirm the code is OK before a full run
```bash
python scripts/04_quick_smoke_test.py --data-path <root>            # fast: data+model+query
python scripts/04_quick_smoke_test.py --data-path <root> --run-fit  # + a 2-step training loop
```
**Expect:** green ✓ lines ending in `Smoke test passed.` The first run downloads
the DINOv2-S backbone (~85 MB) for the model check.

### A5. Fine-tune EoMT (Exp. 8.1)
```bash
python scripts/download_checkpoints.py --which small base large
bash scripts/05_train_segmentation.sh \
  configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml <root> \
  checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin
```
**Expect:** Lightning starts logging train/val loss; checkpoints under
`third_party/eomt/lightning_logs/.../checkpoints/` (or your
`--trainer.default_root_dir`). Swap the config for `base`/`large` as needed.
**wandb:** either `wandb login`, or `export WANDB_MODE=offline` to skip it.

### A6. Extract query-token states (world-model z_t)
```bash
python scripts/06_extract_query_tokens.py \
  --config configs/cholecseg8k/semantic/eomt_small_640_dinov2.yaml \
  --ckpt <path-to>/last.ckpt --data-path <root> --out cache/query_tokens_small
```
**Expect:** one `cache/query_tokens_small/<video>.npz` per video with
`query_tokens (T,num_q,D)`, `class_conf`, `pred_class`.

---

# Part B — Video pipeline (VidEoMT)

VidEoMT is **Detectron2-based** and runs in its own env. Video semantic
segmentation uses the **VSPW** data format; CholecSeg8k converts to it cleanly
(each `videoNN_XXXXX` folder = one 80-frame clip).

### B0. Separate environment + VidEoMT + Detectron2
```bash
conda create -n videomt python=3.12.3 -y
conda activate videomt
# Match torch/CUDA to your 5090 (VidEoMT pins 2.7.0/cu126 — verify at pytorch.org):
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
bash scripts/video/00_setup_videomt.sh
```
**Expect:** `third_party/videomt/` cloned; Detectron2 + panopticapi + requirements
installed.
**Verify:** `python -c "import detectron2, videomt; print('ok')"` from
`third_party/videomt` (or with it on `PYTHONPATH`).

> **Detectron2 build is the #1 pain point.** It compiles against your *installed*
> torch/CUDA. If it fails: ensure `nvcc`/CUDA toolkit matches the torch CUDA
> version, `pip install ninja`, and retry the `--no-build-isolation` install. A
> torch/CUDA mismatch produces `undefined symbol` errors at `import detectron2`.

### B1. Convert CholecSeg8k → VSPW
```bash
export DETECTRON2_DATASETS=$HOME/datasets       # any folder you like
python scripts/video/10_convert_cholecseg8k_to_vspw.py \
  --data-path <root> --out $DETECTRON2_DATASETS/CholecSeg8k_VSPW
```
**Expect:** a summary (`n_clips`, `n_frames`, `clips_per_split`, `parents_per_split`)
and `$DETECTRON2_DATASETS/CholecSeg8k_VSPW/{data,train.txt,val.txt,test.txt}`.
Frames are **symlinked** by default (use `--copy` if your training box can't
follow symlinks into the source dir).

### B2. Get the segmenter init weights
```bash
python scripts/video/download_videomt_checkpoints.py --which vspw_segmenter
```
**Expect:** `checkpoints/videomt/vspw_segmenter.pth`. (This is the per-frame
segmenter that initialises VidEoMT training; its 124-class head won't fit our
13-class head — that mismatch is expected and the head is re-initialised.)

### B3. Train VidEoMT-L on CholecSeg8k
```bash
export WANDB_MODE=offline          # or `wandb login`
python scripts/video/train_videomt.py --num-gpus 1 \
  --config-file configs/cholecseg8k/video_vss/videomt_vitl_cholecseg8k.yaml \
  MODEL.WEIGHTS checkpoints/videomt/vspw_segmenter.pth \
  OUTPUT_DIR output/videomt_vitl_cholecseg8k
```
**Expect:** Detectron2 prints the merged config, "Loading …" (with some skipped
keys for the class head — expected), then iteration logs (`total_loss`, `lr`).
Checkpoints every 2000 iters under `output/videomt_vitl_cholecseg8k/`.
**If you OOM (32 GB):** lower memory in the config or on the CLI, e.g. append
`SOLVER.IMS_PER_BATCH 1 INPUT.SAMPLING_FRAME_NUM 2 INPUT.MIN_SIZE_TEST 400`.

### B4. Evaluate (dump predictions, then compute mIoU offline)
```bash
# 1) dump VSPW-format predictions
python scripts/video/train_videomt.py --num-gpus 1 \
  --config-file configs/cholecseg8k/video_vss/videomt_vitl_cholecseg8k.yaml \
  --eval-only MODEL.WEIGHTS output/videomt_vitl_cholecseg8k/model_final.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR output/videomt_vitl_cholecseg8k/eval

# 2) compute mIoU (13 classes) against the GT masks
python scripts/video/12_eval_miou.py \
  --vspw-root $DETECTRON2_DATASETS/CholecSeg8k_VSPW \
  --pred output/videomt_vitl_cholecseg8k/eval/inference --split val
```
**Expect:** predicted PNGs under `.../eval/inference/<clip>/`, then a report with
`mIoU`, pixel/mean accuracy, FW-IoU, and per-class IoU.

---

# Troubleshooting quick-reference

| Symptom | Fix |
|---|---|
| `import detectron2` → `undefined symbol` / build fails | torch↔CUDA↔detectron2 mismatch. Reinstall torch for your CUDA, `pip install ninja`, rebuild with `--no-build-isolation`. |
| `KeyError: 'cholecseg8k_vss_video_train'` | You ran upstream `train_net_video.py` directly. Use `scripts/video/train_videomt.py` (it registers the dataset first). |
| `No clips found` / `No videos found` | `DETECTRON2_DATASETS` must be the **parent** of `CholecSeg8k_VSPW`; re-run B1. |
| `AssertionError: No CholecSeg8k frames found` | Wrong `--data-path`; it must contain `videoNN/videoNN_XXXXX/frame_*_endo.png`. |
| wandb prompts for login / hangs | `export WANDB_MODE=offline` (both pipelines). |
| Symlinked frames unreadable on training node | Re-run B1 with `--copy`. |
| CUDA OOM (image) | Lower `--data.batch_size`; EoMT-L: try 2. Re-run `scripts/compute_annealing.py` if you change bs/epochs. |
| CUDA OOM (video) | `SOLVER.IMS_PER_BATCH 1 INPUT.SAMPLING_FRAME_NUM 2` and smaller `INPUT.MIN_SIZE_*`. |
| `main.py` can't find `eoworld.*` | Use `scripts/05_train_segmentation.sh` (sets `PYTHONPATH`), or export it yourself. |
| Wrong class colours / labels in figures | Verify encoding with `scripts/02_inspect_masks.py` and fix `class_info.py`. |

## Which backbone / weights?

- **Image:** DINOv2 configs are the ungated default (`docs/CHECKPOINTS.md`). DINOv3
  needs gated Meta access.
- **Video:** only a ViT-L VSS config exists upstream, so
  `videomt_vitl_cholecseg8k.yaml` is ViT-L. Init from `vspw_segmenter.pth`.
