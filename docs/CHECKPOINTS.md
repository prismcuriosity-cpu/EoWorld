# Checkpoints

## TL;DR

Use the **DINOv2** checkpoints. They are **ungated** and are the recommended
starting weights for fine-tuning CholecSeg8k:

```bash
python scripts/download_checkpoints.py --which small base large
```

Then fine-tune with the class head skipped (different class count):

```bash
bash scripts/05_train_segmentation.sh \
  configs/cholecseg8k/semantic/eomt_base_640_dinov2.yaml <data_root> \
  checkpoints/coco_panoptic_eomt_base_640_2x/pytorch_model.bin
```

`scripts/05_train_segmentation.sh` automatically adds
`--model.load_ckpt_class_head False` when you pass a checkpoint.

## What we download (DINOv2, ungated)

| Name | HF repo | Use with |
|---|---|---|
| `small` | `tue-mps/coco_panoptic_eomt_small_640_2x` | `eomt_small_640_dinov2.yaml` |
| `base`  | `tue-mps/coco_panoptic_eomt_base_640_2x`  | `eomt_base_640_dinov2.yaml` |
| `large` | `tue-mps/coco_panoptic_eomt_large_640`    | `eomt_large_640_dinov2.yaml` |
| `large_ade_semantic` | `tue-mps/ade20k_semantic_eomt_large_512` | L, semantic-pretrained alt |
| `large_cityscapes_semantic` | `tue-mps/cityscapes_semantic_eomt_large_1024` | L, semantic-pretrained alt |

COCO-panoptic pretraining transfers well to surgical semantic segmentation; the
`ade`/`cityscapes` semantic-pretrained large variants are alternatives worth an
ablation.

## DINOv3 (GATED — plan around it)

The DINOv3-based EoMT weights (and the `eomt_large_640_dinov3.yaml` config) are
distributed as **deltas relative to Meta's original DINOv3 weights**, which
require **separately requesting gated access**. This is called out in the proposal
(§4.1.1 and the risk table §13) precisely because it can silently eat days of the
fast-track schedule.

**Action:** if you want DINOv3, request access on **day one**
(https://github.com/facebookresearch/dinov3 / the Hugging Face gated repos). Until
it lands, the DINOv2 path above is fully self-sufficient — nothing downstream
depends on which backbone family you start from.

When you do have access, the config sets `delta_weights: True`; add
`--model.delta_weights False` only if your checkpoint is already absolute.

## Fine-tuned outputs

Lightning writes your fine-tuned checkpoints under the trainer's root dir (e.g.
`runs/.../checkpoints/last.ckpt`). Pass that `.ckpt` to
`scripts/06_extract_query_tokens.py --ckpt`.
