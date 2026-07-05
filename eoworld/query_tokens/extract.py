"""Extract EoMT segmentation-query tokens as the world-model latent state z_t.

The proposal's core idea: EoMT's *segmentation-query tokens* — not the rendered
masks — are the compact, object-centric latent state to forecast. In EoMT the
final query embeddings are ``backbone.norm(x)[:, :num_q, :]`` at the last layer;
the class head reads exactly those. We capture them with a forward pre-hook on
``class_head`` (its input is that query slice) and pair them with the matching
per-query class logits (confidence), giving, per frame:

    z_t          : (num_q, D)       — the latent state
    class_logits : (num_q, C+1)     — per-query class distribution (last = no-object)

Frames are processed in temporal order and cached per video as ``.npz`` so the
downstream latent-dynamics model (Exp. 8.2) can train on ``z_1..z_T`` sequences.

Run via ``scripts/06_extract_query_tokens.py`` (adds the EoMT clone to sys.path).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def ensure_eomt_on_path(eomt_dir: str | Path) -> None:
    eomt_dir = str(Path(eomt_dir).resolve())
    if eomt_dir not in sys.path:
        sys.path.insert(0, eomt_dir)


def _strip_prefixes(state_dict: dict) -> dict:
    """Normalise checkpoint keys to match a bare ``EoMT`` module.

    Handles Lightning (``network.``) and torch.compile (``_orig_mod.``) prefixes.
    """
    out = {}
    for k, v in state_dict.items():
        for pref in ("_orig_mod.", "network."):
            if k.startswith(pref):
                k = k[len(pref):]
        out[k] = v
    return out


def load_config_fields(config_path: str | Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    net = cfg["model"]["init_args"]["network"]["init_args"]
    enc = net["encoder"]["init_args"]
    data = cfg.get("data", {}).get("init_args", {})
    img_size = data.get("img_size", [640, 640])
    return {
        "backbone_name": enc["backbone_name"],
        "patch_size": enc.get("patch_size", 16 if "/" in enc["backbone_name"] else 14),
        "num_q": net.get("num_q", 100),
        "img_size": tuple(img_size),
    }


def build_model(
    config_path: str | Path,
    ckpt_path: str | Path,
    num_classes: int,
    masked_attn_enabled: bool = True,
    device: str = "cuda",
) -> tuple[torch.nn.Module, dict]:
    """Build an EoMT network and load a (fine-tuned, absolute) checkpoint."""
    from models.eomt import EoMT
    from models.vit import ViT

    fields = load_config_fields(config_path)
    encoder = ViT(
        img_size=fields["img_size"],
        patch_size=fields["patch_size"],
        backbone_name=fields["backbone_name"],
        ckpt_path=str(ckpt_path),  # skip re-downloading timm weights
    )
    net = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=fields["num_q"],
        masked_attn_enabled=masked_attn_enabled,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = _strip_prefixes(state)
    missing, unexpected = net.load_state_dict(state, strict=False)
    dropped = [k for k in missing if not k.startswith("encoder.pixel_")]
    if dropped:
        print(f"[extract] {len(dropped)} missing keys (first few): {dropped[:5]}")
    if unexpected:
        print(f"[extract] {len(unexpected)} unexpected keys (first few): {unexpected[:5]}")

    net.eval().to(device)
    return net, fields


@torch.no_grad()
def extract_frame_states(
    net: torch.nn.Module,
    imgs: torch.Tensor,  # (B, 3, H, W) float in [0, 1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z, class_logits)`` for the FINAL layer.

    z            : (B, num_q, D)
    class_logits : (B, num_q, C+1)
    """
    captured: list[torch.Tensor] = []

    def pre_hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = net.class_head.register_forward_pre_hook(pre_hook)
    try:
        _mask_logits_per_layer, class_logits_per_layer = net(imgs)
    finally:
        handle.remove()

    z = captured[-1]  # final-layer query embeddings
    class_logits = class_logits_per_layer[-1]
    return z, class_logits


def load_frame(path: str | Path, img_size: tuple[int, int], device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()  # (3,H,W), 0-255
    t = F.interpolate(t[None], size=img_size, mode="bilinear", align_corners=False)[0]
    return (t / 255.0).to(device)


def extract_dataset(
    net: torch.nn.Module,
    data_root: str | Path,
    out_dir: str | Path,
    img_size: tuple[int, int],
    num_classes: int,
    device: str = "cuda",
    batch_size: int = 16,
    videos: Optional[list[str]] = None,
    save_logits: bool = True,
) -> None:
    """Extract query-token states for every frame, cached per video as .npz."""
    from eoworld.viz.dataset_report import _frame_sort_key
    from eoworld.data.discovery import find_samples

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = find_samples(data_root)
    samples.sort(key=_frame_sort_key)

    by_video: dict[str, list] = {}
    for s in samples:
        by_video.setdefault(s[2], []).append(s)

    target_videos = videos or sorted(by_video)
    for vid in target_videos:
        frames = by_video.get(vid, [])
        if not frames:
            continue
        z_all, logits_all, conf_all, pred_all, paths = [], [], [], [], []
        for i in range(0, len(frames), batch_size):
            chunk = frames[i : i + batch_size]
            batch = torch.stack([load_frame(p, img_size, device) for p, _, _ in chunk])
            z, class_logits = extract_frame_states(net, batch)
            probs = class_logits.softmax(-1)[..., :num_classes]  # drop no-object
            conf, pred = probs.max(-1)
            z_all.append(z.half().cpu().numpy())
            conf_all.append(conf.half().cpu().numpy())
            pred_all.append(pred.to(torch.int16).cpu().numpy())
            if save_logits:
                logits_all.append(class_logits.half().cpu().numpy())
            paths.extend(str(p) for p, _, _ in chunk)

        payload = {
            "query_tokens": np.concatenate(z_all),          # (T, num_q, D) f16
            "class_conf": np.concatenate(conf_all),          # (T, num_q)    f16
            "pred_class": np.concatenate(pred_all),          # (T, num_q)    i16
            "frame_paths": np.array(paths),
        }
        if save_logits:
            payload["class_logits"] = np.concatenate(logits_all)  # (T, num_q, C+1)
        np.savez_compressed(out_dir / f"{vid}.npz", **payload)
        print(f"[extract] {vid}: {len(frames)} frames -> {out_dir / (vid + '.npz')}")
