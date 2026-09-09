#!/usr/bin/env python3
"""Create Figure 3: attention rollout or patch/gradient saliency map.

For ViT/DeiT timm models, the script tries to capture attention matrices from
attn_drop layers and compute attention rollout. For CNNs or unsupported models,
it falls back to source-only input-gradient saliency. This is a visualization tool;
it does not access target models during adversarial generation.
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import glob
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scripts.batch_io import get_sample_from_attack, load_clean_tensor
from src.datasets import ImageNetCSVDataset, imagenet_transform
from src.models import create_model
from src.utils import get_device, set_model_cache_dir


def get_sample_from_csv(data_dir, selected_csv, index):
    dataset = ImageNetCSVDataset(data_dir, selected_csv, imagenet_transform())
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index={index} out of range for {selected_csv}")
    return dataset[index]


def tensor_to_np_img(t):
    t = t.detach().cpu().clamp(0, 1)
    return t.permute(1, 2, 0).numpy()


def normalize_map(m):
    m = m.detach().float().cpu()
    m = m - m.min()
    return m / (m.max() + 1e-12)


def overlay_heatmap(image_tensor, heatmap, alpha=0.45):
    img = tensor_to_np_img(image_tensor)
    h = normalize_map(heatmap).numpy()
    cmap = plt.get_cmap("jet")(h)[..., :3]
    out = (1 - alpha) * img + alpha * cmap
    return np.clip(out, 0, 1)


def collect_attention_rollout(model, x):
    """Return heatmap [H,W] if attention rollout succeeds, otherwise None."""
    core = getattr(model, "model", model)
    if not hasattr(core, "blocks"):
        return None

    attn_tensors = []
    handles = []

    def hook(_module, _inp, out):
        if torch.is_tensor(out) and out.ndim == 4:
            attn_tensors.append(out.detach())

    try:
        for blk in core.blocks:
            attn = getattr(blk, "attn", None)
            attn_drop = getattr(attn, "attn_drop", None)
            if attn_drop is not None:
                handles.append(attn_drop.register_forward_hook(hook))
        if not handles:
            return None
        model.eval()
        with torch.no_grad():
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    if not attn_tensors:
        return None

    # Attention tensors: [B, heads, tokens, tokens]
    rollout = None
    for attn in attn_tensors:
        a = attn[0].mean(dim=0)  # [N,N]
        n = a.shape[-1]
        eye = torch.eye(n, device=a.device, dtype=a.dtype)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = a if rollout is None else a @ rollout

    if rollout is None or rollout.shape[0] < 2:
        return None
    cls_to_patch = rollout[0, 1:]
    num_patches = cls_to_patch.numel()
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        return None
    heat = cls_to_patch.reshape(1, 1, side, side)
    heat = F.interpolate(heat, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return normalize_map(heat)


def gradient_saliency(model, x, label=None):
    x_req = x.detach().clone().requires_grad_(True)
    logits = model(x_req)
    if label is None:
        label = int(logits.argmax(dim=1)[0].detach().cpu())
    score = logits[:, int(label)].sum()
    grad = torch.autograd.grad(score, x_req, retain_graph=False, create_graph=False)[0]
    heat = grad.abs().mean(dim=1, keepdim=False)[0]
    return normalize_map(heat)


def attention_or_saliency(model, x, label):
    heat = collect_attention_rollout(model, x)
    if heat is not None:
        return heat, "Attention rollout"
    return gradient_saliency(model, x, label), "Gradient/patch saliency"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--selected-csv", default=None)
    ap.add_argument("--attack-dir", default=None, help="Optional attack dir. If given, also visualizes adversarial image map.")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--source-model", required=True, help="Source model used for visualization, e.g. vit_base_patch16_224 or resnet50")
    ap.add_argument("--models-dir", default="./pretrained_models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Attention / patch saliency visualization")
    args = ap.parse_args()

    set_model_cache_dir(args.models_dir)
    device = get_device(args.device)
    model = create_model(args.source_model, device=device, pretrained=True)

    if args.attack_dir:
        adv, label, relpath = get_sample_from_attack(args.attack_dir, args.index, args.data_dir)
        clean = load_clean_tensor(args.data_dir, relpath)
    else:
        if not args.selected_csv:
            raise ValueError("Pass --selected-csv when --attack-dir is not provided")
        clean, label, relpath = get_sample_from_csv(args.data_dir, args.selected_csv, args.index)
        adv = None

    clean_b = clean.unsqueeze(0).to(device)
    clean_heat, clean_kind = attention_or_saliency(model, clean_b, label)

    if adv is not None:
        adv_b = adv.unsqueeze(0).to(device)
        adv_heat, adv_kind = attention_or_saliency(model, adv_b, label)
        panels = [
            ("Original image", tensor_to_np_img(clean)),
            (f"Clean {clean_kind}", clean_heat.numpy(), "heat"),
            ("Clean overlay", overlay_heatmap(clean, clean_heat)),
            ("Adversarial image", tensor_to_np_img(adv)),
            (f"Adversarial {adv_kind}", adv_heat.numpy(), "heat"),
            ("Adversarial overlay", overlay_heatmap(adv, adv_heat)),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), dpi=180)
        axes = axes.ravel()
    else:
        panels = [
            ("Original image", tensor_to_np_img(clean)),
            (clean_kind, clean_heat.numpy(), "heat"),
            ("Overlay", overlay_heatmap(clean, clean_heat)),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=180)

    fig.suptitle(args.title, fontsize=12)
    for ax, panel in zip(axes, panels):
        name = panel[0]
        img = panel[1]
        if len(panel) >= 3 and panel[2] == "heat":
            ax.imshow(img, cmap="jet")
        else:
            ax.imshow(img)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    fig.text(0.5, 0.02, f"source model={args.source_model} | sample index={args.index} | label={label} | relpath={relpath}", ha="center", fontsize=8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
