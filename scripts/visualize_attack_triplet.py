#!/usr/bin/env python3
"""Create Figure 1: Original image / Perturbation image / Adversarial image."""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scripts.batch_io import get_sample_from_attack, load_clean_tensor


def tensor_to_np_img(t):
    t = t.detach().cpu().clamp(0, 1)
    return t.permute(1, 2, 0).numpy()


def perturbation_to_vis(delta, percentile=99.5):
    d = delta.detach().cpu()
    scale = np.percentile(d.abs().numpy(), percentile)
    scale = max(float(scale), 1e-8)
    vis = (d / scale).clamp(-1, 1) * 0.5 + 0.5
    return vis.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Adversarial example visualization")
    ap.add_argument("--percentile", type=float, default=99.5)
    args = ap.parse_args()

    adv, label, relpath = get_sample_from_attack(args.attack_dir, args.index, args.data_dir)
    clean = load_clean_tensor(args.data_dir, relpath)
    delta = adv - clean

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    fig.suptitle(args.title, fontsize=12)

    panels = [
        ("Original image", tensor_to_np_img(clean)),
        ("Perturbation image\n(rescaled for visibility)", perturbation_to_vis(delta, args.percentile)),
        ("Adversarial image", tensor_to_np_img(adv)),
    ]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.text(0.5, 0.02, f"sample index={args.index} | label={label} | relpath={relpath}", ha="center", fontsize=8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
