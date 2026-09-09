#!/usr/bin/env python3
"""Create Figure 2: FFT spectrum visualization for clean image and perturbations."""
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


def parse_attack_dirs(spec):
    out = []
    # A repeated CLI option preserves commas in directory names. The string
    # form remains compatible with historical --attack-dirs command lines.
    items = spec.split(",") if isinstance(spec, str) else spec
    for item in items:
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(item).name
        out.append((label.strip(), path.strip()))
    if not out:
        raise ValueError("--attack-dirs is empty")
    return out


def grayscale(t):
    if t.ndim == 3:
        return t.mean(dim=0).detach().cpu().numpy()
    return t.detach().cpu().numpy()


def fft_log_spectrum(img2d):
    fft = np.fft.fft2(img2d)
    shifted = np.fft.fftshift(fft)
    spec = np.log1p(np.abs(shifted))
    spec = spec - spec.min()
    spec = spec / (spec.max() + 1e-12)
    return spec


def radial_profile(spec):
    h, w = spec.shape
    yy, xx = np.indices((h, w))
    # fftshift places DC at floor(size / 2), including even image dimensions.
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = r.max()
    sums = np.bincount(r.ravel(), weights=spec.ravel(), minlength=max_r + 1)
    counts = np.bincount(r.ravel(), minlength=max_r + 1)
    return sums / np.maximum(counts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--attack-dirs", help="Legacy comma list: Label=path,Label=path")
    inputs.add_argument("--attack-dir", action="append", help="Repeat Label=path for each method; supports paths containing commas")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radial-out", default=None)
    ap.add_argument("--title", default="Frequency spectrum visualization")
    args = ap.parse_args()

    methods = parse_attack_dirs(args.attack_dir if args.attack_dir is not None else args.attack_dirs)
    first_adv, label, relpath = get_sample_from_attack(methods[0][1], args.index, args.data_dir)
    clean = load_clean_tensor(args.data_dir, relpath)

    panels = [("Original FFT spectrum", fft_log_spectrum(grayscale(clean)))]
    radial_rows = []
    prof = radial_profile(panels[0][1])
    for i, val in enumerate(prof):
        radial_rows.append({"method": "Original", "radius": i, "value": float(val)})

    for method_label, attack_dir in methods:
        adv, _, relpath_i = get_sample_from_attack(attack_dir, args.index, args.data_dir)
        if relpath_i != relpath:
            raise ValueError("Frequency comparisons require the same image across attacks; regenerate using the same selected CSV and order")
        delta = adv - clean
        spec = fft_log_spectrum(grayscale(delta))
        panels.append((f"{method_label} spectrum", spec))
        prof = radial_profile(spec)
        for i, val in enumerate(prof):
            radial_rows.append({"method": method_label, "radius": i, "value": float(val)})

    n = len(panels)
    fig_w = max(12, 2.4 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.2), dpi=180)
    if n == 1:
        axes = [axes]
    fig.suptitle(args.title, fontsize=12)
    for ax, (name, spec) in zip(axes, panels):
        ax.imshow(spec, cmap="magma")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.text(0.5, 0.02, f"sample index={args.index} | label={label} | relpath={relpath}", ha="center", fontsize=8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    if args.radial_out:
        Path(args.radial_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.radial_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "radius", "value"])
            writer.writeheader()
            writer.writerows(radial_rows)
        print("Saved", args.radial_out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
