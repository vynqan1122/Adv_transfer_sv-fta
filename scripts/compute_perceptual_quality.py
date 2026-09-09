#!/usr/bin/env python3
"""Compute PSNR, SSIM and optional LPIPS for Table VI with bounded memory.

The script supports micro-batching each saved adversarial tensor and can delete
consumed .pt files after metrics are safely written.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from scripts.batch_io import delete_consumed_batches, load_clean_batch, reconstruct_adv, resolve_batch_files, safe_torch_load, validate_payload
from src.datasets import imagenet_transform
from src.utils import get_device, save_json


def psnr_batch(x, y):
    """Per-image PSNR for pixels in [0, 1]; identical images yield +infinity."""
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError("PSNR inputs must have matching [B,C,H,W] shapes")
    mse = (x - y).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_batch(x, y, window_size=11):
    """Per-image SSIM using the repository's uniform window and zero padding."""
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError("SSIM inputs must have matching [B,C,H,W] shapes")
    if window_size <= 0 or window_size % 2 != 1:
        raise ValueError("SSIM requires an odd positive window")
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    channels = x.shape[1]
    pad = window_size // 2
    kernel = torch.ones((channels, 1, window_size, window_size), device=x.device, dtype=x.dtype) / (window_size * window_size)
    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)
    sigma_x = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.flatten(1).mean(dim=1)


def load_lpips(device):
    try:
        import lpips
    except ImportError:
        return None
    return lpips.LPIPS(net="alex").to(device).eval()


def delete_attack_batches(attack_dir, batch_files):
    deleted = delete_consumed_batches(attack_dir, batch_files)
    save_json({"deleted_batch_count": deleted, "batch_cleanup": "after_successful_quality_evaluation"}, os.path.join(attack_dir, "batch_cleanup.json"))
    return deleted


def compute_for_attack(
    data_dir,
    attack_dir,
    device,
    max_batches=None,
    use_lpips=True,
    quality_batch_size=4,
):
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if quality_batch_size <= 0:
        raise ValueError("quality_batch_size must be positive")
    batch_files = resolve_batch_files(attack_dir)
    if max_batches is not None:
        batch_files = batch_files[: int(max_batches)]

    lpips_model = load_lpips(device) if use_lpips else None
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    count = 0
    lpips_count = 0

    qbs = max(1, int(quality_batch_size))
    with torch.inference_mode():
        for file_path in tqdm(batch_files, desc=Path(attack_dir).name, leave=False):
            payload = safe_torch_load(file_path)
            stored_cpu, _, relpaths, storage_key = validate_payload(payload, file_path)

            for start in range(0, len(relpaths), qbs):
                end = min(start + qbs, len(relpaths))
                clean = load_clean_batch(data_dir, relpaths[start:end], device)
                stored = stored_cpu[start:end].float().to(device, non_blocking=True)
                adv = reconstruct_adv(clean, stored, storage_key, payload.get("eps"))

                p = psnr_batch(clean, adv)
                s = ssim_batch(clean, adv)
                psnr_sum += float(p.sum().item())
                ssim_sum += float(s.sum().item())
                count += int(end - start)

                if lpips_model is not None:
                    val = lpips_model(clean * 2 - 1, adv * 2 - 1).view(-1)
                    lpips_sum += float(val.sum().item())
                    lpips_count += int(val.numel())
                    del val

                del adv, stored, clean, p, s

            del payload, stored_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if lpips_model is not None:
        del lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    psnr = psnr_sum / max(count, 1)
    ssim = ssim_sum / max(count, 1)
    lp = lpips_sum / max(lpips_count, 1) if lpips_count else ""
    return psnr, ssim, lp, batch_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with family,method,attack_dir")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="runs/table6_quality/table6_quality.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--quality-batch-size", type=int, default=4)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--delete-batches-after-use", action="store_true")
    args = ap.parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        ap.error("--max-batches must be positive")
    if args.quality_batch_size <= 0:
        ap.error("--quality-batch-size must be positive")

    device = get_device(args.device)
    rows = []
    cleanup_plan = []
    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            psnr, ssim, lp, batch_files = compute_for_attack(
                args.data_dir,
                row["attack_dir"],
                device=device,
                max_batches=args.max_batches,
                use_lpips=not args.no_lpips,
                quality_batch_size=args.quality_batch_size,
            )
            rows.append({
                "Family": row.get("family", ""),
                "Method": row.get("method", ""),
                "PSNR": "{:.2f}".format(psnr),
                "SSIM": "{:.4f}".format(ssim),
                "LPIPS": "{:.4f}".format(lp) if lp != "" else "",
                "AttackDir": row["attack_dir"],
            })
            cleanup_plan.append((row["attack_dir"], batch_files))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Family", "Method", "PSNR", "SSIM", "LPIPS", "AttackDir"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Delete only after metric CSV is safely on disk.
    if args.delete_batches_after_use:
        for attack_dir, batch_files in cleanup_plan:
            deleted = delete_attack_batches(attack_dir, batch_files)
            print(f"[cleanup] {attack_dir}: deleted {deleted} adversarial batches")

    if any(r["LPIPS"] == "" for r in rows):
        print("LPIPS was not computed. Install optional package with: pip install lpips")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
