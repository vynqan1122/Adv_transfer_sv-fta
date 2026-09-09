#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import hashlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import build_attack
from src.batching import AdaptiveBatchExecutor, model_loading_oom_hint
from src.datasets import ImageNetCSVDataset, collate_with_paths, imagenet_transform
from src.models import load_models
from src.utils import ensure_dir, get_device, parse_model_list, save_json, set_model_cache_dir, set_seed, tensor_stats, write_csv


def attack_chunk(attack, models, images, labels, device, start, end):
    """Keep each forward/backward scope separate so OOM retries can release it."""
    inputs = images[start:end].to(device, non_blocking=True)
    targets = labels[start:end].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = attack(models, inputs, targets)
    return {
        "adv": result.adv.detach().to(device="cpu", dtype=torch.float32),
        "logs": result.logs,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../dataset")
    parser.add_argument("--selected-csv", default="runs/selected_5000.csv")
    parser.add_argument("--models-dir", default="./pretrained_models")
    parser.add_argument("--surrogates", required=True)
    parser.add_argument("--attack", required=True,
                        choices=["ifgsm", "mifgsm", "difgsm", "tifgsm", "si_ni_fgsm", "freq_only", "vit_aware", "ours", "sv_fca", "svfca", "sv_fta", "svfta", "ddc"])
    parser.add_argument("--variant", default="full_model")
    parser.add_argument("--fusion", default="robust",
                        help="Legacy compatibility option retained for old scripts; SV-FCA does not use a token/spatial fusion branch.")
    parser.add_argument("--freq-mode", default="low_mid")
    parser.add_argument("--num-views", type=int, default=4,
                        help="Number of differentiable source views used by SV-FCA (R).")
    parser.add_argument("--spectral-bands", type=int, default=6,
                        help="Number of radial rFFT bands used by SV-FCA.")
    parser.add_argument("--rho", type=float, default=0.5,
                        help="Legacy SV-FCA option; ignored by SV-FCA.")
    parser.add_argument("--lambda-grid", type=int, default=21,
                        help="Legacy SV-FCA option; ignored by SV-FCA.")
    parser.add_argument("--eps-c", type=float, default=1e-3,
                        help="Legacy precision-weighting option; ignored by SV-FCA.")
    parser.add_argument("--c-min", type=float, default=0.05,
                        help="Legacy precision-weighting option; ignored by SV-FCA.")
    parser.add_argument("--c-max", type=float, default=5.0,
                        help="Legacy precision-weighting option; ignored by SV-FCA.")
    parser.add_argument("--diversity-prob", type=float, default=1.0,
                        help="Input diversity probability for virtual source views")
    parser.add_argument("--disable-precision-weighting", action="store_true",
                        help="Legacy option retained for old shell scripts; SV-FCA has no precision weighting.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--decay", type=float, default=1.0,
                        help="Momentum decay for ours/SV-FCA and compatible attacks")
    parser.add_argument("--soft-sigma", type=float, default=0.55,
                        help="Legacy compatibility option; ignored by SV-FCA.")
    parser.add_argument("--soft-power", type=float, default=4.0,
                        help="Legacy compatibility option; ignored by SV-FCA.")
    parser.add_argument("--low-cutoff", type=float, default=0.25,
                        help="Legacy compatibility option; ignored by SV-FCA.")
    parser.add_argument("--mid-cutoff", type=float, default=0.55,
                        help="Legacy compatibility option; ignored by SV-FCA.")
    parser.add_argument("--band-temperature", type=float, default=0.35,
                        help="SV-FCA softmax temperature for band consensus.")
    parser.add_argument("--low-mid-strength", type=float, default=1.0,
                        help="SV-FCA strength of the smooth low/mid transfer prior.")
    parser.add_argument("--low-prior", type=float, default=1.0,
                        help="Legacy SV-FCA option; ignored by SV-FCA.")
    parser.add_argument("--mid-prior", type=float, default=1.0,
                        help="Legacy SV-FCA option; ignored by SV-FCA.")
    parser.add_argument("--high-prior", type=float, default=0.15,
                        help="Legacy SV-FCA option; ignored by SV-FCA.")
    parser.add_argument("--spectral-decay", type=float, default=0.75,
                        help="SV-FCA EMA decay for spectral band-weight memory.")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA autocast for source-model forward passes. Saves activation VRAM; FFT/statistics stay float32.")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16",
                        help="Autocast dtype used with --amp.")
    parser.add_argument("--log-spectral-energy", action="store_true",
                        help="Legacy SV-FCA option; ignored by SV-FCA.")
    parser.add_argument("--eps", type=float, default=16.0/255.0)
    parser.add_argument("--alpha", type=float, default=1.6/255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-auto-batch", action="store_true", help="Disable automatic CUDA OOM batch-size reduction.")
    parser.add_argument("--min-batch-size", type=int, default=1, help="Smallest CUDA OOM retry chunk; logical image budgets stay unchanged.")
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Limit the number of dataloader batches to generate. Useful for all tables using the same experimental budget.")
    parser.add_argument("--empty-cache-every", type=int, default=1,
                        help="If using CUDA, call torch.cuda.empty_cache() every N batches; <=0 disables it.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--adv-batch-dir",
        default=None,
        help="Central folder to save adversarial batch .pt files",
    )
    parser.add_argument(
        "--storage-mode",
        choices=["adv_fp32", "delta_fp16"],
        default="adv_fp32",
        help="Temporary batch format. delta_fp16 roughly halves disk usage and reconstructs adversarial images from clean inputs during evaluation.",
    )
    parser.add_argument(
        "--clear-adv-batch-dir",
        action="store_true",
        help="Clear generated batch_*.pt files in adv-batch-dir before writing this attack",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if not 1 <= args.min_batch_size <= args.batch_size:
        parser.error("--min-batch-size must be between 1 and --batch-size")
    if args.num_batches is not None and args.num_batches <= 0:
        parser.error("--num-batches must be positive")
    if args.image_size != 224:
        parser.error("This ImageNet pipeline crops inputs to 224; --image-size must be 224")

    set_seed(args.seed)
    set_model_cache_dir(args.models_dir)
    device = get_device(args.device)
    surrogate_names = parse_model_list(args.surrogates)
    if len(surrogate_names) < 1:
        raise ValueError("Pass at least one source model via --surrogates.")

    ensure_dir(args.out_dir)

    if args.adv_batch_dir:
        adv_batch_dir = args.adv_batch_dir
    else:
        # Default to the central runs/adv_batches tree, not out_dir/adv_batches.
        out_abs = os.path.abspath(args.out_dir)
        runs_root = os.path.abspath(os.path.join(ROOT_DIR, "runs"))
        try:
            rel = os.path.relpath(out_abs, runs_root)
            if rel == ".." or rel.startswith(".." + os.sep):
                suffix = hashlib.sha256(os.path.normcase(out_abs).encode("utf-8")).hexdigest()[:12]
                rel = os.path.basename(out_abs) + "_" + suffix
        except ValueError:
            suffix = hashlib.sha256(os.path.normcase(out_abs).encode("utf-8")).hexdigest()[:12]
            rel = os.path.basename(out_abs) + "_" + suffix
        adv_batch_dir = os.path.join(runs_root, "adv_batches", rel)
    adv_batch_dir = os.path.abspath(adv_batch_dir)
    ensure_dir(adv_batch_dir)

    config = vars(args).copy()
    config["resolved_adv_batch_dir"] = adv_batch_dir
    config["num_batches_limit"] = args.num_batches
    config["threat_model"] = "k_source_transfer_based_black_box"
    config["source_setting"] = "single_source" if len(surrogate_names) == 1 else "multi_source"
    config["surrogate_access"] = "white_box_gradient"
    config["target_access_during_attack"] = "none"
    config["target_used_only_for_evaluation"] = True
    config["source_models_used_for_attack"] = surrogate_names
    config["surrogate_models_used_for_attack"] = surrogate_names

    dataset = ImageNetCSVDataset(args.data_dir, args.selected_csv, transform=imagenet_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_paths,
    )
    attack = build_attack(
        args.attack,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        variant=args.variant,
        fusion=args.fusion,
        freq_mode=args.freq_mode,
        num_views=args.num_views,
        spectral_bands=args.spectral_bands,
        rho=args.rho,
        lambda_grid=args.lambda_grid,
        eps_c=args.eps_c,
        c_min=args.c_min,
        c_max=args.c_max,
        diversity_prob=args.diversity_prob,
        disable_precision_weighting=args.disable_precision_weighting,
        image_size=args.image_size,
        resize_size=args.resize_size,
        decay=args.decay,
        soft_sigma=args.soft_sigma,
        soft_power=args.soft_power,
        low_cutoff=args.low_cutoff,
        mid_cutoff=args.mid_cutoff,
        band_temperature=args.band_temperature,
        low_mid_strength=args.low_mid_strength,
        low_prior=args.low_prior,
        mid_prior=args.mid_prior,
        high_prior=args.high_prior,
        spectral_decay=args.spectral_decay,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        log_spectral_energy=args.log_spectral_energy,
    )

    # Invalidate the previous manifest before any tensors are overwritten. A
    # failed rerun must not be evaluated as a mixture of old and new batches.
    manifest_path = Path(args.out_dir) / "attack_batches.csv"
    # Keep an empty manifest while generating so the legacy directory fallback
    # cannot expose an incomplete rerun if generation fails.
    write_csv([], manifest_path, fieldnames=["file", "n"])
    if args.clear_adv_batch_dir:
        for old_file in Path(adv_batch_dir).glob("batch_*.pt"):
            old_file.unlink()
    save_json(config, os.path.join(args.out_dir, "attack_config.json"))
    with model_loading_oom_hint(device, "attack source models"):
        models = load_models(surrogate_names, device=device, pretrained=True)
    # Gradients are needed only with respect to input pixels.
    for model in models.values():
        model.eval()
        model.requires_grad_(False)

    batch_rows = []
    step_rows = []
    executor = AdaptiveBatchExecutor(args.batch_size, device, not args.no_auto_batch, args.min_batch_size,
                                     context="attack " + args.attack)
    for batch_idx, (images, labels, relpaths) in enumerate(tqdm(loader, desc="attack {}".format(args.attack))):
        if args.num_batches is not None and batch_idx >= args.num_batches:
            break

        # The DataLoader and manifest keep their original logical batch size.
        # GPU chunks may shrink, but NUM_BATCHES still processes the same images.
        parts = []
        chunk_sizes = []
        peak_allocated_mb = peak_reserved_mb = 0.0
        retries_before = executor.oom_retries
        process = lambda start, end: attack_chunk(attack, models, images, labels, device, start, end)
        for chunk_idx, (start, end, chunk) in enumerate(executor.iter_batches(len(images), process)):
            parts.append(chunk["adv"])
            chunk_sizes.append(end - start)
            peak_allocated_mb = max(peak_allocated_mb, chunk["peak_allocated_mb"])
            peak_reserved_mb = max(peak_reserved_mb, chunk["peak_reserved_mb"])
            for step_log in chunk["logs"]:
                row = {"batch": batch_idx, "microbatch": chunk_idx, "sample_start": start,
                       "sample_end": end, "microbatch_size": end - start}
                row.update(step_log)
                step_rows.append(row)
        adv_cpu = torch.cat(parts, dim=0)
        delta_cpu = adv_cpu - images
        stats = tensor_stats(delta_cpu)
        labels_cpu = labels.detach().cpu()
        out_file = os.path.join(adv_batch_dir, "batch_{:05d}.pt".format(batch_idx))

        payload = {
            "labels": labels_cpu,
            "relpaths": relpaths,
            "attack": args.attack,
            "variant": args.variant,
            "fusion": args.fusion,
            "source_models": surrogate_names,
            "source_model": ",".join(surrogate_names),
            "surrogates": surrogate_names,
            "surrogates_raw": args.surrogates,
            "threat_model": "k_source_transfer_based_black_box",
            "source_setting": "single_source" if len(surrogate_names) == 1 else "multi_source",
            "target_access_during_attack": "none",
            "storage_mode": args.storage_mode,
            "eps": args.eps,
        }
        if args.storage_mode == "delta_fp16":
            payload["delta"] = delta_cpu.to(dtype=torch.float16)
        else:
            payload["adv"] = adv_cpu
        torch.save(payload, out_file)
        saved_bytes = os.path.getsize(out_file)
        batch_rows.append({
            "batch": batch_idx,
            "file": os.path.relpath(out_file, args.out_dir).replace("\\", "/"),
            "n": int(images.size(0)),
            "requested_batch_size": args.batch_size,
            "effective_batch_size": executor.batch_size,
            "microbatch_sizes": ",".join(map(str, chunk_sizes)),
            "oom_retries": executor.oom_retries - retries_before,
            "source_model": ",".join(surrogate_names),
            "surrogates": args.surrogates,
            "source_setting": "single_source" if len(surrogate_names) == 1 else "multi_source",
            "threat_model": "k_source_transfer_based_black_box",
            "delta_min": stats["min"],
            "delta_max": stats["max"],
            "delta_mean": stats["mean"],
            "delta_std": stats["std"],
            "cuda_peak_allocated_mb": peak_allocated_mb,
            "cuda_peak_reserved_mb": peak_reserved_mb,
            "storage_mode": args.storage_mode,
            "saved_file_mb": saved_bytes / (1024.0 ** 2),
        })
        # Release tensors aggressively. This helps long table runs on 12GB GPUs.
        del chunk, parts, adv_cpu, payload, delta_cpu, labels_cpu, images, labels, process
        if device.type == "cuda" and args.empty_cache_every > 0 and ((batch_idx + 1) % args.empty_cache_every == 0):
            torch.cuda.empty_cache()

    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    write_csv(batch_rows, temporary_manifest)
    temporary_manifest.replace(manifest_path)
    if step_rows:
        fieldnames = sorted(set().union(*[set(r.keys()) for r in step_rows]))
        write_csv(step_rows, os.path.join(args.out_dir, "attack_steps.csv"), fieldnames=fieldnames)
    config["batch_execution"] = executor.metadata
    config["generated_images"] = sum(row["n"] for row in batch_rows)
    config["generated_logical_batches"] = len(batch_rows)
    save_json(config, os.path.join(args.out_dir, "attack_config.json"))
    print("Saved attack outputs to", args.out_dir)


if __name__ == "__main__":
    main()
