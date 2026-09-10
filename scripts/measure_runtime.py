#!/usr/bin/env python3
"""Measure attack runtime per image for Table VII."""
import argparse
import csv
import os
import sys
import time
from itertools import islice

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import build_attack
from src.batching import AdaptiveBatchExecutor, model_loading_oom_hint
from src.datasets import ImageNetCSVDataset, collate_with_paths, imagenet_transform
from src.models import load_models
from src.utils import get_device, parse_model_list, set_model_cache_dir, set_seed

METHOD_LABELS = {
    "ifgsm": "I-FGSM",
    "mifgsm": "MI-FGSM",
    "difgsm": "DI-FGSM",
    "tifgsm": "TI-FGSM",
    "si_ni_fgsm": "SI-NI-FGSM",
    "freq_only": "Freq-Only",
    "vit_aware": "ViT-Aware",
    "ours": "Ours (SV-FCA)",
    "ddc": "Ours",
}


def runtime_chunk(attack, models, images, labels, device, start, end, measure=True):
    """Return only successful attack time; copies and failed attempts are excluded."""
    inputs, targets = images[start:end].to(device), labels[start:end].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter() if measure else None
    result = attack(models, inputs, targets)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if measure else 0.0
    del result
    return elapsed


def measure_attack(attack, models, loader, device, max_images=128, num_batches=None, warmup_batches=1,
                   executor=None, measurement_metadata=None):
    """Warm up separately and retain logical sample budgets if CUDA chunks shrink.

    The historical (ms_per_image, images, logical_batches) return value is kept.
    Pass measurement_metadata={} to also collect actual measured chunk sizes.
    """
    if max_images <= 0 or warmup_batches < 0 or (num_batches is not None and num_batches <= 0):
        raise ValueError("Runtime limits must be positive and warmup_batches nonnegative")
    if executor is None:
        executor = AdaptiveBatchExecutor(getattr(loader, "batch_size", None) or 16, device, context="runtime")
    initial_retries = executor.oom_retries
    for images, labels, _paths in islice(loader, warmup_batches):
        process = lambda start, end: runtime_chunk(attack, models, images, labels, device, start, end, measure=False)
        for _start, _end, _elapsed in executor.iter_batches(len(images), process):
            pass
    warmup_retries = executor.oom_retries - initial_retries
    n_seen = 0
    batches = 0
    microbatches = 0
    measured_sizes = set()
    elapsed = 0.0
    for images, labels, _paths in loader:
        if num_batches is not None and batches >= num_batches:
            break
        if num_batches is None:
            remaining = max_images - n_seen
            if remaining <= 0:
                break
            images, labels = images[:remaining], labels[:remaining]
        process = lambda start, end: runtime_chunk(attack, models, images, labels, device, start, end)
        for start, end, chunk_elapsed in executor.iter_batches(len(images), process):
            elapsed += chunk_elapsed
            microbatches += 1
            measured_sizes.add(end - start)
        n_seen += images.size(0)
        batches += 1
    if not n_seen:
        raise ValueError("No images available for runtime measurement")
    if measurement_metadata is not None:
        measurement_metadata.update(executor.metadata)
        measurement_metadata.update({
            "measured_microbatches": microbatches,
            "measured_batch_sizes": sorted(measured_sizes),
            "warmup_oom_retries": warmup_retries,
            "measured_oom_retries": executor.oom_retries - initial_retries - warmup_retries,
            "timing_scope": "successful_attack_calls_only; excludes transfers and OOM retries",
        })
    return elapsed / n_seen * 1000.0, n_seen, batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--selected-csv", required=True)
    ap.add_argument("--models-dir", default="./pretrained_models")
    ap.add_argument("--surrogates", default="resnet50,densenet121")
    ap.add_argument("--methods", default="difgsm,ifgsm,mifgsm,si_ni_fgsm,tifgsm,vit_aware,freq_only,ours")
    ap.add_argument("--out", default="runs/table7_runtime/table7_runtime.csv")
    ap.add_argument("--max-images", type=int, default=128)
    ap.add_argument("--num-batches", type=int, default=None, help="Measure N batches after a separate warmup pass; overrides --max-images.")
    ap.add_argument("--warmup-batches", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-auto-batch", action="store_true", help="Disable automatic CUDA OOM batch-size reduction.")
    ap.add_argument("--min-batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eps", type=float, default=16.0/255.0)
    ap.add_argument("--alpha", type=float, default=1.6/255.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # Ours-specific knobs
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--spectral-bands", type=int, default=6)
    ap.add_argument("--band-temperature", type=float, default=0.20)
    ap.add_argument("--low-mid-strength", type=float, default=1.0)
    ap.add_argument("--spectral-decay", type=float, default=0.65)
    ap.add_argument("--consensus-gain", type=float, default=2.0)
    ap.add_argument("--energy-strength", type=float, default=0.75)
    ap.add_argument("--band-weight-floor", type=float, default=0.02)
    ap.add_argument("--diversity-prob", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--lambda-grid", type=int, default=21)
    ap.add_argument("--amp", action="store_true", help="Use CUDA autocast for SV-FCA source forwards.")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--log-spectral-energy", action="store_true")
    args = ap.parse_args()
    if args.batch_size <= 0 or args.max_images <= 0 or args.num_workers < 0 or args.warmup_batches < 0:
        ap.error("Batch/image counts must be positive; workers/warmup must be nonnegative")
    if not 1 <= args.min_batch_size <= args.batch_size:
        ap.error("--min-batch-size must be between 1 and --batch-size")
    if args.num_batches is not None and args.num_batches <= 0:
        ap.error("--num-batches must be positive")
    if not parse_model_list(args.surrogates) or not parse_model_list(args.methods):
        ap.error("--surrogates and --methods must each contain at least one name")

    set_seed(args.seed)
    set_model_cache_dir(args.models_dir)
    device = get_device(args.device)
    dataset = ImageNetCSVDataset(args.data_dir, args.selected_csv, transform=imagenet_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_with_paths)
    with model_loading_oom_hint(device, "runtime source models"):
        models = load_models(parse_model_list(args.surrogates), device=device, pretrained=True)
    for model in models.values():
        model.eval()
        model.requires_grad_(False)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = []

    for method in methods:
        set_seed(args.seed)
        attack = build_attack(
            method,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            variant="full_model",
            fusion="robust",
            num_views=args.num_views,
            spectral_bands=args.spectral_bands,
            band_temperature=args.band_temperature,
            low_mid_strength=args.low_mid_strength,
            spectral_decay=args.spectral_decay,
            consensus_gain=args.consensus_gain,
            energy_strength=args.energy_strength,
            weight_floor=args.band_weight_floor,
            diversity_prob=args.diversity_prob,
            decay=args.decay,
            rho=args.rho,
            lambda_grid=args.lambda_grid,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            log_spectral_energy=args.log_spectral_energy,
        )
        executor = AdaptiveBatchExecutor(args.batch_size, device, not args.no_auto_batch, args.min_batch_size,
                                         context="runtime " + method)
        metadata = {}
        ms, measured_images, measured_batches = measure_attack(
            attack, models, loader, device, args.max_images, args.num_batches, args.warmup_batches,
            executor=executor, measurement_metadata=metadata)
        metadata.update({
            "consensus_gain": args.consensus_gain,
            "energy_strength": args.energy_strength,
            "band_weight_floor": args.band_weight_floor,
        })
        results.append({"method_key": method, "Method": METHOD_LABELS.get(method, method),
                        "Time/Image (ms)": ms, "Images": measured_images, "Measured Batches": measured_batches,
                        "batch_execution": metadata})

    base = None
    for r in results:
        if r["method_key"] == "difgsm":
            base = r["Time/Image (ms)"]
            break
    if base is None and results:
        base = results[0]["Time/Image (ms)"]

    out_rows = []
    for r in results:
        metadata = r["batch_execution"]
        out_rows.append({
            "Method": r["Method"], "Images": r["Images"], "Measured Batches": r["Measured Batches"],
            "Time/Image (ms)": "{:.2f}".format(r["Time/Image (ms)"]),
            "Relative Cost": "{:.2f}x".format(r["Time/Image (ms)"] / max(base, 1e-12)),
            "Requested Batch Size": metadata["requested_batch_size"],
            "Effective Batch Size": metadata["effective_batch_size"],
            "Measured Microbatches": metadata["measured_microbatches"],
            "Measured Batch Sizes": ",".join(map(str, metadata["measured_batch_sizes"])),
            "Warmup OOM Retries": metadata["warmup_oom_retries"],
            "Measured OOM Retries": metadata["measured_oom_retries"],
            "Seed": args.seed,
            "Auto Batch": metadata["auto_batch"],
            "RNG Restored On Retry": metadata["rng_restored_on_retry"],
            "Consensus Gain": metadata["consensus_gain"],
            "Energy Strength": metadata["energy_strength"],
            "Band Weight Floor": metadata["band_weight_floor"],
            "Timing Scope": metadata["timing_scope"],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
