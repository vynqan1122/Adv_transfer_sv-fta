#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.batching import AdaptiveBatchExecutor, model_loading_oom_hint
from src.datasets import build_imagenet_dataset, collate_with_paths, resolve_val_dir
from src.models import SURROGATE_ALL, load_models
from src.utils import ensure_dir, get_device, parse_model_list, save_json, set_model_cache_dir, set_seed, write_csv


@torch.no_grad()
def selection_chunk(models, images, labels, device, start, end):
    inputs = images[start:end].to(device, non_blocking=True)
    targets = labels[start:end].to(device, non_blocking=True)
    predictions = {}
    correct_all = torch.ones_like(targets, dtype=torch.bool)
    for name, model in models.items():
        pred = model(inputs).argmax(dim=1)
        predictions[name] = pred.cpu().tolist()
        correct_all &= pred.eq(targets)
    return correct_all.cpu().tolist(), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../dataset")
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--labels-csv", default=None, help="Optional CSV with relpath,label ground truth")
    parser.add_argument("--models-dir", default="./pretrained_models")
    parser.add_argument("--surrogates", default=",".join(SURROGATE_ALL))
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-auto-batch", action="store_true", help="Disable automatic CUDA OOM batch-size reduction.")
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", default="runs/selected_1000.csv")
    args = parser.parse_args()
    if args.num_images <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--num-images and --batch-size must be positive; --num-workers must be nonnegative")
    if not 1 <= args.min_batch_size <= args.batch_size:
        parser.error("--min-batch-size must be between 1 and --batch-size")
    if not parse_model_list(args.surrogates):
        parser.error("--surrogates must contain at least one model")

    set_seed(args.seed)
    set_model_cache_dir(args.models_dir)
    device = get_device(args.device)
    surrogate_names = parse_model_list(args.surrogates)

    dataset = build_imagenet_dataset(args.data_dir, args.val_dir, args.labels_csv)
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_paths,
    )
    with model_loading_oom_hint(device, "selection source models"):
        models = load_models(surrogate_names, device=device, pretrained=True)
    executor = AdaptiveBatchExecutor(args.batch_size, device, not args.no_auto_batch, args.min_batch_size,
                                     context="select clean-correct")

    rows = []
    total_seen = 0
    val_root = resolve_val_dir(args.data_dir, args.val_dir)
    data_root = os.path.abspath(args.data_dir)

    with torch.no_grad():
        for images, labels, relpaths in tqdm(loader, desc="select clean-correct"):
            process = lambda start, end: selection_chunk(models, images, labels, device, start, end)
            for start, end, (mask, predictions) in executor.iter_batches(len(images), process):
                total_seen += end - start
                for offset, ok in enumerate(mask):
                    if not ok:
                        continue
                    index = start + offset
                    relpath = relpaths[index]
                    if args.labels_csv is None:
                        # ImageFolder paths are relative to val root; output paths
                        # must be relative to data-dir for downstream stages.
                        full_path = os.path.join(val_root, relpath)
                        relpath_out = os.path.relpath(full_path, data_root).replace("\\", "/")
                    else:
                        relpath_out = relpath
                    row = {"relpath": relpath_out, "label": int(labels[index])}
                    for name in surrogate_names:
                        row["pred_" + name] = predictions[name][offset]
                    rows.append(row)
                    if len(rows) >= args.num_images:
                        break
                if len(rows) >= args.num_images:
                    break
            if len(rows) >= args.num_images:
                break

    if len(rows) < args.num_images:
        raise RuntimeError(
            "Only found {} clean-correct images out of {} scanned. Check label mapping, dataset layout, or use --labels-csv.".format(
                len(rows), total_seen
            )
        )

    fieldnames = ["relpath", "label"] + ["pred_" + n for n in surrogate_names]
    write_csv(rows, args.out_csv, fieldnames=fieldnames)
    save_json({"seed": args.seed, "selected_images": len(rows), "scanned_images": total_seen,
               "batch_execution": executor.metadata}, str(args.out_csv) + ".metadata.json")
    print("Saved {} selected images to {}".format(len(rows), args.out_csv))


if __name__ == "__main__":
    main()
