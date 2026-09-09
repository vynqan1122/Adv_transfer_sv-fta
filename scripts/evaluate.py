#!/usr/bin/env python3
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import glob
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from scripts.batch_io import delete_consumed_batches, load_clean_batch, reconstruct_adv, resolve_batch_files, safe_torch_load, validate_payload
from src.models import load_models
from src.utils import (
    ensure_dir,
    get_device,
    parse_model_list,
    save_json,
    set_model_cache_dir,
    write_csv,
)


def evaluate_one_model(model, batch_files, data_dir, device, eval_batch_size=None):
    """
    Fair evaluation:
    - clean_acc: target accuracy on clean images
    - asr_all: old/simple ASR over all images
    - asr_clean_correct: fair ASR only over clean-correct images
    """
    if not batch_files:
        raise ValueError("At least one adversarial batch is required")
    if eval_batch_size is not None and eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    n_total = 0
    n_clean_correct = 0
    n_adv_wrong = 0
    n_adv_correct = 0
    n_success_clean_correct = 0
    n_robust_clean_correct = 0

    model.eval()

    with torch.inference_mode():
        for file_path in tqdm(batch_files, desc="eval", leave=False):
            payload = safe_torch_load(file_path)

            stored_cpu, labels, relpaths, storage_key = validate_payload(payload, file_path)

            chunk_size = len(relpaths) if eval_batch_size is None or eval_batch_size <= 0 else int(eval_batch_size)

            for start in range(0, len(relpaths), chunk_size):
                end = min(start + chunk_size, len(relpaths))
                labels_chunk = labels[start:end].long().to(device, non_blocking=True)
                relpaths_chunk = relpaths[start:end]
                clean = load_clean_batch(data_dir, relpaths_chunk, device)

                stored_chunk = stored_cpu[start:end].float().to(device, non_blocking=True)
                adv_chunk = reconstruct_adv(clean, stored_chunk, storage_key, payload.get("eps"))

                clean_logits = model(clean)
                adv_logits = model(adv_chunk)

                clean_pred = clean_logits.argmax(dim=1)
                adv_pred = adv_logits.argmax(dim=1)

                clean_correct = clean_pred.eq(labels_chunk)
                adv_correct = adv_pred.eq(labels_chunk)
                adv_wrong = ~adv_correct

                success_clean_correct = clean_correct & adv_wrong
                robust_clean_correct = clean_correct & adv_correct

                n_total += int(labels_chunk.numel())
                n_clean_correct += int(clean_correct.sum().item())
                n_adv_wrong += int(adv_wrong.sum().item())
                n_adv_correct += int(adv_correct.sum().item())
                n_success_clean_correct += int(success_clean_correct.sum().item())
                n_robust_clean_correct += int(robust_clean_correct.sum().item())

                del adv_chunk, stored_chunk, labels_chunk, clean, clean_logits, adv_logits

            del payload, labels, stored_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    clean_acc = n_clean_correct / max(n_total, 1) * 100.0
    asr_all = n_adv_wrong / max(n_total, 1) * 100.0
    adv_acc_all = n_adv_correct / max(n_total, 1) * 100.0

    # Fair ASR: only count images that target predicted correctly before attack
    asr_clean_correct = n_success_clean_correct / n_clean_correct * 100.0 if n_clean_correct else None
    robust_acc_clean_correct = n_robust_clean_correct / n_clean_correct * 100.0 if n_clean_correct else None

    return {
        "total": n_total,
        "clean_correct": n_clean_correct,
        "clean_acc": clean_acc,
        "adv_wrong": n_adv_wrong,
        "correct_after_attack": n_adv_correct,
        "success_on_clean_correct": n_success_clean_correct,
        "adv_acc_all": adv_acc_all,
        "asr_all": asr_all,
        "asr_clean_correct": asr_clean_correct,
        "robust_acc_clean_correct": robust_acc_clean_correct,

        # Keep old column name for aggregate_results.py.
        # From now on, "asr" means fair ASR.
        "asr": asr_clean_correct,
    }


def weighted_average(rows):
    if not rows:
        raise ValueError("At least one target result is required")
    total = sum(int(r["total"]) for r in rows)
    clean_correct = sum(int(r["clean_correct"]) for r in rows)
    adv_wrong = sum(int(r["adv_wrong"]) for r in rows)
    correct_after_attack = sum(int(r["correct_after_attack"]) for r in rows)
    success_on_clean_correct = sum(int(r["success_on_clean_correct"]) for r in rows)

    robust_on_clean_correct = clean_correct - success_on_clean_correct

    clean_acc = clean_correct / max(total, 1) * 100.0
    asr_all = adv_wrong / max(total, 1) * 100.0
    adv_acc_all = correct_after_attack / max(total, 1) * 100.0
    asr_clean_correct = success_on_clean_correct / clean_correct * 100.0 if clean_correct else None
    robust_acc_clean_correct = robust_on_clean_correct / clean_correct * 100.0 if clean_correct else None

    return {
        "target": "AVG",
        "total": total,
        "clean_correct": clean_correct,
        "clean_acc": clean_acc,
        "adv_wrong": adv_wrong,
        "correct_after_attack": correct_after_attack,
        "success_on_clean_correct": success_on_clean_correct,
        "adv_acc_all": adv_acc_all,
        "asr_all": asr_all,
        "asr_clean_correct": asr_clean_correct,
        "robust_acc_clean_correct": robust_acc_clean_correct,
        "asr": asr_clean_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-dir", required=True)
    parser.add_argument("--data-dir", required=True, help="ImageNet val folder, e.g. ../datasets/imagenet/val")
    parser.add_argument("--models-dir", default="./pretrained_models")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--robustbench-model-dir", default=os.environ.get("ROBUSTBENCH_MODEL_DIR", ""),
                        help="Parent folder containing imagenet/Linf/*.pt for RobustBench Table VIII models.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--num-batches", type=int, default=None, help="Evaluate only the first N adversarial batch files from the manifest.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Split each saved adversarial batch into smaller chunks during evaluation to reduce VRAM.")
    parser.add_argument("--delete-batches-after-eval", action="store_true",
                        help="Delete adversarial .pt batches after all target models are evaluated and CSV/JSON results are safely written.")
    args = parser.parse_args()
    if args.num_batches is not None and args.num_batches <= 0:
        parser.error("--num-batches must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    if not parse_model_list(args.targets):
        parser.error("--targets must contain at least one model")

    set_model_cache_dir(args.models_dir)

    device = get_device(args.device)
    target_names = parse_model_list(args.targets)

    batch_files = resolve_batch_files(args.attack_dir)
    if args.num_batches is not None:
        batch_files = batch_files[: int(args.num_batches)]

    rows = []
    for name in target_names:
        print("[target]", name)

        model = load_models([name], device=device, pretrained=True, robustbench_model_dir=(args.robustbench_model_dir or None))[name]
        metrics = evaluate_one_model(
            model=model,
            batch_files=batch_files,
            data_dir=args.data_dir,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )

        row = {"target": name}
        row.update(metrics)
        rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_row = weighted_average(rows)
    rows.append(avg_row)

    out_csv = args.out_csv or os.path.join(args.attack_dir, "eval_results.csv")

    fieldnames = [
        "target",
        "total",
        "clean_correct",
        "clean_acc",
        "adv_wrong",
        "correct_after_attack",
        "success_on_clean_correct",
        "adv_acc_all",
        "asr_all",
        "asr_clean_correct",
        "robust_acc_clean_correct",
        "asr",
    ]

    write_csv(rows, out_csv, fieldnames=fieldnames)

    save_json(
        {
            "targets": target_names,
            "metric_note": "asr = asr_clean_correct (%); undefined if clean_correct = 0. AVG pools image-target counts, weighted by clean_correct for conditional ASR.",
            "avg_clean_acc": avg_row["clean_acc"],
            "avg_asr_all": avg_row["asr_all"],
            "avg_asr_clean_correct": avg_row["asr_clean_correct"],
            "avg_asr": avg_row["asr"],
            "batch_files": batch_files,
        },
        os.path.join(args.attack_dir, "eval_summary.json"),
    )

    deleted_count = 0
    if args.delete_batches_after_eval:
        deleted_count = delete_consumed_batches(args.attack_dir, batch_files)

        save_json(
            {
                "deleted_batch_count": deleted_count,
                "batch_cleanup": "after_successful_evaluation",
                "note": "Adversarial tensors were deleted after all target models were evaluated.",
            },
            os.path.join(args.attack_dir, "batch_cleanup.json"),
        )

    print("Saved evaluation to", out_csv)
    print("AVG clean_acc: {:.2f}%".format(avg_row["clean_acc"]))
    print("AVG asr_all: {:.2f}%".format(avg_row["asr_all"]))
    fair_asr = avg_row["asr_clean_correct"]
    print("AVG asr_clean_correct:", "undefined (no clean-correct predictions)" if fair_asr is None else f"{fair_asr:.2f}%")
    if args.delete_batches_after_eval:
        print("[cleanup] deleted {} adversarial batch files".format(deleted_count))


if __name__ == "__main__":
    main()
