#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse

import torch
from tqdm import tqdm

from scripts.batch_io import delete_consumed_batches, load_clean_batch, reconstruct_adv, resolve_batch_files, safe_torch_load, validate_payload
from src.batching import AdaptiveBatchExecutor, model_loading_oom_hint
from src.models import load_models
from src.utils import (
    get_device,
    parse_model_list,
    save_json,
    set_model_cache_dir,
    write_csv,
)


def evaluate_one_model(
    model, batch_files, data_dir, device, eval_batch_size=16,
    auto_batch=True, min_batch_size=1,
):
    """
    Fair evaluation:
    - clean_acc: target accuracy on clean images
    - asr_all: old/simple ASR over all images
    - asr_clean_correct: fair ASR only over clean-correct images
    """
    if not batch_files:
        raise ValueError("At least one adversarial batch is required")
    # Keep explicit None compatible with existing callers while bounding memory.
    executor = AdaptiveBatchExecutor(
        16 if eval_batch_size is None else eval_batch_size, device,
        auto_batch=auto_batch, min_batch_size=min_batch_size, context="evaluation",
    )
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

            def process_chunk(start, end):
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

                # Commit counters only after both forwards and all reductions
                # succeed. A retried OOM must never count a sample twice.
                return (
                    int(labels_chunk.numel()),
                    int(clean_correct.sum().item()),
                    int(adv_wrong.sum().item()),
                    int(adv_correct.sum().item()),
                    int(success_clean_correct.sum().item()),
                    int(robust_clean_correct.sum().item()),
                )

            for _, _, counts in executor.iter_batches(len(relpaths), process_chunk):
                n_total += counts[0]
                n_clean_correct += counts[1]
                n_adv_wrong += counts[2]
                n_adv_correct += counts[3]
                n_success_clean_correct += counts[4]
                n_robust_clean_correct += counts[5]

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
        **{
            key: executor.metadata[key] for key in (
                "requested_batch_size", "effective_batch_size",
                "min_successful_batch_size", "max_successful_batch_size",
                "oom_retries", "successful_batches",
            )
        },
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
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Initial evaluation microbatch size (default: 16); halves after CUDA OOM.")
    parser.add_argument("--no-auto-batch", action="store_true", help="Disable automatic CUDA OOM batch reduction.")
    parser.add_argument("--min-batch-size", type=int, default=1, help="Smallest batch cap allowed during CUDA OOM retries (default: 1).")
    parser.add_argument("--delete-batches-after-eval", action="store_true",
                        help="Delete adversarial .pt batches after all target models are evaluated and CSV/JSON results are safely written.")
    args = parser.parse_args()
    if args.num_batches is not None and args.num_batches <= 0:
        parser.error("--num-batches must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    if not 1 <= args.min_batch_size <= args.eval_batch_size:
        parser.error("--min-batch-size must be between 1 and --eval-batch-size")
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

        with model_loading_oom_hint(device, context=f"loading evaluation target {name}"):
            model = load_models([name], device=device, pretrained=True, robustbench_model_dir=(args.robustbench_model_dir or None))[name]
        metrics = evaluate_one_model(
            model=model,
            batch_files=batch_files,
            data_dir=args.data_dir,
            device=device,
            eval_batch_size=args.eval_batch_size,
            auto_batch=not args.no_auto_batch,
            min_batch_size=args.min_batch_size,
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
            "execution": {
                "auto_batch": not args.no_auto_batch,
                "min_batch_size": args.min_batch_size,
                "targets": {
                    row["target"]: {
                        key: row[key] for key in (
                            "requested_batch_size", "effective_batch_size",
                            "min_successful_batch_size", "max_successful_batch_size",
                            "oom_retries", "successful_batches",
                        )
                    }
                    for row in rows if row["target"] != "AVG"
                },
            },
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
