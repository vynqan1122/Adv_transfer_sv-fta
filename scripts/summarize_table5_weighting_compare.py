#!/usr/bin/env python3
"""Summarize a Table V comparison between original full model and no-weighting.

Expected directory layout produced by sh/run_table5_no_weighting_5000.sh:
  <root>/<setting>/<run_name>/eval_results.csv
  <root>/<setting>/<run_name>/attack_config.json

Outputs:
  1) ASR table: Original Full Model vs Full Model (No Weighting)
  2) Config table: key hyperparameters for each run
"""
import argparse
import csv
import json
import os
import math
from pathlib import Path

SETTING_LABELS = {
    "cnn_to_vit": "CNN-to-ViT",
    "vit_to_cnn": "ViT-to-CNN",
    "vit_to_vit": "ViT-to-ViT",
    "mixed_mixed": "Mixed-to-Mixed",
}

DEFAULT_RUN_LABELS = {
    "original_full_model": "Original Full Model (With Weighting)",
    "full_model": "Full Model (No Weighting)",
    "full_model_no_weighting": "Full Model (No Weighting)",
}

CONFIG_KEYS = [
    "surrogates", "attack", "variant", "fusion", "freq_mode", "num_views", "rho",
    "lambda_grid", "eps_c", "c_min", "c_max", "diversity_prob", "eps", "alpha",
    "steps", "batch_size", "selected_csv", "threat_model", "source_setting",
    "target_access_during_attack",
]


def read_avg_asr(eval_csv: Path, metric: str = "asr"):
    with eval_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    avg = next((r for r in rows if r.get("target") == "AVG"), None)
    if avg is None:
        raise ValueError(f"Missing pooled AVG row in {eval_csv}")
    value = avg.get(metric, "")
    if value in (None, "") and metric == "asr":
        value = avg.get("asr_clean_correct", "")
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/table5_weighting_compare_5000")
    ap.add_argument("--out", default="runs/table5_weighting_compare_5000/table5_weighting_compare_5000.csv")
    ap.add_argument("--config-out", default="runs/table5_weighting_compare_5000/table5_weighting_compare_configs.csv")
    ap.add_argument("--metric", default="asr", choices=["asr", "asr_clean_correct", "asr_all", "adv_acc_all", "clean_acc", "robust_acc_clean_correct"])
    ap.add_argument("--settings", default="cnn_to_vit,vit_to_cnn,vit_to_vit,mixed_mixed")
    ap.add_argument("--runs", default="original_full_model,full_model")
    args = ap.parse_args()

    root = Path(args.root)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    rows = []
    for run_name in runs:
        row = {"Run": DEFAULT_RUN_LABELS.get(run_name, run_name)}
        for setting in settings:
            label = SETTING_LABELS.get(setting, setting)
            eval_csv = root / setting / run_name / "eval_results.csv"
            value = read_avg_asr(eval_csv, args.metric) if eval_csv.exists() else None
            row[label] = "" if value is None else "{:.2f}".format(value)
        rows.append(row)

    fieldnames = ["Run"] + [SETTING_LABELS.get(s, s) for s in settings]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print("Saved", args.out)

    config_rows = []
    for setting in settings:
        for run_name in runs:
            cfg_path = root / setting / run_name / "attack_config.json"
            if not cfg_path.exists():
                continue
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            row = {"Setting": SETTING_LABELS.get(setting, setting), "Run": DEFAULT_RUN_LABELS.get(run_name, run_name)}
            for key in CONFIG_KEYS:
                value = cfg.get(key, "")
                if isinstance(value, (list, tuple)):
                    value = ",".join(map(str, value))
                row[key] = value
            config_rows.append(row)

    if config_rows:
        cfg_fields = ["Setting", "Run"] + CONFIG_KEYS
        os.makedirs(os.path.dirname(args.config_out) or ".", exist_ok=True)
        with open(args.config_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cfg_fields)
            w.writeheader(); w.writerows(config_rows)
        print("Saved", args.config_out)


if __name__ == "__main__":
    main()
