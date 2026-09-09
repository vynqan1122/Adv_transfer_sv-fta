#!/usr/bin/env python3
"""Create paper-style CSVs for Tables I-IV from single-source transfer runs.

Expected layout produced by sh/run_table1.sh:
  <root>/<setting>/<source>/<attack>/eval_results.csv

Outputs:
  table1_cnn_to_cnn.csv
  table2_cnn_to_vit.csv
  table3_vit_to_cnn.csv
  table4_vit_to_vit.csv
  tables1_4_long.csv

Each wide CSV has one row per attack and one column per source-target pair.
"""
import argparse
import csv
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

METHOD_ORDER = [
    "difgsm",
    "ifgsm",
    "mifgsm",
    "si_ni_fgsm",
    "tifgsm",
    "vit_aware",
    "freq_only",
    "ours",
]

METHOD_LABELS = {
    "difgsm": "DI-FGSM",
    "ifgsm": "I-FGSM",
    "mifgsm": "MI-FGSM",
    "si_ni_fgsm": "SI-NI-FGSM",
    "tifgsm": "TI-FGSM",
    "vit_aware": "ViT-Aware Attack",
    "freq_only": "Freq-Only",
    "ours": "Ours (SV-FCA)",
}

SOURCE_LABELS = {
    "resnet50": "ResNet50 Surrogate",
    "densenet121": "DenseNet121 Surrogate",
    "vit_base_patch16_224": "ViT-B/16 Surrogate",
    "deit_small_patch16_224": "DeiT-S/16 Surrogate",
}

TARGET_LABELS = {
    "resnet152": "ResNet152",
    "inception_v3": "Inception-v3",
    "vit_large_patch16_224": "ViT-L/16",
    "deit_base_patch16_224": "DeiT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T",
}

TABLE_SPECS = {
    "cnn_to_cnn": {
        "file": "table1_cnn_to_cnn.csv",
        "title": "Table I: CNN-to-CNN",
        "sources": ["densenet121", "resnet50"],
        "targets": ["resnet152", "inception_v3"],
    },
    "cnn_to_vit": {
        "file": "table2_cnn_to_vit.csv",
        "title": "Table II: CNN-to-ViT",
        "sources": ["densenet121", "resnet50"],
        "targets": ["vit_large_patch16_224", "deit_base_patch16_224", "swin_tiny_patch4_window7_224"],
    },
    "vit_to_cnn": {
        "file": "table3_vit_to_cnn.csv",
        "title": "Table III: ViT-to-CNN",
        "sources": ["deit_small_patch16_224", "vit_base_patch16_224"],
        "targets": ["resnet152", "inception_v3"],
    },
    "vit_to_vit": {
        "file": "table4_vit_to_vit.csv",
        "title": "Table IV: ViT-to-ViT",
        "sources": ["deit_small_patch16_224", "vit_base_patch16_224"],
        "targets": ["vit_large_patch16_224", "deit_base_patch16_224", "swin_tiny_patch4_window7_224"],
    },
}


def read_eval(path: Path, metric: str) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        target = row.get("target", "")
        if not target or target == "AVG":
            continue
        value = row.get(metric, "")
        if value in (None, "") and metric in ("asr", "asr_clean_correct"):
            value = row.get("asr_clean_correct" if metric == "asr" else "asr", "")
        if value == "":
            continue
        try:
            numeric = float(value)
            if math.isfinite(numeric):
                out[target] = numeric
        except ValueError:
            pass
    return out


def collect(root: Path, metric: str):
    data = {}
    long_rows = []
    for setting in TABLE_SPECS:
        for source in TABLE_SPECS[setting]["sources"]:
            for method in METHOD_ORDER:
                eval_csv = root / setting / source / method / "eval_results.csv"
                vals = read_eval(eval_csv, metric=metric)
                for target, asr in vals.items():
                    data[(setting, source, method, target)] = asr
                    long_rows.append({
                        "setting": setting,
                        "source": source,
                        "method": method,
                        "target": target,
                        metric: f"{asr:.2f}",
                        "eval_file": str(eval_csv),
                    })
    return data, long_rows


def write_wide_tables(data, out_dir: Path):
    written = []
    for setting, spec in TABLE_SPECS.items():
        columns: List[Tuple[str, str, str]] = []
        for source in spec["sources"]:
            for target in spec["targets"]:
                col_name = f"{SOURCE_LABELS.get(source, source)} | {TARGET_LABELS.get(target, target)}"
                columns.append((source, target, col_name))

        rows = []
        for method in METHOD_ORDER:
            row = {"Attack Method": METHOD_LABELS.get(method, method)}
            any_value = False
            for source, target, col_name in columns:
                value = data.get((setting, source, method, target), None)
                if value is None:
                    row[col_name] = ""
                else:
                    row[col_name] = f"{value:.2f}"
                    any_value = True
            # Keep rows even if empty so missing runs are obvious.
            rows.append(row)

        out_path = out_dir / spec["file"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["Attack Method"] + [c[2] for c in columns]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        written.append(out_path)
        print(f"Saved {spec['title']} -> {out_path}")
    return written


def write_long(rows, out_path: Path, metric: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting", "source", "method", "target", metric, "eval_file"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved long summary ->", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/table1_single_source_transfer")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--metric", default="asr", choices=["asr", "asr_clean_correct", "asr_all"])
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    data, long_rows = collect(root, metric=args.metric)
    write_wide_tables(data, out_dir=out_dir)
    write_long(long_rows, out_dir / "tables1_4_long.csv", metric=args.metric)


if __name__ == "__main__":
    main()
