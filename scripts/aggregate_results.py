#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import glob
import os
from pathlib import Path

from src.utils import write_csv


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root like runs/table1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    files = sorted(glob.glob(os.path.join(root, "**", "eval_results.csv"), recursive=True))
    summary = []
    for f in files:
        rel = os.path.relpath(os.path.dirname(f), root)
        parts = rel.split(os.sep)
        setting = parts[0] if len(parts) >= 1 else ""
        source = parts[1] if len(parts) >= 3 else ""
        method = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else parts[0])
        extra = "/".join(parts[3:]) if len(parts) > 3 else ""
        for row in read_rows(f):
            out = {
                "setting": setting,
                "source": source,
                "method": method,
                "extra": extra,
                "target": row.get("target", ""),
                "asr": row.get("asr", ""),
                "total": row.get("total", ""),
                "eval_file": f,
            }
            summary.append(out)
    write_csv(summary, args.out, fieldnames=["setting", "source", "method", "extra", "target", "asr", "total", "eval_file"])
    print("Saved summary to", args.out)


if __name__ == "__main__":
    main()
