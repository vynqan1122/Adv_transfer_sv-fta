#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

EXPECTED = [
    "Salman2020Do_R50.pt",
    "Mo2022When_ViT-B.pt",
    "Liu2023Comprehensive_Swin-B.pt",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("ROBUSTBENCH_MODEL_DIR", "./models"),
                    help="PARENT folder containing imagenet/Linf")
    ap.add_argument("--targets", default=",".join("robustbench:" + name[:-3] + ":imagenet:Linf" for name in EXPECTED),
                    help="Comma-separated target specs; checks local files for RobustBench targets only")
    args = ap.parse_args()
    root = Path(args.model_dir).expanduser().resolve()
    linf = root / "imagenet" / "Linf"
    print("[Table VIII] ROBUSTBENCH_MODEL_DIR =", root)
    print("[Table VIII] expected Linf folder     =", linf)
    missing = []
    checked = 0
    for target in (item.strip() for item in args.targets.split(",") if item.strip()):
        if not target.startswith("robustbench:"):
            continue
        parts = target.split(":")
        if len(parts) > 4 or len(parts) < 2 or not parts[1]:
            ap.error(f"Invalid RobustBench target: {target}")
        dataset = parts[2] if len(parts) >= 3 and parts[2] else "imagenet"
        threat = parts[3] if len(parts) >= 4 and parts[3] else "Linf"
        path = root / dataset / threat / (parts[1] + ".pt")
        checked += 1
        ok = path.is_file()
        print("  [{}] {}".format("OK" if ok else "MISSING", path))
        if not ok:
            missing.append(str(path))
    if missing:
        raise SystemExit(
            "\nTable VIII defense path is not ready. Set ROBUSTBENCH_MODEL_DIR to the parent "
            "directory that contains imagenet/Linf. Example:\n"
            "  export ROBUSTBENCH_MODEL_DIR=/path/to/models\n"
            "Do NOT point it directly to .../models/imagenet/Linf."
        )
    print(f"[Table VIII] all {checked} requested local defense checkpoints found.")


if __name__ == "__main__":
    main()
