import argparse
import csv
import re
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-dir",
        required=True,
        help="Path to ImageNet val folder, e.g. ../datasets/imagenet/val",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path, e.g. ../datasets/imagenet/imagenet_val_labels.csv",
    )
    args = parser.parse_args()

    val_dir = Path(args.val_dir).resolve()
    out_csv = Path(args.out_csv).resolve()

    if not val_dir.exists():
        raise FileNotFoundError(f"val_dir not found: {val_dir}")

    # ImageNet class folders are WordNet IDs: n01440764, n01443537, ...
    class_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir() and re.fullmatch(r"n\d{8}", p.name)])

    if len(class_dirs) != 1000:
        raise ValueError(f"Expected all 1000 ImageNet class folders, found {len(class_dirs)}. Sorting a partial class set silently assigns incorrect pretrained-model labels. Supply a CSV with canonical 0-based labels for subsets or flat validation folders.")

    wnid_to_label = {p.name: idx for idx, p in enumerate(class_dirs)}

    rows = []
    for class_dir in class_dirs:
        wnid = class_dir.name
        label = wnid_to_label[wnid]

        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                relpath = img_path.relative_to(val_dir).as_posix()
                rows.append((relpath, label, wnid))

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No images found under {val_dir}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "label", "wnid"])
        writer.writerows(rows)

    print(f"Saved: {out_csv}")
    print(f"Classes: {len(class_dirs)}")
    print(f"Images: {len(rows)}")

    if len(rows) != 50000:
        print(f"[WARNING] Expected 50000 ImageNet val images, found {len(rows)}")

    print("\nFirst 5 classes:")
    for p in class_dirs[:5]:
        print(f"{wnid_to_label[p.name]} -> {p.name}")


if __name__ == "__main__":
    main()
