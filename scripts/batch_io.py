"""Shared readers for adversarial batches, metrics, and figure scripts."""
import csv
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from PIL import Image

from src.datasets import imagenet_transform, resolve_image_path


def safe_torch_load(file_path):
    try:
        return torch.load(file_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only.
        return torch.load(file_path, map_location="cpu")


def resolve_batch_files(attack_dir):
    """Fail on incomplete manifests instead of silently changing sample size."""
    attack_dir = Path(attack_dir).resolve()
    manifest = attack_dir / "attack_batches.csv"
    files = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "file" not in reader.fieldnames:
                raise ValueError(f"{manifest} must contain column: file")
            for row in reader:
                raw = (row.get("file") or "").strip()
                if not raw:
                    raise ValueError(f"Empty batch path in {manifest}")
                path = Path(raw.replace("\\", "/"))
                if not path.is_absolute():
                    # New manifests use paths relative to their attack directory;
                    # the CWD fallback preserves historical project-relative CSVs.
                    local = attack_dir / path
                    path = local if local.is_file() else path.resolve()
                files.append(str(path.resolve()))
    else:
        files = [str(p) for p in sorted((attack_dir / "adv_batches").glob("*.pt"))]
    if not files:
        raise RuntimeError(f"No adversarial batches found in {attack_dir}")
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} adversarial batch file(s) missing; first: {missing[0]}. Regenerate the complete attack before evaluation.")
    if len(set(map(os.path.normcase, files))) != len(files):
        raise ValueError(f"Duplicate batch paths in {manifest}")
    return files


def load_clean_tensor(data_dir, relpath):
    with Image.open(resolve_image_path(data_dir, relpath)) as source:
        return imagenet_transform()(source.convert("RGB"))


def load_clean_batch(data_dir, relpaths, device):
    if not relpaths:
        raise ValueError("Cannot load an empty image batch")
    return torch.stack([load_clean_tensor(data_dir, p) for p in relpaths]).to(device, non_blocking=True)


def validate_payload(payload, file_path="batch"):
    relpaths = payload["relpaths"]
    labels = torch.as_tensor(payload["labels"])
    key = "adv" if payload.get("adv") is not None else "delta"
    if payload.get(key) is None:
        raise KeyError(f"{file_path} contains neither 'adv' nor 'delta'")
    images = torch.as_tensor(payload[key])
    if not relpaths or images.ndim != 4 or images.shape[1] != 3 or images.shape[0] != len(relpaths) or labels.shape != (len(relpaths),):
        raise ValueError(f"Inconsistent image, label, and path batch dimensions in {file_path}")
    if not torch.isfinite(images).all():
        raise ValueError(f"Nonfinite adversarial tensor in {file_path}")
    return images, labels, relpaths, key


def reconstruct_adv(clean, stored, storage_key, eps=None):
    if stored.shape != clean.shape:
        raise ValueError(f"Adversarial/clean shape mismatch: {stored.shape} vs {clean.shape}")
    if storage_key == "adv":
        return stored.float()
    delta = stored.float()
    if eps is not None:
        delta = delta.clamp(-float(eps), float(eps))
    return (clean + delta).clamp(0.0, 1.0)


def get_sample_from_attack(attack_dir, index, data_dir=None):
    remaining = int(index)
    if remaining < 0:
        raise IndexError("Sample index must be nonnegative")
    for path in resolve_batch_files(attack_dir):
        payload = safe_torch_load(path)
        images, labels, relpaths, key = validate_payload(payload, path)
        if remaining < len(relpaths):
            image = images[remaining].float()
            if key == "delta":
                if data_dir is None:
                    raise ValueError("data_dir is required to reconstruct delta_fp16 batches")
                clean = load_clean_tensor(data_dir, relpaths[remaining])
                image = reconstruct_adv(clean, image, key, payload.get("eps"))
            return image, int(labels[remaining]), relpaths[remaining]
        remaining -= len(relpaths)
    raise IndexError(f"index={index} is out of range for {attack_dir}")


def delete_consumed_batches(attack_dir, batch_files):
    """Preserve unconsumed and failed-to-delete manifest entries on partial runs."""
    deleted = set()
    parents = set()
    for file_path in batch_files:
        path = Path(file_path).resolve()
        try:
            path.unlink()
            deleted.add(os.path.normcase(str(path)))
            parents.add(path.parent)
        except OSError as exc:
            print(f"[cleanup warning] {path}: {exc}")
    manifest = Path(attack_dir) / "attack_batches.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        remaining = []
        for row in rows:
            path = Path(row["file"].replace("\\", "/"))
            if not path.is_absolute():
                local = manifest.parent / path
                local_resolved = os.path.normcase(str(local.resolve()))
                path = local if local.is_file() or local_resolved in deleted else path
            if os.path.normcase(str(path.resolve())) not in deleted:
                remaining.append(row)
        if remaining:
            temporary = manifest.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(remaining)
            temporary.replace(manifest)
        else:
            manifest.unlink()
    for parent in sorted(parents, key=lambda p: len(str(p)), reverse=True):
        try:
            parent.rmdir()  # Only succeeds for an empty directory.
        except OSError:
            pass
    return len(deleted)
