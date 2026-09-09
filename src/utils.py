import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path, fieldnames=None):
    rows = list(rows)
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    # Preserve requested/inferred order but retain keys from every row.
    fieldnames = list(fieldnames)
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_model_list(text):
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def set_model_cache_dir(model_dir):
    # Must be called before timm/torchvision downloads model weights.
    model_dir = os.path.abspath(model_dir)
    os.environ.setdefault("TORCH_HOME", model_dir)
    os.environ.setdefault("HF_HOME", os.path.join(model_dir, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(model_dir, "hf", "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(model_dir, "xdg"))
    ensure_dir(model_dir)
    return model_dir


def clamp_linf(x_adv, x_clean, eps):
    delta = torch.clamp(x_adv - x_clean, min=-eps, max=eps)
    return torch.clamp(x_clean + delta, 0.0, 1.0)


def tensor_stats(x):
    return {
        "min": float(x.min().detach().cpu()),
        "max": float(x.max().detach().cpu()),
        "mean": float(x.mean().detach().cpu()),
        "std": float(x.std(unbiased=False).detach().cpu()),
    }
