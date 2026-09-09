#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


import argparse
import torch

from src.models import SURROGATE_ALL, TARGET_ALL, load_models
from src.utils import get_device, parse_model_list, set_model_cache_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="./pretrained_models")
    parser.add_argument("--models", default=",".join(SURROGATE_ALL + TARGET_ALL))
    parser.add_argument("--device", default="cpu", help="Use cpu for download to avoid GPU memory use")
    args = parser.parse_args()

    set_model_cache_dir(args.models_dir)
    device = get_device(args.device)
    names = parse_model_list(args.models)
    if not names:
        parser.error("--models must contain at least one name")
    print("Downloading/loading pretrained models into cache:", args.models_dir)
    for name in names:
        print("[model]", name)
        loaded = load_models([name], device=device, pretrained=True)
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
