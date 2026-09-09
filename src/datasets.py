import csv
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def resolve_val_dir(data_dir, val_dir=None):
    if val_dir:
        return os.path.abspath(val_dir)
    data_dir = os.path.abspath(data_dir)
    candidate = os.path.join(data_dir, "val")
    if os.path.isdir(candidate):
        return candidate
    return data_dir


def resolve_image_path(data_dir, relpath):
    """Resolve CSV paths consistently on Windows and Linux, including val roots."""
    relpath = str(relpath).replace("\\", "/")
    path = os.path.join(data_dir, relpath)
    if not os.path.isfile(path) and relpath.startswith("val/"):
        path = os.path.join(data_dir, relpath[4:])
    if not os.path.isfile(path) and not relpath.startswith("val/"):
        path = os.path.join(data_dir, "val", relpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Clean image not found: {path} (data_dir={data_dir}, relpath={relpath})")
    return path


class ImageNetCSVDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir = os.path.abspath(data_dir)
        self.transform = transform
        self.samples = []
        with open(labels_csv, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"relpath", "label"}.issubset(reader.fieldnames):
                raise ValueError("labels_csv must contain columns: relpath,label")
            for line, row in enumerate(reader, start=2):
                relpath = (row["relpath"] or "").strip().replace("\\", "/")
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid label at {labels_csv}:{line}") from exc
                if not relpath or not 0 <= label < 1000:
                    raise ValueError(f"Expected nonempty relpath and 0-based ImageNet label in [0, 999] at {labels_csv}:{line}")
                self.samples.append((relpath, label))
        if not self.samples:
            raise ValueError(f"No images listed in {labels_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relpath, label = self.samples[index]
        path = resolve_image_path(self.data_dir, relpath)
        with Image.open(path) as source:
            img = source.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label), relpath


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        relpath = os.path.relpath(path, self.root).replace("\\", "/")
        return img, int(label), relpath


def imagenet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def build_imagenet_dataset(data_dir, val_dir=None, labels_csv=None):
    transform = imagenet_transform()
    if labels_csv:
        return ImageNetCSVDataset(data_dir=data_dir, labels_csv=labels_csv, transform=transform)
    root = resolve_val_dir(data_dir, val_dir)
    dataset = ImageFolderWithPaths(root=root, transform=transform)
    if len(dataset.classes) != 1000:
        raise ValueError("ImageFolder requires all 1000 ImageNet class directories to preserve pretrained-model labels. For a subset, pass --labels-csv with canonical 0-based labels.")
    return dataset


def collate_with_paths(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long), list(paths)
