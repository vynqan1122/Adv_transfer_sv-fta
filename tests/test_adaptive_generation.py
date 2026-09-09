"""No datasets/downloads/GPU: validate generation budgets under simulated CUDA OOM."""
import csv
import json
import tempfile
import unittest
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from attacks.base import AttackResult
from scripts import measure_runtime, run_attack, select_imagenet_subset
from src.batching import AdaptiveBatchExecutor


@contextmanager
def fake_cuda():
    """Run CUDA branches against CPU tensors; never initialize a GPU context."""
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)):
            args = (torch.device("cpu"),) + args[1:]
        if "device" in kwargs:
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "to", cpu_to))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        for name in ("empty_cache", "synchronize", "reset_peak_memory_stats"):
            stack.enter_context(patch("torch.cuda." + name))
        for name in ("max_memory_allocated", "max_memory_reserved"):
            stack.enter_context(patch("torch.cuda." + name, return_value=0))
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def toy_loader():
    images = torch.arange(7, dtype=torch.float32).view(7, 1, 1, 1).expand(7, 3, 4, 4) / 10
    labels = torch.zeros(7, dtype=torch.long)
    paths = [f"sample_{index}.png" for index in range(7)]
    return [(images[:4], labels[:4], paths[:4]), (images[4:], labels[4:], paths[4:])]


class LimitedAttack:
    def __init__(self):
        self.sizes = []
    def __call__(self, _models, images, _labels):
        self.sizes.append(len(images))
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return AttackResult(images + 0.01, [{"step": 0, "loss": 1.0}])


class LimitedClassifier(torch.nn.Module):
    def forward(self, images):
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return torch.tensor([[1.0, 0.0]]).repeat(len(images), 1)


class AdaptiveGenerationTests(unittest.TestCase):
    def test_attack_manifest_preserves_logical_batches_paths_and_all_images(self):
        for storage in ("adv_fp32", "delta_fp16"):
            with self.subTest(storage=storage), tempfile.TemporaryDirectory() as tmp, fake_cuda():
                root = Path(tmp)
                attack = LimitedAttack()
                argv = ["attack", "--device", "cuda", "--surrogates", "toy", "--attack", "ifgsm",
                        "--out-dir", str(root), "--adv-batch-dir", str(root / "batches"),
                        "--batch-size", "4", "--num-batches", "2", "--num-workers", "0",
                        "--models-dir", str(root / "cache"), "--storage-mode", storage]
                with patch("sys.argv", argv), patch.object(run_attack, "ImageNetCSVDataset"), \
                        patch.object(run_attack, "DataLoader", return_value=toy_loader()), \
                        patch.object(run_attack, "build_attack", return_value=attack), \
                        patch.object(run_attack, "load_models", return_value={"toy": torch.nn.Identity()}):
                    run_attack.main()
                with (root / "attack_batches.csv").open(newline="", encoding="utf-8") as stream:
                    manifest = list(csv.DictReader(stream))
                self.assertEqual([int(row["n"]) for row in manifest], [4, 3])
                self.assertEqual([row["microbatch_sizes"] for row in manifest], ["2,2", "2,1"])
                paths = []
                for row in manifest:
                    payload = torch.load(root / row["file"], weights_only=False)
                    paths.extend(payload["relpaths"])
                    self.assertEqual(len(payload["labels"]), int(row["n"]))
                self.assertEqual(paths, [f"sample_{index}.png" for index in range(7)])
                self.assertEqual(attack.sizes, [4, 2, 2, 2, 1])
                config = json.loads((root / "attack_config.json").read_text())
                self.assertEqual(config["generated_images"], 7)
                self.assertEqual(config["batch_execution"]["effective_batch_size"], 2)

    def test_selection_keeps_original_order_and_requested_count(self):
        with tempfile.TemporaryDirectory() as tmp, fake_cuda():
            root = Path(tmp)
            selected = root / "selected.csv"
            argv = ["select", "--device", "cuda", "--surrogates", "toy", "--num-images", "7",
                    "--labels-csv", "fake.csv", "--out-csv", str(selected), "--data-dir", str(root),
                    "--batch-size", "4", "--num-workers", "0", "--models-dir", str(root / "cache")]
            with patch("sys.argv", argv), patch.object(select_imagenet_subset, "build_imagenet_dataset"), \
                    patch.object(select_imagenet_subset, "DataLoader", return_value=toy_loader()), \
                    patch.object(select_imagenet_subset, "resolve_val_dir", return_value=str(root)), \
                    patch.object(select_imagenet_subset, "load_models", return_value={"toy": LimitedClassifier()}):
                select_imagenet_subset.main()
            with selected.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["relpath"] for row in rows], [f"sample_{index}.png" for index in range(7)])
            metadata = json.loads(Path(str(selected) + ".metadata.json").read_text())
            self.assertEqual(metadata["selected_images"], 7)
            self.assertEqual(metadata["batch_execution"]["oom_retries"], 1)

    def test_runtime_excludes_failed_attempt_and_preserves_num_batches_budget(self):
        attack = LimitedAttack()
        metadata = {}
        with fake_cuda(), patch.object(measure_runtime.time, "perf_counter", side_effect=[0, 10, 11, 20, 22]):
            ms, images, batches = measure_runtime.measure_attack(
                attack, {}, toy_loader(), torch.device("cuda"), num_batches=1, warmup_batches=0,
                executor=AdaptiveBatchExecutor(4, "cuda"), measurement_metadata=metadata)
        self.assertEqual((ms, images, batches), (750.0, 4, 1))
        self.assertEqual(metadata["measured_microbatches"], 2)
        self.assertEqual(metadata["measured_batch_sizes"], [2])
        self.assertEqual(metadata["measured_oom_retries"], 1)

    def test_runtime_warmup_reduction_keeps_entire_measurement_budget(self):
        attack = LimitedAttack()
        metadata = {}
        with fake_cuda(), patch.object(measure_runtime.time, "perf_counter", side_effect=range(8)):
            _ms, images, batches = measure_runtime.measure_attack(
                attack, {}, toy_loader(), torch.device("cuda"), max_images=7, warmup_batches=1,
                executor=AdaptiveBatchExecutor(4, "cuda"), measurement_metadata=metadata)
        self.assertEqual((images, batches), (7, 2))
        self.assertEqual(attack.sizes, [4, 2, 2, 2, 2, 2, 1])
        self.assertEqual(metadata["warmup_oom_retries"], 1)
        self.assertEqual(metadata["measured_oom_retries"], 0)


if __name__ == "__main__":
    unittest.main()
