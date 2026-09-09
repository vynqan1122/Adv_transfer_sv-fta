"""Run selection -> attack -> evaluation -> quality without downloaded models."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from scripts import evaluate, run_attack, select_imagenet_subset
from scripts.batch_io import get_sample_from_attack, load_clean_tensor, resolve_batch_files
from scripts.compute_perceptual_quality import compute_for_attack


class ThresholdClassifier(torch.nn.Module):
    def forward(self, x):
        score = 10 * (x.mean(dim=(1, 2, 3)) - 0.5)
        return torch.stack((score, -score), dim=1)


def toy_models(names, device, **kwargs):
    return {name: ThresholdClassifier().to(device) for name in names}


class PipelineTests(unittest.TestCase):
    def test_cli_pipeline_both_storage_formats_and_partial_final_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "images"
            data.mkdir()
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["relpath", "label"])
                for index, value in enumerate((153, 160, 170)):
                    filename = f"sample_{index}.png"
                    Image.new("RGB", (240, 256), (value,) * 3).save(data / filename)
                    writer.writerow([filename, 0])
            selected = root / "selected.csv"
            common = ["--data-dir", str(data), "--models-dir", str(root / "cache"),
                      "--device", "cpu", "--num-workers", "0", "--batch-size", "2"]
            selection_args = ["select", *common, "--labels-csv", str(labels),
                              "--surrogates", "toy_source", "--num-images", "3", "--out-csv", str(selected)]
            with patch.object(select_imagenet_subset, "load_models", toy_models), patch("sys.argv", selection_args):
                select_imagenet_subset.main()
            for storage in ("adv_fp32", "delta_fp16"):
                with self.subTest(storage=storage):
                    output = root / storage
                    attack_args = ["attack", *common, "--selected-csv", str(selected),
                                   "--surrogates", "toy_source", "--attack", "sv_fca", "--steps", "2",
                                   "--eps", "0.2", "--alpha", "0.1", "--num-views", "2",
                                   "--spectral-bands", "3", "--diversity-prob", "0",
                                   "--out-dir", str(output), "--adv-batch-dir", str(root / "batches" / storage),
                                   "--storage-mode", storage]
                    with patch.object(run_attack, "load_models", toy_models), patch("sys.argv", attack_args):
                        run_attack.main()
                    self.assertEqual(len(resolve_batch_files(output)), 2)
                    for index in range(3):
                        image, label, path = get_sample_from_attack(output, index, data)
                        clean = load_clean_tensor(data, path)
                        self.assertEqual(label, 0)
                        self.assertLessEqual((image - clean).abs().max().item(), 0.200001)
                        self.assertLess(image.mean().item(), 0.5)
                    eval_args = ["evaluate", "--attack-dir", str(output), "--data-dir", str(data),
                                 "--models-dir", str(root / "cache"), "--targets", "toy_target",
                                 "--eval-batch-size", "1", "--device", "cpu"]
                    with patch.object(evaluate, "load_models", toy_models), patch("sys.argv", eval_args):
                        evaluate.main()
                    with (output / "eval_results.csv").open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(int(rows[0]["total"]), 3)
                    self.assertEqual(float(rows[0]["clean_acc"]), 100)
                    self.assertEqual(float(rows[0]["asr"]), 100)
                    psnr, ssim, lpips, files = compute_for_attack(
                        data, output, torch.device("cpu"), use_lpips=False, quality_batch_size=1)
                    self.assertTrue(0 < psnr < 100)
                    self.assertTrue(0 < ssim < 1)
                    self.assertEqual(lpips, "")
                    self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
