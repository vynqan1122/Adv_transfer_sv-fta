import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from scripts.batch_io import (
    delete_consumed_batches,
    get_sample_from_attack,
    load_clean_tensor,
    reconstruct_adv,
    resolve_batch_files,
    validate_payload,
)
from scripts.compute_perceptual_quality import psnr_batch, ssim_batch
from scripts.evaluate import evaluate_one_model, weighted_average
from scripts.measure_runtime import measure_attack
from scripts.summarize_table5_prime import read_avg_metric
from scripts.summarize_table5_weighting_compare import read_avg_asr
from scripts.summarize_tables1_4_transfer import read_eval
from scripts.visualize_frequency_spectrum import radial_profile
from src.datasets import ImageNetCSVDataset, imagenet_transform
from src.metrics import pairwise_cosine_agreement


def write_manifest(directory, rows):
    with (Path(directory) / "attack_batches.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "n"])
        writer.writeheader()
        writer.writerows(rows)


class BatchIOTests(unittest.TestCase):
    def test_manifest_requires_all_batches_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "first.pt").touch()
            write_manifest(directory, [{"file": "first.pt", "n": 1}, {"file": "missing.pt", "n": 1}])
            with self.assertRaises(FileNotFoundError):
                resolve_batch_files(directory)
            write_manifest(directory, [{"file": "first.pt", "n": 1}, {"file": "first.pt", "n": 1}])
            with self.assertRaises(ValueError):
                resolve_batch_files(directory)

    def test_partial_cleanup_keeps_remaining_manifest_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("first.pt", "second.pt"):
                (directory / name).touch()
            write_manifest(directory, [{"file": "first.pt", "n": 1}, {"file": "second.pt", "n": 1}])
            self.assertEqual(delete_consumed_batches(directory, [directory / "first.pt"]), 1)
            self.assertEqual(resolve_batch_files(directory), [str(directory / "second.pt")])

    def test_half_delta_reconstructs_for_figures_and_honors_epsilon(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            Image.new("RGB", (224, 224), (128, 128, 128)).save(directory / "clean.png")
            clean = load_clean_tensor(directory, "clean.png")
            eps = 0.1
            delta = torch.full_like(clean, 0.1001).half().unsqueeze(0)
            torch.save({"delta": delta, "eps": eps, "labels": torch.tensor([4]), "relpaths": ["clean.png"]}, directory / "first.pt")
            write_manifest(directory, [{"file": "first.pt", "n": 1}])
            adv, label, relpath = get_sample_from_attack(directory, 0, directory)
            self.assertEqual((label, relpath), (4, "clean.png"))
            self.assertLessEqual(float((adv - clean).abs().max()), eps + 1e-7)
            with self.assertRaises(IndexError):
                get_sample_from_attack(directory, -1, directory)

    def test_rejects_invalid_tensor_dimensions(self):
        payload = {"adv": torch.zeros(2, 3, 12, 12), "labels": torch.tensor([0]), "relpaths": ["one.png"]}
        with self.assertRaises(ValueError):
            validate_payload(payload)
        with self.assertRaises(ValueError):
            reconstruct_adv(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 9, 9), "delta")


class DatasetTests(unittest.TestCase):
    def test_csv_paths_port_between_windows_and_linux_and_val_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "class").mkdir()
            Image.new("RGB", (230, 240)).save(directory / "class" / "clean.png")
            path = directory / "labels.csv"
            path.write_text("relpath,label\nval\\class\\clean.png,999\n", encoding="utf-8-sig")
            dataset = ImageNetCSVDataset(directory, path, imagenet_transform())
            image, label, relpath = dataset[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertEqual(label, 999)
            self.assertEqual(relpath, "val/class/clean.png")

    def test_empty_or_noncanonical_labels_fail_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            for text in ("", "relpath,label\n", "relpath,label\none.png,1000\n", "relpath,label\none.png,-1\n"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    ImageNetCSVDataset(temporary, path)


class MetricTests(unittest.TestCase):
    def test_psnr_known_mse_and_identical_image(self):
        clean = torch.zeros(2, 3, 16, 16, dtype=torch.float64)
        adv = clean.clone()
        adv[1] = 0.1
        psnr = psnr_batch(clean, adv)
        self.assertTrue(torch.isinf(psnr[0]))
        self.assertAlmostEqual(float(psnr[1]), 20.0, places=9)

    def test_ssim_identity_and_point_window_analytic_value(self):
        clean = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
        self.assertAlmostEqual(float(ssim_batch(clean, clean)[0]), 1.0, places=12)
        adv = torch.full_like(clean, 0.1)
        expected = 0.01 ** 2 / (0.1 ** 2 + 0.01 ** 2)
        self.assertAlmostEqual(float(ssim_batch(clean, adv, window_size=1)[0]), expected, places=12)
        with self.assertRaises(ValueError):
            ssim_batch(clean, adv, window_size=2)

    def test_target_average_pools_counts_exactly_and_undefined_asr_stays_missing(self):
        rows = [
            {"total": 10, "clean_correct": 2, "adv_wrong": 9, "correct_after_attack": 1, "success_on_clean_correct": 1},
            {"total": 10, "clean_correct": 8, "adv_wrong": 9, "correct_after_attack": 1, "success_on_clean_correct": 7},
        ]
        result = weighted_average(rows)
        self.assertEqual(result["asr"], 80.0)
        self.assertEqual(result["robust_acc_clean_correct"], 20.0)
        self.assertIsNone(weighted_average([dict(rows[0], clean_correct=0, success_on_clean_correct=0)])["asr"])

    def test_evaluation_conditions_on_clean_correct_and_chunks_identically(self):
        class MeanModel(torch.nn.Module):
            def forward(self, x):
                score = x.mean((1, 2, 3))
                return torch.stack((0.5 - score, score - 0.5), dim=1)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            Image.new("RGB", (224, 224), (0, 0, 0)).save(directory / "a.png")
            Image.new("RGB", (224, 224), (255, 255, 255)).save(directory / "b.png")
            torch.save({"adv": torch.ones(2, 3, 224, 224), "labels": torch.tensor([0, 0]), "relpaths": ["a.png", "b.png"]}, directory / "batch.pt")
            for chunk in (None, 1):
                result = evaluate_one_model(MeanModel(), [directory / "batch.pt"], directory, torch.device("cpu"), chunk)
                self.assertEqual(result["total"], 2)
                self.assertEqual(result["clean_correct"], 1)
                self.assertEqual(result["asr"], 100.0)
                self.assertEqual(result["clean_acc"], 50.0)

    def test_fft_radial_profile_uses_actual_shifted_dc_pixel(self):
        spectrum = np.zeros((8, 8))
        spectrum[4, 4] = 1.0
        profile = radial_profile(spectrum)
        self.assertEqual(profile[0], 1.0)
        self.assertTrue(np.all(profile[1:] == 0.0))

    def test_empty_gradient_list_has_clear_error(self):
        with self.assertRaises(ValueError):
            pairwise_cosine_agreement([])


class RuntimeTests(unittest.TestCase):
    def test_warmup_does_not_consume_measurement_budget(self):
        loader = [(torch.zeros(4, 3, 8, 8), torch.zeros(4, dtype=torch.long), ["x"] * 4)]
        sizes = []
        def attack(_models, images, _labels):
            sizes.append(len(images))
        with mock.patch("scripts.measure_runtime.time.perf_counter", side_effect=[1.0, 2.0]):
            ms, images, batches = measure_attack(attack, {}, loader, torch.device("cpu"), num_batches=1, warmup_batches=1)
        self.assertEqual(sizes, [4, 4])
        self.assertEqual((ms, images, batches), (250.0, 4, 1))

    def test_last_measured_batch_is_actually_trimmed(self):
        loader = [(torch.zeros(4, 3, 8, 8), torch.zeros(4, dtype=torch.long), ["x"] * 4)] * 2
        sizes = []
        def attack(_models, images, _labels):
            sizes.append(len(images))
        with mock.patch("scripts.measure_runtime.time.perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]):
            ms, images, batches = measure_attack(attack, {}, loader, torch.device("cpu"), max_images=5, warmup_batches=0)
        self.assertEqual(sizes, [4, 1])
        self.assertEqual((ms, images, batches), (400.0, 5, 2))


class SummaryTests(unittest.TestCase):
    def test_missing_requested_metric_cannot_silently_substitute_conditional_asr(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eval.csv"
            path.write_text("target,asr\nmodel,42\nAVG,42\n", encoding="utf-8")
            self.assertEqual(read_eval(path, "asr_all"), {})
            self.assertIsNone(read_avg_asr(path, "asr_all"))
            self.assertEqual(read_eval(path, "asr_clean_correct"), {"model": 42.0})

    def test_missing_average_is_not_replaced_with_last_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eval.csv"
            path.write_text("target,asr\nfirst,10\nlast,90\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_avg_metric(path, "asr")
            with self.assertRaises(ValueError):
                read_avg_asr(path)


if __name__ == "__main__":
    unittest.main()
