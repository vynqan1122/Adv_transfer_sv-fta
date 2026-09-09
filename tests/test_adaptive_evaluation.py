"""Tiny CPU fixtures simulate CUDA OOM; no GPU, images or weights are needed."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import warnings

import torch

from scripts import compute_perceptual_quality as quality
from scripts import evaluate


class ThresholdModel(torch.nn.Module):
    def __init__(self, failure=None):
        super().__init__()
        self.failure = failure
        self.sizes = []

    def forward(self, images):
        self.sizes.append(len(images))
        # Fail on the adversarial forward after a successful clean forward.
        if len(self.sizes) == 2 and self.failure is not None:
            raise self.failure
        score = images.mean((1, 2, 3))
        return torch.stack((0.5 - score, score - 0.5), dim=1)


class AdaptiveEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.clean_values = torch.tensor([0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1])
        adv_values = torch.tensor([0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9])
        self.payloads = {}
        for filename, start, end in (("first.pt", 0, 5), ("second.pt", 5, 8)):
            self.payloads[filename] = {
                "adv": adv_values[start:end, None, None, None].expand(-1, 3, 2, 2).clone(),
                "labels": torch.zeros(end - start, dtype=torch.long),
                "relpaths": [str(index) for index in range(start, end)],
            }

    def cpu_fixtures(self, module):
        stack = ExitStack()
        stack.enter_context(patch.object(module, "safe_torch_load", side_effect=self.payloads.__getitem__))
        stack.enter_context(patch.object(module, "load_clean_batch", side_effect=self.load_clean))
        # Exercise the real retry executor using a CUDA device, but keep all
        # tensor arithmetic on CPU so the test cannot allocate GPU memory.
        stack.enter_context(patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        stack.enter_context(patch("torch.cuda.empty_cache"))
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore", RuntimeWarning)
        return stack

    def load_clean(self, data_dir, relpaths, device):
        values = self.clean_values[[int(path) for path in relpaths]]
        return values[:, None, None, None].expand(-1, 3, 2, 2).clone()

    def test_second_forward_oom_retries_without_double_counting_and_keeps_cap(self):
        model = ThresholdModel(torch.cuda.OutOfMemoryError("synthetic CUDA out of memory"))
        with self.cpu_fixtures(evaluate):
            result = evaluate.evaluate_one_model(
                model, list(self.payloads), "unused", self.device, eval_batch_size=4,
            )
        self.assertEqual(result["total"], 8)
        self.assertEqual(result["clean_correct"], 5)
        self.assertEqual(result["adv_wrong"], 4)
        self.assertEqual(result["success_on_clean_correct"], 3)
        self.assertEqual(result["asr"], 60.0)
        self.assertEqual(result["robust_acc_clean_correct"], 40.0)
        self.assertEqual(result["requested_batch_size"], 4)
        self.assertEqual(result["effective_batch_size"], 2)
        self.assertEqual(result["oom_retries"], 1)
        self.assertEqual(result["successful_batches"], 5)
        self.assertEqual(model.sizes, [4, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1])

    def test_evaluation_defaults_to_sixteen(self):
        with self.cpu_fixtures(evaluate):
            result = evaluate.evaluate_one_model(ThresholdModel(), list(self.payloads), "unused", self.device)
        self.assertEqual(result["requested_batch_size"], 16)
        self.assertEqual(result["effective_batch_size"], 16)

    def test_unrelated_error_and_disabled_fallback_propagate(self):
        for failure, auto_batch in (
            (RuntimeError("invalid model shape"), True),
            (torch.cuda.OutOfMemoryError("synthetic CUDA out of memory"), False),
        ):
            with self.subTest(auto_batch=auto_batch):
                model = ThresholdModel(failure)
                with self.cpu_fixtures(evaluate), self.assertRaises(type(failure)) as raised:
                    evaluate.evaluate_one_model(
                        model, list(self.payloads), "unused", self.device,
                        eval_batch_size=4, auto_batch=auto_batch,
                    )
                self.assertIs(raised.exception, failure)
                self.assertEqual(model.sizes, [4, 4])

    def test_lpips_oom_does_not_partially_commit_other_quality_metrics(self):
        calls = []

        def fake_lpips(clean, adv):
            calls.append(len(clean))
            if len(clean) > 2:
                raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
            return (clean - adv).abs().flatten(1).mean(1)

        def fake_psnr(clean, adv):
            return clean.flatten(1).mean(1) * 10

        def fake_ssim(clean, adv):
            return adv.flatten(1).mean(1)

        stats = {}
        with self.cpu_fixtures(quality), \
                patch.object(quality, "resolve_batch_files", return_value=list(self.payloads)), \
                patch.object(quality, "load_lpips", return_value=fake_lpips), \
                patch.object(quality, "psnr_batch", side_effect=fake_psnr), \
                patch.object(quality, "ssim_batch", side_effect=fake_ssim):
            result = quality.compute_for_attack("unused", "attack", self.device, execution_stats=stats)

        self.assertEqual(len(result), 4)  # Preserve the public tuple contract.
        self.assertAlmostEqual(result[0], 4.0, places=5)
        self.assertAlmostEqual(result[1], 0.5, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)
        self.assertEqual(result[3], list(self.payloads))
        self.assertEqual(stats["total_images"], 8)
        self.assertEqual(stats["lpips_images"], 8)
        self.assertEqual(stats["requested_batch_size"], 16)
        self.assertEqual(stats["effective_batch_size"], 2)
        self.assertEqual(stats["oom_retries"], 1)
        self.assertEqual(calls, [5, 2, 2, 1, 2, 1])


if __name__ == "__main__":
    unittest.main()
