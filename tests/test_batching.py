"""Exercise CUDA retry policy using tiny CPU tensors and simulated allocation errors."""
import random
import unittest
import warnings
import weakref
from unittest.mock import patch

import numpy as np
import torch

from src.batching import AdaptiveBatchExecutor, model_loading_oom_hint


class AdaptiveBatchTests(unittest.TestCase):
    def test_halves_only_failed_chunks_and_keeps_learned_size(self):
        executor = AdaptiveBatchExecutor(8, "cuda")
        attempts = []

        def process(start, end):
            attempts.append((start, end))
            if end - start > 2:
                raise torch.cuda.OutOfMemoryError("simulated CUDA allocation")
            return list(range(start, end))

        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.empty_cache") as empty:
            with self.assertWarns(RuntimeWarning):
                chunks = list(executor.iter_batches(8, process))
            subsequent = list(executor.iter_batches(5, process))
        self.assertEqual(attempts[:4], [(0, 8), (0, 4), (0, 2), (2, 4)])
        self.assertEqual([x for _, _, part in chunks for x in part], list(range(8)))
        self.assertEqual([x for _, _, part in subsequent for x in part], list(range(5)))
        self.assertEqual(empty.call_count, 2)
        self.assertEqual(executor.metadata["oom_retries"], 2)
        self.assertEqual(executor.metadata["effective_batch_size"], 2)
        self.assertEqual(executor.metadata["successful_batch_sizes"], [1, 2])

    def test_failed_scope_is_released_before_cache_cleanup_and_rng_is_restored(self):
        executor = AdaptiveBatchExecutor(2, "cuda")
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        draws, refs = [], []

        def process(start, end):
            temporary = torch.ones(2)
            refs.append(weakref.ref(temporary))
            draws.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
            if end - start > 1:
                raise RuntimeError("CUDA out of memory. simulated legacy exception")
            return start

        def clear_cache():
            self.assertIsNone(refs[0](), "failed callback retains temporary tensors")

        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.empty_cache", side_effect=clear_cache):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual([part for _, _, part in executor.iter_batches(2, process)], [0, 1])
        self.assertEqual(draws[0], draws[1])
        self.assertNotEqual(draws[1], draws[2])

    def test_late_oom_does_not_replay_an_earlier_successful_chunk(self):
        executor = AdaptiveBatchExecutor(4, "cuda")
        attempts = []
        def process(start, end):
            attempts.append((start, end))
            if start == 4 and end - start > 2:
                raise torch.cuda.OutOfMemoryError("simulated")
            return list(range(start, end))
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.empty_cache"):
            with self.assertWarns(RuntimeWarning):
                chunks = list(executor.iter_batches(8, process))
        self.assertEqual(attempts, [(0, 4), (4, 8), (4, 6), (6, 8)])
        self.assertEqual([sample for _, _, chunk in chunks for sample in chunk], list(range(8)))

    def test_unrelated_errors_cpu_oom_and_disabled_retry_are_not_swallowed(self):
        for device, enabled, error in [
            ("cuda", True, RuntimeError("shape mismatch")),
            ("cpu", True, torch.cuda.OutOfMemoryError("out of memory")),
            ("cuda", False, torch.cuda.OutOfMemoryError("out of memory")),
        ]:
            with self.subTest(device=device, enabled=enabled):
                executor = AdaptiveBatchExecutor(8, device, auto_batch=enabled)
                def fail(_start, _end):
                    raise error
                with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.empty_cache") as empty:
                    with self.assertRaises(type(error)) as caught:
                        list(executor.iter_batches(8, fail))
                self.assertIs(caught.exception, error)
                empty.assert_not_called()

    def test_minimum_failure_is_bounded_and_describes_limit(self):
        executor = AdaptiveBatchExecutor(8, "cuda", min_batch_size=2)
        attempts = []
        def fail(start, end):
            attempts.append(end - start)
            raise torch.cuda.OutOfMemoryError("simulated")
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.empty_cache"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(RuntimeError, "min-batch-size=2"):
                list(executor.iter_batches(8, fail))
        self.assertEqual(attempts, [8, 4, 2])

    def test_model_load_failure_has_actionable_message(self):
        with self.assertRaisesRegex(RuntimeError, "Reducing batch size cannot resolve"):
            with model_loading_oom_hint("cuda", "source models"):
                raise torch.cuda.OutOfMemoryError("simulated")

    def test_invalid_limits_fail_before_running_callback(self):
        for size, minimum in [(0, 1), (16, 0), (2, 3), (1.5, 1)]:
            with self.subTest(size=size, minimum=minimum), self.assertRaises(ValueError):
                AdaptiveBatchExecutor(size, "cpu", min_batch_size=minimum)


if __name__ == "__main__":
    unittest.main()
