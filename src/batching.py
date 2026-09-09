"""Bounded CUDA OOM retries without changing a logical batch's sample budget.

Callbacks own their temporary GPU tensors and return CPU tensors/plain values.
Only a failed chunk is retried; successful chunks are never run twice. Restoring
RNG state removes draws from failed attempts, but changing batch partitions can
still change stochastic attacks and batch-dependent reductions.
"""
import gc
import random
import warnings
from contextlib import contextmanager

import numpy as np
import torch


def is_cuda_oom(error, device):
    """Distinguish CUDA allocation failures from CPU OOM and unrelated errors."""
    if torch.device(device).type != "cuda":
        return False
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "cuda out of memory" in str(error).lower()
    )


def _capture_rng_state(device):
    cuda = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda = (device, torch.cuda.get_rng_state(device))
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda


def _restore_rng_state(state):
    python, numpy, cpu, cuda = state
    random.setstate(python)
    np.random.set_state(numpy)
    torch.set_rng_state(cpu)
    if cuda is not None:
        device, cuda_state = cuda
        torch.cuda.set_rng_state(cuda_state, device)


@contextmanager
def model_loading_oom_hint(device, context="model loading"):
    """Explain why shrinking input chunks cannot resolve model-weight OOM."""
    try:
        yield
    except RuntimeError as error:
        if not is_cuda_oom(error, device):
            raise
        raise RuntimeError(
            f"{context}: CUDA ran out of memory while loading model weights. "
            "Reducing batch size cannot resolve this allocation; free GPU memory "
            "or use a GPU with enough memory for the selected models."
        ) from error


class AdaptiveBatchExecutor:
    def __init__(self, batch_size, device, auto_batch=True, min_batch_size=1, context="batch"):
        if batch_size < 1 or int(batch_size) != batch_size:
            raise ValueError("batch_size must be a positive integer")
        if min_batch_size < 1 or int(min_batch_size) != min_batch_size or min_batch_size > batch_size:
            raise ValueError("min_batch_size must be a positive integer <= batch_size")
        self.requested_batch_size = int(batch_size)
        self.batch_size = int(batch_size)
        self.min_batch_size = int(min_batch_size)
        self.device = torch.device(device)
        self.auto_batch = bool(auto_batch)
        self.context = context
        self.oom_retries = 0
        self.successful_batches = 0
        self._successful_sizes = set()

    @property
    def metadata(self):
        sizes = sorted(self._successful_sizes)
        return {
            "requested_batch_size": self.requested_batch_size,
            "effective_batch_size": self.batch_size,
            "min_batch_size": self.min_batch_size,
            "auto_batch": self.auto_batch,
            "min_successful_batch_size": min(sizes) if sizes else None,
            "max_successful_batch_size": max(sizes) if sizes else None,
            "successful_batch_sizes": sizes,
            "oom_retries": self.oom_retries,
            "successful_batches": self.successful_batches,
            "rng_restored_on_retry": True,
        }

    def iter_batches(self, total, process):
        """Yield (start, end, CPU result) with ranges relative to this logical batch."""
        if total < 0 or int(total) != total:
            raise ValueError("total must be a nonnegative integer")
        start = 0
        while start < total:
            end = min(start + self.batch_size, total)
            size = end - start
            can_retry = self.auto_batch and self.device.type == "cuda"
            rng_state = _capture_rng_state(self.device) if can_retry else None
            try:
                result = process(start, end)
            except RuntimeError as error:
                if not can_retry or not is_cuda_oom(error, self.device):
                    raise
                if size <= self.min_batch_size:
                    raise RuntimeError(
                        f"{self.context}: CUDA out of memory at batch size {size}; "
                        f"cannot reduce below --min-batch-size={self.min_batch_size}. "
                        "Free GPU memory, enable AMP for supported attacks, or use a larger GPU."
                    ) from error
                reduced = max(self.min_batch_size, size // 2)
                # Drop callback traceback frames before collection/cache release.
                # They otherwise retain the failed forward/backward graph.
                error.__traceback__ = None
            else:
                self.successful_batches += 1
                self._successful_sizes.add(size)
                yield start, end, result
                del result
                start = end
                continue

            gc.collect()
            torch.cuda.empty_cache()
            _restore_rng_state(rng_state)
            self.oom_retries += 1
            self.batch_size = reduced
            warnings.warn(
                f"{self.context}: CUDA OOM for samples [{start}:{end}); reducing batch size "
                f"{size} -> {reduced} and retrying only this chunk (sample budget unchanged).",
                RuntimeWarning,
                stacklevel=2,
            )
