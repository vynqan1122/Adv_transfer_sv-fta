import random
import math

import torch
import torch.nn.functional as F

from src.utils import clamp_linf


def assert_single_source(models, context="attack"):
    if len(models) != 1:
        raise ValueError(
            f"{context} is configured for single-source transfer black-box evaluation. "
            "Pass exactly one source model to scripts/run_attack.py via --surrogates. "
            "Target models must be evaluated separately with scripts/evaluate.py."
        )


def loss_ce(logits, y):
    return F.cross_entropy(logits, y)


def gradients_per_model(models, x, y, transform_fn=None, require_single_source=False):
    if not models:
        raise ValueError("Gradient-based attacks require at least one source model")
    if require_single_source:
        assert_single_source(models, context="Gradient-based attack")
    grads = []
    losses = []
    names = []
    for name, model in models.items():
        x_req = x.detach().clone().requires_grad_(True)
        x_in = transform_fn(x_req) if transform_fn is not None else x_req
        logits = model(x_in)
        loss = loss_ce(logits, y)
        grad = torch.autograd.grad(loss, x_req, retain_graph=False, create_graph=False)[0]
        grads.append(grad.detach())
        losses.append(float(loss.detach().cpu()))
        names.append(name)
    return names, grads, losses


def mean_grad(grads):
    return torch.stack(grads, dim=0).mean(dim=0)


def normalize_grad_l1(grad):
    # Accumulate in fp32 for half inputs: 1e-12 underflows to zero in fp16.
    if grad.dtype in (torch.float16, torch.bfloat16):
        grad = grad.float()
    denom = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return grad / denom


def gaussian_kernel2d(kernel_size=15, sigma=3.0, channels=3, device="cpu", dtype=torch.float32):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def smooth_grad_ti(grad, kernel_size=15, sigma=3.0):
    kernel = gaussian_kernel2d(kernel_size, sigma, grad.shape[1], grad.device, grad.dtype)
    pad = kernel_size // 2
    return F.conv2d(grad, kernel, padding=pad, groups=grad.shape[1])


def input_diversity(x, prob=0.5, image_size=224, resize_size=256):
    if not 0.0 <= prob <= 1.0:
        raise ValueError("diversity probability must be in [0, 1]")
    if image_size < 1 or resize_size < image_size:
        raise ValueError("resize_size must be >= image_size > 0")
    if random.random() >= prob:
        return x
    rnd = random.randint(image_size, resize_size)
    rescaled = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pad_total = resize_size - rnd
    pad_top = random.randint(0, pad_total)
    pad_bottom = pad_total - pad_top
    pad_left = random.randint(0, pad_total)
    pad_right = pad_total - pad_left
    padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0.0)
    return F.interpolate(padded, size=(image_size, image_size), mode="bilinear", align_corners=False)


def pgd_update(x_adv, x_clean, update_grad, alpha, eps):
    x_adv = x_adv + alpha * update_grad.sign()
    return clamp_linf(x_adv, x_clean, eps)


class AttackResult(object):
    def __init__(self, adv, logs):
        self.adv = adv
        self.logs = logs


class Attack(object):
    name = "base"

    def __init__(self, eps=16/255.0, alpha=1.6/255.0, steps=10, **kwargs):
        self.eps = float(eps)
        self.alpha = float(alpha)
        if not math.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be finite and nonnegative")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.steps = int(steps)
        if self.steps < 1 or self.steps != steps:
            raise ValueError("steps must be a positive integer")
        self.kwargs = kwargs

    def __call__(self, models, x, y):
        raise NotImplementedError
