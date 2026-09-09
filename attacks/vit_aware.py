import torch
import torch.nn.functional as F

from attacks.base import Attack, AttackResult, gradients_per_model, mean_grad, pgd_update
from src.metrics import pairwise_cosine_agreement
from src.models import is_vit_like


def patch_saliency_from_grads(
    names,
    grads,
    patch_size=16,
    use_vit_only=True,
    hard_mask=False,
    topk_ratio=0.25,
):
    if not grads or len(names) != len(grads):
        raise ValueError("names and grads must be nonempty and have the same length")
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError("topk_ratio must be in (0, 1]")
    selected = []

    for name, grad in zip(names, grads):
        if (not use_vit_only) or is_vit_like(name):
            selected.append(grad)

    if not selected:
        # CNN-only sources fall back to input-gradient patch saliency.
        selected = grads

    g = torch.stack(selected, dim=0).mean(dim=0)
    sal = g.abs().mean(dim=1, keepdim=True)

    b, _, h, w = sal.shape
    ph = max(1, h // patch_size)
    pw = max(1, w // patch_size)

    pooled = F.adaptive_avg_pool2d(sal, output_size=(ph, pw))

    if hard_mask:
        flat = pooled.flatten(1)
        k = max(1, int(flat.shape[1] * float(topk_ratio)))

        # Select exactly k patches even when values tie at the cutoff.
        indices = torch.topk(flat, k=k, dim=1).indices
        pooled = torch.zeros_like(flat).scatter_(1, indices, 1.0).view_as(pooled)

    dense = F.interpolate(pooled, size=(h, w), mode="nearest")

    if not hard_mask:
        dense = dense / dense.mean(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        dense = dense.clamp(0.0, 5.0)

    return dense


class ViTAwareAttack(Attack):
    name = "vit_aware"

    def __call__(self, models, x, y):
        patch_size = int(self.kwargs.get("patch_size", 16))
        topk_ratio = float(self.kwargs.get("topk_ratio", 0.25))

        x_clean = x.detach()
        x_adv = x_clean.clone()
        logs = []

        for step in range(self.steps):
            names, grads, losses = gradients_per_model(models, x_adv, y)

            saliency = patch_saliency_from_grads(
                names,
                grads,
                patch_size=patch_size,
                use_vit_only=True,
                hard_mask=True,
                topk_ratio=topk_ratio,
            )

            grad = saliency * mean_grad(grads)

            x_adv = pgd_update(x_adv, x_clean, grad, self.alpha, self.eps).detach()

            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": float(pairwise_cosine_agreement(grads).detach().cpu()),
                "lambda": 0.0,
            })

        return AttackResult(x_adv, logs)
