import torch
import math

from attacks.base import Attack, AttackResult, gradients_per_model, mean_grad, normalize_grad_l1, pgd_update
from src.metrics import pairwise_cosine_agreement


class MIFGSM(Attack):
    name = "mifgsm"

    def __call__(self, models, x, y):
        decay = float(self.kwargs.get("decay", 1.0))
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("decay must be finite and nonnegative")
        x_clean = x.detach()
        x_adv = x_clean.clone()
        momentum = torch.zeros_like(x_adv)
        logs = []
        for step in range(self.steps):
            _, grads, losses = gradients_per_model(models, x_adv, y)
            grad = normalize_grad_l1(mean_grad(grads))
            momentum = decay * momentum + grad
            x_adv = pgd_update(x_adv, x_clean, momentum, self.alpha, self.eps).detach()
            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": float(pairwise_cosine_agreement(grads).detach().cpu()),
                "lambda": "",
            })
        return AttackResult(x_adv, logs)
