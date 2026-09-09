import torch
import math

from attacks.base import Attack, AttackResult, loss_ce, normalize_grad_l1, pgd_update
from src.metrics import pairwise_cosine_agreement


class SINIFGSM(Attack):
    name = "si_ni_fgsm"

    def _scaled_grad_per_model(self, models, x, y, scales):
        if not models:
            raise ValueError("SI-NI-FGSM requires at least one source model")
        grads = []
        losses = []
        for _, model in models.items():
            x_req = x.detach().clone().requires_grad_(True)
            total_loss = 0.0
            grad = torch.zeros_like(x_req)
            for i in range(scales):
                logits = model(x_req / float(2 ** i))
                loss = loss_ce(logits, y)
                grad.add_(torch.autograd.grad(loss, x_req, retain_graph=False, create_graph=False)[0])
                total_loss += float(loss.detach().cpu())
            grads.append(grad.div(float(scales)).detach())
            losses.append(total_loss / float(scales))
        return grads, losses

    def __call__(self, models, x, y):
        decay = float(self.kwargs.get("decay", 1.0))
        scales = int(self.kwargs.get("scales", 5))
        if scales < 1:
            raise ValueError("scales must be positive")
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("decay must be finite and nonnegative")
        x_clean = x.detach()
        x_adv = x_clean.clone()
        momentum = torch.zeros_like(x_adv)
        logs = []
        for step in range(self.steps):
            lookahead = x_adv + decay * self.alpha * momentum
            # The Nesterov point need not be feasible; project only the iterate.
            grads, losses = self._scaled_grad_per_model(models, lookahead, y, scales=scales)
            grad = normalize_grad_l1(torch.stack(grads, dim=0).mean(dim=0))
            momentum = decay * momentum + grad
            x_adv = pgd_update(x_adv, x_clean, momentum, self.alpha, self.eps).detach()
            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": float(pairwise_cosine_agreement(grads).detach().cpu()),
                "lambda": "",
            })
        return AttackResult(x_adv, logs)
