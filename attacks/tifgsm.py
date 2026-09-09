from attacks.base import Attack, AttackResult, gradients_per_model, mean_grad, pgd_update, smooth_grad_ti
from src.metrics import pairwise_cosine_agreement


class TIFGSM(Attack):
    name = "tifgsm"

    def __call__(self, models, x, y):
        kernel_size = int(self.kwargs.get("kernel_size", 15))
        sigma = float(self.kwargs.get("sigma", 3.0))
        x_clean = x.detach()
        x_adv = x_clean.clone()
        logs = []
        for step in range(self.steps):
            _, grads, losses = gradients_per_model(models, x_adv, y)
            grad = smooth_grad_ti(mean_grad(grads), kernel_size=kernel_size, sigma=sigma)
            x_adv = pgd_update(x_adv, x_clean, grad, self.alpha, self.eps).detach()
            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": float(pairwise_cosine_agreement(grads).detach().cpu()),
                "lambda": "",
            })
        return AttackResult(x_adv, logs)
