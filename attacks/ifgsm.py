from attacks.base import Attack, AttackResult, gradients_per_model, mean_grad, pgd_update
from src.metrics import pairwise_cosine_agreement


class IFGSM(Attack):
    name = "ifgsm"

    def __call__(self, models, x, y):
        x_clean = x.detach()
        x_adv = x_clean.clone()
        logs = []
        for step in range(self.steps):
            _, grads, losses = gradients_per_model(models, x_adv, y)
            grad = mean_grad(grads)
            x_adv = pgd_update(x_adv, x_clean, grad, self.alpha, self.eps).detach()
            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": float(pairwise_cosine_agreement(grads).detach().cpu()),
                "lambda": "",
            })
        return AttackResult(x_adv, logs)
