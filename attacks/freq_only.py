from attacks.base import Attack, AttackResult, gradients_per_model, mean_grad, pgd_update
from src.frequency import frequency_filter_fft
from src.metrics import pairwise_cosine_agreement


class FreqOnlyAttack(Attack):
    name = "freq_only"

    def __call__(self, models, x, y):
        mode = self.kwargs.get("freq_mode", "low_mid")
        x_clean = x.detach()
        x_adv = x_clean.clone()
        logs = []
        for step in range(self.steps):
            _, grads, losses = gradients_per_model(models, x_adv, y)
            agreement = float(pairwise_cosine_agreement(grads).detach().cpu())
            grad = frequency_filter_fft(mean_grad(grads), mode=mode, agreement=agreement)
            x_adv = pgd_update(x_adv, x_clean, grad, self.alpha, self.eps).detach()
            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": agreement,
                "lambda": 1.0,
            })
        return AttackResult(x_adv, logs)
