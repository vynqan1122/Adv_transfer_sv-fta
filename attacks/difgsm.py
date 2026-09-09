from attacks.base import Attack, AttackResult, gradients_per_model, input_diversity, mean_grad, pgd_update
from src.metrics import pairwise_cosine_agreement


class DIFGSM(Attack):
    name = "difgsm"

    def __call__(self, models, x, y):
        prob = float(self.kwargs.get("diversity_prob", 0.7))
        image_size = int(self.kwargs.get("image_size", 224))
        resize_size = int(self.kwargs.get("resize_size", 256))
        x_clean = x.detach()
        x_adv = x_clean.clone()
        logs = []
        for step in range(self.steps):
            transform = lambda z: input_diversity(z, prob=prob, image_size=image_size, resize_size=resize_size)
            _, grads, losses = gradients_per_model(models, x_adv, y, transform_fn=transform)
            grad = mean_grad(grads)
            x_adv = pgd_update(x_adv, x_clean, grad, self.alpha, self.eps).detach()
            logs.append({
                "step": step,
                "loss_mean": sum(losses) / len(losses),
                "agreement": float(pairwise_cosine_agreement(grads).detach().cpu()),
                "lambda": "",
            })
        return AttackResult(x_adv, logs)
