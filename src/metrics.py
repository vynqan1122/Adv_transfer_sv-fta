import torch


def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item(), pred


def attack_success_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    success = (pred != y).float()
    return success.mean().item(), pred


def pairwise_cosine_agreement(grads):
    if not grads:
        raise ValueError("At least one gradient is required")
    if len(grads) < 2:
        return torch.tensor(1.0, device=grads[0].device)
    vals = []
    for i in range(len(grads)):
        gi = grads[i].flatten(1)
        for j in range(i + 1, len(grads)):
            gj = grads[j].flatten(1)
            cos = torch.nn.functional.cosine_similarity(gi, gj, dim=1, eps=1e-12)
            vals.append(cos.mean())
    return torch.stack(vals).mean()
