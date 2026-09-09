# Single-source transfer black-box patch notes

This patch changes the implementation from the original multi-surrogate paper-style setting to a single-source -> black-box-target transfer setting.

## Threat model

- `scripts/run_attack.py` now requires exactly one `--surrogates` model.
- Target models are never loaded or queried during adversarial example generation.
- Target models are used only in `scripts/evaluate.py`.
- Baseline gradient attacks also reject multiple source models through `attacks.base.gradients_per_model`.

## DDCAttack changes

`attacks/ddc.py` was rewritten as a single-source variance-calibrated dynamic dual-domain transfer attack.

Main changes:

1. Cross-model consistency was replaced by cross-view consistency.
   - Gradients are sampled from one source model under multiple differentiable input views.
   - Mean and variance are computed across virtual source views.

2. Exponential consistency weighting was replaced by variance-calibrated precision weighting:

   `C(p) = 1 / (sqrt(V(p)) + eps_c)`

   followed by mean normalization and clipping.

3. Dynamic fusion was changed from a heuristic sigmoid to source-only robust optimization:

   `lambda* = argmax_lambda cos(d(lambda), mean_grad) - rho * uncertainty(d(lambda))`

   where `d(lambda) = lambda * frequency_branch + (1-lambda) * patch_branch`.

4. Token branch was renamed logically to token/patch branch.
   - If the source is ViT-like, it behaves as a token/patch saliency proxy.
   - If the source is CNN, it becomes a patch-level gradient saliency proxy.

## Important commands

Example source-to-target workflow:

```bash
python scripts/run_attack.py \
  --data-dir ../datasets/imagenet/val \
  --selected-csv runs/selected_1000.csv \
  --models-dir ./pretrained_models \
  --surrogates resnet50 \
  --attack ours \
  --variant full_model \
  --fusion robust \
  --out-dir runs/resnet50_to_vit/ours

python scripts/evaluate.py \
  --attack-dir runs/resnet50_to_vit/ours \
  --data-dir ../datasets/imagenet/val \
  --models-dir ./pretrained_models \
  --targets vit_large_patch16_224,deit_base_patch16_224,swin_tiny_patch4_window7_224
```

## Validation performed

- Python syntax check: `python3 -m py_compile attacks/*.py scripts/*.py src/*.py`
- Smoke test with a tiny dummy model for all attacks:
  - ifgsm
  - mifgsm
  - difgsm
  - tifgsm
  - si_ni_fgsm
  - freq_only
  - vit_aware
  - ours
- Multi-source guard tested: passing two source models raises `ValueError`.
