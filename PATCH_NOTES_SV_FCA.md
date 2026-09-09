# SV-FCA patch

The previous token/precision/fusion components were removed from the proposed full model because the ablation did not show a meaningful contribution. The new method is frequency-centric.

## Full model

1. Build a K-source x R-view normalized gradient pool.
2. Decompose every pool gradient into low, mid and high FFT bands.
3. For every band, compute the mean band gradient and source-view cosine consensus.
4. Convert consensus + low/mid transfer prior into per-image band weights with a softmax.
5. Stabilize the band weights across attack iterations with spectral EMA memory.
6. Reconstruct the spatial attack direction from the weighted band means.
7. Apply ordinary gradient momentum and projected L_inf update.

No target gradients, logits, labels, confidence or queries are used during generation.

## Table V variants

- `full_model`: complete SV-FCA.
- `without_sv_pool`: identity view only (R=1).
- `without_frequency_coordination`: uses source-view mean gradient directly.
- `without_band_consensus`: fixed frequency weights from the transfer prior.
- `without_low_mid_prior`: consensus only, no low/mid preference.
- `without_spectral_memory`: instantaneous band weights, no EMA stabilization.

## Run

```bash
BATCH_SIZE=4 NUM_BATCHES=16 DEVICE=cuda bash sh/run_table5_sv_fca.sh
```

5000 images:

```bash
BATCH_SIZE=4 DEVICE=cuda bash sh/run_table5_sv_fca_5000.sh
```

All `.pt` adversarial batches are written only under `runs/adv_batches/...` through the central batch path defined in `sh/common.sh`.
