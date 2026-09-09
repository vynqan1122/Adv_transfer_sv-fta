# 12GB VRAM + low-disk patch

This patch is designed for RTX 3060 12GB-class GPUs.

## Main changes

1. **Streaming SV-FCA spectral statistics**
   - Does not retain the full KxR gradient pool.
   - Does not retain 3xKxR low/mid/high band tensors.
   - Each source/view gradient is FFT-processed and accumulated immediately, then released.
   - Band consensus uses a one-pass resultant-length ratio:
     `||sum g_b||_2 / sum ||g_b||_2`.

2. **Safe default batch profile**
   - `BATCH_SIZE=4`
   - `EVAL_BATCH_SIZE=4`
   - `NUM_WORKERS=2`
   - `EMPTY_CACHE_EVERY=8`
   - `SVFCA_AMP=0` by default for FP32 reproducibility.
   - Set `SVFCA_AMP=1` if a setting still OOMs.

3. **Temporary batch storage is halved**
   - Default `ADV_STORAGE_MODE=delta_fp16`.
   - Saves `delta = x_adv - x` in FP16 instead of `x_adv` in FP32.
   - Evaluation reconstructs `x_adv = clamp(x_clean + delta.float(), 0, 1)`.
   - Set `ADV_STORAGE_MODE=adv_fp32` for exact old storage behavior.

4. **Delete batches after use**
   - Default `DELETE_ADV_AFTER_USE=1`.
   - Evaluation deletes `.pt` files only after every requested target has consumed them and CSV/JSON metrics were saved.
   - The stale `attack_batches.csv` manifest is also removed.
   - Table VI now runs one method at a time: attack -> quality metrics -> delete batches, so disk does not accumulate all methods.
   - Figure temporary batches are deleted after figures are rendered.

5. **VRAM diagnostics**
   - `attack_batches.csv` records `cuda_peak_allocated_mb` and `cuda_peak_reserved_mb` before cleanup.

## Recommended commands

### Table V, 1000 images

```bash
export DEVICE=cuda
export BATCH_SIZE=4
export EVAL_BATCH_SIZE=4
export NUM_WORKERS=2
export DELETE_ADV_AFTER_USE=1
export ADV_STORAGE_MODE=delta_fp16
export TABLE5_NUM_IMAGES=1000
bash sh/run_table5_sv_fca.sh
```

### Table V, 5000 images

```bash
export DEVICE=cuda
export BATCH_SIZE=4
export EVAL_BATCH_SIZE=4
export NUM_WORKERS=2
export DELETE_ADV_AFTER_USE=1
export ADV_STORAGE_MODE=delta_fp16
bash sh/run_table5_sv_fca_5000.sh
```

### If OOM still occurs

```bash
export BATCH_SIZE=2
export EVAL_BATCH_SIZE=2
export SVFCA_AMP=1
export SVFCA_AMP_DTYPE=fp16
bash sh/run_table5_sv_fca_5000.sh
```

### Fixed number of batches across Table I-VIII

```bash
export DEVICE=cuda
export BATCH_SIZE=4
export EVAL_BATCH_SIZE=4
export NUM_BATCHES=125
export NUM_WORKERS=2
export DELETE_ADV_AFTER_USE=1
export ADV_STORAGE_MODE=delta_fp16
bash sh/run_tables_1_8.sh
```

`NUM_BATCHES=125` with batch size 4 means at most 500 images per attack run. Choose the value required by the experiment.

## Optional diagnostics

Spectral energy logging costs an extra FFT per attack step and is disabled by default. Enable only when making analysis figures:

```bash
export SVFCA_LOG_SPECTRAL_ENERGY=1
```
