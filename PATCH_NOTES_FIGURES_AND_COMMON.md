# Patch notes: figures + common.sh

## Added figure generation

New scripts:

- `sh/run_figures.sh`
- `sh/run_all_outputs.sh`
- `scripts/visualize_attack_triplet.py`
- `scripts/visualize_frequency_spectrum.py`
- `scripts/visualize_attention_map.py`

The figure script generates:

1. `figure1_original_perturbation_adversarial.png`
   - Original image / rescaled perturbation / adversarial image.
2. `figure2_frequency_spectrum.png`
   - Original FFT spectrum plus perturbation FFT spectra for configured attacks.
   - Also exports `figure2_radial_frequency_profile.csv`.
3. `figure3_attention_map.png`
   - ViT/DeiT attention rollout when supported.
   - Fallback to source-only gradient/patch saliency for CNN or unsupported models.

## Updated common.sh

`sh/common.sh` now centralizes configuration for all tables and figures:

- Dataset/model paths
- Source/target model groups
- Table VIII defense targets
- Figure options
- Shared helpers: `ensure_selected`, `attack_label`, `run_attack_once`

Table VIII defaults now use exactly three defense target columns:

- `resnet152`
- `vit_large_patch16_224`
- `swin_tiny_patch4_window7_224`

You can override with RobustBench specs using `DEFENSE_TARGETS` and `DEFENSE_COLUMNS`.

## Running

Run tables only:

```bash
bash sh/run_tables_1_8.sh
```

Run figures only:

```bash
bash sh/run_figures.sh
```

Run tables + figures:

```bash
bash sh/run_all_outputs.sh
```

Or run tables and figures in one call:

```bash
RUN_FIGURES=1 bash sh/run_tables_1_8.sh
```
