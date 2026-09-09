# Patch notes for Tables V-VIII

This patch converts the attack implementation from a single-source-only prototype
into the general K-source framework described in the revised paper.

## Main code changes

- `scripts/run_attack.py`
  - Allows `--surrogates` to contain one or more source models.
  - Keeps the target strictly outside attack generation; targets are evaluated only by `scripts/evaluate.py`.

- `attacks/base.py`
  - Baseline attacks now support multi-source transfer by averaging gradients across all source models.

- `attacks/sinifgsm.py`
  - SI-NI-FGSM now supports multiple source models.

- `attacks/ddc.py`
  - Implements K-source x R-view source-view gradient pool.
  - Implements variance-calibrated precision weighting.
  - Implements robust dynamic fusion.
  - Supports ablation variants used by Table V:
    - `full_model`
    - `without_sv_pool`
    - `without_frequency`
    - `without_token`
    - `without_weighting`
    - `without_fusion`

- `src/models.py`
  - Adds optional model specs:
    - `timm:<model_name>`
    - `robustbench:<model_name>[:dataset][:threat_model]`

## New scripts

- `sh/run_table5_ablation.sh`
  - Runs Table V ablations for CNN-to-ViT, ViT-to-CNN, ViT-to-ViT, and Mixed-to-Mixed.

- `scripts/summarize_table5_ablation.py`
  - Converts Table V eval files into a paper-ready CSV.

- `sh/run_table6_quality.sh`
  - Generates adversarial examples for each method and computes perceptual metrics.

- `scripts/compute_perceptual_quality.py`
  - Computes PSNR, SSIM, and optional LPIPS.
  - Install LPIPS with `pip install lpips` if you want the LPIPS column.

- `sh/run_table7_runtime.sh`
  - Measures runtime per image.

- `scripts/measure_runtime.py`
  - Outputs time/image and relative cost.

- `sh/run_table8_defense.sh`
  - Runs attacks and evaluates against robust/defense targets.
  - Set `DEFENSE_TARGETS` to RobustBench specs if robustbench is installed.

- `scripts/summarize_table8_defense.py`
  - Converts defense eval files into a paper-ready CSV.

## Example usage

```bash
export DATA_DIR=../datasets/imagenet/val
export OUT_DIR=./runs
export MODEL_DIR=./pretrained_models
export SELECTED_CSV=./runs/selected_1000.csv
export BATCH_SIZE=16
export DEVICE=cuda

bash sh/run_table5_ablation.sh
bash sh/run_table6_quality.sh
bash sh/run_table7_runtime.sh
bash sh/run_table8_defense.sh
```

For Table VIII with RobustBench targets:

```bash
pip install robustbench
export DEFENSE_TARGETS="robustbench:Engstrom2019Robustness:imagenet:Linf,robustbench:Salman2020Do_R50:imagenet:Linf,robustbench:Salman2020Do_R18:imagenet:Linf"
export DEFENSE_COLUMNS="Robust CNN,Robust CNN 2,Robust CNN 3"
bash sh/run_table8_defense.sh
```

Replace the RobustBench model names with the models you choose to report.
