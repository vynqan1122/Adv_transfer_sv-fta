# Patch: Table V no-weighting comparison on 5000 images

## What changed

This patch adds a dedicated experiment script:

```bash
bash sh/run_table5_no_weighting_5000.sh
```

It compares:

1. `Original Full Model (With Weighting)`
   - `--variant full_model`
   - variance-calibrated precision map is enabled:
     `C_t = 1 / (sqrt(V_t) + eps_c)`

2. `Full Model (No Weighting)`
   - `--variant without_weighting`
   - true no-weighting ablation:
     `C_t = 1`

The script uses 5000 selected clean-correct ImageNet images by default:

```bash
NUM_IMAGES=5000
SELECTED_CSV=runs/selected_5000.csv
```

## Outputs

```text
runs/table5_weighting_compare_5000/table5_weighting_compare_5000.csv
runs/table5_weighting_compare_5000/table5_weighting_compare_configs.csv
```

## Important detail

`without_weighting` disables the precision weighting multiplication only. Robust fusion may still use `V_t` in the uncertainty penalty. This is intentional because the original Table V ablation item is “Without Weighting”, not “Without Uncertainty”.

To remove variance influence from robust fusion as well, run:

```bash
TABLE5_NO_WEIGHTING_RHO=0 bash sh/run_table5_no_weighting_5000.sh
```

## Recommended run command for RTX 3060 12GB

```bash
export DEVICE=cuda
export BATCH_SIZE=4
bash sh/run_table5_no_weighting_5000.sh
```

For Mixed-to-Mixed, use `BATCH_SIZE=2` if CUDA OOM occurs.
