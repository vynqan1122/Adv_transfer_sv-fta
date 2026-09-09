# Central adversarial batches + fixed NUM_BATCHES patch

## What changed

1. All table scripts can now use one shared batch budget:

```bash
export NUM_BATCHES=32
export BATCH_SIZE=4
bash sh/run_tables_1_8.sh
```

`NUM_BATCHES` is passed to `scripts/run_attack.py`, `scripts/evaluate.py`, Table VI quality computation, and Table VII runtime measurement.

2. All adversarial `.pt` batch files are now written only under:

```text
runs/adv_batches/
```

Each run gets a unique subfolder under that root, mirroring the table output path. For example:

```text
runs/table5_ablation/cnn_to_vit/full_model/attack_batches.csv
runs/adv_batches/table5_ablation/cnn_to_vit/full_model/batch_00000.pt
runs/adv_batches/table5_ablation/cnn_to_vit/full_model/batch_00001.pt
```

No table script writes `.pt` files to `runs/table*/.../adv_batches/` anymore.

3. Evaluation reads the manifest file `attack_batches.csv`, so the `.pt` batches may live in the central folder without breaking evaluation or quality metrics.

4. Memory optimizations:
   - `run_attack.py` supports `--num-batches`.
   - `run_attack.py` uses `pin_memory` only when CUDA is available.
   - `run_attack.py` releases tensors and calls `torch.cuda.empty_cache()` after each batch by default.
   - `evaluate.py` supports `--eval-batch-size` to split saved adversarial batches into smaller chunks during target evaluation.
   - `measure_runtime.py` supports `--num-batches`.

## Main variables

```bash
export OUT_DIR=./runs
export ADV_BATCH_ROOT=./runs/adv_batches
export NUM_BATCHES=32
export BATCH_SIZE=4
export EVAL_BATCH_SIZE=4
```

## Force regeneration

If a previous `attack_batches.csv` exists but has a different number of batches, `run_attack_once` reruns automatically. To force all attacks to regenerate:

```bash
FORCE_ATTACKS=1 bash sh/run_tables_1_8.sh
```
