# Patch Notes: run scripts for Table I-VIII

Added/updated:

- `sh/run_tables_1_8.sh`
  - Runs Table I-IV through `sh/run_table1.sh`.
  - Runs Table V-VIII through `sh/run_tables_5_8.sh`.
  - Supports skip flags: `SKIP_TABLES1_4`, `SKIP_TABLE5`, `SKIP_TABLE6`, `SKIP_TABLE7`, `SKIP_TABLE8`.

- `sh/run_tables_5_8.sh`
  - Runs Table V, VI, VII, VIII in order.
  - Supports skip flags for each table.
  - Works even when called from outside the project root.

- `scripts/summarize_tables1_4_transfer.py`
  - Converts the single-source transfer outputs from `sh/run_table1.sh` into paper-style CSVs:
    - `table1_cnn_to_cnn.csv`
    - `table2_cnn_to_vit.csv`
    - `table3_vit_to_cnn.csv`
    - `table4_vit_to_vit.csv`
    - `tables1_4_long.csv`

- `sh/run_table1.sh`
  - Now automatically calls `scripts/summarize_tables1_4_transfer.py` after evaluation.

- `sh/run_all_tables.sh`
  - Kept as a backward-compatible alias for `sh/run_tables_1_8.sh`.

Main usage:

```bash
bash sh/run_tables_1_8.sh
```

or:

```bash
bash sh/run_tables_5_8.sh
```

Example environment:

```bash
export DATA_DIR=../datasets/imagenet/val
export LABELS_CSV=../datasets/imagenet/imagenet_val_labels.csv
export MODEL_DIR=./pretrained_models
export OUT_DIR=./runs
export SELECTED_CSV=./runs/selected_1000.csv
export BATCH_SIZE=16
export DEVICE=cuda
```
