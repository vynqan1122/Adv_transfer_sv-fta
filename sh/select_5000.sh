#!/usr/bin/env bash
set -euo pipefail
export NUM_IMAGES="${NUM_IMAGES:-5000}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PY" scripts/select_imagenet_subset.py \
  --data-dir "$DATA_DIR" \
  "${maybe_val_args[@]}" \
  "${maybe_label_args[@]}" \
  --models-dir "$MODEL_DIR" \
  --surrogates "$MIXED_SURROGATES" \
  --num-images "$NUM_IMAGES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out-csv "$SELECTED_CSV"
