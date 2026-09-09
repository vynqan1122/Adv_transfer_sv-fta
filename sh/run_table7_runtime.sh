#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_selected

ROOT="$OUT_DIR/table7_runtime"
mkdir -p "$ROOT"

"$PY" scripts/measure_runtime.py \
  --data-dir "$DATA_DIR" \
  --selected-csv "$SELECTED_CSV" \
  --models-dir "$MODEL_DIR" \
  --surrogates "$MIXED_SURROGATES" \
  --methods "difgsm,ifgsm,mifgsm,si_ni_fgsm,tifgsm,vit_aware,freq_only,ours" \
  --out "$ROOT/table7_runtime.csv" \
  --max-images "${RUNTIME_IMAGES:-128}" \
  "${num_batch_args[@]}" \
  "${svfca_amp_args[@]}" \
  "${svfca_core_args[@]}" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo "[done] $ROOT/table7_runtime.csv"
