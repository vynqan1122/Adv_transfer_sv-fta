#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# IMPORTANT: this must be the parent of imagenet/Linf, not the Linf folder itself.
export ROBUSTBENCH_MODEL_DIR="${ROBUSTBENCH_MODEL_DIR:-$PROJECT_ROOT/models}"
"$PY" scripts/check_defense_models.py --model-dir "$ROBUSTBENCH_MODEL_DIR" --targets "$DEFENSE_TARGETS"
ensure_selected

ROOT="$OUT_DIR/table8_defense"
mkdir -p "$ROOT"
IFS=',' read -r -a METHODS <<< "${DEFENSE_METHODS:-difgsm,ifgsm,mifgsm,si_ni_fgsm,tifgsm,vit_aware,freq_only,ours}"

for method_raw in "${METHODS[@]}"; do
  method="$(echo "$method_raw" | xargs)"
  [[ -z "$method" ]] && continue
  dir="$ROOT/$method"
  mkdir -p "$dir"
  echo "[Table VIII] method=$method"
  echo "[Table VIII] sources=$MIXED_SURROGATES"
  echo "[Table VIII] targets=$DEFENSE_TARGETS"
  echo "[Table VIII] robustbench_model_dir=$ROBUSTBENCH_MODEL_DIR"

  run_attack_once "$method" "$MIXED_SURROGATES" "$dir" "full_model" "robust"

  "$PY" scripts/evaluate.py \
    --attack-dir "$dir" \
    --data-dir "$DATA_DIR" \
    --models-dir "$MODEL_DIR" \
    --robustbench-model-dir "$ROBUSTBENCH_MODEL_DIR" \
    --targets "$DEFENSE_TARGETS" \
    "${num_batch_args[@]}" \
    "${eval_batch_args[@]}" \
    "${delete_adv_args[@]}" \
    --device "$DEVICE"
done

"$PY" scripts/summarize_table8_defense.py \
  --root "$ROOT" \
  --out "$ROOT/table8_defense.csv" \
  --columns "$DEFENSE_COLUMNS" \
  --targets "$DEFENSE_TARGETS" \
  --methods "$(IFS=','; echo "${METHODS[*]}")"

echo "[done] $ROOT/table8_defense.csv"
