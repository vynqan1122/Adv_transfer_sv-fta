#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_selected

TABLE_DIR="$OUT_DIR/table1_single_source_transfer"
ATTACKS=(ifgsm mifgsm difgsm tifgsm si_ni_fgsm freq_only vit_aware ours)

run_one_transfer() {
  local setting="$1"
  local source_model="$2"
  local targets="$3"

  for attack in "${ATTACKS[@]}"; do
    local attack_dir="$TABLE_DIR/$setting/$source_model/$attack"
    echo "[Table 1 / Single-source transfer] setting=$setting source=$source_model targets=$targets attack=$attack"

    run_attack_once "$attack" "$source_model" "$attack_dir"

    run_eval "$attack_dir" "$targets"
  done
}

run_one_transfer "cnn_to_cnn" "resnet50" "$CNN_TARGETS"
run_one_transfer "cnn_to_cnn" "densenet121" "$CNN_TARGETS"
run_one_transfer "cnn_to_vit" "resnet50" "$VIT_TARGETS"
run_one_transfer "cnn_to_vit" "densenet121" "$VIT_TARGETS"
run_one_transfer "vit_to_vit" "vit_base_patch16_224" "$VIT_TARGETS"
run_one_transfer "vit_to_vit" "deit_small_patch16_224" "$VIT_TARGETS"
run_one_transfer "vit_to_cnn" "vit_base_patch16_224" "$CNN_TARGETS"
run_one_transfer "vit_to_cnn" "deit_small_patch16_224" "$CNN_TARGETS"

"$PY" scripts/aggregate_results.py --root "$TABLE_DIR" --out "$TABLE_DIR/table1_single_source_summary.csv"

# Also create paper-style Table I-IV CSV files from the same single-source runs.
"$PY" scripts/summarize_tables1_4_transfer.py \
  --root "$TABLE_DIR" \
  --out-dir "$TABLE_DIR" \
  --metric asr

echo "[done] Table I-IV CSVs saved under $TABLE_DIR"
