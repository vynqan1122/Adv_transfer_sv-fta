#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_selected

TABLE_DIR="$OUT_DIR/table1_single_surrogate_transfer"

# Baseline and proposed attacks.
ATTACKS=(ifgsm mifgsm difgsm tifgsm si_ni_fgsm freq_only vit_aware ours)

run_one_transfer() {
  local setting="$1"
  local surrogate="$2"
  local targets="$3"

  for attack in "${ATTACKS[@]}"; do
    local attack_dir="$TABLE_DIR/$setting/$surrogate/$attack"

    echo "============================================================"
    echo "[Transfer Black-box / Single Surrogate]"
    echo "setting   = $setting"
    echo "surrogate = $surrogate"
    echo "targets   = $targets"
    echo "attack    = $attack"
    echo "============================================================"

    run_attack_once "$attack" "$surrogate" "$attack_dir"

    run_eval "$attack_dir" "$targets"
  done
}

# Single CNN surrogate -> CNN targets
run_one_transfer "resnet50_to_cnn" "resnet50" "$CNN_TARGETS"
run_one_transfer "densenet121_to_cnn" "densenet121" "$CNN_TARGETS"

# Single CNN surrogate -> ViT targets
run_one_transfer "resnet50_to_vit" "resnet50" "$VIT_TARGETS"
run_one_transfer "densenet121_to_vit" "densenet121" "$VIT_TARGETS"

# Single ViT surrogate -> ViT targets
run_one_transfer "vit_base_to_vit" "vit_base_patch16_224" "$VIT_TARGETS"
run_one_transfer "deit_small_to_vit" "deit_small_patch16_224" "$VIT_TARGETS"

# Single ViT surrogate -> CNN targets
run_one_transfer "vit_base_to_cnn" "vit_base_patch16_224" "$CNN_TARGETS"
run_one_transfer "deit_small_to_cnn" "deit_small_patch16_224" "$CNN_TARGETS"

"$PY" scripts/aggregate_results.py \
  --root "$TABLE_DIR" \
  --out "$TABLE_DIR/table1_single_surrogate_summary.csv"

echo "[DONE] Saved results to $TABLE_DIR"
