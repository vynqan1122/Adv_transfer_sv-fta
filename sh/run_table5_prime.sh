#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Final Table V: SV-FCA ablation.
# Columns: CNN->ViT | ViT->CNN | ViT->ViT | Mixed->Mixed
# Rows: Full Model and five component-removal ablations.

if [[ -n "${TABLE5P_NUM_IMAGES:-}" ]]; then
  NUM_IMAGES="$TABLE5P_NUM_IMAGES"
  require_positive_integer TABLE5P_NUM_IMAGES "$TABLE5P_NUM_IMAGES"
  SELECTED_CSV="$OUT_DIR/selected_${NUM_IMAGES}.csv"
fi
SELECTED_CSV="${TABLE5P_SELECTED_CSV:-$SELECTED_CSV}"

ROOT="${TABLE5P_ROOT:-$OUT_DIR/table5_prime${TABLE5P_NUM_IMAGES:+_${TABLE5P_NUM_IMAGES}}}"
mkdir -p "$ROOT"
ensure_selected

IFS=',' read -r -a SETTINGS <<< "${TABLE5P_SETTINGS:-cnn_to_vit,vit_to_cnn,vit_to_vit,mixed_mixed}"
IFS=',' read -r -a VARIANTS <<< "${TABLE5P_VARIANTS:-full_model,without_sv_pool,without_frequency_coordination,without_band_consensus,without_low_mid_prior,without_spectral_memory}"

sources_for_setting() {
  case "$1" in
    cnn_to_vit) echo "$CNN_SURROGATES" ;;
    vit_to_cnn) echo "$VIT_SURROGATES" ;;
    vit_to_vit) echo "$VIT_SURROGATES" ;;
    mixed_mixed) echo "$MIXED_SURROGATES" ;;
    *) echo "Unknown setting: $1" >&2; exit 1 ;;
  esac
}

targets_for_setting() {
  case "$1" in
    cnn_to_vit) echo "$VIT_TARGETS" ;;
    vit_to_cnn) echo "$CNN_TARGETS" ;;
    vit_to_vit) echo "$VIT_TARGETS" ;;
    mixed_mixed) echo "$MIXED_TARGETS" ;;
    *) echo "Unknown setting: $1" >&2; exit 1 ;;
  esac
}

for setting_raw in "${SETTINGS[@]}"; do
  setting="$(echo "$setting_raw" | xargs)"
  [[ -z "$setting" ]] && continue
  sources="$(sources_for_setting "$setting")"
  targets="$(targets_for_setting "$setting")"

  for variant_raw in "${VARIANTS[@]}"; do
    variant="$(echo "$variant_raw" | xargs)"
    [[ -z "$variant" ]] && continue
    dir="$ROOT/$setting/$variant"
    mkdir -p "$dir"

    signature="$(attack_signature sv_fca "$sources" "$variant" robust)"
    if [[ "$FORCE_ATTACKS" != "1" && -f "$dir/eval_results.csv" && -f "$dir/.shell_eval_targets" ]] && \
      attack_signature_matches "$dir" "$signature" && \
      [[ "$(cat "$dir/.shell_eval_targets")" == "$targets" ]]; then
      echo "[reuse][Table V] $setting/$variant"
      continue
    fi

    echo "[Table V / SV-FCA] images=$NUM_IMAGES setting=$setting variant=$variant"
    run_attack_once sv_fca "$sources" "$dir" "$variant" robust

    run_eval "$dir" "$targets"
    printf '%s\n' "$targets" > "$dir/.shell_eval_targets"
  done
done

settings_csv="$(IFS=','; echo "${SETTINGS[*]}")"
variants_csv="$(IFS=','; echo "${VARIANTS[*]}")"
"$PY" scripts/summarize_table5_prime.py \
  --root "$ROOT" \
  --out "$ROOT/table5_prime.csv" \
  --metric asr \
  --settings "$settings_csv" \
  --variants "$variants_csv"

echo "[done] $ROOT/table5_prime.csv"
