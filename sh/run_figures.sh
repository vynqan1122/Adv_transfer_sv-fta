#!/usr/bin/env bash
set -euo pipefail
# Use exact FP32 temporary storage for publication figures by default, then
# delete it after figures are written. FIG_ADV_STORAGE_MODE can override this.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ADV_STORAGE_MODE="$FIG_ADV_STORAGE_MODE"
memory_attack_args=(--empty-cache-every "$EMPTY_CACHE_EVERY" --storage-mode "$ADV_STORAGE_MODE")
ensure_selected

mkdir -p "$FIG_ROOT"
FIG_ATTACK_ROOT="$FIG_ROOT/attack_outputs"
mkdir -p "$FIG_ATTACK_ROOT"

IFS=',' read -r -a FIG_METHOD_ARRAY <<< "$FIG_METHODS"

attack_dir_args=()
for method_raw in "${FIG_METHOD_ARRAY[@]}"; do
  method="$(echo "$method_raw" | xargs)"
  [[ -z "$method" ]] && continue
  dir="$FIG_ATTACK_ROOT/$method"
  run_attack_once "$method" "$FIG_SURROGATES" "$dir" "full_model" "robust"
  label="$(attack_label "$method")"
  attack_dir_args+=(--attack-dir "$label=$dir")
done

TRIPLET_DIR="$FIG_ATTACK_ROOT/$FIG_TRIPLET_METHOD"
if [[ ",$FIG_METHODS," != *",$FIG_TRIPLET_METHOD,"* ]]; then
  run_attack_once "$FIG_TRIPLET_METHOD" "$FIG_SURROGATES" "$TRIPLET_DIR" "full_model" "robust"
fi

"$PY" scripts/visualize_attack_triplet.py \
  --attack-dir "$TRIPLET_DIR" \
  --data-dir "$DATA_DIR" \
  --index "$FIG_INDEX" \
  --out "$FIG_ROOT/figure1_original_perturbation_adversarial.png" \
  --title "Original / perturbation / adversarial image ($(attack_label "$FIG_TRIPLET_METHOD"))"

"$PY" scripts/visualize_frequency_spectrum.py \
  --data-dir "$DATA_DIR" \
  "${attack_dir_args[@]}" \
  --index "$FIG_INDEX" \
  --out "$FIG_ROOT/figure2_frequency_spectrum.png" \
  --radial-out "$FIG_ROOT/figure2_radial_frequency_profile.csv" \
  --title "FFT spectrum of original image and attack perturbations"

# Use the first model in FIG_ATTENTION_MODEL if a comma-separated source list is passed.
ATTN_MODEL="${FIG_ATTENTION_MODEL%%,*}"
"$PY" scripts/visualize_attention_map.py \
  --data-dir "$DATA_DIR" \
  --selected-csv "$SELECTED_CSV" \
  --attack-dir "$TRIPLET_DIR" \
  --index "$FIG_INDEX" \
  --source-model "$ATTN_MODEL" \
  --models-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --out "$FIG_ROOT/figure3_attention_map.png" \
  --title "Attention / patch-saliency map from source model"

cat > "$FIG_ROOT/README_figures.txt" <<TXT
Generated figures:
1. figure1_original_perturbation_adversarial.png
   Original image / rescaled perturbation image / adversarial image.
2. figure2_frequency_spectrum.png
   Original FFT spectrum plus FFT spectra of perturbations from: $FIG_METHODS.
   The accompanying radial profile CSV is figure2_radial_frequency_profile.csv.
3. figure3_attention_map.png
   Attention rollout if the source model exposes ViT/DeiT attention maps; otherwise gradient/patch saliency fallback.

Configuration:
FIG_INDEX=$FIG_INDEX
FIG_SURROGATES=$FIG_SURROGATES
FIG_METHODS=$FIG_METHODS
FIG_TRIPLET_METHOD=$FIG_TRIPLET_METHOD
FIG_ATTENTION_MODEL=$ATTN_MODEL
TXT

if [[ "$DELETE_ADV_AFTER_USE" == "1" ]]; then
  for method_raw in "${FIG_METHOD_ARRAY[@]}" "$FIG_TRIPLET_METHOD"; do
    method="$(echo "$method_raw" | xargs)"
    [[ -z "$method" ]] && continue
    batch_dir="$(central_adv_batch_dir "$FIG_ATTACK_ROOT/$method")"
    if [[ -d "$batch_dir" ]]; then
      find "$batch_dir" -maxdepth 1 -type f -name 'batch_*.pt' -delete
    fi
  done
  echo "[cleanup] removed temporary figure adversarial batches"
fi

echo "[done] Figures saved to $FIG_ROOT"
