#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_selected

ROOT="$OUT_DIR/table6_quality"
mkdir -p "$ROOT"
FINAL="$ROOT/table6_quality.csv"
TMP_MANIFEST="$ROOT/.one_attack_manifest.csv"
TMP_RESULT="$ROOT/.one_attack_quality.csv"
METHODS=(difgsm ifgsm mifgsm si_ni_fgsm tifgsm vit_aware freq_only ours)

label_method() {
  case "$1" in
    difgsm) echo "DI-FGSM" ;;
    ifgsm) echo "I-FGSM" ;;
    mifgsm) echo "MI-FGSM" ;;
    si_ni_fgsm) echo "SI-NI-FGSM" ;;
    tifgsm) echo "TI-FGSM" ;;
    vit_aware) echo "ViT-Aware" ;;
    freq_only) echo "Freq-Only" ;;
    ours) echo "Ours (SV-FCA)" ;;
    *) echo "$1" ;;
  esac
}

# Process one attack at a time: generate -> quality metrics -> delete .pt batches.
# This bounds peak disk usage to a single method/family run.
printf "Family,Method,PSNR,SSIM,LPIPS,AttackDir\n" > "$FINAL"

for family_key in cnn_surrogates vit_surrogates; do
  if [[ "$family_key" == "cnn_surrogates" ]]; then
    family="CNN Surrogates"; sources="$CNN_SURROGATES"
  else
    family="ViT Surrogates"; sources="$VIT_SURROGATES"
  fi

  for method in "${METHODS[@]}"; do
    dir="$ROOT/$family_key/$method"
    mkdir -p "$dir"
    echo "[Table VI] family=$family method=$method"

    run_attack_once "$method" "$sources" "$dir"

    "$PY" - "$TMP_MANIFEST" "$family" "$(label_method "$method")" "$dir" <<'PY_QUALITY_MANIFEST'
import csv, sys
with open(sys.argv[1], 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['family', 'method', 'attack_dir'])
    writer.writerow(sys.argv[2:])
PY_QUALITY_MANIFEST

    quality_delete_args=()
    if [[ "$DELETE_ADV_AFTER_USE" == "1" ]]; then
      quality_delete_args+=(--delete-batches-after-use)
    fi

    "$PY" scripts/compute_perceptual_quality.py \
      --manifest "$TMP_MANIFEST" \
      --data-dir "$DATA_DIR" \
      --out "$TMP_RESULT" \
      "${quality_batch_args[@]}" \
      --quality-batch-size "$QUALITY_BATCH_SIZE" \
      "${adaptive_batch_args[@]}" \
      "${quality_delete_args[@]}" \
      --device "$DEVICE"

    tail -n +2 "$TMP_RESULT" >> "$FINAL"
    rm -f "$TMP_MANIFEST" "$TMP_RESULT" "${TMP_RESULT%.csv}.execution.json"
  done
done

echo "[done] $FINAL"
echo "[storage] adversarial .pt tensors are removed after each quality run when DELETE_ADV_AFTER_USE=1"
