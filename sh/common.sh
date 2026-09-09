#!/usr/bin/env bash
# Shared configuration for all table and figure scripts.
# Every variable can be overridden from the shell, e.g.:
#   DATA_DIR=/data/imagenet/val DEVICE=cuda bash sh/run_tables_1_8.sh

set -euo pipefail

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$COMMON_SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${PY:=python3}"
: "${DATA_DIR:=../datasets/imagenet/val}"
: "${VAL_DIR:=}"
: "${LABELS_CSV=../datasets/imagenet/imagenet_val_labels.csv}"
: "${MODEL_DIR:=./pretrained_models}"
: "${OUT_DIR:=./runs}"

# All adversarial .pt batches are stored only under this central folder.
# Each attack run gets a unique subfolder under ADV_BATCH_ROOT, mirroring its OUT_DIR path.
: "${ADV_BATCH_ROOT:=$OUT_DIR/adv_batches}"
: "${CLEAR_ADV_BATCH_DIR:=1}"

: "${BATCH_SIZE:=4}"
: "${NUM_BATCHES:=}"
: "${EVAL_BATCH_SIZE:=$BATCH_SIZE}"
: "${NUM_WORKERS:=2}"
: "${DEVICE:=auto}"
: "${SEED:=0}"

# 12GB-VRAM-oriented defaults. Streaming SV-FCA is FP32 by default for
# reproducibility; enable SVFCA_AMP=1 only if a setting still exceeds VRAM.
: "${EMPTY_CACHE_EVERY:=8}"
: "${DELETE_ADV_AFTER_USE:=1}"
: "${SVFCA_AMP:=${SVFTA_AMP:-0}}"
: "${SVFCA_AMP_DTYPE:=${SVFTA_AMP_DTYPE:-fp16}}"
# Legacy aliases retained for older scripts.
: "${SVFTA_AMP:=$SVFCA_AMP}"
: "${SVFTA_AMP_DTYPE:=$SVFCA_AMP_DTYPE}"

# Final SV-FCA defaults used across Tables I-VIII.
: "${SVFCA_NUM_VIEWS:=4}"
: "${SVFCA_SPECTRAL_BANDS:=6}"
: "${SVFCA_DIVERSITY_PROB:=1.0}"
: "${SVFCA_BAND_TEMPERATURE:=0.35}"
: "${SVFCA_LOW_MID_STRENGTH:=1.0}"
: "${SVFCA_SPECTRAL_DECAY:=0.75}"
: "${SVFCA_DECAY:=1.0}"
: "${ADV_STORAGE_MODE:=delta_fp16}"

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[config] $name must be a positive integer; got '$value'" >&2
    return 1
  fi
}
require_positive_integer BATCH_SIZE "$BATCH_SIZE"
require_positive_integer EVAL_BATCH_SIZE "$EVAL_BATCH_SIZE"
if [[ -n "$NUM_BATCHES" ]]; then
  require_positive_integer NUM_BATCHES "$NUM_BATCHES"
fi

# If NUM_IMAGES is not explicitly set and NUM_BATCHES is given, select exactly enough images.
# Otherwise default to 1000 images.
if [[ -z "${NUM_IMAGES+x}" ]]; then
  if [[ -n "$NUM_BATCHES" ]]; then
    NUM_IMAGES=$(( NUM_BATCHES * BATCH_SIZE ))
  else
    NUM_IMAGES=1000
  fi
fi
require_positive_integer NUM_IMAGES "$NUM_IMAGES"
: "${SELECTED_CSV:=$OUT_DIR/selected_${NUM_IMAGES}.csv}"

# Model groups used throughout Table I-VIII.
: "${CNN_SURROGATES:=resnet50,densenet121}"
: "${VIT_SURROGATES:=vit_base_patch16_224,deit_small_patch16_224}"
: "${MIXED_SURROGATES:=resnet50,densenet121,vit_base_patch16_224,deit_small_patch16_224}"

: "${CNN_TARGETS:=resnet152,inception_v3}"
: "${VIT_TARGETS:=vit_large_patch16_224,deit_base_patch16_224,swin_tiny_patch4_window7_224}"
: "${MIXED_TARGETS:=resnet152,inception_v3,vit_large_patch16_224,deit_base_patch16_224,swin_tiny_patch4_window7_224}"

# Table VIII defaults. Override DEFENSE_TARGETS with RobustBench specs when available.
: "${ROBUSTBENCH_MODEL_DIR:=$PROJECT_ROOT/models}"
export ROBUSTBENCH_MODEL_DIR
: "${DEFAULT_DEFENSE_TARGETS:=robustbench:Salman2020Do_R50:imagenet:Linf,robustbench:Mo2022When_ViT-B:imagenet:Linf,robustbench:Liu2023Comprehensive_Swin-B:imagenet:Linf}"
: "${DEFENSE_TARGETS:=$DEFAULT_DEFENSE_TARGETS}"
: "${DEFENSE_COLUMNS:=}"

# Figure defaults.
: "${FIG_ROOT:=$OUT_DIR/figures}"
: "${FIG_INDEX:=0}"
: "${FIG_SURROGATES:=resnet50}"
: "${FIG_METHODS:=ifgsm,mifgsm,difgsm,tifgsm,si_ni_fgsm,vit_aware,freq_only,ours}"
: "${FIG_TRIPLET_METHOD:=ours}"
: "${FIG_ATTENTION_MODEL:=$FIG_SURROGATES}"
: "${FORCE_FIGURES:=0}"
: "${FORCE_ATTACKS:=0}"

maybe_val_args=()
if [[ -n "$VAL_DIR" ]]; then
  maybe_val_args+=(--val-dir "$VAL_DIR")
fi

maybe_label_args=()
if [[ -n "$LABELS_CSV" ]]; then
  maybe_label_args+=(--labels-csv "$LABELS_CSV")
fi

ensure_selected() {
  mkdir -p "$(dirname "$SELECTED_CSV")"
  local need_select=0
  if [[ ! -f "$SELECTED_CSV" ]]; then
    need_select=1
  else
    local existing_rows
    existing_rows=$("$PY" - "$SELECTED_CSV" <<'PY_COUNT_ROWS'
import csv, sys
path = sys.argv[1]
try:
    with open(path, newline='', encoding='utf-8') as f:
        print(max(sum(1 for _ in csv.DictReader(f)), 0))
except Exception:
    print(0)
PY_COUNT_ROWS
)
    if [[ "$existing_rows" -ne "$NUM_IMAGES" ]]; then
      echo "[select] $SELECTED_CSV has $existing_rows rows; need exactly $NUM_IMAGES. Reselecting."
      need_select=1
    fi
  fi

  if [[ "$need_select" == "1" ]]; then
    echo "[select] Selecting $NUM_IMAGES clean-correct images -> $SELECTED_CSV"
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
  fi
}

num_batch_args=()
if [[ -n "$NUM_BATCHES" ]]; then
  num_batch_args+=(--num-batches "$NUM_BATCHES")
fi

eval_batch_args=()
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  eval_batch_args+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi

quality_batch_args=()
if [[ -n "$NUM_BATCHES" ]]; then
  quality_batch_args+=(--max-batches "$NUM_BATCHES")
fi

central_adv_batch_dir() {
  local run_dir="$1"
  "$PY" - "$OUT_DIR" "$ADV_BATCH_ROOT" "$run_dir" <<'PY_CENTRAL_ADV'
import hashlib, os, sys
out_dir, adv_root, run_dir = sys.argv[1:4]
out_abs = os.path.abspath(out_dir)
run_abs = os.path.abspath(run_dir)
adv_root_abs = os.path.abspath(adv_root)
try:
    rel = os.path.relpath(run_abs, out_abs)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        rel = os.path.join('external', hashlib.sha256(run_abs.encode()).hexdigest()[:16], os.path.basename(run_abs))
except ValueError:
    rel = os.path.join('external', hashlib.sha256(run_abs.encode()).hexdigest()[:16], os.path.basename(run_abs))
print(os.path.join(adv_root_abs, rel))
PY_CENTRAL_ADV
}

manifest_has_live_batches() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  "$PY" - "$manifest" <<'PY_CHECK_MANIFEST'
import csv, pathlib, sys
try:
    with open(sys.argv[1], newline='', encoding='utf-8') as f:
        files = [row.get('file', '') for row in csv.DictReader(f)]
    parent = pathlib.Path(sys.argv[1]).parent
    ok = bool(files) and all(path and ((parent / path).is_file() or pathlib.Path(path).is_file()) for path in files)
except (OSError, csv.Error):
    ok = False
raise SystemExit(0 if ok else 1)
PY_CHECK_MANIFEST
}

# Include the selected images, attack settings, and Python source revision in
# the reuse key. Old runs without a key are regenerated once.
attack_signature() {
  "$PY" - "$SELECTED_CSV" "$DATA_DIR" "$MODEL_DIR" "$NUM_IMAGES" \
    "$NUM_BATCHES" "$BATCH_SIZE" "$SEED" "$DEVICE" "$ADV_STORAGE_MODE" \
    "${svfca_core_args[@]}" "${svfca_amp_args[@]}" "$@" <<'PY_SIGNATURE'
import hashlib, json, pathlib, sys
digest = hashlib.sha256(json.dumps(sys.argv[1:], ensure_ascii=False).encode())
digest.update(pathlib.Path(sys.argv[1]).read_bytes())
for folder in ('attacks', 'src', 'scripts'):
    for path in sorted(pathlib.Path(folder).glob('*.py')):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
print(digest.hexdigest())
PY_SIGNATURE
}

attack_signature_matches() {
  local out_dir="$1" signature="$2"
  [[ -f "$out_dir/.shell_attack_signature" ]] && \
    [[ "$(cat "$out_dir/.shell_attack_signature")" == "$signature" ]]
}

clear_adv_args=()
if [[ "$CLEAR_ADV_BATCH_DIR" == "1" ]]; then
  clear_adv_args+=(--clear-adv-batch-dir)
fi

memory_attack_args=(--empty-cache-every "$EMPTY_CACHE_EVERY" --storage-mode "$ADV_STORAGE_MODE")
svfca_amp_args=()
if [[ "$SVFCA_AMP" == "1" ]]; then
  svfca_amp_args+=(--amp --amp-dtype "$SVFCA_AMP_DTYPE")
fi
# Backward-compatible array name used by older table scripts.
svfta_amp_args=("${svfca_amp_args[@]}")

svfca_core_args=(
  --num-views "$SVFCA_NUM_VIEWS"
  --spectral-bands "$SVFCA_SPECTRAL_BANDS"
  --diversity-prob "$SVFCA_DIVERSITY_PROB"
  --band-temperature "$SVFCA_BAND_TEMPERATURE"
  --low-mid-strength "$SVFCA_LOW_MID_STRENGTH"
  --spectral-decay "$SVFCA_SPECTRAL_DECAY"
  --decay "$SVFCA_DECAY"
)

delete_adv_args=()
if [[ "$DELETE_ADV_AFTER_USE" == "1" ]]; then
  delete_adv_args+=(--delete-batches-after-eval)
fi

attack_label() {
  case "$1" in
    difgsm) echo "DI-FGSM" ;;
    ifgsm) echo "I-FGSM" ;;
    mifgsm) echo "MI-FGSM" ;;
    si_ni_fgsm) echo "SI-NI-FGSM" ;;
    tifgsm) echo "TI-FGSM" ;;
    vit_aware) echo "ViT-Aware" ;;
    freq_only) echo "Freq-Only" ;;
    ours|sv_fca|svfca|sv_fta|svfta|ddc) echo "Ours (SV-FCA)" ;;
    *) echo "$1" ;;
  esac
}

run_attack_once() {
  local method="$1"
  local sources="$2"
  local out_dir="$3"
  local variant="${4:-full_model}"
  local fusion="${5:-robust}"
  local signature
  signature="$(attack_signature "$method" "$sources" "$variant" "$fusion")"

  if [[ "$FORCE_ATTACKS" != "1" && "$FORCE_FIGURES" != "1" ]] && \
    attack_signature_matches "$out_dir" "$signature" && \
    manifest_has_live_batches "$out_dir/attack_batches.csv"; then
    echo "[reuse] $out_dir/attack_batches.csv"
    return 0
  fi

  mkdir -p "$out_dir"
  rm -f "$out_dir/.shell_attack_signature" "$out_dir/.shell_eval_targets"
  echo "[attack] method=$method sources=$sources out=$out_dir"
  "$PY" scripts/run_attack.py \
    --data-dir "$DATA_DIR" \
    --selected-csv "$SELECTED_CSV" \
    --models-dir "$MODEL_DIR" \
    --surrogates "$sources" \
    --attack "$method" \
    --variant "$variant" \
    --fusion "$fusion" \
    --out-dir "$out_dir" \
    --adv-batch-dir "$(central_adv_batch_dir "$out_dir")" \
    "${clear_adv_args[@]}" \
    "${num_batch_args[@]}" \
    "${memory_attack_args[@]}" \
    "${svfca_amp_args[@]}" \
    "${svfca_core_args[@]}" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
  printf '%s\n' "$signature" > "$out_dir/.shell_attack_signature"
}
