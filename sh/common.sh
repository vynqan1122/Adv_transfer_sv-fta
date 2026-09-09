#!/usr/bin/env bash
# Shared experiment launcher. Edit experiment.venv; keep workflow logic here.
# Preview resolved settings without loading models: bash sh/common.sh
set -euo pipefail

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$COMMON_SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 01. Load configuration. Optional overlays also use Bash default assignments
# (: "${NAME:=value}") so one-run environment overrides always take priority.
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/experiment.venv}"
[[ "$CONFIG_FILE" == /* ]] || CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
if [[ ! -f "$CONFIG_FILE" || "$CONFIG_FILE" != *.venv ]]; then
  echo "[config] CONFIG_FILE must be an existing .venv file: $CONFIG_FILE" >&2
  exit 1
fi
source "$CONFIG_FILE"
if [[ "$CONFIG_FILE" != "$PROJECT_ROOT/experiment.venv" ]]; then
  source "$PROJECT_ROOT/experiment.venv"
fi
export CONFIG_FILE ROBUSTBENCH_MODEL_DIR

# 02. Validate and resolve dependent values before creating outputs.
require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[config] $name must be a positive integer; got '$value'" >&2
    return 1
  fi
}
require_nonnegative_integer() {
  local name="$1" value="$2"
  if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "[config] $name must be a nonnegative integer; got '$value'" >&2
    return 1
  fi
}
for config_key in BATCH_SIZE EVAL_BATCH_SIZE QUALITY_BATCH_SIZE MIN_BATCH_SIZE STEPS DEFENSE_STEPS RUNTIME_IMAGES SVFCA_NUM_VIEWS SVFCA_SPECTRAL_BANDS; do
  require_positive_integer "$config_key" "${!config_key}"
done
if (( SVFCA_SPECTRAL_BANDS < 2 )); then
  echo "[config] SVFCA_SPECTRAL_BANDS must be at least 2" >&2
  exit 1
fi
for config_key in NUM_WORKERS EMPTY_CACHE_EVERY FIG_INDEX SEED; do
  require_nonnegative_integer "$config_key" "${!config_key}"
done
for config_key in AUTO_BATCH CLEAR_ADV_BATCH_DIR DELETE_ADV_AFTER_USE SVFCA_AMP FORCE_ATTACKS FORCE_FIGURES SKIP_TABLES SKIP_TABLES1_4 SKIP_TABLE5 SKIP_TABLE6 SKIP_TABLE7 SKIP_TABLE8 RUN_FIGURES SKIP_FIGURES; do
  if [[ "${!config_key}" != 0 && "${!config_key}" != 1 ]]; then
    echo "[config] $config_key must be 0 or 1; got '${!config_key}'" >&2
    exit 1
  fi
done
for config_key in BATCH_SIZE EVAL_BATCH_SIZE QUALITY_BATCH_SIZE; do
  if (( MIN_BATCH_SIZE > ${!config_key} )); then
    echo "[config] MIN_BATCH_SIZE cannot exceed $config_key" >&2
    exit 1
  fi
done
if [[ -n "$NUM_BATCHES" ]]; then
  require_positive_integer NUM_BATCHES "$NUM_BATCHES"
fi
if [[ -z "$NUM_IMAGES" ]]; then
  if [[ -n "$NUM_BATCHES" ]]; then
    NUM_IMAGES=$(( NUM_BATCHES * BATCH_SIZE ))
  else
    NUM_IMAGES=1000
  fi
fi
require_positive_integer NUM_IMAGES "$NUM_IMAGES"
: "${SELECTED_CSV:=$OUT_DIR/selected_${NUM_IMAGES}.csv}"

# Parse decimal or numerator/denominator notation; never evaluate shell/Python
# expressions. Downstream CLIs receive ordinary finite decimal numbers.
resolved_budget=$("$PY" - "$EPS" "$ALPHA" "$DEFENSE_EPS" "$DEFENSE_ALPHA" \
  "$SVFCA_DIVERSITY_PROB" "$SVFCA_BAND_TEMPERATURE" "$SVFCA_LOW_MID_STRENGTH" \
  "$SVFCA_SPECTRAL_DECAY" "$SVFCA_DECAY" <<'PY_BUDGET'
from fractions import Fraction
import math
import sys
values = []
for name, raw in zip(('EPS', 'ALPHA', 'DEFENSE_EPS', 'DEFENSE_ALPHA'), sys.argv[1:]):
    try:
        parts = raw.split('/')
        if len(parts) not in (1, 2):
            raise ValueError('use a decimal or numerator/denominator')
        value = Fraction(parts[0]) / (Fraction(parts[1]) if len(parts) == 2 else 1)
        if not 0 <= value <= 1:
            raise ValueError('must be in [0, 1] for input pixels in [0, 1]')
    except (ValueError, ZeroDivisionError) as exc:
        raise SystemExit(f'[config] invalid {name}={raw!r}: {exc}') from None
    values.append(format(float(value), '.17g'))
rules = (
    ('SVFCA_DIVERSITY_PROB', lambda v: 0 <= v <= 1, 'in [0, 1]'),
    ('SVFCA_BAND_TEMPERATURE', lambda v: v > 0, 'positive'),
    ('SVFCA_LOW_MID_STRENGTH', lambda v: v >= 0, 'nonnegative'),
    ('SVFCA_SPECTRAL_DECAY', lambda v: 0 <= v < 1, 'in [0, 1)'),
    ('SVFCA_DECAY', lambda v: v >= 0, 'nonnegative'),
)
for (name, valid, requirement), raw in zip(rules, sys.argv[5:]):
    try:
        value = float(raw)
        if not math.isfinite(value) or not valid(value):
            raise ValueError(f'must be finite and {requirement}')
    except ValueError as exc:
        raise SystemExit(f'[config] invalid {name}={raw!r}: {exc}') from None
print(' '.join(values))
PY_BUDGET
)
read -r EPS ALPHA DEFENSE_EPS DEFENSE_ALPHA <<< "$resolved_budget"
unset resolved_budget config_key
case "$ADV_STORAGE_MODE" in
  adv_fp32|delta_fp16) ;;
  *) echo "[config] ADV_STORAGE_MODE must be adv_fp32 or delta_fp16" >&2; exit 1 ;;
esac
case "$SVFCA_AMP_DTYPE" in
  fp16|bf16) ;;
  *) echo "[config] SVFCA_AMP_DTYPE must be fp16 or bf16" >&2; exit 1 ;;
esac

# 03. Shared CLI arguments. OOM recovery is implemented inside Python, keeping
# logical batches/sample limits intact while reducing only GPU microbatches.
attack_budget_args=(--eps "$EPS" --alpha "$ALPHA" --steps "$STEPS")
adaptive_batch_args=(--min-batch-size "$MIN_BATCH_SIZE")
[[ "$AUTO_BATCH" == 1 ]] || adaptive_batch_args+=(--no-auto-batch)

maybe_val_args=()
if [[ -n "$VAL_DIR" ]]; then
  maybe_val_args+=(--val-dir "$VAL_DIR")
fi

maybe_label_args=()
if [[ -n "$LABELS_CSV" ]]; then
  maybe_label_args+=(--labels-csv "$LABELS_CSV")
fi

# 04. Selection, output paths and reuse signatures.
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
      "${adaptive_batch_args[@]}" \
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
    "${svfca_core_args[@]}" "${svfca_amp_args[@]}" \
    "$EPS" "$ALPHA" "$STEPS" "$DEFENSE_EPS" "$DEFENSE_ALPHA" "$DEFENSE_STEPS" \
    "$AUTO_BATCH" "$MIN_BATCH_SIZE" "$EVAL_BATCH_SIZE" "$QUALITY_BATCH_SIZE" "$@" <<'PY_SIGNATURE'
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
  local EPS="${6:-$EPS}" ALPHA="${7:-$ALPHA}" STEPS="${8:-$STEPS}"
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
    --eps "$EPS" --alpha "$ALPHA" --steps "$STEPS" \
    "${adaptive_batch_args[@]}" \
    "${svfca_amp_args[@]}" \
    "${svfca_core_args[@]}" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
  printf '%s\n' "$signature" > "$out_dir/.shell_attack_signature"
}

# 05. Reusable evaluation command; target checks remain table-specific.
run_eval() {
  local out_dir="$1" targets="$2"
  "$PY" scripts/evaluate.py \
    --attack-dir "$out_dir" \
    --data-dir "$DATA_DIR" \
    --models-dir "$MODEL_DIR" \
    --robustbench-model-dir "$ROBUSTBENCH_MODEL_DIR" \
    --targets "$targets" \
    "${num_batch_args[@]}" \
    "${eval_batch_args[@]}" \
    "${adaptive_batch_args[@]}" \
    "${delete_adv_args[@]}" \
    --device "$DEVICE"
}

# Executing common.sh is a lightweight configuration check, not an experiment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%-23s %s\n' \
    'Configuration:' "$CONFIG_FILE" \
    'Python / device:' "$PY / $DEVICE" \
    'Data:' "$DATA_DIR" \
    'Models:' "$MODEL_DIR" \
    'Outputs:' "$OUT_DIR" \
    'Selected images:' "$NUM_IMAGES" \
    'Original batch limit:' "${NUM_BATCHES:-all}" \
    'Attack / eval / quality:' "$BATCH_SIZE / $EVAL_BATCH_SIZE / $QUALITY_BATCH_SIZE" \
    'Auto batch / minimum:' "$AUTO_BATCH / $MIN_BATCH_SIZE" \
    'EPS / ALPHA / STEPS:' "$EPS / $ALPHA / $STEPS" \
    'Defense budget:' "$DEFENSE_EPS / $DEFENSE_ALPHA / $DEFENSE_STEPS" \
    'Selected CSV:' "$SELECTED_CSV"
fi
