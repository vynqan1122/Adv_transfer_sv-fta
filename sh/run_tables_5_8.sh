#!/usr/bin/env bash
set -euo pipefail

# Run from project root even if this script is launched from another folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

# You can skip any table by setting SKIP_TABLE5=1, SKIP_TABLE6=1, etc.
# Example: SKIP_TABLE6=1 bash sh/run_tables_5_8.sh

echo "=========================================="
echo "[RUN] Table V: SV-FCA Ablation Study"
echo "=========================================="
if [[ "${SKIP_TABLE5:-0}" != "1" ]]; then
  bash sh/run_table5_prime.sh
else
  echo "[SKIP] Table V"
fi

echo "=========================================="
echo "[RUN] Table VI: Perceptual Quality"
echo "=========================================="
if [[ "${SKIP_TABLE6:-0}" != "1" ]]; then
  bash sh/run_table6_quality.sh
else
  echo "[SKIP] Table VI"
fi

echo "=========================================="
echo "[RUN] Table VII: Runtime"
echo "=========================================="
if [[ "${SKIP_TABLE7:-0}" != "1" ]]; then
  bash sh/run_table7_runtime.sh
else
  echo "[SKIP] Table VII"
fi

echo "=========================================="
echo "[RUN] Table VIII: Defense Evaluation"
echo "=========================================="
if [[ "${SKIP_TABLE8:-0}" != "1" ]]; then
  bash sh/run_table8_defense.sh
else
  echo "[SKIP] Table VIII"
fi

echo "=========================================="
echo "[DONE] Finished Table V-VIII"
echo "Outputs:"
echo "  ${OUT_DIR:-./runs}/table5_prime/table5_prime.csv"
echo "  ${OUT_DIR:-./runs}/table6_quality/table6_quality.csv"
echo "  ${OUT_DIR:-./runs}/table7_runtime/table7_runtime.csv"
echo "  ${OUT_DIR:-./runs}/table8_defense/table8_defense.csv"
echo "=========================================="
