#!/usr/bin/env bash
set -euo pipefail

# Run all numeric tables and all visualization figures.
# Use SKIP_TABLES=1 or SKIP_FIGURES=1 if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ "${SKIP_TABLES:-0}" != "1" ]]; then
  RUN_FIGURES=0 bash sh/run_tables_1_8.sh
else
  echo "[SKIP] Tables I-VIII"
fi

if [[ "${SKIP_FIGURES:-0}" != "1" ]]; then
  bash sh/run_figures.sh
else
  echo "[SKIP] Figures"
fi
