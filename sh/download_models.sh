#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PY" scripts/download_models.py \
  --models-dir "$MODEL_DIR" \
  --device cpu
