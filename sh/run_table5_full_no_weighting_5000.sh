#!/usr/bin/env bash
set -euo pipefail
echo "[deprecated] The current model is SV-FCA. Running the 5000-image Table V component ablations."
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_table5_prime_5000.sh"
