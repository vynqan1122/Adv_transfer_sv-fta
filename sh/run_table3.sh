#!/usr/bin/env bash
set -euo pipefail
echo "[deprecated] Legacy fusion modes are ignored by SV-FCA; running the current component ablations in Table V." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_table5_prime.sh"
