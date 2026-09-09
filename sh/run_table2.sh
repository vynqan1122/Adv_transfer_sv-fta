#!/usr/bin/env bash
set -euo pipefail
echo "[deprecated] Legacy token/frequency ablations were removed; running the current SV-FCA Table V." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_table5_prime.sh"
