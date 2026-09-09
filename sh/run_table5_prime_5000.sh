#!/usr/bin/env bash
set -euo pipefail
export TABLE5P_NUM_IMAGES="${TABLE5P_NUM_IMAGES:-5000}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_table5_prime.sh"
