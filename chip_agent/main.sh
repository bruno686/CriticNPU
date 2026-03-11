#!/usr/bin/env bash
# main.sh — Run the full multi-agent NPU design pipeline.
#
# Usage:
#   ./main.sh <model_name> [options]
#
# Example:
#   ./main.sh llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --disaggregated

set -euo pipefail
cd "$(dirname "$0")"

python main.py "$@"
