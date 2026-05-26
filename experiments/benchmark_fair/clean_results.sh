#!/bin/bash
# Clean all cached benchmark results for a fresh start.
# Must be run before every new experiment batch.
#
# Usage: bash experiments/benchmark_fair/clean_results.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning cached benchmark results..."
rm -f "$SCRIPT_DIR/fair_benchmark_results.json"
rm -f "$SCRIPT_DIR/ablation_results.json"
rm -rf "$SCRIPT_DIR/raw_outputs"
rm -rf "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/logs"
echo "  Removed: fair_benchmark_results.json"
echo "  Removed: ablation_results.json"
echo "  Removed: raw_outputs/"
echo "  Cleared: logs/"
echo "Done. Ready for clean run."
