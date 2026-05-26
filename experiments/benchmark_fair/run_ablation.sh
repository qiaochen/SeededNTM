#!/bin/bash
# Ablation launcher: 36 experiments, up to 6 parallel processes
# Interleaved by dataset so parallel slots run different datasets (avoids GPU memory contention)
#
# Usage: bash run_ablation.sh
#
# Environment: micromamba activate seededntm

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/run_rep_prior_ablation.py"

echo "=========================================="
echo "Representation x Prior Mode Ablation"
echo "=========================================="
echo "Script: $SCRIPT"
echo "Project: $PROJECT_ROOT"
echo "Parallel: 6 processes"
echo "Total experiments: 36"
echo "=========================================="

cd "$PROJECT_ROOT"

# Activate environment (set +u needed for conda/micromamba deactivation hooks)
eval "$(micromamba shell hook --shell bash)"
set +u
micromamba activate seededntm
set -u

# Show experiment matrix before launching
python "$SCRIPT" --show

echo ""
echo "Launching 36 experiments with xargs -P 6..."
echo ""

# Run indices 0-35 with up to 6 parallel processes
seq 0 35 | xargs -I{} -P 6 python "$SCRIPT" --idx {}

echo ""
echo "=========================================="
echo "All experiments complete. Running analysis..."
echo "=========================================="

python "$SCRIPT" --analyze
