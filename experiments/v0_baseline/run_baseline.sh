#!/bin/bash
# Run SeedTopic baseline (v0) on all 4 benchmark datasets.
# This reproduces the published SeedTopic results for comparison.
#
# Usage:
#   bash experiments/v0_baseline/run_baseline.sh
#
# Environment: requires seededntm conda env
# Expected runtime: ~30 min total (2 min NPC + 24 min Xenium + 2 min CRC_I + 2 min CRC_II)
#
# Output: experiments/v0_baseline/raw_outputs/{dataset}_proportions.csv
#         experiments/v0_baseline/metrics_summary.json
#         experiments/RESULTS_LOG.md (appended)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "SeedTopic Baseline Reproduction (v0)"
echo "Project: $PROJECT_DIR"
echo "Started: $(date)"
echo "========================================"

cd "$PROJECT_DIR"

# Use the seededntm environment python
PYTHON="/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python"

# Verify environment
echo "Python: $PYTHON"
$PYTHON -c "import seededntm; print(f'seededntm loaded OK')" 2>/dev/null || {
    echo "ERROR: seededntm not importable. Using system python with path workaround."
    PYTHON="python"
}

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo ""
echo "--- Running all 4 datasets ---"
$PYTHON experiments/run_experiment.py \
    --version v0_baseline \
    --datasets visium_NPC xenium_BC visiumHD_CRC_I visiumHD_CRC_II \
    --notes "baseline reproduction"

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "Results: experiments/v0_baseline/metrics_summary.json"
echo "Log: experiments/RESULTS_LOG.md"
echo "========================================"
