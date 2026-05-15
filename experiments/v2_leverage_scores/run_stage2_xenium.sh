#!/bin/bash
# Stage 2: Run leverage-score experiment on xenium_BC (large dataset)
# Expected runtime: ~60-80 min (159K spots x 3 power values)
#
# Usage:
#   bash experiments/v2_leverage_scores/run_stage2_xenium.sh

set -euo pipefail

PYTHON="/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python3.11"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Leverage-Score Gene Weighting: Stage 2 (Xenium BC) ==="
echo "Start time: $(date)"
echo ""

$PYTHON experiments/v2_leverage_scores/run_leverage_test.py \
    --dataset xenium_BC \
    --power 0.25 0.5 1.0

echo ""
echo "End time: $(date)"
echo "Done! Check experiments/v2_leverage_scores/leverage_results.json"
