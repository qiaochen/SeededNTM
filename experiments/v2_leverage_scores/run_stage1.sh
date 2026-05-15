#!/bin/bash
# Stage 1: Run leverage-score experiment on small datasets (NPC + CRC)
# Expected runtime: ~30-45 min total (3 datasets x 3 power values x ~3-5 min each)
#
# Usage:
#   bash experiments/v2_leverage_scores/run_stage1.sh

set -euo pipefail

PYTHON="/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python3.11"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Leverage-Score Gene Weighting: Stage 1 (Small Datasets) ==="
echo "Start time: $(date)"
echo ""

$PYTHON experiments/v2_leverage_scores/run_leverage_test.py \
    --dataset visium_NPC visiumHD_CRC_I visiumHD_CRC_II \
    --power 0.25 0.5 1.0

echo ""
echo "End time: $(date)"
echo "Done! Run stage2 (xenium_BC) next if stage1 looks promising."
