#!/bin/bash
# Validate auto leverage scaling and PCA vs CountSketch
#
# Tests:
# 1. Auto scaling on CRC_II (should match or beat power=0.5-1.0 results)
# 2. Auto scaling on NPC (should NOT have the ARI regression seen with power=0.5+)
# 3. CountSketch vs PCA comparison on both
#
# Expected runtime: ~15-20 min (4 runs x ~3-5 min each)
#
# Usage:
#   bash experiments/v2_leverage_scores/run_validate_auto.sh

set -euo pipefail

PYTHON="/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python3.11"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Validate Auto Leverage + PCA vs Sketch ==="
echo "Start time: $(date)"
echo ""

# Run auto (no --power) with both pca and sketch on NPC + CRC_II
$PYTHON experiments/v2_leverage_scores/run_leverage_test.py \
    --dataset visium_NPC visiumHD_CRC_II \
    --method pca sketch

echo ""
echo "End time: $(date)"
echo "Results: experiments/v2_leverage_scores/leverage_results_v2.json"
