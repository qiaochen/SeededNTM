#!/bin/bash
# Run leverage-score experiment on CRC datasets (primary targets)
# These are where FlashDeconv/RCTD outperform SeedTopic most.
#
# Sweeps power={0.25, 0.5, 1.0} on NPC (quick validation) + both CRCs.
#
# Usage:
#   bash experiments/v2_leverage_scores/run_crc.sh
#
# Expected runtime: ~3-5 min per dataset per power value, ~30-45 min total

set -euo pipefail

PYTHON="/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Leverage-Score Gene Weighting Experiment ==="
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo ""

# Run on all datasets with available reference
# Power sweep: 0.25 (gentle), 0.5 (sqrt), 1.0 (raw)
$PYTHON experiments/v2_leverage_scores/run_leverage_test.py \
    --dataset visium_NPC visiumHD_CRC_I visiumHD_CRC_II xenium_BC \
    --power 0.25 0.5 1.0

echo ""
echo "End time: $(date)"
echo "Done! Check experiments/v2_leverage_scores/leverage_results.json for results."
