#!/usr/bin/env bash
set -eo pipefail

# Phase 10: Multi-Direction CRC_II Improvement
# 8 configs x 4 datasets = 32 experiments
# Directions: entropy gating, seed-gene NNLS, post-blend smoothing
#
# Indices (36 configs/dataset, phase10 at positions 28-35):
#   visium_NPC:      28-35
#   xenium_BC:       64-71
#   visiumHD_CRC_I:  100-107
#   visiumHD_CRC_II: 136-143

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate seededntm
set -u

cd "$PROJ_DIR"

echo "=== Phase 10: Multi-Direction CRC_II Improvement ==="
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Start time: $(date)"
echo "Experiment script: experiments/v3_reffree_leverage/run_v28_twofactor.py"
echo ""

run_exp() {
    local idx=$1
    echo "[$(date '+%H:%M:%S')] Starting idx=$idx"
    python experiments/v3_reffree_leverage/run_v28_twofactor.py --idx "$idx" 2>&1 | \
        tail -5
    echo "[$(date '+%H:%M:%S')] Finished idx=$idx"
}

# Interleave datasets to avoid GPU memory contention
# Process in groups of 6 (max parallel), interleaving CRC_I/CRC_II/NPC/xenium
# Group 1: 1 from each dataset + 2 extra (6 total)
run_exp 28 &   # NPC entgate_A
run_exp 64 &   # xenium entgate_A
run_exp 100 &  # CRC_I entgate_A
run_exp 136 &  # CRC_II entgate_A
run_exp 29 &   # NPC entgate_B
run_exp 65 &   # xenium entgate_B
wait

# Group 2
run_exp 101 &  # CRC_I entgate_B
run_exp 137 &  # CRC_II entgate_B
run_exp 30 &   # NPC entgate_C
run_exp 66 &   # xenium entgate_C
run_exp 102 &  # CRC_I entgate_C
run_exp 138 &  # CRC_II entgate_C
wait

# Group 3
run_exp 31 &   # NPC entgate_D
run_exp 67 &   # xenium entgate_D
run_exp 103 &  # CRC_I entgate_D
run_exp 139 &  # CRC_II entgate_D
run_exp 32 &   # NPC seednnls_A
run_exp 68 &   # xenium seednnls_A
wait

# Group 4
run_exp 104 &  # CRC_I seednnls_A
run_exp 140 &  # CRC_II seednnls_A
run_exp 33 &   # NPC seednnls_B
run_exp 69 &   # xenium seednnls_B
run_exp 105 &  # CRC_I seednnls_B
run_exp 141 &  # CRC_II seednnls_B
wait

# Group 5
run_exp 34 &   # NPC postsmooth_A
run_exp 70 &   # xenium postsmooth_A
run_exp 106 &  # CRC_I postsmooth_A
run_exp 142 &  # CRC_II postsmooth_A
run_exp 35 &   # NPC postsmooth_B
run_exp 71 &   # xenium postsmooth_B
wait

# Group 6 (final 2)
run_exp 107 &  # CRC_I postsmooth_B
run_exp 143 &  # CRC_II postsmooth_B
wait

echo ""
echo "=== Phase 10 complete ==="
echo "End time: $(date)"
echo "Results: experiments/v3_reffree_leverage/reffree_ablation_v28_twofactor.json"
