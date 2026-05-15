#!/bin/bash
# Run reference-free leverage experiments on a computing node.
# Usage: submit this script to your job scheduler, or run interactively on a compute node.
#
# Example (interactive):
#   srun --gres=gpu:1 --mem=32G --time=2:00:00 bash experiments/v3_reffree_leverage/run_on_compute.sh
#
# Example (batch):
#   sbatch experiments/v3_reffree_leverage/run_on_compute.sh

#SBATCH --job-name=reffree_leverage
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=experiments/v3_reffree_leverage/slurm_%j.log

set -e

# Activate the seededntm environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate seededntm

PYTHON=python
PROJECT_DIR=/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic

cd "$PROJECT_DIR"

echo "=== Reference-Free Leverage Experiment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# Run NPC first (small, ~2 min per condition)
echo "--- Stage 1: visium_NPC (small dataset, quick validation) ---"
$PYTHON experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visium_NPC \
    --methods none pseudo_sig self_leverage_k seed_specificity \
    --dim-red pca \
    --include-ref-based \
    2>&1 | tee experiments/v3_reffree_leverage/log_npc.txt

echo ""
echo "--- Stage 2: xenium_BC_leiden (primary test case, ~20 min per condition) ---"
$PYTHON experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset xenium_BC_leiden \
    --methods none pseudo_sig self_leverage_k seed_specificity \
    --dim-red pca \
    2>&1 | tee experiments/v3_reffree_leverage/log_xenium.txt

echo ""
echo "--- Stage 3: CountSketch comparison on NPC ---"
$PYTHON experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visium_NPC \
    --methods pseudo_sig self_leverage_k \
    --dim-red pca sketch \
    2>&1 | tee experiments/v3_reffree_leverage/log_sketch.txt

echo ""
echo "=== All stages complete ==="
echo "Results: experiments/v3_reffree_leverage/reffree_results.json"
