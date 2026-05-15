#!/bin/bash
# Ablation: TF-IDF vs raw counts, PCA vs CountSketch
#
# Tests the full 2x2 matrix:
#   TF-IDF + PCA (existing baseline)
#   TF-IDF + CountSketch
#   Raw counts + PCA (ablate TF-IDF)
#   Raw counts + CountSketch (ablate both)
#
# For each: none (no weights) + seed_specificity + pseudo_sig
#
# Usage (on compute node):
#   cd /illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic
#   bash experiments/v3_reffree_leverage/run_ablation.sh

set +e

eval "$(micromamba shell hook --shell bash)"
micromamba activate seededntm

cd /illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic

echo "=== Ablation: TF-IDF vs Raw, PCA vs Sketch ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# CRC datasets first (small, fast)
echo "--- CRC_I + CRC_II: all preprocessing combos ---"
python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visiumHD_CRC_I visiumHD_CRC_II \
    --methods none seed_specificity pseudo_sig \
    --dim-red sketch raw_pca raw_sketch

echo ""
echo "--- NPC: all preprocessing combos ---"
python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visium_NPC \
    --methods none seed_specificity pseudo_sig \
    --dim-red raw_pca raw_sketch

echo ""
echo "--- Xenium BC: all preprocessing combos ---"
python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset xenium_BC_leiden \
    --methods none seed_specificity pseudo_sig \
    --dim-red sketch raw_pca raw_sketch

echo ""
echo "=== Ablation complete ==="
cat experiments/v3_reffree_leverage/reffree_results.json
