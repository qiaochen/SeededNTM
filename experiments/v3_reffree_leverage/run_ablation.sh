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
# Runs 4 datasets in parallel.
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

# Run all 4 datasets in parallel
python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visiumHD_CRC_I \
    --methods none seed_specificity pseudo_sig \
    --dim-red sketch raw_pca raw_sketch \
    2>&1 | tee experiments/v3_reffree_leverage/log_ablation_crc1.txt &
PID1=$!

python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visiumHD_CRC_II \
    --methods none seed_specificity pseudo_sig \
    --dim-red sketch raw_pca raw_sketch \
    2>&1 | tee experiments/v3_reffree_leverage/log_ablation_crc2.txt &
PID2=$!

python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visium_NPC \
    --methods none seed_specificity pseudo_sig \
    --dim-red raw_pca raw_sketch \
    2>&1 | tee experiments/v3_reffree_leverage/log_ablation_npc.txt &
PID3=$!

python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset xenium_BC_leiden \
    --methods none seed_specificity pseudo_sig \
    --dim-red sketch raw_pca raw_sketch \
    2>&1 | tee experiments/v3_reffree_leverage/log_ablation_xenium.txt &
PID4=$!

echo "Launched 4 parallel jobs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; echo "CRC_I done (exit $?)"
wait $PID2; echo "CRC_II done (exit $?)"
wait $PID3; echo "NPC done (exit $?)"
wait $PID4; echo "Xenium done (exit $?)"

echo ""
echo "=== Ablation complete ==="
echo "Results:"
cat experiments/v3_reffree_leverage/reffree_results.json
