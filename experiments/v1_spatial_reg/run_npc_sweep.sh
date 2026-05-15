#!/bin/bash
# Test spatial regularization on visium_NPC (fast iteration: 1331 spots, ~2 min)
#
# Sweeps spatial_reg_lambda values: 0.001, 0.01, 0.05, 0.1
# Evaluates each against baseline.
#
# Usage:
#   bash experiments/v1_spatial_reg/run_npc_sweep.sh
#
# Output: experiments/v1_spatial_reg/raw_outputs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python3.11"

run_seedtopic() {
    $PYTHON -c "import sys; sys.argv = sys.argv[1:]; from seededntm.main import main; main()" "infer_seededntm" "$@"
}

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

ADATA="/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visium_NPC/exp_seeding_ref_scrnaseq/seededntm_adata_seeding_ref_scrnaseq.h5ad"
SEEDS="/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visium_NPC/NPC_topic_seeds_ref_scRNAseq.txt"

echo "========================================"
echo "Spatial Reg Sweep: visium_NPC"
echo "Started: $(date)"
echo "========================================"

OUTBASE="$SCRIPT_DIR/raw_outputs"
mkdir -p "$OUTBASE"

for LAMBDA in 0.001 0.01 0.05 0.1; do
    echo ""
    echo "--- Lambda = $LAMBDA ---"
    OUTDIR="$OUTBASE/npc_lambda_${LAMBDA}"
    mkdir -p "$OUTDIR"
    
    run_seedtopic \
        --adata_h5ad_path "$ADATA" \
        --condition_feat_path "$SEEDS" \
        --key_input tfidf_pca \
        --key_count_out rna_count \
        --num_topics 7 \
        --key_topic_prior topic_prior \
        --reg_topic_prior 0.9 \
        --wt_fusion_top_seed 1.0 \
        --batch_size 8192 \
        --spatial_reg_lambda "$LAMBDA" \
        --spatial_k_neighbors 6 \
        --exp_outdir "$OUTDIR" \
        2>&1 | tail -5
    
    echo "  Output: $OUTDIR/df_topic.csv"
done

echo ""
echo "--- Evaluating all results ---"
$PYTHON experiments/v1_spatial_reg/evaluate_sweep.py

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "========================================"
