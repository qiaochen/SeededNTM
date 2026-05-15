#!/bin/bash
# Run reference-free leverage experiments on a computing node.
#
# Usage (on compute node):
#   cd /illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic
#   bash experiments/v3_reffree_leverage/run_on_compute.sh

#SBATCH --job-name=reffree_leverage
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=experiments/v3_reffree_leverage/slurm_%j.log

set +e

eval "$(micromamba shell hook --shell bash)"
micromamba activate seededntm

PROJECT_DIR=/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic
cd "$PROJECT_DIR"

echo "=== Reference-Free Leverage Experiment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# Step 1: Recover metrics from previously saved xenium proportions
echo "--- Recovering xenium_BC_leiden metrics from saved proportions ---"
python -c "
import json, sys
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from experiments.evaluate import compute_metrics, load_ground_truth
from pathlib import Path

out_dir = Path('experiments/v3_reffree_leverage/raw_outputs')
results_path = Path('experiments/v3_reffree_leverage/reffree_results.json')

existing = {}
if results_path.exists():
    existing = json.loads(results_path.read_text())

gt = load_ground_truth('xenium_BC')
seeds_path = '/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/xenium_breast_cancer/bc_topic_seeds_leiden_chatgpt.txt'
with open(seeds_path) as f:
    seeds = json.load(f)
idx_to_name = {v['topic_index']: k for k, v in seeds.items()}
ct_names = [idx_to_name[i] for i in range(len(seeds))]

for method in ['none', 'pseudo_sig']:
    key = f'xenium_BC_leiden_{method}_pca'
    csv_path = out_dir / f'{key}_proportions.csv'
    if csv_path.exists() and key not in existing:
        df = pd.read_csv(csv_path)
        n = min(len(gt), len(df))
        metrics = compute_metrics(df.values[:n], list(df.columns), gt[:n])
        existing[key] = {'dataset': 'xenium_BC_leiden', 'method': method, 'dim_red': 'pca', 'metrics': metrics, 'runtime_s': 'recovered', 'status': 'ok'}
        print(f'  Recovered {key}: ARI={metrics[\"ARI\"]:.3f}')

results_path.write_text(json.dumps(existing, indent=2, default=str))
print('Done recovering.')
"

echo ""
echo "--- Stage 1: xenium_BC_leiden (remaining: self_leverage_k, seed_specificity) ---"
python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset xenium_BC_leiden \
    --methods self_leverage_k seed_specificity \
    --dim-red pca

echo ""
echo "--- Stage 2: CountSketch comparison on NPC ---"
python experiments/v3_reffree_leverage/run_reffree_test.py \
    --dataset visium_NPC \
    --methods pseudo_sig self_leverage_k \
    --dim-red sketch

echo ""
echo "=== All stages complete ==="
echo "Results: experiments/v3_reffree_leverage/reffree_results.json"
cat experiments/v3_reffree_leverage/reffree_results.json
