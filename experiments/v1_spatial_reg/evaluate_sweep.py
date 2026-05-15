#!/usr/bin/env python
"""Evaluate spatial regularization sweep results on visium_NPC."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.evaluate import compute_metrics, load_ground_truth

SWEEP_DIR = Path(__file__).parent / "raw_outputs"
DATASET = "visium_NPC"

# Baseline reference
BASELINE = {"f1": 0.547, "AUPRC": 0.584, "ARI": 0.692, "AMI": 0.485}


def load_seedtopic_output(outdir):
    """Load df_topic.csv from SeedTopic output and extract proportions."""
    seeds_path = "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visium_NPC/NPC_topic_seeds_ref_scRNAseq.txt"
    with open(seeds_path) as f:
        marker_genes = json.load(f)
    idx_to_name = {v["topic_index"]: k for k, v in marker_genes.items()}
    ct_names = [idx_to_name[i] for i in range(len(marker_genes))]

    df = pd.read_csv(outdir / "df_topic.csv", index_col=0)
    return df.values, ct_names


def main():
    gt_labels = load_ground_truth(DATASET)

    print(f"\n{'='*70}")
    print(f"Spatial Regularization Sweep Results: {DATASET}")
    print(f"{'='*70}")
    print(f"\n{'Lambda':<10} {'F1':<8} {'AUPRC':<8} {'ARI':<8} {'AMI':<8} {'vs baseline'}")
    print(f"{'-'*70}")
    print(f"{'baseline':<10} {BASELINE['f1']:<8.3f} {BASELINE['AUPRC']:<8.3f} {BASELINE['ARI']:<8.3f} {BASELINE['AMI']:<8.3f} (reference)")

    results = {}
    for subdir in sorted(SWEEP_DIR.glob("npc_lambda_*")):
        lambda_val = subdir.name.replace("npc_lambda_", "")
        topic_path = subdir / "df_topic.csv"
        if not topic_path.exists():
            print(f"{lambda_val:<10} MISSING")
            continue

        proportions, ct_names = load_seedtopic_output(subdir)
        n = min(len(gt_labels), proportions.shape[0])
        metrics = compute_metrics(proportions[:n], ct_names, gt_labels[:n])

        ari_diff = metrics["ARI"] - BASELINE["ARI"]
        f1_diff = metrics["f1"] - BASELINE["f1"]
        indicator = "+" if ari_diff > 0 else ""

        print(f"{lambda_val:<10} {metrics['f1']:<8.3f} {metrics['AUPRC']:<8.3f} "
              f"{metrics['ARI']:<8.3f} {metrics['AMI']:<8.3f} "
              f"ARI {indicator}{ari_diff:.3f}, F1 {'+' if f1_diff > 0 else ''}{f1_diff:.3f}")

        results[lambda_val] = metrics

    # Save results
    out_path = Path(__file__).parent / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
