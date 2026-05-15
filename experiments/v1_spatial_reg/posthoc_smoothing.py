#!/usr/bin/env python
"""Post-hoc spatial smoothing of SeedTopic proportions.

Applies a simple weighted average with k-NN spatial neighbors as post-processing.
This is more effective than training-time regularization because it doesn't
interfere with the ELBO optimization.

Usage:
    python experiments/v1_spatial_reg/posthoc_smoothing.py --alpha 0.3
    python experiments/v1_spatial_reg/posthoc_smoothing.py --alpha 0.2 --datasets xenium_BC
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.evaluate import (
    compute_metrics,
    compute_per_class_metrics,
    load_baseline_proportions,
    load_ground_truth,
    load_rctd_pixel_mask,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_ROOT / "experiments" / "configs" / "datasets.json"


def spatial_smooth(proportions, coords, alpha=0.3, k_neighbors=6):
    """Apply post-hoc spatial smoothing to topic proportions.

    theta_smooth = (1-alpha)*theta + alpha * mean(theta_neighbors)

    Args:
        proportions: (N, K) array.
        coords: (N, 2) spatial coordinates.
        alpha: smoothing strength in [0, 1]. 0 = no smoothing.
        k_neighbors: number of spatial neighbors.

    Returns:
        Smoothed (N, K) proportions (row-normalized).
    """
    A = kneighbors_graph(coords, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
    A = ((A + A.T) > 0).astype(np.float32)
    A_csr = csr_matrix(A)
    degrees = np.array(A_csr.sum(axis=1)).flatten()

    neighbor_avg = A_csr.dot(proportions) / degrees[:, np.newaxis]
    smoothed = (1 - alpha) * proportions + alpha * neighbor_avg
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Post-hoc spatial smoothing evaluation")
    parser.add_argument("--alpha", type=float, default=0.3, help="Smoothing strength")
    parser.add_argument("--k-neighbors", type=int, default=6, help="Spatial k-NN")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    args = parser.parse_args()

    with open(CONFIGS_PATH) as f:
        configs = json.load(f)

    if "all" in args.datasets:
        dataset_names = list(configs.keys())
    else:
        dataset_names = args.datasets

    results = {}
    print(f"\nPost-hoc Spatial Smoothing (alpha={args.alpha}, k={args.k_neighbors})")
    print("=" * 75)
    print(f"{'Dataset':<18} {'F1':<8} {'AUPRC':<8} {'ARI':<8} {'AMI':<8} {'delta_ARI':<10} {'delta_F1'}")
    print("-" * 75)

    for ds_name in dataset_names:
        cfg = configs[ds_name]
        proportions, ct_names = load_baseline_proportions(ds_name, "seedtopic")
        gt_labels = load_ground_truth(ds_name)

        adata = sc.read_h5ad(cfg["st_path"])
        if "spatial" not in adata.obsm:
            print(f"{ds_name:<18} SKIP (no spatial coords)")
            continue

        coords = adata.obsm["spatial"]
        n = min(len(gt_labels), proportions.shape[0], coords.shape[0])

        # Baseline
        metrics_bl = compute_metrics(proportions[:n], ct_names, gt_labels[:n])

        # Smoothed
        smoothed = spatial_smooth(proportions[:n], coords[:n], alpha=args.alpha, k_neighbors=args.k_neighbors)
        metrics = compute_metrics(smoothed, ct_names, gt_labels[:n])

        ari_diff = metrics["ARI"] - metrics_bl["ARI"]
        f1_diff = metrics["f1"] - metrics_bl["f1"]

        print(f"{ds_name:<18} {metrics['f1']:<8.3f} {metrics['AUPRC']:<8.3f} "
              f"{metrics['ARI']:<8.3f} {metrics['AMI']:<8.3f} "
              f"{'+' if ari_diff > 0 else ''}{ari_diff:<9.3f} "
              f"{'+' if f1_diff > 0 else ''}{f1_diff:.3f}")

        results[ds_name] = {
            "baseline": metrics_bl,
            "smoothed": metrics,
            "alpha": args.alpha,
            "k_neighbors": args.k_neighbors,
            "delta_ARI": ari_diff,
            "delta_F1": f1_diff,
        }

        # Save smoothed proportions
        out_dir = Path(__file__).parent / "raw_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(smoothed, columns=ct_names)
        df.to_csv(out_dir / f"{ds_name}_smoothed_alpha{args.alpha}.csv", index=False)

    # Save results
    out_path = Path(__file__).parent / f"posthoc_results_alpha{args.alpha}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
