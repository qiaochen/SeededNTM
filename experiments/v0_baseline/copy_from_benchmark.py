#!/usr/bin/env python
"""Copy baseline proportions from the SeededNTM benchmark as v0_baseline.

Since the benchmark already ran SeedTopic with the exact same parameters and
code (the infer_seededntm binary points to this repo's seededntm package),
we use those saved results as our v0_baseline reference.

This avoids a redundant 30-minute GPU run. A fresh reproduction can be done
via run_baseline.sh if needed to verify exact reproducibility.

Usage:
    python experiments/v0_baseline/copy_from_benchmark.py
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.evaluate import (
    compute_metrics,
    compute_per_class_metrics,
    load_baseline_proportions,
    load_ground_truth,
    load_rctd_pixel_mask,
)

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(
    "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM"
    "/experiments/benchmark_deconv/raw_outputs"
)
OUTPUT_DIR = Path(__file__).parent / "raw_outputs"

DATASETS = ["visium_NPC", "xenium_BC", "visiumHD_CRC_I", "visiumHD_CRC_II"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        src_prop = BENCHMARK_DIR / ds_name / "seedtopic_proportions.csv"
        if not src_prop.exists():
            print(f"  SKIP: {src_prop} not found")
            continue

        # Copy proportions
        dst_prop = OUTPUT_DIR / f"{ds_name}_proportions.csv"
        shutil.copy2(src_prop, dst_prop)
        print(f"  Copied: {dst_prop}")

        # Compute metrics
        proportions, ct_names = load_baseline_proportions(ds_name, "seedtopic")
        gt_labels = load_ground_truth(ds_name)
        n = min(len(gt_labels), proportions.shape[0])
        metrics = compute_metrics(proportions[:n], ct_names, gt_labels[:n])
        metrics["version"] = "v0_baseline"
        metrics["dataset"] = ds_name

        print(f"  F1={metrics['f1']:.3f}  AUPRC={metrics['AUPRC']:.3f}  "
              f"ARI={metrics['ARI']:.3f}  AMI={metrics['AMI']:.3f}")

        per_class = compute_per_class_metrics(proportions[:n], ct_names, gt_labels[:n])
        per_class.to_csv(OUTPUT_DIR / f"{ds_name}_per_class_metrics.csv", index=False)

        # Subset metrics
        mask = load_rctd_pixel_mask(ds_name)
        subset_metrics = None
        if mask is not None and len(mask) == n:
            subset_metrics = compute_metrics(proportions[mask], ct_names, gt_labels[mask])
            subset_metrics["version"] = "v0_baseline"
            subset_metrics["dataset"] = f"{ds_name} (RCTD subset)"
            print(f"  Subset: F1={subset_metrics['f1']:.3f}  "
                  f"AUPRC={subset_metrics['AUPRC']:.3f}  ARI={subset_metrics['ARI']:.3f}")

        all_results[ds_name] = {
            "full": metrics,
            "subset": subset_metrics,
        }

    # Save summary
    summary_path = Path(__file__).parent / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
