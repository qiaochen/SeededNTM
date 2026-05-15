#!/usr/bin/env python
"""Run SeedTopic experiment and evaluate against baselines.

This script runs SeedTopic (via the infer_seededntm CLI) on one or more datasets,
saves the proportions as CSV, computes metrics, and appends results to RESULTS_LOG.md.

Usage:
    # Run baseline on all datasets:
    python experiments/run_experiment.py --version v0_baseline --datasets all

    # Run on a single dataset with custom args:
    python experiments/run_experiment.py --version v1_spatial_reg --datasets visium_NPC \
        --extra-args "--spatial_reg_lambda 0.01"

    # Evaluate existing proportions (skip SeedTopic run):
    python experiments/run_experiment.py --version v0_baseline --datasets all --eval-only

Environment:
    Requires the seededntm conda env with infer_seededntm installed.
    Default: /illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/infer_seededntm
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.evaluate import (
    compute_metrics,
    compute_per_class_metrics,
    load_baseline_proportions,
    load_ground_truth,
    load_rctd_pixel_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_PATH = EXPERIMENTS_DIR / "configs" / "datasets.json"

SEEDEDNTM_BIN = Path(
    "/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs"
    "/chatdna-prd/seededntm/bin/infer_seededntm"
)


def load_configs() -> dict:
    with open(CONFIGS_PATH) as f:
        return json.load(f)


def run_seedtopic(dataset_name: str, cfg: dict, output_dir: Path, extra_args: list = None) -> tuple:
    """Run SeedTopic CLI and return (proportions, cell_type_names, runtime_seconds)."""
    seeds_path = cfg["condition_feat_path"]
    with open(seeds_path) as f:
        marker_genes = json.load(f)
    idx_to_name = {v["topic_index"]: k for k, v in marker_genes.items()}
    ct_names = [idx_to_name[i] for i in range(len(marker_genes))]

    tmp_dir = str(output_dir / "seedtopic_output")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        str(SEEDEDNTM_BIN),
        "--adata_h5ad_path", cfg["seedtopic_adata_path"],
        "--condition_feat_path", seeds_path,
        "--key_input", "tfidf_pca",
        "--key_count_out", "rna_count",
        "--num_topics", str(cfg["num_topics"]),
        "--key_topic_prior", "topic_prior",
        "--reg_topic_prior", str(cfg["reg_topic_prior"]),
        "--wt_fusion_top_seed", str(cfg["wt_fusion_top_seed"]),
        "--batch_size", "8192",
        "--exp_outdir", tmp_dir,
    ] + cfg.get("extra_args", [])

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running SeedTopic on {dataset_name}: {' '.join(cmd[-10:])}")

    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    runtime = time.time() - t0

    if result.returncode != 0:
        logger.error(f"SeedTopic FAILED on {dataset_name} (exit {result.returncode})")
        logger.error(f"STDERR (last 2000 chars): {result.stderr[-2000:]}")
        raise RuntimeError(f"infer_seededntm failed: {result.stderr[-500:]}")

    df_topic = pd.read_csv(os.path.join(tmp_dir, "df_topic.csv"), index_col=0)
    logger.info(f"SeedTopic finished on {dataset_name}: shape={df_topic.shape}, runtime={runtime:.1f}s")

    return df_topic.values, ct_names, runtime


def evaluate_dataset(dataset_name: str, proportions: np.ndarray, ct_names: list, version: str, runtime: float = None) -> dict:
    """Evaluate proportions against ground truth and baselines."""
    gt_labels = load_ground_truth(dataset_name)

    if len(gt_labels) != proportions.shape[0]:
        logger.warning(
            f"Shape mismatch: GT has {len(gt_labels)} spots, proportions has {proportions.shape[0]}. "
            f"Truncating to min."
        )
        n = min(len(gt_labels), proportions.shape[0])
        gt_labels = gt_labels[:n]
        proportions = proportions[:n]

    metrics = compute_metrics(proportions, ct_names, gt_labels)
    metrics["version"] = version
    metrics["dataset"] = dataset_name
    metrics["runtime_s"] = f"{runtime:.1f}" if runtime else "?"

    logger.info(
        f"  {dataset_name}: F1={metrics['f1']:.3f} AUPRC={metrics['AUPRC']:.3f} "
        f"ARI={metrics['ARI']:.3f} AMI={metrics['AMI']:.3f}"
    )

    per_class = compute_per_class_metrics(proportions, ct_names, gt_labels)

    # Fair subset comparison (RCTD-retained spots)
    mask = load_rctd_pixel_mask(dataset_name)
    subset_metrics = None
    if mask is not None and len(mask) == proportions.shape[0]:
        subset_metrics = compute_metrics(proportions[mask], ct_names, gt_labels[mask])
        subset_metrics["version"] = version
        subset_metrics["dataset"] = f"{dataset_name} (RCTD subset)"
        logger.info(
            f"  {dataset_name} (subset): F1={subset_metrics['f1']:.3f} "
            f"AUPRC={subset_metrics['AUPRC']:.3f} ARI={subset_metrics['ARI']:.3f}"
        )

    return {
        "full": metrics,
        "subset": subset_metrics,
        "per_class": per_class,
    }


def append_results_log(metrics: dict, notes: str = ""):
    """Append metrics to RESULTS_LOG.md (append-only)."""
    log_path = EXPERIMENTS_DIR / "RESULTS_LOG.md"
    metrics["notes"] = notes

    row = (
        f"| {metrics.get('version', '?')} | {metrics.get('dataset', '?')} "
        f"| {metrics['f1']:.3f} | {metrics.get('precision', 0):.3f} | {metrics.get('recall', 0):.3f} "
        f"| {metrics.get('AUROC', 0):.3f} | {metrics['AUPRC']:.3f} "
        f"| {metrics['ARI']:.3f} | {metrics['AMI']:.3f} "
        f"| {metrics.get('runtime_s', '?')} | {notes} |\n"
    )

    with open(log_path, "a") as f:
        f.write(row)


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate SeedTopic experiments")
    parser.add_argument("--version", required=True, help="Version tag (e.g., v0_baseline, v1_spatial_reg)")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        help="Dataset names or 'all'. Choices: visium_NPC, xenium_BC, visiumHD_CRC_I, visiumHD_CRC_II")
    parser.add_argument("--extra-args", nargs="*", default=[],
                        help="Extra CLI arguments to pass to infer_seededntm")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip running SeedTopic, just evaluate existing proportions")
    parser.add_argument("--notes", default="", help="Notes to append to results log")
    args = parser.parse_args()

    configs = load_configs()

    if "all" in args.datasets:
        dataset_names = list(configs.keys())
    else:
        dataset_names = args.datasets

    version_dir = EXPERIMENTS_DIR / args.version
    os.makedirs(version_dir / "raw_outputs", exist_ok=True)

    all_results = {}

    for ds_name in dataset_names:
        if ds_name not in configs:
            logger.error(f"Unknown dataset: {ds_name}")
            continue

        cfg = configs[ds_name]
        output_dir = version_dir / "raw_outputs"
        prop_path = output_dir / f"{ds_name}_proportions.csv"

        if args.eval_only:
            if not prop_path.exists():
                logger.warning(f"No proportions file for {ds_name}, skipping")
                continue
            df = pd.read_csv(prop_path, index_col=0)
            proportions = df.values
            ct_names = df.columns.tolist()
            runtime = None
        else:
            proportions, ct_names, runtime = run_seedtopic(
                ds_name, cfg, version_dir, extra_args=args.extra_args
            )
            df = pd.DataFrame(proportions, columns=ct_names)
            df.to_csv(prop_path, index=False)
            logger.info(f"  Saved proportions: {prop_path}")

        results = evaluate_dataset(ds_name, proportions, ct_names, args.version, runtime)
        all_results[ds_name] = results

        append_results_log(results["full"], notes=args.notes)
        if results["subset"]:
            append_results_log(results["subset"], notes=args.notes + " (RCTD subset)")

        per_class_path = output_dir / f"{ds_name}_per_class_metrics.csv"
        results["per_class"].to_csv(per_class_path, index=False)

    # Save full results as JSON
    summary_path = version_dir / "metrics_summary.json"
    json_results = {}
    for ds_name, r in all_results.items():
        json_results[ds_name] = {
            "full": r["full"],
            "subset": r["subset"],
        }
    with open(summary_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    logger.info(f"Results saved: {summary_path}")


if __name__ == "__main__":
    main()
