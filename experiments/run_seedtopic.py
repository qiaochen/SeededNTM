#!/usr/bin/env python
"""Unified SeedTopic runner with auto-configuration.

Accepts any spatial transcriptomics dataset (h5ad + seed file) and automatically
selects optimal hyperparameters based on dataset characteristics. Reports all 6
metrics (AMI, ARI, F1, Precision, Recall, AUPRC) plus per-class F1 for rare types.

Usage:
    python experiments/run_seedtopic.py \\
        --adata /path/to/adata.h5ad \\
        --seeds /path/to/seeds.json \\
        --gt_dataset visium_NPC \\
        --outdir /path/to/output

    # Override auto-config:
    python experiments/run_seedtopic.py \\
        --adata /path/to/adata.h5ad \\
        --seeds /path/to/seeds.json \\
        --gt_dataset xenium_BC \\
        --reg 0.99 --prior_mode standard --K 19
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, str(Path(__file__).parent.parent))
from seededntm.util import (
    auto_configure,
    compute_topic_prior,
    compute_topic_prior_idf,
    spatial_smooth_prior,
    compute_reffree_leverage,
    rebalance_prior,
)
from seededntm.main import do_exp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Soft mask annealing (optional, off by default)
# ============================================================

_ANNEAL_CFG = {
    "anneal_start": 200,
    "mask_final": 1.0,
    "mask_initial": 10.0,
    "n_epochs": 800,
    "current_epoch": 0,
    "enabled": False,
}

_original_masked_fill = torch.Tensor.masked_fill

import seededntm.experiment as _exp_module
_original_train_step = _exp_module.train_step


def _get_mask_strength():
    epoch = _ANNEAL_CFG["current_epoch"]
    if epoch < _ANNEAL_CFG["anneal_start"]:
        return float('inf')
    progress = min(1.0, (epoch - _ANNEAL_CFG["anneal_start"]) / max(1, _ANNEAL_CFG["n_epochs"] - _ANNEAL_CFG["anneal_start"]))
    return _ANNEAL_CFG["mask_initial"] + (_ANNEAL_CFG["mask_final"] - _ANNEAL_CFG["mask_initial"]) * progress


def _soft_masked_fill(self_tensor, mask, value):
    if _ANNEAL_CFG["enabled"] and value == float('-inf') and not math.isinf(_get_mask_strength()):
        return _original_masked_fill(self_tensor, mask, -_get_mask_strength())
    return _original_masked_fill(self_tensor, mask, value)


def _epoch_tracking_train_step(dataloader, model, elbo, adam, clip_norm, params, reg_topic_prior, epoch):
    _ANNEAL_CFG["current_epoch"] = epoch
    return _original_train_step(dataloader, model, elbo, adam, clip_norm, params, reg_topic_prior, epoch)


# ============================================================
# Metrics
# ============================================================

def compute_metrics_hungarian(proportions, K, gt_labels, gt_types=None):
    """Full metrics via Hungarian matching for any K."""
    if gt_types is None:
        gt_types = sorted(set(gt_labels.tolist()))
    n_gt = len(gt_types)
    gt_type_to_idx = {t: i for i, t in enumerate(gt_types)}

    predicted_topic_idx = np.argmax(proportions, axis=1)
    gt_idx = np.array([gt_type_to_idx.get(g, -1) for g in gt_labels])
    valid = gt_idx >= 0
    predicted_topic_idx = predicted_topic_idx[valid]
    gt_idx = gt_idx[valid]
    proportions_valid = proportions[valid]

    cost_matrix = np.zeros((K, n_gt), dtype=np.float64)
    for k in range(K):
        mask_k = predicted_topic_idx == k
        if mask_k.sum() > 0:
            for g in range(n_gt):
                cost_matrix[k, g] = (gt_idx[mask_k] == g).sum()

    if K <= n_gt:
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        topic_to_gt = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        for k in range(K):
            if k not in topic_to_gt:
                topic_to_gt[k] = np.argmax(cost_matrix[k])
    else:
        topic_to_gt = {}
        topic_counts = cost_matrix.sum(axis=1)
        top_topics = np.argsort(-topic_counts)[:n_gt]
        sub_cost = cost_matrix[top_topics]
        row_ind, col_ind = linear_sum_assignment(-sub_cost)
        for i in range(len(row_ind)):
            topic_to_gt[top_topics[row_ind[i]]] = col_ind[i]
        for k in range(K):
            if k not in topic_to_gt:
                topic_to_gt[k] = np.argmax(cost_matrix[k])

    merged_proportions = np.zeros((len(proportions_valid), n_gt), dtype=np.float64)
    for k in range(K):
        merged_proportions[:, topic_to_gt[k]] += proportions_valid[:, k]
    row_sums = merged_proportions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    merged_proportions = merged_proportions / row_sums

    pred_gt_idx = np.argmax(merged_proportions, axis=1)

    metrics = {
        "f1": float(f1_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "precision": float(precision_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "recall": float(recall_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "ARI": float(adjusted_rand_score(gt_idx, pred_gt_idx)),
        "AMI": float(adjusted_mutual_info_score(gt_idx, pred_gt_idx)),
    }

    gt_onehot = np.zeros((len(gt_idx), n_gt), dtype=np.float64)
    gt_onehot[np.arange(len(gt_idx)), gt_idx] = 1.0
    try:
        metrics["AUPRC"] = float(average_precision_score(gt_onehot, merged_proportions, average="macro"))
    except ValueError:
        metrics["AUPRC"] = float("nan")

    # Per-class F1 for rare type analysis
    per_class_f1 = f1_score(gt_idx, pred_gt_idx, labels=list(range(n_gt)), average=None, zero_division=0)
    metrics["per_class_f1"] = {gt_types[i]: float(per_class_f1[i]) for i in range(n_gt)}

    return metrics


# ============================================================
# Representation computation
# ============================================================

def compute_representation(X_counts, var_names, seeds, representation, n_topics):
    """Compute input representation based on auto-config recommendation."""
    from sklearn.decomposition import PCA

    if issparse(X_counts):
        row_sums = np.asarray(X_counts.sum(axis=1)).ravel()
    else:
        row_sums = X_counts.sum(axis=1)
    row_sums[row_sums == 0] = 1

    if issparse(X_counts):
        tf = X_counts.multiply(1.0 / row_sums[:, np.newaxis])
    else:
        tf = X_counts / row_sums[:, np.newaxis]

    use_idf = "idf" in representation
    if use_idf:
        N = X_counts.shape[0]
        if issparse(X_counts):
            doc_freq = np.asarray((X_counts > 0).sum(axis=0)).ravel()
        else:
            doc_freq = (X_counts > 0).sum(axis=0)
        idf = np.log(N / (doc_freq + 1))
        if issparse(tf):
            weighted = tf.multiply(idf.reshape(1, -1))
        else:
            weighted = tf * idf.reshape(1, -1)
    else:
        weighted = tf

    use_lev = "lev" in representation
    if use_lev:
        if "pseudo_sig" in representation:
            lev_method = "pseudo_sig"
        else:
            lev_method = "seed_specificity"
        leverage = compute_reffree_leverage(
            X_counts, seeds, var_names, method=lev_method, n_topics=n_topics
        )
        lev = np.asarray(leverage, dtype=np.float32).ravel()
        if issparse(weighted):
            weighted = weighted.multiply(lev.reshape(1, -1))
        else:
            weighted = weighted * lev.reshape(1, -1)

    if issparse(weighted):
        weighted = weighted.toarray()
    weighted = np.asarray(weighted, dtype=np.float32)
    n_comp = min(100, weighted.shape[0] - 1, weighted.shape[1] - 1)
    rep = PCA(n_components=n_comp).fit_transform(weighted).astype(np.float32)
    return rep


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified SeedTopic runner with auto-configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to h5ad file")
    parser.add_argument("--seeds", type=str, required=True, help="Path to seed genes JSON")
    parser.add_argument("--gt_dataset", type=str, default=None,
                        help="Ground truth dataset name for evaluation (optional)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")

    # Overrides (optional, auto-configured if not provided)
    parser.add_argument("--reg", type=float, default=None, help="Override reg_topic_prior")
    parser.add_argument("--prior_mode", type=str, default=None,
                        choices=["standard", "spatial", "idf_spatial"])
    parser.add_argument("--representation", type=str, default=None,
                        help="Override representation (e.g. idf_pca, idf_lev_seed_specificity_pca)")
    parser.add_argument("--K", type=int, default=None, help="Override number of topics")
    parser.add_argument("--rebalance", action="store_true",
                        help="Apply prior marginal rebalancing")
    parser.add_argument("--soft_mask", action="store_true", help="Enable soft mask annealing")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading adata: {args.adata}")
    adata = sc.read_h5ad(args.adata)
    logger.info(f"  Shape: {adata.shape}")

    logger.info(f"Loading seeds: {args.seeds}")
    with open(args.seeds) as f:
        seeds = json.load(f)
    logger.info(f"  {len(seeds)} cell types")

    # Auto-configure
    logger.info("Running auto-configuration...")
    cfg = auto_configure(adata, seeds)
    diag = cfg["diagnostics"]
    logger.info(f"  seed_cov={diag['seed_cov']:.4f}, overlap={diag['overlap_ratio']:.3f}")
    logger.info(f"  platform={'targeted' if diag['targeted_panel'] else 'genome-wide'}")

    # Apply overrides
    reg = args.reg if args.reg is not None else cfg["reg_topic_prior"]
    prior_mode = args.prior_mode if args.prior_mode is not None else cfg["prior_mode"]
    representation = args.representation if args.representation is not None else cfg["representation"]
    K = args.K if args.K is not None else cfg["K"]
    soft_mask = args.soft_mask
    early_stop = not args.no_early_stop and cfg["early_stop"]

    logger.info(f"  Config: reg={reg}, prior_mode={prior_mode}, repr={representation}, K={K}")
    if args.rebalance:
        logger.info(f"  Prior rebalancing: ENABLED")

    # Output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(args.adata).parent / "seedtopic_output"
    outdir.mkdir(parents=True, exist_ok=True)

    # Compute representation
    X_counts = adata.X
    var_names_arr = np.array(adata.var_names)
    rep = compute_representation(X_counts, var_names_arr, seeds, representation, K)
    logger.info(f"  Representation shape: {rep.shape}")

    input_key = "seedtopic_input_rep"
    adata.obsm[input_key] = rep

    # Compute prior
    if prior_mode == "idf_spatial":
        raw_prior = compute_topic_prior_idf(adata, seeds, temperature=0.8)
        coords = np.asarray(adata.obsm['spatial'])
        topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=6)
    elif prior_mode == "spatial":
        raw_prior = compute_topic_prior(adata, seeds, temperature=0.8)
        coords = np.asarray(adata.obsm['spatial'])
        topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=6)
    else:
        topic_prior = compute_topic_prior(adata, seeds, temperature=0.8)

    # Apply rebalancing if requested
    if args.rebalance:
        topic_prior = rebalance_prior(topic_prior)
        logger.info(f"  Prior rebalanced: mean per topic = {topic_prior.mean(axis=0).round(4)}")

    # Handle multi-scale K
    n_seed_types = len(seeds)
    if K > n_seed_types:
        extended_seeds = dict(seeds)
        for extra_idx in range(n_seed_types, K):
            extended_seeds[f"_extra_topic_{extra_idx}"] = {
                "topic_index": extra_idx,
                "features": [],
            }
        seed_path = str(outdir / f"seeds_K{K}.json")
        with open(seed_path, "w") as f_s:
            json.dump(extended_seeds, f_s)
        n_extra = K - n_seed_types
        uniform_extra = np.full((topic_prior.shape[0], n_extra), 1.0 / K)
        extended_prior = np.hstack([topic_prior * (1 - n_extra / K), uniform_extra])
        extended_prior = extended_prior / extended_prior.sum(axis=1, keepdims=True)
        adata.obsm["topic_prior"] = extended_prior
    else:
        seed_path = args.seeds
        adata.obsm["topic_prior"] = topic_prior

    # Save prepared adata
    h5ad_path = outdir / "prepared_adata.h5ad"
    adata.write_h5ad(h5ad_path)

    exp_outdir = str(outdir / "model_output")
    os.makedirs(exp_outdir, exist_ok=True)

    # Setup soft mask
    if soft_mask:
        _ANNEAL_CFG["enabled"] = True
        _exp_module.train_step = _epoch_tracking_train_step
        torch.Tensor.masked_fill = _soft_masked_fill

    # Run
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Training SeedTopic (device={device})...")
    t0 = time.time()

    do_exp(
        adata_h5ad_path=str(h5ad_path),
        condition_feat_path=seed_path,
        key_input=input_key,
        key_count_out='rna_count',
        key_topic_prior='topic_prior',
        n_topics=K,
        reg_topic_prior=reg,
        wt_fusion_top_seed=1.0,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_nb_obs=True,
        early_stop=early_stop,
        exp_outdir=exp_outdir,
        device=device,
    )

    runtime = time.time() - t0
    logger.info(f"Training complete ({runtime:.0f}s)")

    # Restore soft mask
    if soft_mask:
        torch.Tensor.masked_fill = _original_masked_fill
        _exp_module.train_step = _original_train_step
        _ANNEAL_CFG["enabled"] = False

    # Load results
    df_topic_path = os.path.join(exp_outdir, "df_topic.csv")
    if not os.path.exists(df_topic_path):
        logger.error("No output (df_topic.csv missing). Exiting.")
        sys.exit(1)

    df_topic = pd.read_csv(df_topic_path, index_col=0)
    proportions = df_topic.values

    logger.info(f"Final config: reg={reg}, prior_mode={prior_mode}, repr={representation}")

    # Evaluate if GT available
    result = {
        "config": {
            "reg_topic_prior": reg,
            "prior_mode": prior_mode,
            "K": K,
            "soft_mask": soft_mask,
            "early_stop": early_stop,
            "representation": representation,
            "rebalance": args.rebalance,
            "use_nb_obs": True,
        },
        "auto_config": cfg,
        "runtime_s": round(runtime, 1),
    }

    if args.gt_dataset:
        try:
            from experiments.evaluate import load_ground_truth
            gt_labels = load_ground_truth(args.gt_dataset)
            gt_types = sorted(set(gt_labels.tolist()))
            n = min(len(gt_labels), proportions.shape[0])

            metrics = compute_metrics_hungarian(proportions[:n], K, gt_labels[:n], gt_types)
            result["metrics"] = metrics

            logger.info(f"\n{'='*60}")
            logger.info(f"RESULTS:")
            logger.info(f"  F1={metrics['f1']:.3f}  ARI={metrics['ARI']:.3f}  "
                        f"AMI={metrics['AMI']:.3f}  AUPRC={metrics['AUPRC']:.3f}")
            logger.info(f"  Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}")

            # Rare type analysis
            counts = Counter(gt_labels[:n].tolist())
            rare_threshold = 0.02
            rare_types = [t for t in gt_types if counts.get(t, 0) / n < rare_threshold]
            common_types = [t for t in gt_types if t not in rare_types]

            if rare_types:
                rare_f1 = np.mean([metrics["per_class_f1"].get(t, 0) for t in rare_types])
                common_f1 = np.mean([metrics["per_class_f1"].get(t, 0) for t in common_types])
                logger.info(f"\n  Common types (>{rare_threshold*100:.0f}%) avg F1: {common_f1:.3f}")
                logger.info(f"  Rare types (<{rare_threshold*100:.0f}%) avg F1: {rare_f1:.3f}")
                logger.info(f"  Per rare type:")
                for t in sorted(rare_types, key=lambda x: -metrics["per_class_f1"].get(x, 0)):
                    logger.info(f"    {t:<30s} F1={metrics['per_class_f1'][t]:.3f} "
                                f"(n={counts.get(t,0)}, {100*counts.get(t,0)/n:.1f}%)")
                result["rare_type_analysis"] = {
                    "rare_types": rare_types,
                    "rare_avg_f1": float(rare_f1),
                    "common_avg_f1": float(common_f1),
                }

            logger.info(f"{'='*60}")
        except Exception as e:
            logger.warning(f"Could not evaluate: {e}")

    # Save results
    results_path = outdir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
