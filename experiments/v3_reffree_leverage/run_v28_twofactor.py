#!/usr/bin/env python
"""Two-factor adaptive blend: stability x max_prop.

Pushes CRC performance toward FlashDeconv/RCTD while keeping NPC/xenium non-degraded.
Two-factor weight: w_i = w_min + (w_max-w_min) * sig(g1*(stab-t1)) * sig(g2*(maxp-t2))

Usage:
    python experiments/v3_reffree_leverage/run_v28_twofactor.py --idx 0
"""

import argparse
import fcntl
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scipy.sparse import issparse, diags
from scipy.optimize import nnls as scipy_nnls
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as scipy_entropy
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score,
    average_precision_score, f1_score, precision_score, recall_score,
)
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from seededntm.util import compute_topic_prior, EarlyStopper
from experiments.run_seedtopic import compute_representation
from experiments.evaluate import load_ground_truth

import pyro
from pyro.infer import TraceMeanField_ELBO
from torch.utils.data import DataLoader
from tqdm import tqdm
import seededntm.experiment as _exp_module
from seededntm.model import SeededNTM
from seededntm.data import MyDataset
from seededntm.experiment import preprocess_ST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

RESULTS_PATH = Path(__file__).parent / "reffree_ablation_v28_twofactor.json"
RAW_OUTPUT_DIR = Path(__file__).parent / "raw_outputs_v28_twofactor"

DATASETS = {
    "visium_NPC": {
        "adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visium_NPC/spatial_adata.h5ad",
        "seeds_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visium_NPC/seeds.json",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visium_NPC/reference_subset.h5ad",
        "ref_celltype_key": "Cluster",
        "K": 7,
    },
    "xenium_BC": {
        "adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/xenium_BC/spatial_adata.h5ad",
        "seeds_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/xenium_BC/seeds.json",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/xenium_BC/reference_subset.h5ad",
        "ref_celltype_key": "Annotation",
        "K": 19,
    },
    "visiumHD_CRC_I": {
        "adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visiumHD_CRC_I/spatial_adata.h5ad",
        "seeds_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visiumHD_CRC_I/seeds.json",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visiumHD_CRC_I/reference_subset.h5ad",
        "ref_celltype_key": "Cluster",
        "K": 5,
    },
    "visiumHD_CRC_II": {
        "adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visiumHD_CRC_II/spatial_adata.h5ad",
        "seeds_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visiumHD_CRC_II/seeds.json",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeedTopic/experiments/benchmark_fair/data/processed/visiumHD_CRC_II/reference_subset.h5ad",
        "ref_celltype_key": "Cluster",
        "K": 6,
    },
}

REPRESENTATION = "idf_lev_seed_spec"
PRIOR_MODE = "standard"
BASE_ALPHA = 199
BASE_FLOOR = 0.8
TEMPERATURE = 0.8
N_EPOCHS = 800
BATCH_SIZE = 8192
LR = 0.01
ENC_HID_DIM = 64
CLIP_NORM = 5.0
POS_SCALE = 0.5
DROPOUT = 0.2
SEED = 0
WT_FUSION_TOP_SEED = 1.0
USE_NB_OBS = True

# ============================================================
# Gene selection
# ============================================================

def select_hvg(Y, n_top=2000):
    if issparse(Y):
        Y_dense = np.asarray(Y.todense(), dtype=np.float64)
    else:
        Y_dense = np.asarray(Y, dtype=np.float64)
    means = Y_dense.mean(axis=0)
    variances = Y_dense.var(axis=0)
    dispersions = np.zeros_like(means)
    valid = means > 0
    dispersions[valid] = variances[valid] / means[valid]
    return np.argsort(dispersions)[::-1][:n_top]


def select_markers(X_norm, n_markers=50):
    K_ref = X_norm.shape[0]
    all_markers = []
    for k in range(K_ref):
        if K_ref > 1:
            others = np.delete(X_norm, k, axis=0)
            fold = X_norm[k] / (others.max(axis=0) + 1e-10)
            markers_k = np.argsort(fold)[::-1][:n_markers]
        else:
            markers_k = np.argsort(X_norm[k])[::-1][:n_markers]
        all_markers.extend(markers_k)
    return np.unique(all_markers)


# ============================================================
# NNLS with stability and max_prop
# ============================================================

def compute_nnls_with_stability_and_sharpness(adata, adata_ref, ref_celltype_key, seeds,
                                               n_splits=3, seed_gene_mode=None):
    """Compute NNLS proportions, per-spot split-half stability, and max_prop.

    Args:
        seed_gene_mode: None (default, backward compat), "include" (add seed genes
            to gene selection), or "weighted" (add + sqrt(2) weight multiplier).

    Returns (nnls_aligned, stability, max_prop) where max_prop is from
    the FINAL normalized aligned proportions (post-smoothing).
    """
    spatial_genes = np.array(adata.var_names)
    ref_genes = np.array(adata_ref.var_names)
    common_genes = np.intersect1d(spatial_genes, ref_genes)
    logger.info(f"  NNLS: {len(common_genes)} common genes")

    spatial_lookup = {g: i for i, g in enumerate(spatial_genes)}
    ref_lookup = {g: i for i, g in enumerate(ref_genes)}
    spatial_idx = np.array([spatial_lookup[g] for g in common_genes])
    ref_idx = np.array([ref_lookup[g] for g in common_genes])

    Y_raw = adata.X[:, spatial_idx]
    expr_ref = adata_ref.X[:, ref_idx]

    cell_types = np.array(adata_ref.obs[ref_celltype_key])
    unique_types = sorted(np.unique(cell_types).tolist())
    K_ref = len(unique_types)
    n_common = len(common_genes)

    # ASSERTION: all seed types must exist in reference
    missing = [ct for ct in seeds.keys() if ct not in unique_types]
    assert not missing, f"Seed types missing from reference: {missing}"

    X_sig = np.zeros((K_ref, n_common), dtype=np.float64)
    for i, ct in enumerate(unique_types):
        mask = cell_types == ct
        subset = expr_ref[mask]
        if issparse(subset):
            X_sig[i] = np.asarray(subset.mean(axis=0)).ravel()
        else:
            X_sig[i] = np.mean(subset, axis=0)

    gene_idx = np.union1d(select_hvg(Y_raw, n_top=2000), select_markers(X_sig, n_markers=50))

    # Seed-gene inclusion for dslevel_seednnls
    seed_in_common = np.array([], dtype=np.intp)
    if seed_gene_mode in ("include", "weighted"):
        seed_gene_names = set()
        for ct_info in seeds.values():
            seed_gene_names.update(ct_info.get("features", []))
        seed_in_common = np.array([i for i, g in enumerate(common_genes) if g in seed_gene_names],
                                  dtype=np.intp)
        if len(seed_in_common) > 0:
            n_before = len(gene_idx)
            gene_idx = np.union1d(gene_idx, seed_in_common)
            logger.info(f"  NNLS seed-gene mode={seed_gene_mode}: "
                        f"added {len(gene_idx) - n_before} new seed genes (total now {len(gene_idx)})")

    Y_sub = Y_raw[:, gene_idx]
    X_sub = X_sig[:, gene_idx]
    G = len(gene_idx)
    logger.info(f"  NNLS: selected {G} informative genes (G/K={G/K_ref:.0f})")

    # log-CPM preprocessing
    if issparse(Y_sub):
        lib_size = np.array(Y_sub.sum(axis=1)).flatten()
        lib_size[lib_size == 0] = 1.0
        Y_norm = np.asarray((diags(1e4 / lib_size) @ Y_sub).todense())
        Y_norm = np.log1p(Y_norm)
    else:
        Y_dense = np.asarray(Y_sub, dtype=np.float64)
        ls = Y_dense.sum(axis=1, keepdims=True)
        ls[ls == 0] = 1.0
        Y_norm = np.log1p(Y_dense / ls * 1e4)

    X_norm = np.log1p(X_sub / (X_sub.sum(axis=1, keepdims=True) + 1e-10) * 1e4)

    # Full NNLS solve
    N = Y_norm.shape[0]
    beta = np.zeros((N, K_ref), dtype=np.float64)
    A_full = X_norm.T  # shape (G, K_ref)

    # For weighted mode, apply sqrt(2) multiplier to seed gene rows
    if seed_gene_mode == "weighted" and len(seed_in_common) > 0:
        seed_mask_in_selected = np.isin(gene_idx, seed_in_common)
        gene_weights = np.ones(G, dtype=np.float64)
        gene_weights[seed_mask_in_selected] = np.sqrt(2.0)
        A_weighted = A_full * gene_weights[:, None]  # broadcast: (G,1) * (G,K)
        for i in range(N):
            b_weighted = Y_norm[i] * gene_weights
            x, _ = scipy_nnls(A_weighted, b_weighted)
            beta[i] = x
            if i % 10000 == 0 and i > 0:
                logger.info(f"    NNLS weighted solve progress: {i}/{N}")
    else:
        for i in range(N):
            x, _ = scipy_nnls(A_full, Y_norm[i])
            beta[i] = x
            if i % 10000 == 0 and i > 0:
                logger.info(f"    NNLS full solve progress: {i}/{N}")

    # Split-half stability
    rng = np.random.RandomState(42)
    stability_accum = np.zeros(N, dtype=np.float64)

    for split_idx in range(n_splits):
        perm = rng.permutation(G)
        half1 = perm[:G // 2]
        half2 = perm[G // 2:]
        A1 = X_norm[:, half1].T
        A2 = X_norm[:, half2].T

        for i in range(N):
            x1, _ = scipy_nnls(A1, Y_norm[i, half1])
            x2, _ = scipy_nnls(A2, Y_norm[i, half2])
            s1, s2 = x1.sum(), x2.sum()
            p1 = x1 / s1 if s1 > 1e-10 else np.ones(K_ref) / K_ref
            p2 = x2 / s2 if s2 > 1e-10 else np.ones(K_ref) / K_ref
            dot = np.dot(p1, p2)
            norm1 = np.sqrt(np.dot(p1, p1))
            norm2 = np.sqrt(np.dot(p2, p2))
            stability_accum[i] += dot / (norm1 * norm2 + 1e-10)

        logger.info(f"    Split-half {split_idx+1}/{n_splits} done")

    stability = (stability_accum / n_splits).astype(np.float64)
    logger.info(f"  Stability: mean={stability.mean():.4f}, median={np.median(stability):.4f}")

    # Spatial smoothing
    coords = None
    if "spatial" in adata.obsm:
        coords = np.array(adata.obsm["spatial"])
    elif "X_spatial" in adata.obsm:
        coords = np.array(adata.obsm["X_spatial"])
    if coords is not None:
        nn = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        neighbor_idx = indices[:, 1:]
        for _ in range(3):
            neighbor_mean = beta[neighbor_idx].mean(axis=1)
            beta = 0.7 * beta + 0.3 * neighbor_mean

    row_sums = np.maximum(beta.sum(axis=1, keepdims=True), 1e-10)
    nnls_props = beta / row_sums

    # Reorder to seed topic indices
    K_total = len(seeds)
    nnls_aligned = np.full((N, K_total), 1.0 / K_total, dtype=np.float32)
    for ct_name, record in seeds.items():
        topic_idx = record["topic_index"]
        if ct_name in unique_types:
            src_idx = unique_types.index(ct_name)
            nnls_aligned[:, topic_idx] = nnls_props[:, src_idx]
    row_sums = nnls_aligned.sum(axis=1, keepdims=True)
    mask_valid = (row_sums.squeeze() > 1e-10)
    nnls_aligned[mask_valid] = nnls_aligned[mask_valid] / row_sums[mask_valid]
    nnls_aligned[~mask_valid] = 1.0 / K_total

    # max_prop from FINAL normalized aligned proportions
    max_prop = np.max(nnls_aligned, axis=1).astype(np.float64)
    logger.info(f"  max_prop: mean={max_prop.mean():.4f}, median={np.median(max_prop):.4f}")

    return nnls_aligned, stability, max_prop


# ============================================================
# Adaptive weight functions
# ============================================================

def compute_adaptive_weights_stability(stability, gamma, tau, w_min=0.05, w_max=0.95):
    """Single-factor: stability-only (control)."""
    logit = gamma * (stability - tau)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    return (w_min + (w_max - w_min) * sigmoid).astype(np.float32)


def compute_adaptive_weights_twofactor(stability, max_prop, g1, t1, g2, t2,
                                       w_min=0.05, w_max=0.95):
    """Two-factor: stability x max_prop."""
    logit1 = g1 * (stability - t1)
    logit2 = g2 * (max_prop - t2)
    sig1 = 1.0 / (1.0 + np.exp(-np.clip(logit1, -50, 50)))
    sig2 = 1.0 / (1.0 + np.exp(-np.clip(logit2, -50, 50)))
    return (w_min + (w_max - w_min) * sig1 * sig2).astype(np.float32)


# ============================================================
# Full 10-metric evaluation
# ============================================================

def compute_all_metrics(proportions, K, gt_labels):
    """Compute all 10 metrics via Hungarian matching.

    Returns dict with: F1, Precision, Recall, AMI, ARI, AUPRC, Pearson, Spearman, RMSE, JSD
    """
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

    merged = np.zeros((len(proportions_valid), n_gt), dtype=np.float64)
    for k in range(K):
        merged[:, topic_to_gt[k]] += proportions_valid[:, k]
    row_sums = merged.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    merged = merged / row_sums

    pred_gt_idx = np.argmax(merged, axis=1)

    metrics = {
        "F1": float(f1_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "Precision": float(precision_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "Recall": float(recall_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "ARI": float(adjusted_rand_score(gt_idx, pred_gt_idx)),
        "AMI": float(adjusted_mutual_info_score(gt_idx, pred_gt_idx)),
    }

    gt_onehot = np.zeros((len(gt_idx), n_gt), dtype=np.float64)
    gt_onehot[np.arange(len(gt_idx)), gt_idx] = 1.0
    try:
        metrics["AUPRC"] = float(average_precision_score(gt_onehot, merged, average="macro"))
    except ValueError:
        metrics["AUPRC"] = float("nan")

    prop_flat = merged.ravel()
    gt_flat = gt_onehot.ravel()
    try:
        metrics["Pearson"] = float(pearsonr(prop_flat, gt_flat)[0])
    except (ValueError, FloatingPointError):
        metrics["Pearson"] = float("nan")
    try:
        metrics["Spearman"] = float(spearmanr(prop_flat, gt_flat)[0])
    except (ValueError, FloatingPointError):
        metrics["Spearman"] = float("nan")

    metrics["RMSE"] = float(np.sqrt(np.mean((merged - gt_onehot) ** 2)))

    jsd_vals = []
    for i in range(len(gt_idx)):
        p = gt_onehot[i]
        q = merged[i]
        q_safe = np.clip(q, 1e-10, None)
        q_safe = q_safe / q_safe.sum()
        jsd_vals.append(jensenshannon(p, q_safe) ** 2)
    metrics["JSD"] = float(np.mean(jsd_vals))

    return metrics


# ============================================================
# Experiment Configs (12 per dataset, 48 total)
# ============================================================

CONFIGS_PER_DATASET = [
    {"method": "baseline", "strategy": "none"},
    {"method": "adapt_g10_t95", "strategy": "adaptive_stab", "gamma": 10.0, "tau": 0.95},
    {"method": "fixed_w50", "strategy": "fixed", "blend_w": 0.5},
    {"method": "twofactor_A", "strategy": "twofactor", "g1": 15.0, "t1": 0.85, "g2": 10.0, "t2": 0.70},
    {"method": "twofactor_B", "strategy": "twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.70},
    {"method": "twofactor_C", "strategy": "twofactor", "g1": 20.0, "t1": 0.85, "g2": 8.0, "t2": 0.65},
    {"method": "twofactor_D", "strategy": "twofactor", "g1": 25.0, "t1": 0.80, "g2": 10.0, "t2": 0.65},
    {"method": "twofactor_E", "strategy": "twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.60},
    {"method": "twofactor_F", "strategy": "twofactor", "g1": 15.0, "t1": 0.85, "g2": 12.0, "t2": 0.65},
    {"method": "twofactor_G", "strategy": "twofactor", "g1": 20.0, "t1": 0.85, "g2": 10.0, "t2": 0.65},
    {"method": "twofactor_H", "strategy": "twofactor", "g1": 15.0, "t1": 0.80, "g2": 10.0, "t2": 0.65},
    {"method": "hybrid_twofactor", "strategy": "hybrid_twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.70, "blend_alpha": 0.5},
    {"method": "twofactor_E_wmax55", "strategy": "twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.60, "w_max": 0.55},
    {"method": "twofactor_E_wmax60", "strategy": "twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.60, "w_max": 0.60},
    {"method": "twofactor_E_wmax65", "strategy": "twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.60, "w_max": 0.65},
    {"method": "twofactor_E_wmax70", "strategy": "twofactor", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.60, "w_max": 0.70},
    {"method": "dslevel_A", "strategy": "dslevel", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95},
    {"method": "dslevel_B", "strategy": "dslevel", "g1": 20.0, "t1": 0.80, "g2": 15.0, "t2": 0.85, "w_max": 0.95},
    {"method": "dslevel_C", "strategy": "dslevel", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.80, "w_max": 0.70},
    {"method": "dslevel_D", "strategy": "dslevel", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.70},
    {"method": "dslevel_E", "strategy": "dslevel", "g1": 20.0, "t1": 0.80, "g2": 20.0, "t2": 0.85, "w_max": 0.95},
    {"method": "dslevel_F", "strategy": "dslevel", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.82, "w_max": 0.80},
    {"method": "dslevel_sharp_A", "strategy": "dslevel_sharp", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "sharpen_T": 0.7, "sharpen_w_thresh": 0.3},
    {"method": "dslevel_sharp_B", "strategy": "dslevel_sharp", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "sharpen_T": 0.5, "sharpen_w_thresh": 0.3},
    {"method": "dslevel_sharp_C", "strategy": "dslevel_sharp", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "sharpen_T": 0.3, "sharpen_w_thresh": 0.3},
    {"method": "dslevel_sharp_D", "strategy": "dslevel_sharp", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "sharpen_T": 0.7, "sharpen_w_thresh": 0.0},
    {"method": "dslevel_sharp_E", "strategy": "dslevel_sharp", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "sharpen_T": 0.5, "sharpen_w_thresh": 0.0},
    {"method": "dslevel_sharp_F", "strategy": "dslevel_sharp", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "sharpen_T": 0.8, "sharpen_w_thresh": 0.3},
    # Phase 10: entropy gating
    {"method": "dslevel_entgate_A", "strategy": "dslevel_entgate", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "gamma_ent": 10.0, "H_thresh": 0.5},
    {"method": "dslevel_entgate_B", "strategy": "dslevel_entgate", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "gamma_ent": 10.0, "H_thresh": 0.3},
    {"method": "dslevel_entgate_C", "strategy": "dslevel_entgate", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "gamma_ent": 5.0, "H_thresh": 0.5},
    {"method": "dslevel_entgate_D", "strategy": "dslevel_entgate", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "gamma_ent": 5.0, "H_thresh": 0.3},
    # Phase 10: seed-gene NNLS
    {"method": "dslevel_seednnls_A", "strategy": "dslevel_seednnls", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "seed_gene_mode": "include"},
    {"method": "dslevel_seednnls_B", "strategy": "dslevel_seednnls", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "seed_gene_mode": "weighted"},
    # Phase 10: post-blend smoothing
    {"method": "dslevel_postsmooth_A", "strategy": "dslevel_postsmooth", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "iter_post": 1, "alpha_post": 0.85},
    {"method": "dslevel_postsmooth_B", "strategy": "dslevel_postsmooth", "g1": 20.0, "t1": 0.80, "g2": 10.0, "t2": 0.85, "w_max": 0.95, "iter_post": 2, "alpha_post": 0.90},
]


def build_experiment_matrix():
    configs = []
    for ds_name in DATASETS:
        for cfg in CONFIGS_PER_DATASET:
            exp = dict(cfg)
            exp["dataset"] = ds_name
            configs.append(exp)
    return configs


EXPERIMENT_MATRIX = build_experiment_matrix()


# ============================================================
# IndexedDataset wrapper
# ============================================================

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["indices"] = idx
        return item


# ============================================================
# Training loop
# ============================================================

def run_training(exp_data, K, alpha_np, device, early_stop=False):
    """Standard SeedTopic training with canonical hyperparameters."""
    pyro.clear_param_store()
    _exp_module.seed_everything(SEED)
    pyro.set_rng_seed(SEED)

    elbo = TraceMeanField_ELBO(num_particles=1)
    n_batches = 1
    if exp_data.batch_labels is not None:
        n_batches = np.unique(exp_data.batch_labels).shape[0]

    model = SeededNTM(
        input_dim=exp_data.input_rep.shape[1],
        out_dim_rna=exp_data.out_counts.shape[1] if exp_data.out_counts is not None else 0,
        out_dim_normal=exp_data.out_normal.shape[1] if exp_data.out_normal is not None else 0,
        init_bg_count=exp_data.init_bg_rna,
        init_bg_normal=exp_data.init_bg_gau,
        condition_mask=exp_data.condition_mask,
        n_topics=K,
        enc_hid_dim=ENC_HID_DIM,
        clamp_logvar_max=None,
        scale_normal_feat=-1,
        wt_fusion_top_seed=WT_FUSION_TOP_SEED,
        is_group_mode=False,
        pos_scale=POS_SCALE,
        device=device,
        n_batches=n_batches,
        use_nb_obs=USE_NB_OBS,
    )
    model.to(device)

    dataset = MyDataset(
        input=exp_data.input_rep,
        out_counts=exp_data.out_counts,
        out_normal=exp_data.out_normal,
        batch_labels=exp_data.batch_labels,
        topic_prior=exp_data.topic_prior,
    )
    indexed_dataset = IndexedDataset(dataset)

    init_loader = DataLoader(dataset=indexed_dataset, batch_size=2, shuffle=False)
    for batch_data in init_loader:
        elbo.differentiable_loss(
            model.model, model.guide,
            **{key: val.to(device) for key, val in batch_data.items()
               if key in {"input", "out_counts", "out_normal", "batch_labels"}}
        )
        break

    params = [v for _, v in pyro.get_param_store().named_parameters()]
    adam = torch.optim.AdamW(params, lr=LR, betas=(0.90, 0.999))
    alpha_tensor = torch.FloatTensor(alpha_np).to(device)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(dataset=indexed_dataset, batch_size=BATCH_SIZE, generator=g, shuffle=True)

    es = EarlyStopper(20)
    p_bar = tqdm(range(N_EPOCHS), desc="Training")

    for epoch in p_bar:
        epoch_losses = []
        for batch_data in train_loader:
            bs = batch_data["input"].shape[0]
            elbo_loss = elbo.differentiable_loss(
                model.model, model.guide,
                **{key: val.to(device) for key, val in batch_data.items()
                   if key in {"input", "out_counts", "out_normal", "batch_labels"}}
            ) / bs

            prior_loss = torch.zeros(1, device=device)
            if "topic_prior" in batch_data and "indices" in batch_data:
                idx = batch_data["indices"]
                batch_alpha = alpha_tensor[idx]
                tp = batch_data["topic_prior"].to(device)
                logtheta = model.encode(
                    **{k: v.to(device) for k, v in batch_data.items()
                       if k in {"input", "batch_labels"}})
                theta = F.softmax(logtheta, dim=-1)
                valid = tp.sum(dim=1) > 0.5
                if valid.sum() > 0:
                    ce = -(tp[valid] * torch.log(theta[valid] + 1e-8)).sum(dim=-1)
                    prior_loss = (batch_alpha[valid] * ce).mean()

            loss = elbo_loss + prior_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            adam.step()
            adam.zero_grad()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        p_bar.set_description(f"loss={mean_loss:.4f}")

        if early_stop and epoch > 100 and es.early_stop(mean_loss):
            logger.info("Early Stopping")
            break

    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    topics = []
    for batch_data in loader:
        model.eval()
        with torch.no_grad():
            theta = model.infer_topic(
                **{k: v.to(device) for k, v in batch_data.items()
                   if k in {"input", "batch_labels"}})
            topics.append(theta.cpu())
    topics = torch.concat(topics, dim=0).numpy()
    return topics


# ============================================================
# Single Run
# ============================================================

def run_single(exp_cfg, device="cuda:0"):
    """Run a single two-factor experiment."""
    ds_name = exp_cfg["dataset"]
    ds = DATASETS[ds_name]
    method = exp_cfg["method"]
    strategy = exp_cfg["strategy"]
    K = ds["K"]

    logger.info(f"=== {method} on {ds_name} (strategy={strategy}) ===")

    adata = sc.read_h5ad(ds["adata_path"])
    with open(ds["seeds_path"]) as f:
        seeds = json.load(f)

    X_counts = adata.X
    var_names = np.array(adata.var_names)

    nnls_prior = None
    stability = None
    max_prop = None
    if strategy in ("fixed", "adaptive_stab", "twofactor", "hybrid_twofactor",
                    "dslevel", "dslevel_sharp", "dslevel_entgate", "dslevel_seednnls", "dslevel_postsmooth"):
        adata_ref = sc.read_h5ad(ds["ref_path"])
        logger.info(f"  Reference: {adata_ref.shape}")
        seed_gene_mode = exp_cfg.get("seed_gene_mode", None)
        nnls_prior, stability, max_prop = compute_nnls_with_stability_and_sharpness(
            adata, adata_ref, ds["ref_celltype_key"], seeds, seed_gene_mode=seed_gene_mode)
        logger.info(f"  NNLS prior: shape={nnls_prior.shape}")
        del adata_ref
        gc.collect()

    # Input representation
    rep = compute_representation(X_counts, var_names, seeds, REPRESENTATION, K)
    input_key = "seedtopic_input_rep"
    adata.obsm[input_key] = rep

    if "rna_count" not in adata.obsm:
        if issparse(X_counts):
            adata.obsm["rna_count"] = np.asarray(X_counts.todense(), dtype=np.float32)
        else:
            adata.obsm["rna_count"] = np.asarray(X_counts, dtype=np.float32)

    # Seed-based topic prior
    seed_prior = compute_topic_prior(adata, seeds, temperature=TEMPERATURE)

    # For hybrid: blend prior
    if strategy == "hybrid_twofactor":
        blend_alpha = exp_cfg.get("blend_alpha", 0.5)
        topic_prior = blend_alpha * nnls_prior + (1.0 - blend_alpha) * seed_prior
        tp_sums = topic_prior.sum(axis=1, keepdims=True)
        topic_prior = topic_prior / np.clip(tp_sums, 1e-10, None)
    else:
        topic_prior = seed_prior

    adata.obsm["topic_prior"] = topic_prior.astype(np.float32)

    # Per-spot alpha
    per_spot_ent = scipy_entropy(topic_prior, axis=1)
    max_ent = np.log(K)
    confidence = np.clip(1.0 - per_spot_ent / max_ent, 0.0, 1.0)
    alpha_np = (BASE_ALPHA * (BASE_FLOOR + (1.0 - BASE_FLOOR) * confidence)).astype(np.float32)

    # Save seeds for preprocess_ST
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_seeds_path = RAW_OUTPUT_DIR / f"tmp_seeds_{ds_name}_{method}.json"
    with open(tmp_seeds_path, "w") as f:
        json.dump(seeds, f)

    if issparse(X_counts):
        out_counts = np.asarray(X_counts.todense(), dtype=np.float32)
    else:
        out_counts = np.asarray(X_counts, dtype=np.float32)

    exp_data = preprocess_ST(
        adata=adata,
        n_topics=K,
        input_rep=adata.obsm[input_key],
        out_counts=out_counts,
        out_normal=None,
        condition_feat_path=str(tmp_seeds_path),
        topic_prior=topic_prior.astype(np.float32),
    )

    early_stop = (ds_name == "xenium_BC")
    dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    t_start = time.time()
    theta = run_training(exp_data, K, alpha_np, dev, early_stop=early_stop)

    # Post-hoc blend
    w_per_spot = None
    if strategy == "fixed":
        w = exp_cfg["blend_w"]
        proportions = (1 - w) * theta + w * nnls_prior
    elif strategy == "adaptive_stab":
        gamma = exp_cfg["gamma"]
        tau = exp_cfg["tau"]
        w_per_spot = compute_adaptive_weights_stability(stability, gamma, tau)
        w_2d = w_per_spot[:, None]
        proportions = (1.0 - w_2d) * theta + w_2d * nnls_prior
    elif strategy in ("twofactor", "hybrid_twofactor"):
        g1, t1 = exp_cfg["g1"], exp_cfg["t1"]
        g2, t2 = exp_cfg["g2"], exp_cfg["t2"]
        cfg_w_max = exp_cfg.get("w_max", 0.95)
        w_per_spot = compute_adaptive_weights_twofactor(stability, max_prop, g1, t1, g2, t2, w_max=cfg_w_max)
        logger.info(f"  TwoFactor weights: mean={w_per_spot.mean():.3f}, "
                    f"std={w_per_spot.std():.3f}, "
                    f"min={w_per_spot.min():.3f}, max={w_per_spot.max():.3f}")
        w_2d = w_per_spot[:, None]
        proportions = (1.0 - w_2d) * theta + w_2d * nnls_prior
    elif strategy == "dslevel":
        g1, t1 = exp_cfg["g1"], exp_cfg["t1"]
        g2, t2 = exp_cfg["g2"], exp_cfg["t2"]
        cfg_w_max = exp_cfg.get("w_max", 0.95)
        mean_stab = float(np.mean(stability))
        mean_maxp = float(np.mean(max_prop))
        w_scalar = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), g1, t1, g2, t2, w_max=cfg_w_max)[0]
        logger.info(f"  Dataset-level weight: w={w_scalar:.4f} "
                    f"(mean_stab={mean_stab:.4f}, mean_maxp={mean_maxp:.4f})")
        w_per_spot = np.full(len(stability), w_scalar, dtype=np.float32)
        proportions = (1.0 - w_scalar) * theta + w_scalar * nnls_prior
    elif strategy == "dslevel_sharp":
        g1, t1 = exp_cfg["g1"], exp_cfg["t1"]
        g2, t2 = exp_cfg["g2"], exp_cfg["t2"]
        cfg_w_max = exp_cfg.get("w_max", 0.95)
        mean_stab = float(np.mean(stability))
        mean_maxp = float(np.mean(max_prop))
        w_scalar = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), g1, t1, g2, t2, w_max=cfg_w_max)[0]
        logger.info(f"  Dataset-level weight: w={w_scalar:.4f} "
                    f"(mean_stab={mean_stab:.4f}, mean_maxp={mean_maxp:.4f})")
        w_per_spot = np.full(len(stability), w_scalar, dtype=np.float32)
        proportions = (1.0 - w_scalar) * theta + w_scalar * nnls_prior
        sharpen_T = exp_cfg.get("sharpen_T", 1.0)
        sharpen_w_thresh = exp_cfg.get("sharpen_w_thresh", 0.3)
        if w_scalar > sharpen_w_thresh and sharpen_T < 1.0:
            proportions = np.clip(proportions, 1e-10, None)
            proportions = proportions ** (1.0 / sharpen_T)
            proportions = proportions / proportions.sum(axis=1, keepdims=True)
            logger.info(f"  Sharpening applied: T={sharpen_T}, exponent={1.0/sharpen_T:.2f}")
        else:
            logger.info(f"  Sharpening skipped: w_scalar={w_scalar:.4f} <= thresh={sharpen_w_thresh}")
    elif strategy == "dslevel_entgate":
        g1, t1 = exp_cfg["g1"], exp_cfg["t1"]
        g2, t2 = exp_cfg["g2"], exp_cfg["t2"]
        cfg_w_max = exp_cfg.get("w_max", 0.95)
        mean_stab = float(np.mean(stability))
        mean_maxp = float(np.mean(max_prop))
        w_scalar = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), g1, t1, g2, t2, w_max=cfg_w_max)[0]
        logger.info(f"  Dataset-level weight: w={w_scalar:.4f} "
                    f"(mean_stab={mean_stab:.4f}, mean_maxp={mean_maxp:.4f})")
        # Entropy gating: reduce w where theta is confident (low entropy)
        gamma_ent = exp_cfg["gamma_ent"]
        H_thresh = exp_cfg["H_thresh"]
        H = -np.sum(theta * np.log(theta + 1e-10), axis=1)
        H_norm = H / np.log(K)
        gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - H_thresh)))
        w_per_spot = (w_scalar * gate).astype(np.float32)
        logger.info(f"  Entropy gating: gamma_ent={gamma_ent}, H_thresh={H_thresh}, "
                    f"gate_mean={gate.mean():.3f}, effective_w_mean={w_per_spot.mean():.4f}")
        w_2d = w_per_spot[:, None]
        proportions = (1.0 - w_2d) * theta + w_2d * nnls_prior
    elif strategy == "dslevel_seednnls":
        g1, t1 = exp_cfg["g1"], exp_cfg["t1"]
        g2, t2 = exp_cfg["g2"], exp_cfg["t2"]
        cfg_w_max = exp_cfg.get("w_max", 0.95)
        mean_stab = float(np.mean(stability))
        mean_maxp = float(np.mean(max_prop))
        w_scalar = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), g1, t1, g2, t2, w_max=cfg_w_max)[0]
        logger.info(f"  Dataset-level weight: w={w_scalar:.4f} "
                    f"(mean_stab={mean_stab:.4f}, mean_maxp={mean_maxp:.4f})")
        w_per_spot = np.full(len(stability), w_scalar, dtype=np.float32)
        proportions = (1.0 - w_scalar) * theta + w_scalar * nnls_prior
    elif strategy == "dslevel_postsmooth":
        g1, t1 = exp_cfg["g1"], exp_cfg["t1"]
        g2, t2 = exp_cfg["g2"], exp_cfg["t2"]
        cfg_w_max = exp_cfg.get("w_max", 0.95)
        mean_stab = float(np.mean(stability))
        mean_maxp = float(np.mean(max_prop))
        w_scalar = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), g1, t1, g2, t2, w_max=cfg_w_max)[0]
        logger.info(f"  Dataset-level weight: w={w_scalar:.4f} "
                    f"(mean_stab={mean_stab:.4f}, mean_maxp={mean_maxp:.4f})")
        w_per_spot = np.full(len(stability), w_scalar, dtype=np.float32)
        proportions = (1.0 - w_scalar) * theta + w_scalar * nnls_prior
        # Post-blend spatial smoothing
        iter_post = exp_cfg.get("iter_post", 1)
        alpha_post = exp_cfg.get("alpha_post", 0.85)
        coords = None
        if "spatial" in adata.obsm:
            coords = np.array(adata.obsm["spatial"])
        elif "X_spatial" in adata.obsm:
            coords = np.array(adata.obsm["X_spatial"])
        if coords is not None:
            nn_post = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
            nn_post.fit(coords)
            _, indices_post = nn_post.kneighbors(coords)
            neighbor_idx_post = indices_post[:, 1:]
            for _ in range(iter_post):
                neighbor_mean = proportions[neighbor_idx_post].mean(axis=1)
                proportions = alpha_post * proportions + (1 - alpha_post) * neighbor_mean
            proportions = proportions / proportions.sum(axis=1, keepdims=True)
            logger.info(f"  Post-blend smooth: iter={iter_post}, alpha={alpha_post}")
        else:
            logger.warning("  Post-blend smooth skipped: no spatial coords found")
    else:
        proportions = theta

    # Normalize
    row_sums = proportions.sum(axis=1, keepdims=True)
    proportions = proportions / np.maximum(row_sums, 1e-10)

    runtime = time.time() - t_start

    # Evaluate all 10 metrics
    gt_labels = load_ground_truth(ds_name)
    n = min(len(gt_labels), proportions.shape[0])
    metrics = compute_all_metrics(proportions[:n], K, gt_labels[:n])
    logger.info(f"  F1={metrics['F1']:.3f} AMI={metrics['AMI']:.3f} "
                f"Pearson={metrics['Pearson']:.3f} RMSE={metrics['RMSE']:.3f}")

    # Save raw output
    out_path = RAW_OUTPUT_DIR / f"{ds_name}_{method}.npz"
    np.savez_compressed(out_path, proportions=proportions,
                        stability=stability, max_prop=max_prop,
                        w=w_per_spot if w_per_spot is not None else np.array([]))

    # Diagnostics
    stab_stats = None
    if stability is not None:
        stab_stats = {
            "mean": float(stability.mean()),
            "median": float(np.median(stability)),
            "std": float(stability.std()),
        }
    maxp_stats = None
    if max_prop is not None:
        maxp_stats = {
            "mean": float(max_prop.mean()),
            "median": float(np.median(max_prop)),
            "std": float(max_prop.std()),
        }
    w_stats = None
    if w_per_spot is not None:
        w_stats = {
            "mean": float(w_per_spot.mean()),
            "std": float(w_per_spot.std()),
            "min": float(w_per_spot.min()),
            "max": float(w_per_spot.max()),
        }

    result = {
        "dataset": ds_name,
        "method": method,
        "strategy": strategy,
        "K": K,
        **{k: exp_cfg.get(k) for k in ("g1", "t1", "g2", "t2", "gamma", "tau", "blend_w", "blend_alpha",
                                         "gamma_ent", "H_thresh", "seed_gene_mode", "iter_post", "alpha_post",
                                         "w_max", "sharpen_T", "sharpen_w_thresh")},
        "metrics": metrics,
        "stability_stats": stab_stats,
        "max_prop_stats": maxp_stats,
        "w_stats": w_stats,
        "runtime_s": round(runtime, 1),
        "status": "success",
    }

    del adata, proportions, exp_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ============================================================
# Result saving (thread-safe)
# ============================================================

def save_result(result):
    """Thread-safe append to results JSON."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = RESULTS_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if RESULTS_PATH.exists():
                with open(RESULTS_PATH) as f:
                    content = f.read().strip()
                if content:
                    data = json.loads(content)
                else:
                    data = {"experiment": "v28_twofactor", "results": []}
            else:
                data = {"experiment": "v28_twofactor", "results": []}
            data["results"].append(result)
            data["n_results"] = len(data["results"])
            with open(RESULTS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def is_done(ds_name, method):
    """Check if already completed."""
    if not RESULTS_PATH.exists():
        return False
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if r.get("dataset") == ds_name and r.get("method") == method:
                return True
    except (json.JSONDecodeError, KeyError):
        pass
    return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.idx is not None:
        configs = [EXPERIMENT_MATRIX[args.idx]]
    elif args.dataset:
        configs = [c for c in EXPERIMENT_MATRIX if c["dataset"] == args.dataset]
    else:
        configs = EXPERIMENT_MATRIX

    logger.info(f"Running {len(configs)} experiments (total matrix size: {len(EXPERIMENT_MATRIX)})")

    for cfg in configs:
        ds_name = cfg["dataset"]
        method = cfg["method"]

        if is_done(ds_name, method):
            logger.info(f"SKIP (done): {method} on {ds_name}")
            continue

        try:
            result = run_single(cfg, device=args.device)
            save_result(result)
            logger.info(f"DONE: {method} on {ds_name} -> F1={result['metrics']['F1']:.3f}")
        except Exception as e:
            logger.error(f"FAIL: {method} on {ds_name}: {e}", exc_info=True)
            save_result({
                "dataset": ds_name,
                "method": method,
                "strategy": cfg["strategy"],
                "status": "error",
                "error": str(e),
            })

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()

