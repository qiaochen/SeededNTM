#!/usr/bin/env python
"""Fair benchmark: ALL methods use the SAME spatial_adata.h5ad per dataset.

6 methods evaluated on identical input data:
  - FlashDeconv (reference-based regression, env: transpa)
  - RCTD (reference-based regression, env: transpa)
  - MarkerScore (reference-based scoring, env: transpa)
  - STAMP (unsupervised topic model, env: seededntm)
  - SeedTopic ref-based (seed-guided topic model, env: seededntm)
  - SeedTopic ref-free (seed-guided topic model, env: seededntm)

Usage:
    # Run reference-based methods (in transpa env):
    python run_fair_benchmark.py --dataset visiumHD_CRC_I --methods flashdeconv rctd marker_scoring

    # Run topic models (in seededntm env):
    python run_fair_benchmark.py --dataset visiumHD_CRC_I --methods seedtopic seedtopic_reffree stamp

    # Run all datasets, all methods:
    python run_fair_benchmark.py --dataset all

Environment:
    - FlashDeconv/RCTD/MarkerScore: micromamba activate transpa
    - SeedTopic/SeedTopic-reffree/STAMP: micromamba activate seededntm
    Use run_benchmark.sh to orchestrate both envs automatically.

All data paths are relative to PROJECT_ROOT (the SeedTopic repo root).
Required data layout under experiments/benchmark_fair/data/processed/:
    {dataset}/spatial_adata.h5ad          - Preprocessed spatial (RAW COUNTS in .X)
    {dataset}/reference_subset.h5ad       - K-type subsetted reference
    {dataset}/ground_truth.csv            - Ground truth labels
    {dataset}/seeds.json                  - Ref-based seed genes for SeedTopic
    {dataset}/seeds_leiden_chatgpt.json   - Ref-free seed genes for SeedTopic

Run `python preprocess_data.py --all` to generate processed data from raw inputs.
See experiments/benchmark_fair/README.md for full pipeline instructions.
"""

import argparse
import fcntl
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# All paths relative to the SeedTopic project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "experiments" / "benchmark_fair" / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "benchmark_fair"
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw_outputs"

DATASETS = {
    "visium_NPC": {
        "processed_dir": PROCESSED_DIR / "visium_NPC",
        "ref_cell_type_key": "Cluster",
        "ground_label": "cell_type",
        "target_types": ["B", "Myeloid", "T", "Treg", "fibroblast", "normal", "tumor"],
        "K": 7,
    },
    "xenium_BC": {
        "processed_dir": PROCESSED_DIR / "xenium_BC",
        "ref_cell_type_key": "Annotation",
        "ground_label": "cell_type",
        "target_types": [
            "B_Cells", "CD4+_T_Cells", "CD8+_T_Cells", "DCIS_1", "DCIS_2",
            "Endothelial", "IRF7+_DCs", "Invasive_Tumor", "LAMP3+_DCs",
            "Macrophages_1", "Macrophages_2", "Mast_Cells", "Myoepi_ACTA2+",
            "Myoepi_KRT15+", "Perivascular-Like", "Prolif_Invasive_Tumor",
            "Stromal", "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid",
        ],
        "K": 19,
    },
    "visiumHD_CRC_I": {
        "processed_dir": PROCESSED_DIR / "visiumHD_CRC_I",
        "ref_cell_type_key": "Cluster",
        "ground_label": "cell_type",
        "target_types": ["CAF", "Endothelial", "Macrophage", "Pericytes", "Tumor III"],
        "K": 5,
    },
    "visiumHD_CRC_II": {
        "processed_dir": PROCESSED_DIR / "visiumHD_CRC_II",
        "ref_cell_type_key": "Cluster",
        "ground_label": "cell_type",
        "target_types": ["CAF", "Endothelial", "Macrophage", "Neutrophil", "Pericytes", "Tumor III"],
        "K": 6,
    },
}

SEEDTOPIC_CONFIG = {
    "representation": "idf_lev_seed_spec",  # ablation winner (avg rank #1 across 4 datasets)
    "prior_mode": "standard",               # ablation winner
    "alpha": 199,
    "floor": 0.8,
    "n_epochs": 800,
    "batch_size": 8192,
    "lr": 0.01,
    "use_nb_obs": True,
    "enc_hid_dim": 64,
    "wt_fusion_top_seed": 1.0,
    "pos_scale": 0.5,
    "dropout": 0.2,
    "is_group_mode": False,
    "seed": 0,
    "clip_norm": 5.0,
    "temperature": 0.8,
    "k_neighbors": 6,
}

ALL_METHODS = ["flashdeconv", "rctd", "marker_scoring", "stamp", "seedtopic", "seedtopic_reffree"]


def load_spatial_data(dataset_name):
    """Load preprocessed spatial transcriptomics AnnData from processed dir."""
    import scanpy as sc
    cfg = DATASETS[dataset_name]
    st_path = cfg["processed_dir"] / "spatial_adata.h5ad"
    logger.info(f"Loading ST data: {st_path}")
    adata_st = sc.read_h5ad(str(st_path))
    adata_st.var_names_make_unique()
    logger.info(f"  ST shape: {adata_st.shape}")
    return adata_st


def load_reference_subsetted(dataset_name):
    """Load pre-subsetted reference from processed dir."""
    import scanpy as sc
    cfg = DATASETS[dataset_name]
    ref_path = cfg["processed_dir"] / "reference_subset.h5ad"
    logger.info(f"Loading subsetted reference: {ref_path}")
    adata_ref_subset = sc.read_h5ad(str(ref_path))

    ct_key = cfg["ref_cell_type_key"]
    remaining_types = sorted(adata_ref_subset.obs[ct_key].unique().tolist())
    type_counts = adata_ref_subset.obs[ct_key].value_counts()
    logger.info(f"  Subsetted reference: {adata_ref_subset.shape}, "
                f"{len(remaining_types)} types")
    for t in sorted(remaining_types):
        logger.info(f"    {t}: {type_counts.get(t, 0)}")

    return adata_ref_subset


# ============================================================
# Metrics
# ============================================================

def compute_metrics(proportions, cell_type_names, ground_truth_labels):
    """Compute all evaluation metrics."""
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score,
        average_precision_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )
    from scipy.stats import pearsonr, spearmanr
    from scipy.spatial.distance import jensenshannon

    predicted_labels = np.array(cell_type_names)[np.argmax(proportions, axis=1)]
    all_labels = sorted(set(ground_truth_labels.tolist()) | set(predicted_labels.tolist()))
    cls2id = {cls: idx for idx, cls in enumerate(all_labels)}
    gt_ids = np.array([cls2id[x] for x in ground_truth_labels])
    pred_ids = np.array([cls2id[x] for x in predicted_labels])

    n_classes = len(all_labels)
    gt_onehot = np.zeros((len(gt_ids), n_classes), dtype=np.float64)
    gt_onehot[np.arange(len(gt_ids)), gt_ids] = 1.0

    prop_aligned = np.zeros((len(gt_ids), n_classes), dtype=np.float64)
    for i, ct_name in enumerate(cell_type_names):
        if ct_name in cls2id:
            prop_aligned[:, cls2id[ct_name]] = proportions[:, i]

    gt_classes_present = np.where(gt_onehot.sum(axis=0) > 0)[0]
    gt_onehot_f = gt_onehot[:, gt_classes_present]
    prop_f = prop_aligned[:, gt_classes_present]

    try:
        auprc = float(average_precision_score(gt_onehot_f, prop_f, average="macro"))
    except ValueError:
        auprc = float("nan")

    # Pearson/Spearman on flattened proportion vs one-hot
    prop_flat = prop_f.ravel()
    gt_flat = gt_onehot_f.ravel()
    try:
        pearson_r = float(pearsonr(prop_flat, gt_flat)[0])
    except (ValueError, FloatingPointError):
        pearson_r = float("nan")
    try:
        spearman_r = float(spearmanr(prop_flat, gt_flat)[0])
    except (ValueError, FloatingPointError):
        spearman_r = float("nan")

    # RMSE on proportions vs one-hot
    rmse = float(np.sqrt(np.mean((prop_f - gt_onehot_f) ** 2)))

    # Mean JSD per spot
    jsd_vals = []
    for i in range(len(gt_ids)):
        p = gt_onehot_f[i]
        q = prop_f[i]
        q_safe = np.clip(q, 1e-10, None)
        q_safe = q_safe / q_safe.sum()
        jsd_vals.append(jensenshannon(p, q_safe) ** 2)
    mean_jsd = float(np.mean(jsd_vals))

    return {
        "F1": float(f1_score(gt_ids, pred_ids, average="macro", zero_division=0)),
        "Precision": float(precision_score(gt_ids, pred_ids, average="macro", zero_division=0)),
        "Recall": float(recall_score(gt_ids, pred_ids, average="macro", zero_division=0)),
        "AMI": float(adjusted_mutual_info_score(gt_ids, pred_ids)),
        "ARI": float(adjusted_rand_score(gt_ids, pred_ids)),
        "AUPRC": auprc,
        "Pearson": pearson_r,
        "Spearman": spearman_r,
        "RMSE": rmse,
        "JSD": mean_jsd,
    }


def compute_per_class_metrics(proportions, cell_type_names, ground_truth_labels):
    """Compute per-class F1, precision, recall, and AUPRC."""
    from sklearn.metrics import average_precision_score, classification_report
    from collections import Counter

    predicted_labels = np.array(cell_type_names)[np.argmax(proportions, axis=1)]
    gt_classes = sorted(set(ground_truth_labels.tolist()))
    report = classification_report(
        ground_truth_labels, predicted_labels,
        labels=gt_classes, output_dict=True, zero_division=0,
    )

    ct_to_idx = {ct: i for i, ct in enumerate(cell_type_names)}
    dist = Counter(ground_truth_labels.tolist())

    rows = []
    for ct in gt_classes:
        r = report.get(ct, {})
        ap = 0.0
        if ct in ct_to_idx:
            idx = ct_to_idx[ct]
            y_true_bin = (np.array(ground_truth_labels) == ct).astype(int)
            y_score = proportions[:, idx]
            if y_true_bin.sum() > 0:
                ap = float(average_precision_score(y_true_bin, y_score))
        rows.append({
            "cell_type": ct,
            "count": dist.get(ct, 0),
            "pct": 100 * dist.get(ct, 0) / len(ground_truth_labels),
            "f1": r.get("f1-score", 0),
            "precision": r.get("precision", 0),
            "recall": r.get("recall", 0),
            "auprc": ap,
        })
    return pd.DataFrame(rows)


# ============================================================
# Method runners
# ============================================================

def run_flashdeconv(adata_st, adata_ref_subset, cell_type_key):
    """Run FlashDeconv with K-type subsetted reference."""
    import flashdeconv as fd

    adata_st_copy = adata_st.copy()
    logger.info("  Running FlashDeconv with subsetted reference...")
    fd.tl.deconvolve(adata_st_copy, adata_ref_subset, cell_type_key=cell_type_key)
    prop_df = adata_st_copy.obsm["flashdeconv"]
    logger.info(f"  FlashDeconv done: {prop_df.shape}")
    return prop_df.values, prop_df.columns.tolist()


def run_rctd(adata_st, adata_ref_subset, cell_type_key, device="cuda"):
    """Run RCTD with K-type subsetted reference."""
    from rctd import RCTDConfig, Reference, run_rctd

    logger.info("  Running RCTD (full mode) with subsetted reference...")
    reference = Reference(adata_ref_subset, cell_type_col=cell_type_key, cell_min=1)
    config = RCTDConfig(compile=False, device=device)
    result = run_rctd(adata_st, reference, mode="full", config=config)

    weights = result.weights
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    weights = weights / row_sums

    logger.info(f"  RCTD done: {weights.shape}, "
                f"filtered {(~result.pixel_mask).sum()}/{len(result.pixel_mask)} spots")
    return weights, result.cell_type_names, result.pixel_mask


def run_seedtopic(dataset_name, device="cuda", seeds_path=None):
    """Run SeedTopic with canonical hyperparameters.

    Loads spatial_adata.h5ad from the processed directory (same data all methods use).
    Seeds are loaded from the provided seeds_path parameter.
    """
    import scanpy as sc
    import torch
    from scipy.stats import entropy as scipy_entropy
    from scipy.sparse import issparse

    from seededntm.util import (
        compute_topic_prior, compute_topic_prior_idf,
        spatial_smooth_prior, compute_reffree_leverage,
    )
    from seededntm.main import do_exp
    from seededntm.experiment import preprocess_ST
    from experiments.run_seedtopic import compute_representation

    cfg = DATASETS[dataset_name]
    processed_dir = cfg["processed_dir"]
    K = cfg["K"]

    # Default seeds_path: ref-based seeds.json
    if seeds_path is None:
        seeds_path = processed_dir / "seeds.json"

    # Load the SAME spatial adata that all methods use
    adata_path = processed_dir / "spatial_adata.h5ad"
    logger.info(f"  Loading spatial adata: {adata_path}")
    adata = sc.read_h5ad(str(adata_path))
    adata.var_names_make_unique()
    logger.info(f"  Shape: {adata.shape}")

    # Load seeds
    logger.info(f"  Loading seeds: {seeds_path}")
    with open(seeds_path) as f:
        seeds = json.load(f)
    ct_names = [None] * K
    for name, info in seeds.items():
        ct_names[info["topic_index"]] = name
    logger.info(f"  Seeds: K={K}, types={ct_names}")

    # === ASSERTIONS: verify all seed genes exist in adata ===
    missing_genes = {}
    for ct_name, info in seeds.items():
        genes = info.get("features", info.get("genes", []))
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            missing_genes[ct_name] = missing
    if missing_genes:
        msg = "FATAL: Seed genes not found in spatial_adata.var_names:\n"
        for ct, genes in missing_genes.items():
            msg += f"  {ct}: {genes}\n"
        raise ValueError(msg)

    # === ASSERTIONS: canonical hyperparameters ===
    assert SEEDTOPIC_CONFIG["use_nb_obs"] is True, "use_nb_obs must be True"
    assert SEEDTOPIC_CONFIG["n_epochs"] == 800, "n_epochs must be 800"
    assert SEEDTOPIC_CONFIG["batch_size"] == 8192, "batch_size must be 8192"
    assert SEEDTOPIC_CONFIG["alpha"] == 199, "alpha must be 199"
    assert SEEDTOPIC_CONFIG["floor"] == 0.8, "floor must be 0.8"

    logger.info(f"  Canonical hyperparameters verified: use_nb_obs=True, "
                f"n_epochs=800, batch_size=8192, alpha=199, floor=0.8")

    # Get count matrix (X is raw counts)
    X_counts = adata.X
    var_names = np.array(adata.var_names)

    # Compute representation (universal config — not per-dataset)
    rep_mode = SEEDTOPIC_CONFIG["representation"]
    logger.info(f"  Computing representation: {rep_mode}")
    rep = compute_representation(X_counts, var_names, seeds, rep_mode, K)
    logger.info(f"  Representation: {rep.shape}")

    # Store representation in adata
    input_key = "seedtopic_input_rep"
    adata.obsm[input_key] = rep

    # Store rna_count in obsm
    if "rna_count" not in adata.obsm:
        if issparse(X_counts):
            adata.obsm["rna_count"] = np.asarray(X_counts.todense(), dtype=np.float32)
        else:
            adata.obsm["rna_count"] = np.asarray(X_counts, dtype=np.float32)

    # Compute topic prior (universal config — not per-dataset)
    prior_mode = SEEDTOPIC_CONFIG["prior_mode"]
    logger.info(f"  Computing prior: {prior_mode}")
    if "idf" in prior_mode:
        raw_prior = compute_topic_prior_idf(adata, seeds, temperature=SEEDTOPIC_CONFIG["temperature"])
    else:
        raw_prior = compute_topic_prior(adata, seeds, temperature=SEEDTOPIC_CONFIG["temperature"])

    if "spatial" in prior_mode and "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=SEEDTOPIC_CONFIG["k_neighbors"])
    else:
        topic_prior = raw_prior

    # Per-spot alpha
    alpha = SEEDTOPIC_CONFIG["alpha"]
    floor = SEEDTOPIC_CONFIG["floor"]
    per_spot_ent = scipy_entropy(topic_prior, axis=1)
    max_ent = np.log(K)
    confidence = np.clip(1.0 - per_spot_ent / max_ent, 0.0, 1.0)
    alpha_np = (alpha * (floor + (1.0 - floor) * confidence)).astype(np.float32)
    logger.info(f"  Alpha: mean={alpha_np.mean():.1f}, min={alpha_np.min():.1f}, max={alpha_np.max():.1f}")

    # Build ExpData using preprocess_ST
    if issparse(X_counts):
        out_counts = np.asarray(X_counts.todense(), dtype=np.float32)
    else:
        out_counts = np.asarray(X_counts, dtype=np.float32)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='seeds_') as sf:
        json.dump(seeds, sf)
        seeds_tmp_path = sf.name

    try:
        exp_data = preprocess_ST(
            adata=adata,
            n_topics=K,
            input_rep=rep,
            out_counts=out_counts,
            out_normal=None,
            condition_feat_path=seeds_tmp_path,
            topic_prior=topic_prior.astype(np.float32),
        )
    finally:
        os.unlink(seeds_tmp_path)

    # Run with per-spot Dirichlet
    import importlib.util
    _trainer_path = str(
        PROJECT_ROOT / "experiments" / "v3_reffree_leverage" / "run_ablation_v21d_universal_prior.py"
    )
    _spec = importlib.util.spec_from_file_location("trainer_module", _trainer_path)
    _trainer_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_trainer_mod)
    run_perspot_dirichlet = _trainer_mod.run_perspot_dirichlet

    # early_stop: True ONLY for xenium_BC
    early_stop = (dataset_name == "xenium_BC")

    with tempfile.TemporaryDirectory(prefix="seedtopic_fair_") as tmp_dir:
        df_topic = run_perspot_dirichlet(
            exp_data, K, tmp_dir, alpha_np, device,
            early_stop=early_stop,
            n_epochs=SEEDTOPIC_CONFIG["n_epochs"],
            batch_size=SEEDTOPIC_CONFIG["batch_size"],
            wt_fusion_top_seed=SEEDTOPIC_CONFIG["wt_fusion_top_seed"],
            use_nb_obs=SEEDTOPIC_CONFIG["use_nb_obs"],
        )

    proportions = df_topic.values
    logger.info(f"  SeedTopic done: {proportions.shape}")
    return proportions, ct_names


def run_stamp(dataset_name, device="cuda"):
    """Run STAMP (unsupervised spatial topic model) from the scTM package.

    STAMP is unsupervised — no reference, no seeds. K = number of target types.
    Results are evaluated via Hungarian matching since topics are unnamed.
    """
    try:
        SCTM_PATH = "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/scTM"
        if SCTM_PATH not in sys.path:
            sys.path.insert(0, SCTM_PATH)
        import unittest.mock as _mock
        with _mock.patch.dict(sys.modules, {"discotoolkit": _mock.MagicMock()}):
            import sctm.stamp
        import scanpy as sc
        import squidpy as sq
        import torch
        import random
    except ImportError as e:
        logger.warning(f"  STAMP skipped: missing package ({e}). Need seededntm env with sctm.")
        return None, None

    cfg = DATASETS[dataset_name]
    processed_dir = cfg["processed_dir"]
    K = cfg["K"]

    adata_path = processed_dir / "spatial_adata.h5ad"
    logger.info(f"  Loading spatial adata for STAMP: {adata_path}")
    adata = sc.read_h5ad(str(adata_path))
    adata.var_names_make_unique()
    logger.info(f"  Shape: {adata.shape}")

    # Compute spatial neighbors (required by STAMP)
    if "spatial" in adata.obsm:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_rings=2)
    else:
        sq.gr.spatial_neighbors(adata, n_rings=2)

    # Reproducibility
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    model = sctm.stamp.STAMP(adata, n_topics=K)
    seed_everything(0)
    model.train(max_epochs=800, batch_size=4096, learning_rate=0.01)

    topics = model.get_cell_by_topic()
    proportions = topics.values.astype(np.float64)

    # Normalize to sum to 1 per spot
    row_sums = proportions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proportions = proportions / row_sums

    logger.info(f"  STAMP done: {proportions.shape}, K={K}")
    return proportions, K


def run_marker_scoring(dataset_name):
    """Marker gene scoring baseline using scanpy's sc.tl.score_genes.

    Derives DE markers from reference (Wilcoxon, top 50, logFC>0.5, padj<0.01),
    scores spatial data per cell type, normalizes via softmax.
    """
    try:
        import scanpy as sc
    except ImportError as e:
        logger.warning(f"  MarkerScore skipped: missing package ({e}).")
        return None, None

    from scipy.special import softmax

    cfg = DATASETS[dataset_name]
    processed_dir = cfg["processed_dir"]
    ct_key = cfg["ref_cell_type_key"]

    # Load reference
    ref_path = processed_dir / "reference_subset.h5ad"
    logger.info(f"  Loading reference for MarkerScore: {ref_path}")
    adata_ref = sc.read_h5ad(str(ref_path))

    # Compute DE markers on a normalized copy of reference
    ref = adata_ref.copy()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    sc.tl.rank_genes_groups(ref, groupby=ct_key, method="wilcoxon", n_genes=200)

    ct_names = sorted(ref.obs[ct_key].unique().tolist())

    markers_per_ct = {}
    for ct in ct_names:
        result_df = sc.get.rank_genes_groups_df(ref, group=ct)
        top_markers = result_df[
            (result_df["logfoldchanges"] > 0.5) & (result_df["pvals_adj"] < 0.01)
        ]["names"].head(50).tolist()
        if len(top_markers) < 10:
            top_markers = result_df[result_df["logfoldchanges"] > 0]["names"].head(50).tolist()
        markers_per_ct[ct] = top_markers

    logger.info(f"  Computed markers: {[(ct, len(g)) for ct, g in markers_per_ct.items()]}")

    # Save marker gene lists for reproducibility
    out_dir = RAW_OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    markers_path = out_dir / "marker_scoring_gene_lists.json"
    with open(markers_path, "w") as f:
        json.dump(markers_per_ct, f, indent=2)
    logger.info(f"  Saved marker gene lists: {markers_path}")

    # Load spatial adata and normalize a COPY for scoring
    st_path = processed_dir / "spatial_adata.h5ad"
    logger.info(f"  Loading spatial adata for MarkerScore: {st_path}")
    st = sc.read_h5ad(str(st_path))
    st.var_names_make_unique()
    sc.pp.normalize_total(st, target_sum=1e4)
    sc.pp.log1p(st)

    # Score each cell type
    for ct in ct_names:
        gene_list = [g for g in markers_per_ct[ct] if g in st.var_names]
        if len(gene_list) < 3:
            st.obs[f"score_{ct}"] = 0.0
            continue
        sc.tl.score_genes(st, gene_list=gene_list, score_name=f"score_{ct}", random_state=0)

    score_cols = [f"score_{ct}" for ct in ct_names]
    scores = st.obs[score_cols].values.astype(np.float64)
    proportions = softmax(scores, axis=1)

    logger.info(f"  MarkerScore complete: {proportions.shape}, {len(ct_names)} types")
    return proportions, ct_names


def compute_metrics_hungarian(proportions, K, gt_labels, gt_types=None):
    """Full metrics via Hungarian matching for unsupervised methods (STAMP)."""
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score,
        average_precision_score, f1_score, precision_score, recall_score,
    )
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr, spearmanr
    from scipy.spatial.distance import jensenshannon

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

    # Build cost matrix for Hungarian assignment
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

    # Merge topic proportions according to Hungarian assignment
    merged_proportions = np.zeros((len(proportions_valid), n_gt), dtype=np.float64)
    for k in range(K):
        merged_proportions[:, topic_to_gt[k]] += proportions_valid[:, k]
    row_sums = merged_proportions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    merged_proportions = merged_proportions / row_sums

    pred_gt_idx = np.argmax(merged_proportions, axis=1)

    metrics = {
        "F1": float(f1_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "Precision": float(precision_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "Recall": float(recall_score(gt_idx, pred_gt_idx, average="macro", zero_division=0)),
        "AMI": float(adjusted_mutual_info_score(gt_idx, pred_gt_idx)),
        "ARI": float(adjusted_rand_score(gt_idx, pred_gt_idx)),
    }

    gt_onehot = np.zeros((len(gt_idx), n_gt), dtype=np.float64)
    gt_onehot[np.arange(len(gt_idx)), gt_idx] = 1.0
    try:
        metrics["AUPRC"] = float(average_precision_score(gt_onehot, merged_proportions, average="macro"))
    except ValueError:
        metrics["AUPRC"] = float("nan")

    # Pearson/Spearman on flattened proportion vs one-hot
    prop_flat = merged_proportions.ravel()
    gt_flat = gt_onehot.ravel()
    try:
        metrics["Pearson"] = float(pearsonr(prop_flat, gt_flat)[0])
    except (ValueError, FloatingPointError):
        metrics["Pearson"] = float("nan")
    try:
        metrics["Spearman"] = float(spearmanr(prop_flat, gt_flat)[0])
    except (ValueError, FloatingPointError):
        metrics["Spearman"] = float("nan")

    metrics["RMSE"] = float(np.sqrt(np.mean((merged_proportions - gt_onehot) ** 2)))

    jsd_vals = []
    for i in range(len(gt_idx)):
        p = gt_onehot[i]
        q = merged_proportions[i]
        q_safe = np.clip(q, 1e-10, None)
        q_safe = q_safe / q_safe.sum()
        jsd_vals.append(jensenshannon(p, q_safe) ** 2)
    metrics["JSD"] = float(np.mean(jsd_vals))

    return metrics


def log_startup_info(dataset_name, methods):
    """Log reproducibility information at the start of a run."""
    logger.info("=" * 70)
    logger.info("STARTUP PROVENANCE LOG")
    logger.info("=" * 70)

    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
        logger.info(f"  Git commit: {git_hash}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("  Git commit: UNKNOWN (git not available)")

    # Package versions
    pkg_versions = {}
    for pkg_name in ["torch", "pyro", "scanpy", "seededntm", "sctm", "flashdeconv", "rctd"]:
        try:
            mod = __import__(pkg_name)
            pkg_versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkg_versions[pkg_name] = "NOT INSTALLED"
    logger.info(f"  Package versions: {pkg_versions}")

    # Input file paths and existence
    cfg = DATASETS[dataset_name]
    processed_dir = cfg["processed_dir"]
    files_to_check = {
        "spatial_adata": processed_dir / "spatial_adata.h5ad",
        "reference_subset": processed_dir / "reference_subset.h5ad",
        "ground_truth": processed_dir / "ground_truth.csv",
        "seeds_refbased": processed_dir / "seeds.json",
        "seeds_reffree": processed_dir / "seeds_leiden_chatgpt.json",
    }
    for name, path in files_to_check.items():
        exists = path.exists()
        logger.info(f"  {name}: {path} [{'EXISTS' if exists else 'MISSING'}]")

    # Canonical hyperparameters
    logger.info(f"  SeedTopic config: {json.dumps(SEEDTOPIC_CONFIG, indent=None)}")
    logger.info(f"  Dataset: {dataset_name}, K={cfg['K']}, target_types={cfg['target_types']}")
    logger.info(f"  Methods: {methods}")
    logger.info("=" * 70)

def run_single_dataset(dataset_name, methods, device="cuda"):
    """Run all requested methods on one dataset and save results."""
    cfg = DATASETS[dataset_name]
    target_types = cfg["target_types"]
    K = cfg["K"]
    processed_dir = cfg["processed_dir"]

    # Startup logging and assertions
    log_startup_info(dataset_name, methods)

    logger.info(f"\n{'='*70}")
    logger.info(f"FAIR BENCHMARK: {dataset_name} (K={K} types)")
    logger.info(f"Target types: {target_types}")
    logger.info(f"Methods: {methods}")
    logger.info(f"{'='*70}")

    # Check that processed data exists
    if not processed_dir.exists():
        logger.error(
            f"Processed data not found: {processed_dir}\n"
            f"Run 'python preprocess_data.py --dataset {dataset_name}' first."
        )
        return []

    out_dir = RAW_OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load spatial data and ground truth
    adata_st = load_spatial_data(dataset_name)

    # Log data shapes
    logger.info(f"  Spatial adata shape: {adata_st.shape}")

    # Load ground truth from the standardized CSV
    gt_csv = processed_dir / "ground_truth.csv"
    gt_df = pd.read_csv(str(gt_csv))
    gt_labels = gt_df["ground_truth"].values.astype(str)

    # Save ground truth
    gt_df = pd.DataFrame({
        "ground_truth": gt_labels,
        "coord_x": adata_st.obsm["spatial"][:, 0],
        "coord_y": adata_st.obsm["spatial"][:, 1],
    })
    gt_df.to_csv(out_dir / "ground_truth.csv", index=False)

    # Load and subset reference for methods that need it
    adata_ref_subset = None
    if any(m in methods for m in ["flashdeconv", "rctd", "marker_scoring"]):
        adata_ref_subset = load_reference_subsetted(dataset_name)

    results = []

    for method in methods:
        logger.info(f"\n--- {method} on {dataset_name} ---")
        t0 = time.time()
        result = {"dataset": dataset_name, "method": method, "K": K}

        try:
            if method == "flashdeconv":
                proportions, ct_names = run_flashdeconv(
                    adata_st, adata_ref_subset, cfg["ref_cell_type_key"]
                )
                gt_eval = gt_labels
                pixel_mask = None

            elif method == "rctd":
                weights, ct_names, pixel_mask = run_rctd(
                    adata_st, adata_ref_subset, cfg["ref_cell_type_key"], device
                )
                proportions = weights
                if pixel_mask is not None:
                    gt_eval = gt_labels[pixel_mask]
                    result["spots_filtered"] = int((~pixel_mask).sum())
                    result["spots_evaluated"] = int(pixel_mask.sum())
                    np.save(out_dir / "rctd_pixel_mask.npy", pixel_mask)
                else:
                    gt_eval = gt_labels

            elif method == "marker_scoring":
                proportions, ct_names = run_marker_scoring(dataset_name)
                if proportions is None:
                    result["status"] = "skipped"
                    result["error"] = "missing package"
                    results.append(result)
                    continue
                gt_eval = gt_labels
                pixel_mask = None

            elif method == "seedtopic":
                proportions, ct_names = run_seedtopic(
                    dataset_name, device,
                    seeds_path=processed_dir / "seeds.json",
                )
                gt_eval = gt_labels
                pixel_mask = None

            elif method == "seedtopic_reffree":
                proportions, ct_names = run_seedtopic(
                    dataset_name, device,
                    seeds_path=processed_dir / "seeds_leiden_chatgpt.json",
                )
                gt_eval = gt_labels
                pixel_mask = None

            elif method == "stamp":
                proportions, stamp_K = run_stamp(dataset_name, device)
                if proportions is None:
                    result["status"] = "skipped"
                    result["error"] = "missing package"
                    results.append(result)
                    continue
                gt_eval = gt_labels
                pixel_mask = None
                # STAMP uses Hungarian matching (unsupervised)
                metrics = compute_metrics_hungarian(proportions, K, gt_eval, target_types)
                result.update(metrics)
                result["status"] = "success"
                result["n_output_types"] = K
                result["output_types"] = [f"topic_{i}" for i in range(K)]
                result["matching"] = "hungarian"

                prop_df = pd.DataFrame(proportions, columns=[f"topic_{i}" for i in range(K)])
                prop_df.to_csv(out_dir / f"{method}_proportions.csv", index=False)

                elapsed = time.time() - t0
                result["elapsed_seconds"] = round(elapsed, 1)
                result["eval_set"] = "full"
                results.append(result)

                logger.info(f"  Results: F1={metrics['F1']:.4f} AUPRC={metrics['AUPRC']:.4f} "
                            f"ARI={metrics['ARI']:.4f} AMI={metrics['AMI']:.4f} "
                            f"Pearson={metrics['Pearson']:.4f} JSD={metrics['JSD']:.4f}")
                gc.collect()
                continue

            else:
                raise ValueError(f"Unknown method: {method}")

            # Compute metrics (direct label — for all non-STAMP methods)
            metrics = compute_metrics(proportions, ct_names, gt_eval)
            result.update(metrics)
            result["status"] = "success"
            result["n_output_types"] = len(ct_names)
            result["output_types"] = ct_names
            result["matching"] = "direct"

            # Save proportions
            prop_df = pd.DataFrame(proportions, columns=ct_names)
            prop_df.to_csv(out_dir / f"{method}_proportions.csv", index=False)
            logger.info(f"  Saved: {out_dir / method}_proportions.csv")

            logger.info(f"  Results: F1={metrics['F1']:.4f} AUPRC={metrics['AUPRC']:.4f} "
                        f"ARI={metrics['ARI']:.4f} AMI={metrics['AMI']:.4f} "
                        f"Pearson={metrics['Pearson']:.4f} JSD={metrics['JSD']:.4f}")

            # Per-class F1 breakdown
            per_class_df = compute_per_class_metrics(proportions, ct_names, gt_eval)
            per_class_df["method"] = method
            per_class_df["dataset"] = dataset_name
            per_class_path = out_dir / f"{method}_per_class_metrics.csv"
            per_class_df.to_csv(per_class_path, index=False)
            logger.info(f"  Per-class F1 saved: {per_class_path}")
            for _, row in per_class_df.iterrows():
                logger.info(f"    {row['cell_type']:20s} F1={row['f1']:.3f} "
                            f"AUPRC={row['auprc']:.3f} ({row['count']:>5d} spots, {row['pct']:.1f}%)")

            # Also compute metrics on RCTD subset for non-RCTD methods
            if method != "rctd":
                rctd_mask_path = out_dir / "rctd_pixel_mask.npy"
                if rctd_mask_path.exists():
                    rctd_mask = np.load(rctd_mask_path)
                    if len(rctd_mask) == len(gt_labels):
                        subset_metrics = compute_metrics(
                            proportions[rctd_mask], ct_names, gt_labels[rctd_mask]
                        )
                        subset_result = dict(result)
                        subset_result["eval_set"] = "rctd_subset"
                        subset_result.update(subset_metrics)
                        results.append(subset_result)

        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)

        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        result["eval_set"] = result.get("eval_set", "full")
        results.append(result)
        gc.collect()

    return results


def _load_results_locked(results_path):
    """Load results JSON with shared file lock for concurrent reads."""
    if not results_path.exists():
        return []
    lock_path = results_path.with_suffix(".lock")
    with open(lock_path, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_SH)
        try:
            with open(results_path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _save_results_locked(results_path, new_results, datasets, methods):
    """Save results JSON with exclusive file lock for concurrent writes.

    Merges new_results into existing file, replacing entries for the given
    dataset+method combinations.
    """
    lock_path = results_path.with_suffix(".lock")
    with open(lock_path, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing = []
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except (json.JSONDecodeError, IOError):
                    existing = []

            current_combos = {(d, m) for d in datasets for m in methods}
            merged = [
                r for r in existing
                if (r.get("dataset"), r.get("method")) not in current_combos
            ]
            merged.extend(new_results)

            with open(results_path, "w") as f:
                json.dump(merged, f, indent=2, default=str)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        help="Dataset name or 'all'. Choices: " + ", ".join(DATASETS.keys()),
    )
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        choices=ALL_METHODS,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, using CPU")
    except ImportError:
        device = "cpu"

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    # Check that processed data directory exists
    if not PROCESSED_DIR.exists():
        logger.error(
            f"Processed data directory not found: {PROCESSED_DIR}\n"
            f"Run 'python preprocess_data.py --all' first to generate processed data."
        )
        sys.exit(1)

    results_path = OUTPUT_DIR / "fair_benchmark_results.json"
    all_results = []

    for ds in datasets:
        if ds not in DATASETS:
            logger.error(f"Unknown dataset: {ds}")
            continue
        ds_results = run_single_dataset(ds, args.methods, device)
        all_results.extend(ds_results)

        _save_results_locked(results_path, all_results, [ds], args.methods)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("FAIR BENCHMARK COMPLETE")
    logger.info(f"{'='*70}")

    final_results = _load_results_locked(results_path)
    summary_rows = [r for r in final_results if r.get("eval_set") == "full" and r.get("status") == "success"]
    if summary_rows:
        cols = ["dataset", "method", "K", "F1", "Precision", "Recall", "AMI", "ARI", "AUPRC", "Pearson", "Spearman", "RMSE", "JSD"]
        available_cols = [c for c in cols if c in pd.DataFrame(summary_rows).columns]
        df = pd.DataFrame(summary_rows)[available_cols]
        print("\n" + df.to_string(index=False))

    print(f"\nResults saved: {results_path}")
    print(f"Raw outputs:  {RAW_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
