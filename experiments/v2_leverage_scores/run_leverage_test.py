#!/usr/bin/env python
"""Test leverage-score-weighted inputs for SeedTopic.

Supports two modes:
  1. Automatic leverage-weighted TF-IDF + PCA (FlashDeconv parameterless design)
  2. Leverage-weighted TF-IDF + CountSketch (randomized alternative to PCA)

Usage:
    # Auto leverage (default, no power parameter needed):
    python experiments/v2_leverage_scores/run_leverage_test.py --dataset visiumHD_CRC_II

    # Compare PCA vs CountSketch:
    python experiments/v2_leverage_scores/run_leverage_test.py --dataset visiumHD_CRC_II --method pca sketch

    # Legacy power sweep (for comparison with old results):
    python experiments/v2_leverage_scores/run_leverage_test.py --dataset visium_NPC --power 0.25 0.5 1.0

    # All datasets, auto mode:
    python experiments/v2_leverage_scores/run_leverage_test.py --dataset all

Environment:
    Requires the seededntm conda env with GPU access.
    Expected runtime: ~2 min per small dataset, ~24 min for xenium_BC
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from seededntm.util import (
    compute_tfidf_rep,
    compute_leverage_scores,
    compute_countsketch_rep,
    compute_topic_prior,
)
from experiments.evaluate import compute_metrics, load_ground_truth, load_baseline_proportions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_ROOT / "experiments" / "configs" / "datasets.json"
OUTPUT_DIR = Path(__file__).parent / "raw_outputs"


def compute_signature_matrix(adata_ref, ct_key: str, target_genes) -> np.ndarray:
    """Compute normalized mean expression per cell type, aligned to target genes.

    Returns (K, G_target) matrix where K = number of cell types.
    """
    cell_types = sorted(adata_ref.obs[ct_key].unique())
    logger.info(f"  {len(cell_types)} cell types, {adata_ref.n_obs} cells")

    ref_gene_list = list(adata_ref.var_names)
    ref_gene_set = set(ref_gene_list)
    ref_gene_to_idx = {g: i for i, g in enumerate(ref_gene_list)}

    target_in_ref = [g for g in target_genes if g in ref_gene_set]
    target_ref_idx = [ref_gene_to_idx[g] for g in target_in_ref]
    target_out_idx = [i for i, g in enumerate(target_genes) if g in ref_gene_set]

    logger.info(f"  Gene overlap: {len(target_in_ref)}/{len(target_genes)} target genes found in ref")

    signatures = np.zeros((len(cell_types), len(target_genes)), dtype=np.float64)

    for ct_i, ct in enumerate(cell_types):
        mask = (adata_ref.obs[ct_key] == ct).values
        X_ct = adata_ref.X[mask][:, target_ref_idx]
        if issparse(X_ct):
            X_ct = X_ct.toarray()

        row_sums = X_ct.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_norm = np.log1p(X_ct / row_sums * 1e4)
        mean_expr = X_norm.mean(axis=0)
        signatures[ct_i, target_out_idx] = mean_expr

    return signatures, cell_types


def load_reference(cfg: dict) -> "sc.AnnData":
    """Load reference data, handling both h5ad and 10x h5 + xlsx cases."""
    ref_path = cfg.get("ref_path")
    if ref_path is not None:
        logger.info(f"Loading reference h5ad: {ref_path}")
        return sc.read_h5ad(ref_path)

    ref_h5 = cfg.get("ref_10x_h5_path")
    ref_xlsx = cfg.get("ref_annotation_xlsx_path")
    if ref_h5 is not None:
        logger.info(f"Building reference from 10x h5: {ref_h5}")
        adata = sc.read_10x_h5(ref_h5)
        adata.var_names_make_unique()
        if ref_xlsx is not None:
            anno = pd.read_excel(ref_xlsx)
            anno = anno.set_index("Barcode")
            anno["Annotation"] = anno["Annotation"].apply(lambda x: x.replace(" ", "_"))
            adata = adata[adata.obs_names.isin(anno.index.values)]
            adata.obs["Annotation"] = anno.loc[adata.obs_names, "Annotation"].values
        sc.pp.filter_cells(adata, min_genes=200)
        return adata

    return None


def prepare_input(dataset_name: str, cfg: dict, output_dir: Path,
                  method: str = "pca", power: float = None,
                  sketch_dim: int = 512) -> Path:
    """Compute leverage-weighted input and save as new h5ad.

    Args:
        method: "pca" for TF-IDF + leverage + PCA, "sketch" for TF-IDF + CountSketch.
        power: if None, uses automatic FlashDeconv-style scaling.
            If a float, applies legacy power softening (for backward compat).
        sketch_dim: dimension for CountSketch (only used if method="sketch").

    Returns:
        Path to the new h5ad file, or None on failure.
    """
    logger.info(f"Loading spatial data: {cfg['seedtopic_adata_path']}")
    adata = sc.read_h5ad(cfg["seedtopic_adata_path"])

    adata_ref = load_reference(cfg)
    if adata_ref is None:
        logger.warning(f"{dataset_name}: no reference data available")
        return None

    # Compute signature matrix
    logger.info("Computing per-type signature matrix...")
    target_genes = list(adata.var_names)
    X_ref, ref_types = compute_signature_matrix(
        adata_ref, cfg["ref_cell_type_key"], target_genes
    )
    logger.info(f"Signature matrix: {X_ref.shape} ({len(ref_types)} types x {len(target_genes)} genes)")
    del adata_ref

    # Compute leverage scores (auto-scaled by default)
    leverage = compute_leverage_scores(X_ref)
    logger.info(f"Auto leverage weights: min={leverage.min():.3f}, max={leverage.max():.3f}, "
                f"mean={leverage.mean():.3f}, std={leverage.std():.3f}")

    if power is not None:
        # Legacy mode: apply additional power transform on top of auto weights
        leverage = np.power(np.maximum(leverage, 0), power)
        leverage = leverage / (leverage.mean() + 1e-8)
        logger.info(f"After power={power}: min={leverage.min():.3f}, max={leverage.max():.3f}")

    # Save leverage scores
    suffix = f"_p{power}" if power is not None else "_auto"
    np.save(output_dir / f"{dataset_name}_leverage_scores{suffix}.npy", leverage)

    # Diagnostic: top genes
    top_idx = np.argsort(leverage)[-5:][::-1]
    top_genes = [(target_genes[i], f"{leverage[i]:.2f}") for i in top_idx]
    logger.info(f"Top leverage genes: {top_genes}")

    # Compute input representation
    X_counts = adata.obsm["rna_count"]
    input_key = f"tfidf_{method}_leverage"

    if method == "pca":
        logger.info(f"Computing TF-IDF + leverage + PCA (input: {X_counts.shape})")
        rep = compute_tfidf_rep(X_counts, n_pcs=100, gene_weights=leverage)
    elif method == "sketch":
        logger.info(f"Computing TF-IDF + leverage + CountSketch(d={sketch_dim}) (input: {X_counts.shape})")
        # First compute TF-IDF, then sketch
        tfidf = compute_tfidf_rep(X_counts, n_pcs=None, gene_weights=leverage)
        rep = compute_countsketch_rep(tfidf, sketch_dim=sketch_dim,
                                      leverage_scores=leverage, seed=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Output representation: {rep.shape}")
    adata.obsm[input_key] = rep

    out_path = (output_dir / f"{dataset_name}_{method}{suffix}.h5ad").resolve()
    adata.write_h5ad(out_path)
    logger.info(f"Saved: {out_path}")

    # Free memory before subprocess invocation
    del adata, rep
    import gc
    gc.collect()

    return out_path, input_key


def run_seedtopic_with_input(dataset_name: str, cfg: dict, h5ad_path: Path,
                             input_key: str, output_dir: Path) -> tuple:
    """Run SeedTopic on a custom h5ad with specified input key (in-process)."""
    import torch

    seeds_path = cfg["condition_feat_path"]
    with open(seeds_path) as f:
        marker_genes = json.load(f)
    idx_to_name = {v["topic_index"]: k for k, v in marker_genes.items()}
    ct_names = [idx_to_name[i] for i in range(len(marker_genes))]

    exp_outdir = str((output_dir / f"{dataset_name}_{input_key}_output").resolve())
    os.makedirs(exp_outdir, exist_ok=True)

    from seededntm.main import do_exp

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    extra_args = cfg.get("extra_args", [])
    early_stop = "--early_stop" in extra_args
    use_nb_obs = "--use_nb_obs" in extra_args

    t0 = time.time()
    logger.info(f"CALLING do_exp with condition_feat_path={seeds_path}")
    try:
        do_exp(
            adata_h5ad_path=str(h5ad_path),
            condition_feat_path=seeds_path,
            key_input=input_key,
            key_count_out='rna_count',
            key_topic_prior='topic_prior',
            n_topics=cfg["num_topics"],
            reg_topic_prior=cfg["reg_topic_prior"],
            wt_fusion_top_seed=cfg["wt_fusion_top_seed"],
            batch_size=8192,
            exp_outdir=exp_outdir,
            early_stop=early_stop,
            use_nb_obs=use_nb_obs,
            device=device,
        )
    except Exception as e:
        logger.error(f"SeedTopic raised: {e}")
        runtime = time.time() - t0
        return None, ct_names, runtime

    runtime = time.time() - t0

    df_topic_path = os.path.join(exp_outdir, "df_topic.csv")
    if os.path.exists(df_topic_path):
        df_topic = pd.read_csv(df_topic_path, index_col=0)
        return df_topic.values, ct_names, runtime

    logger.error(f"No output produced in {exp_outdir}")
    return None, ct_names, runtime


def main():
    parser = argparse.ArgumentParser(description="Test leverage-score weighted inputs")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--method", nargs="+", default=["pca"],
                        choices=["pca", "sketch"],
                        help="Dimensionality reduction method(s) to test")
    parser.add_argument("--power", nargs="+", type=float, default=None,
                        help="Legacy power exponents. Omit to use automatic scaling.")
    parser.add_argument("--sketch-dim", type=int, default=512,
                        help="CountSketch dimension (default 512)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare h5ad files, don't run SeedTopic")
    args = parser.parse_args()

    with open(CONFIGS_PATH) as f:
        configs = json.load(f)

    if "all" in args.dataset:
        dataset_names = list(configs.keys())
    else:
        dataset_names = args.dataset

    # If no power specified, use automatic (None)
    power_values = args.power if args.power else [None]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for ds_name in dataset_names:
        cfg = configs[ds_name]
        if cfg.get("ref_path") is None and cfg.get("ref_10x_h5_path") is None:
            logger.info(f"Skipping {ds_name}: no reference data")
            continue

        for method in args.method:
            for power in power_values:
                power_str = f"p{power}" if power is not None else "auto"
                run_key = f"{ds_name}_{method}_{power_str}"
                print(f"\n{'='*60}")
                print(f"Dataset: {ds_name} | Method: {method} | Scaling: {power_str}")
                print(f"{'='*60}")

                # Step 1: Prepare input
                result = prepare_input(
                    ds_name, cfg, OUTPUT_DIR,
                    method=method, power=power, sketch_dim=args.sketch_dim
                )
                if result is None:
                    continue
                h5ad_path, input_key = result

                if args.prepare_only:
                    continue

                # Step 2: Run SeedTopic
                logger.info(f"Running SeedTopic with {method} input (key={input_key})...")
                proportions, ct_names, runtime = run_seedtopic_with_input(
                    ds_name, cfg, h5ad_path, input_key, OUTPUT_DIR
                )

                if proportions is None:
                    logger.error(f"SeedTopic failed on {ds_name}")
                    continue

                # Step 3: Evaluate
                gt_labels = load_ground_truth(ds_name)
                n = min(len(gt_labels), proportions.shape[0])
                metrics = compute_metrics(proportions[:n], ct_names, gt_labels[:n])

                bl_proportions, bl_ct_names = load_baseline_proportions(ds_name, "seedtopic")
                bl_metrics = compute_metrics(bl_proportions[:n], bl_ct_names, gt_labels[:n])

                print(f"\n  Baseline:         F1={bl_metrics['f1']:.3f}  AUPRC={bl_metrics['AUPRC']:.3f}  ARI={bl_metrics['ARI']:.3f}")
                print(f"  {method}({power_str}): F1={metrics['f1']:.3f}  AUPRC={metrics['AUPRC']:.3f}  ARI={metrics['ARI']:.3f}")
                print(f"  Delta:            F1={metrics['f1']-bl_metrics['f1']:+.3f}  "
                      f"AUPRC={metrics['AUPRC']-bl_metrics['AUPRC']:+.3f}  "
                      f"ARI={metrics['ARI']-bl_metrics['ARI']:+.3f}")
                print(f"  Runtime: {runtime:.1f}s")

                # Save proportions
                df = pd.DataFrame(proportions, columns=ct_names)
                df.to_csv(OUTPUT_DIR / f"{ds_name}_{method}_{power_str}_proportions.csv",
                          index=False)

                results[run_key] = {
                    "dataset": ds_name,
                    "method": method,
                    "scaling": power_str,
                    "baseline": bl_metrics,
                    "result": metrics,
                    "runtime_s": runtime,
                    "delta_F1": metrics["f1"] - bl_metrics["f1"],
                    "delta_AUPRC": metrics["AUPRC"] - bl_metrics["AUPRC"],
                    "delta_ARI": metrics["ARI"] - bl_metrics["ARI"],
                }

    # Save results
    if results:
        out_path = Path(__file__).parent / "leverage_results_v2.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

        # Summary table
        print(f"\n{'='*80}")
        print(f"{'Dataset':<16} {'Method':<8} {'Scale':<7} {'F1':>6} {'AUPRC':>7} {'ARI':>6} | {'dF1':>6} {'dAUPRC':>7} {'dARI':>6}")
        print(f"{'-'*80}")
        for key, r in results.items():
            print(f"{r['dataset']:<16} {r['method']:<8} {r['scaling']:<7} "
                  f"{r['result']['f1']:>6.3f} {r['result']['AUPRC']:>7.3f} {r['result']['ARI']:>6.3f}"
                  f" | {r['delta_F1']:>+6.3f} {r['delta_AUPRC']:>+7.3f} {r['delta_ARI']:>+6.3f}")


if __name__ == "__main__":
    main()
