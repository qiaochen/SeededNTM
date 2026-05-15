#!/usr/bin/env python
"""Test leverage-score-weighted TF-IDF on SeedTopic.

This script:
1. Loads the reference scRNA-seq data
2. Computes per-gene leverage scores from the reference
3. Recomputes the TF-IDF + PCA input with leverage weighting
4. Saves a new h5ad file with the weighted input
5. Runs SeedTopic with the new input
6. Evaluates against baseline

Usage:
    python experiments/v2_leverage_scores/run_leverage_test.py --dataset visiumHD_CRC_I
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
from seededntm.util import compute_tfidf_rep, compute_leverage_scores, compute_topic_prior
from experiments.evaluate import compute_metrics, load_ground_truth, load_baseline_proportions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_ROOT / "experiments" / "configs" / "datasets.json"
OUTPUT_DIR = Path(__file__).parent / "raw_outputs"


def compute_signature_matrix(adata_ref, ct_key: str, target_genes) -> np.ndarray:
    """Compute normalized mean expression per cell type, aligned to target genes.

    Returns (K, G_target) matrix where K = number of cell types.
    Uses chunked processing to handle large references (>100K cells).
    """
    cell_types = sorted(adata_ref.obs[ct_key].unique())
    logger.info(f"  {len(cell_types)} cell types, {adata_ref.n_obs} cells")

    # Build gene index mapping: target_gene -> ref_col_idx
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
        n_cells = mask.sum()
        X_ct = adata_ref.X[mask][:, target_ref_idx]
        if issparse(X_ct):
            X_ct = X_ct.toarray()

        # Normalize per cell (CPM), then log1p, then mean across cells
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
            import pandas as pd
            anno = pd.read_excel(ref_xlsx)
            anno = anno.set_index("Barcode")
            anno["Annotation"] = anno["Annotation"].apply(lambda x: x.replace(" ", "_"))
            adata = adata[adata.obs_names.isin(anno.index.values)]
            adata.obs["Annotation"] = anno.loc[adata.obs_names, "Annotation"].values
        sc.pp.filter_cells(adata, min_genes=200)
        return adata

    return None


def prepare_leverage_weighted_input(dataset_name: str, cfg: dict, output_dir: Path,
                                    power: float = 0.5) -> Path:
    """Compute leverage-weighted TF-IDF+PCA and save as new h5ad.

    Args:
        power: exponent to soften leverage scores. 1.0 = raw, 0.5 = sqrt, 0.0 = no effect.
            Lower values reduce the dynamic range of weights.

    Returns path to new h5ad file.
    """
    logger.info(f"Loading spatial data: {cfg['seedtopic_adata_path']}")
    adata = sc.read_h5ad(cfg["seedtopic_adata_path"])

    adata_ref = load_reference(cfg)
    if adata_ref is None:
        logger.warning(f"{dataset_name}: no reference data available")
        return None

    # Compute signature matrix aligned to spatial data genes
    logger.info("Computing per-type signature matrix...")
    target_genes = list(adata.var_names)
    X_ref, ref_types = compute_signature_matrix(
        adata_ref, cfg["ref_cell_type_key"], target_genes
    )
    logger.info(f"Signature matrix: {X_ref.shape} ({len(ref_types)} types x {len(target_genes)} genes)")
    del adata_ref  # free memory

    # Compute leverage scores
    leverage_raw = compute_leverage_scores(X_ref)
    logger.info(f"Raw leverage: min={leverage_raw.min():.3f}, max={leverage_raw.max():.3f}, "
                f"mean={leverage_raw.mean():.3f}, std={leverage_raw.std():.3f}")

    # Apply power softening: w = leverage^power (preserves mean≈1 after renorm)
    leverage = np.power(np.maximum(leverage_raw, 0), power)
    leverage = leverage / (leverage.mean() + 1e-8)  # renormalize mean to 1
    logger.info(f"Softened (power={power}): min={leverage.min():.3f}, max={leverage.max():.3f}, "
                f"mean={leverage.mean():.3f}, std={leverage.std():.3f}")

    # Save leverage scores for inspection
    np.save(output_dir / f"{dataset_name}_leverage_scores_raw.npy", leverage_raw)
    np.save(output_dir / f"{dataset_name}_leverage_scores_p{power}.npy", leverage)

    # Diagnostic: top-weighted genes
    top_idx = np.argsort(leverage)[-10:][::-1]
    top_genes = [(target_genes[i], f"{leverage[i]:.2f}") for i in top_idx]
    logger.info(f"Top leverage genes: {top_genes[:5]}")

    # Recompute TF-IDF + PCA with leverage weights
    X_counts = adata.obsm["rna_count"]
    logger.info(f"Recomputing TF-IDF+PCA with leverage weights (input shape: {X_counts.shape})")
    tfidf_pca_weighted = compute_tfidf_rep(X_counts, n_pcs=100, gene_weights=leverage)
    logger.info(f"Weighted TF-IDF+PCA shape: {tfidf_pca_weighted.shape}")

    # Save new h5ad with updated input
    adata.obsm["tfidf_pca_leverage"] = tfidf_pca_weighted

    out_path = output_dir / f"{dataset_name}_leverage_p{power}.h5ad"
    adata.write_h5ad(out_path)
    logger.info(f"Saved: {out_path}")

    return out_path


def run_seedtopic_with_input(dataset_name: str, cfg: dict, h5ad_path: Path,
                             input_key: str, output_dir: Path) -> tuple:
    """Run SeedTopic on a custom h5ad with specified input key."""
    import subprocess

    PYTHON = "/illumina-sdcolo-02/scratch/deep_learning/cqiao/software/micromamba/envs/chatdna-prd/seededntm/bin/python3.11"

    seeds_path = cfg["condition_feat_path"]
    with open(seeds_path) as f:
        marker_genes = json.load(f)
    idx_to_name = {v["topic_index"]: k for k, v in marker_genes.items()}
    ct_names = [idx_to_name[i] for i in range(len(marker_genes))]

    exp_outdir = str(output_dir / f"{dataset_name}_output")
    os.makedirs(exp_outdir, exist_ok=True)

    cmd_code = f"""
import sys
sys.argv = ['infer_seededntm',
    '--adata_h5ad_path', '{h5ad_path}',
    '--condition_feat_path', '{seeds_path}',
    '--key_input', '{input_key}',
    '--key_count_out', 'rna_count',
    '--num_topics', '{cfg["num_topics"]}',
    '--key_topic_prior', 'topic_prior',
    '--reg_topic_prior', '{cfg["reg_topic_prior"]}',
    '--wt_fusion_top_seed', '{cfg["wt_fusion_top_seed"]}',
    '--batch_size', '8192',
    '--exp_outdir', '{exp_outdir}',
] + {cfg.get("extra_args", [])}
from seededntm.main import main
main()
"""

    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    t0 = time.time()
    result = subprocess.run([PYTHON, "-c", cmd_code], capture_output=True, text=True, env=env)
    runtime = time.time() - t0

    if result.returncode != 0:
        logger.error(f"FAILED: {result.stderr[-1000:]}")
        return None, ct_names, runtime

    df_topic = pd.read_csv(os.path.join(exp_outdir, "df_topic.csv"), index_col=0)
    return df_topic.values, ct_names, runtime


def main():
    parser = argparse.ArgumentParser(description="Test leverage-score weighted TF-IDF")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--power", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                        help="Power exponents to soften leverage scores (sweep)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare h5ad files, don't run SeedTopic")
    args = parser.parse_args()

    with open(CONFIGS_PATH) as f:
        configs = json.load(f)

    if "all" in args.dataset:
        dataset_names = list(configs.keys())
    else:
        dataset_names = args.dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for ds_name in dataset_names:
        cfg = configs[ds_name]
        if cfg.get("ref_path") is None and cfg.get("ref_10x_h5_path") is None:
            logger.info(f"Skipping {ds_name}: no reference data")
            continue

        for power in args.power:
            run_key = f"{ds_name}_p{power}"
            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name} | Power: {power}")
            print(f"{'='*60}")

            # Step 1: Prepare leverage-weighted input
            h5ad_path = prepare_leverage_weighted_input(ds_name, cfg, OUTPUT_DIR, power=power)
            if h5ad_path is None:
                continue

            if args.prepare_only:
                continue

            # Step 2: Run SeedTopic with leverage-weighted input
            logger.info("Running SeedTopic with leverage-weighted TF-IDF+PCA...")
            proportions, ct_names, runtime = run_seedtopic_with_input(
                ds_name, cfg, h5ad_path, "tfidf_pca_leverage", OUTPUT_DIR
            )

            if proportions is None:
                logger.error(f"SeedTopic failed on {ds_name}")
                continue

            # Step 3: Evaluate
            gt_labels = load_ground_truth(ds_name)
            n = min(len(gt_labels), proportions.shape[0])
            metrics = compute_metrics(proportions[:n], ct_names, gt_labels[:n])

            # Compare with baseline
            bl_proportions, bl_ct_names = load_baseline_proportions(ds_name, "seedtopic")
            bl_metrics = compute_metrics(bl_proportions[:n], bl_ct_names, gt_labels[:n])

            print(f"\n  Baseline:        F1={bl_metrics['f1']:.3f}  AUPRC={bl_metrics['AUPRC']:.3f}  ARI={bl_metrics['ARI']:.3f}")
            print(f"  Leverage(p={power}): F1={metrics['f1']:.3f}  AUPRC={metrics['AUPRC']:.3f}  ARI={metrics['ARI']:.3f}")
            print(f"  Delta:           F1={metrics['f1']-bl_metrics['f1']:+.3f}  "
                  f"AUPRC={metrics['AUPRC']-bl_metrics['AUPRC']:+.3f}  "
                  f"ARI={metrics['ARI']-bl_metrics['ARI']:+.3f}")
            print(f"  Runtime: {runtime:.1f}s")

            # Save proportions
            df = pd.DataFrame(proportions, columns=ct_names)
            df.to_csv(OUTPUT_DIR / f"{ds_name}_leverage_p{power}_proportions.csv", index=False)

            results[run_key] = {
                "dataset": ds_name,
                "power": power,
                "baseline": bl_metrics,
                "leverage": metrics,
                "runtime_s": runtime,
                "delta_F1": metrics["f1"] - bl_metrics["f1"],
                "delta_AUPRC": metrics["AUPRC"] - bl_metrics["AUPRC"],
                "delta_ARI": metrics["ARI"] - bl_metrics["ARI"],
            }

    # Save results summary
    if results:
        out_path = Path(__file__).parent / "leverage_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

        # Print summary table
        print(f"\n{'='*70}")
        print(f"{'Dataset':<16} {'Power':<7} {'F1':>6} {'AUPRC':>7} {'ARI':>6} | {'dF1':>6} {'dAUPRC':>7} {'dARI':>6}")
        print(f"{'-'*70}")
        for key, r in results.items():
            print(f"{r['dataset']:<16} {r['power']:<7.2f} "
                  f"{r['leverage']['f1']:>6.3f} {r['leverage']['AUPRC']:>7.3f} {r['leverage']['ARI']:>6.3f}"
                  f" | {r['delta_F1']:>+6.3f} {r['delta_AUPRC']:>+7.3f} {r['delta_ARI']:>+6.3f}")


if __name__ == "__main__":
    main()
