#!/usr/bin/env python
"""Test reference-free leverage scoring strategies for SeedTopic.

Compares three approaches + two dimensionality reduction methods:
  - Approach 1: Pseudo-signatures from seed-guided topic_prior
  - Approach 2a: Self-leverage from top-K SVD of TF-IDF
  - Approach 3: Seed specificity weighting
  - Dimensionality reduction: PCA(100) vs CountSketch(512)

Also runs baselines:
  - No leverage (original pipeline)
  - Reference-based leverage (for comparison on datasets with references)

Usage:
    python experiments/v3_reffree_leverage/run_reffree_test.py --dataset xenium_BC_leiden
    python experiments/v3_reffree_leverage/run_reffree_test.py --dataset visium_NPC
    python experiments/v3_reffree_leverage/run_reffree_test.py --dataset all
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
    compute_reffree_leverage,
)
from experiments.evaluate import compute_metrics, load_ground_truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_ROOT / "experiments" / "configs" / "datasets.json"
OUTPUT_DIR = Path(__file__).parent / "raw_outputs"

REFFREE_DATASETS = {
    "xenium_BC_leiden": {
        "description": "Xenium Breast Cancer with leiden+ChatGPT seeds (reference-free scenario)",
        "seedtopic_adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/xenium_breast_cancer/exp_seeding_leidenchatgpt/seededntm_adata_seeding_leidenChatGPT.h5ad",
        "condition_feat_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/xenium_breast_cancer/bc_topic_seeds_leiden_chatgpt.txt",
        "ground_label": "Cluster",
        "num_topics": 19,
        "reg_topic_prior": 0.9,
        "wt_fusion_top_seed": 1.0,
        "extra_args": ["--early_stop"],
        "has_reference": True,
        "ref_seeds_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/xenium_breast_cancer/bc_topic_seeds_ref_scRNAseq.txt",
        "ref_adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/xenium_breast_cancer/exp_seeding_ref_scrnaseq/seededntm_adata_seeding_ref_scrnaseq.h5ad",
    },
    "visium_NPC": {
        "description": "Visium NPC with reference seeds (test ref-free methods on ref-derived seeds)",
        "seedtopic_adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visium_NPC/exp_seeding_ref_scrnaseq/seededntm_adata_seeding_ref_scrnaseq.h5ad",
        "condition_feat_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visium_NPC/NPC_topic_seeds_ref_scRNAseq.txt",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visium_NPC/NPC_single_cell_ref_adata.h5ad",
        "ref_cell_type_key": "Cluster",
        "ground_label": "cell_type",
        "num_topics": 7,
        "reg_topic_prior": 0.9,
        "wt_fusion_top_seed": 1.0,
        "extra_args": [],
        "has_reference": True,
    },
    "visiumHD_CRC_I": {
        "description": "VisiumHD CRC dataset 1, 2580 spots, 5 GT types",
        "seedtopic_adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visiumHD_CRC/dataset1/exp_seeding_ref_scrnaseq/SeededNTM_1_10_top20markers.h5ad",
        "condition_feat_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visiumHD_CRC/dataset1/exp_seeding_ref_scrnaseq/topic_seeds_1_10_20.txt",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visiumHD_CRC/CRC16um/Colon_Cancer/filtered_feature_bc_matrix_CRC_reference.h5ad",
        "ref_cell_type_key": "Cluster",
        "ground_label": "DeconvolutionLabel1_name",
        "num_topics": 5,
        "reg_topic_prior": 0.99,
        "wt_fusion_top_seed": 1.0,
        "extra_args": ["--use_nb_obs"],
        "has_reference": True,
    },
    "visiumHD_CRC_II": {
        "description": "VisiumHD CRC dataset 2, 2275 spots, 6 GT types",
        "seedtopic_adata_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visiumHD_CRC/dataset2/exp_seeding_ref_scrnaseq/SeededNTM_2_10_top20markers.h5ad",
        "condition_feat_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visiumHD_CRC/dataset2/exp_seeding_ref_scrnaseq/topic_seeds_2_10_20.txt",
        "ref_path": "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM/experiments/visiumHD_CRC/CRC16um/Colon_Cancer/filtered_feature_bc_matrix_CRC_reference.h5ad",
        "ref_cell_type_key": "Cluster",
        "ground_label": "DeconvolutionLabel1_name",
        "num_topics": 6,
        "reg_topic_prior": 0.99,
        "wt_fusion_top_seed": 1.0,
        "extra_args": ["--use_nb_obs"],
        "has_reference": True,
    },
}


def load_seeds(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_signature_matrix(adata_ref, ct_key: str, target_genes) -> np.ndarray:
    """Compute normalized mean expression per cell type, aligned to target genes."""
    cell_types = sorted(adata_ref.obs[ct_key].unique())
    logger.info(f"  Reference: {len(cell_types)} cell types, {adata_ref.n_obs} cells")

    ref_gene_list = list(adata_ref.var_names)
    ref_gene_set = set(ref_gene_list)
    ref_gene_to_idx = {g: i for i, g in enumerate(ref_gene_list)}

    target_in_ref = [g for g in target_genes if g in ref_gene_set]
    target_ref_idx = [ref_gene_to_idx[g] for g in target_in_ref]
    target_out_idx = [i for i, g in enumerate(target_genes) if g in ref_gene_set]

    logger.info(f"  Gene overlap: {len(target_in_ref)}/{len(target_genes)} target genes in ref")

    signatures = np.zeros((len(cell_types), len(target_genes)), dtype=np.float64)
    for ct_i, ct in enumerate(cell_types):
        mask = (adata_ref.obs[ct_key] == ct).values
        X_ct = adata_ref.X[mask][:, target_ref_idx]
        if issparse(X_ct):
            X_ct = X_ct.toarray()
        row_sums = X_ct.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_norm = np.log1p(X_ct / row_sums * 1e4)
        signatures[ct_i, target_out_idx] = X_norm.mean(axis=0)

    return signatures, cell_types


def load_reference_for_dataset(ds_name, cfg):
    """Load reference and compute leverage for comparison."""
    ref_path = cfg.get("ref_path")
    if ref_path is not None:
        logger.info(f"Loading reference h5ad: {ref_path}")
        return sc.read_h5ad(ref_path), cfg["ref_cell_type_key"]
    elif ds_name == "xenium_BC_leiden":
        from experiments.v2_leverage_scores.run_leverage_test import load_reference
        with open(CONFIGS_PATH) as f:
            all_configs = json.load(f)
        return load_reference(all_configs["xenium_BC"]), "Annotation"
    return None, None


def prepare_reffree_input(ds_name, cfg, output_dir, method="pseudo_sig",
                          dim_red="pca", sketch_dim=512):
    """Compute ref-free leverage-weighted input and save as h5ad.

    Args:
        method: "pseudo_sig", "self_leverage_k", "seed_specificity", or "none" (baseline).
        dim_red: "pca" or "sketch" for dimensionality reduction.
    """
    logger.info(f"Loading spatial data: {cfg['seedtopic_adata_path']}")
    adata = sc.read_h5ad(cfg["seedtopic_adata_path"])
    X_counts = adata.obsm["rna_count"]
    var_names = list(adata.var_names)

    seeds = load_seeds(cfg["condition_feat_path"])

    if method == "none":
        leverage = None
        logger.info("No leverage weighting (baseline)")
    elif method == "ref_based":
        adata_ref, ct_key = load_reference_for_dataset(ds_name, cfg)
        if adata_ref is None:
            logger.warning("No reference available for ref_based")
            return None
        signatures, _ = compute_signature_matrix(adata_ref, ct_key, var_names)
        leverage = compute_leverage_scores(signatures)
        del adata_ref
        logger.info(f"Ref-based leverage: min={leverage.min():.3f}, max={leverage.max():.3f}, "
                    f"mean={leverage.mean():.3f}")
    else:
        leverage = compute_reffree_leverage(
            X_counts, seeds, var_names, method=method,
            n_topics=cfg["num_topics"]
        )
        logger.info(f"{method} leverage: min={leverage.min():.3f}, max={leverage.max():.3f}, "
                    f"mean={leverage.mean():.3f}")

    # Compute representation
    if dim_red == "pca":
        rep = compute_tfidf_rep(X_counts, n_pcs=100, gene_weights=leverage)
        input_key = f"tfidf_pca_{method}"
    elif dim_red == "sketch":
        tfidf = compute_tfidf_rep(X_counts, n_pcs=None, gene_weights=leverage)
        rep = compute_countsketch_rep(tfidf, sketch_dim=sketch_dim,
                                      leverage_scores=leverage, seed=42)
        input_key = f"tfidf_sketch_{method}"
    elif dim_red == "raw_pca":
        # Ablation: skip TF-IDF, apply weights directly to log-normalized counts
        from sklearn.decomposition import PCA
        X = np.asarray(X_counts.toarray() if issparse(X_counts) else X_counts, dtype=np.float32)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_norm = np.log1p(X / row_sums * 1e4)
        if leverage is not None:
            X_norm = X_norm * leverage.reshape(1, -1)
        rep = PCA(n_components=100).fit_transform(X_norm).astype(np.float32)
        input_key = f"raw_pca_{method}"
    elif dim_red == "raw_sketch":
        # Ablation: skip TF-IDF, apply weights + CountSketch on log-normalized counts
        X = np.asarray(X_counts.toarray() if issparse(X_counts) else X_counts, dtype=np.float32)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_norm = np.log1p(X / row_sums * 1e4)
        if leverage is not None:
            X_norm = X_norm * leverage.reshape(1, -1)
        rep = compute_countsketch_rep(X_norm, sketch_dim=sketch_dim,
                                      leverage_scores=leverage, seed=42)
        input_key = f"raw_sketch_{method}"
    else:
        raise ValueError(f"Unknown dim_red: {dim_red}")

    logger.info(f"Output representation: {rep.shape}")
    adata.obsm[input_key] = rep

    # Recompute topic_prior from seeds at runtime (not stale stored value)
    topic_prior = compute_topic_prior(adata, seeds, temperature=0.8)
    adata.obsm["topic_prior"] = topic_prior

    out_path = output_dir / f"{ds_name}_{method}_{dim_red}.h5ad"
    adata.write_h5ad(out_path)
    logger.info(f"Saved: {out_path}")

    del adata, rep
    import gc
    gc.collect()

    return out_path, input_key


def run_seedtopic(ds_name, cfg, h5ad_path, input_key, output_dir):
    """Run SeedTopic in-process with the specified input."""
    import torch

    seeds_path = cfg["condition_feat_path"]
    seeds = load_seeds(seeds_path)
    idx_to_name = {v["topic_index"]: k for k, v in seeds.items()}
    ct_names = [idx_to_name[i] for i in range(len(seeds))]

    exp_outdir = str((output_dir / f"{ds_name}_{input_key}_output").resolve())
    os.makedirs(exp_outdir, exist_ok=True)

    from seededntm.main import do_exp

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    extra_args = cfg.get("extra_args", [])
    early_stop = "--early_stop" in extra_args
    use_nb_obs = "--use_nb_obs" in extra_args

    t0 = time.time()
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
        return None, ct_names, time.time() - t0

    runtime = time.time() - t0

    df_topic_path = os.path.join(exp_outdir, "df_topic.csv")
    if os.path.exists(df_topic_path):
        df_topic = pd.read_csv(df_topic_path, index_col=0)
        return df_topic.values, ct_names, runtime

    logger.error(f"No output produced in {exp_outdir}")
    return None, ct_names, runtime


def main():
    parser = argparse.ArgumentParser(description="Test reference-free leverage scoring")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--methods", nargs="+",
                        default=["none", "pseudo_sig", "self_leverage_k", "seed_specificity"],
                        help="Leverage methods to test")
    parser.add_argument("--dim-red", nargs="+", default=["pca"],
                        choices=["pca", "sketch", "raw_pca", "raw_sketch"],
                        help="Dimensionality reduction method(s)")
    parser.add_argument("--sketch-dim", type=int, default=512)
    parser.add_argument("--include-ref-based", action="store_true",
                        help="Include reference-based leverage for comparison")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare h5ad files, don't run SeedTopic")
    args = parser.parse_args()

    if "all" in args.dataset:
        dataset_names = list(REFFREE_DATASETS.keys())
    else:
        dataset_names = args.dataset

    methods = list(args.methods)
    if args.include_ref_based:
        methods.append("ref_based")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for ds_name in dataset_names:
        if ds_name not in REFFREE_DATASETS:
            logger.warning(f"Unknown dataset: {ds_name}")
            continue

        cfg = REFFREE_DATASETS[ds_name]

        for method in methods:
            for dim_red in args.dim_red:
                run_key = f"{ds_name}_{method}_{dim_red}"
                print(f"\n{'='*60}")
                print(f"Dataset: {ds_name} | Method: {method} | DimRed: {dim_red}")
                print(f"{'='*60}")

                result = prepare_reffree_input(
                    ds_name, cfg, OUTPUT_DIR,
                    method=method, dim_red=dim_red, sketch_dim=args.sketch_dim
                )
                if result is None:
                    continue
                h5ad_path, input_key = result

                if args.prepare_only:
                    continue

                logger.info(f"Running SeedTopic with {input_key}...")
                proportions, ct_names, runtime = run_seedtopic(
                    ds_name, cfg, h5ad_path, input_key, OUTPUT_DIR
                )

                if proportions is None:
                    logger.error(f"SeedTopic failed for {run_key}")
                    results[run_key] = {
                        "dataset": ds_name, "method": method, "dim_red": dim_red,
                        "status": "failed",
                    }
                    continue

                gt_ds_name = ds_name.replace("_leiden", "")
                gt_labels = load_ground_truth(gt_ds_name)
                n = min(len(gt_labels), proportions.shape[0])
                metrics = compute_metrics(proportions[:n], ct_names, gt_labels[:n])

                print(f"\n  {method}/{dim_red}: F1={metrics['f1']:.3f}  "
                      f"AUPRC={metrics['AUPRC']:.3f}  ARI={metrics['ARI']:.3f}  "
                      f"Runtime={runtime:.1f}s")

                df = pd.DataFrame(proportions, columns=ct_names)
                df.to_csv(OUTPUT_DIR / f"{run_key}_proportions.csv", index=False)

                results[run_key] = {
                    "dataset": ds_name,
                    "method": method,
                    "dim_red": dim_red,
                    "metrics": metrics,
                    "runtime_s": runtime,
                    "status": "ok",
                }

    # Save results (merge with existing)
    if results:
        out_path = Path(__file__).parent / "reffree_results.json"
        existing = {}
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
        existing.update(results)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"\nSaved results: {out_path}")

        # Summary table (show all results including previously saved)
        all_results = existing
        print(f"\n{'='*90}")
        print(f"{'Dataset':<18} {'Method':<18} {'DimRed':<7} "
              f"{'F1':>6} {'AUPRC':>7} {'ARI':>6} {'AMI':>6} {'Time':>6}")
        print(f"{'-'*90}")
        for key, r in all_results.items():
            if r.get("status") != "ok":
                print(f"{r['dataset']:<18} {r['method']:<18} {r['dim_red']:<7} FAILED")
                continue
            m = r["metrics"]
            rt = r['runtime_s']
            rt_str = f"{rt:>5.0f}s" if isinstance(rt, (int, float)) else f"{rt:>6}"
            print(f"{r['dataset']:<18} {r['method']:<18} {r['dim_red']:<7} "
                  f"{m['f1']:>6.3f} {m['AUPRC']:>7.3f} {m['ARI']:>6.3f} "
                  f"{m['AMI']:>6.3f} {rt_str}")

        # Compute deltas relative to no-leverage baseline
        print(f"\n{'='*70}")
        print("Deltas relative to no-leverage baseline:")
        print(f"{'Dataset':<18} {'Method':<18} {'DimRed':<7} {'dARI':>7} {'dAUPRC':>7} {'dF1':>7}")
        print(f"{'-'*70}")
        for ds in set(r["dataset"] for r in all_results.values()):
            baseline_key = f"{ds}_none_pca"
            if baseline_key not in all_results or all_results[baseline_key].get("status") != "ok":
                continue
            bl = all_results[baseline_key]["metrics"]
            for key, r in all_results.items():
                if not key.startswith(ds) or r.get("status") != "ok":
                    continue
                if key == baseline_key:
                    continue
                m = r["metrics"]
                print(f"{r['dataset']:<18} {r['method']:<18} {r['dim_red']:<7} "
                      f"{m['ARI']-bl['ARI']:>+7.3f} {m['AUPRC']-bl['AUPRC']:>+7.3f} "
                      f"{m['f1']-bl['f1']:>+7.3f}")


if __name__ == "__main__":
    main()
