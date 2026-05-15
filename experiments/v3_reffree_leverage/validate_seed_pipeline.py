#!/usr/bin/env python
"""Validate seed construction pipeline on xenium_BC.

This script demonstrates the pipeline on xenium_BC data and compares the
automatically constructed seeds to the manually produced leiden+ChatGPT seeds.

The comparison is structural (Jaccard overlap of seed gene sets), since running
SeedTopic requires a GPU computing node.

For the full downstream ARI comparison, run on a compute node:
    python experiments/v3_reffree_leverage/run_on_compute.sh

Usage (login node, no GPU needed for structural comparison):
    python experiments/v3_reffree_leverage/validate_seed_pipeline.py --structural-only

Usage (compute node, full comparison):
    python experiments/v3_reffree_leverage/validate_seed_pipeline.py --full
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

XENIUM_BC_H5AD = (
    "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM"
    "/experiments/xenium_breast_cancer/exp_seeding_leidenchatgpt"
    "/seededntm_adata_seeding_leidenChatGPT.h5ad"
)
MANUAL_SEEDS_PATH = (
    "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM"
    "/experiments/xenium_breast_cancer/bc_topic_seeds_leiden_chatgpt.txt"
)
OUTPUT_DIR = Path(__file__).parent / "seed_pipeline_validation"


def load_seeds(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_seed_sets(auto_seeds: dict, manual_seeds: dict) -> dict:
    """Compare two seed gene dictionaries structurally.

    Returns per-type Jaccard similarity and overall statistics.
    """
    results = {}

    # Get all genes in each set
    auto_all_genes = set()
    manual_all_genes = set()
    for v in auto_seeds.values():
        auto_all_genes.update(v["features"])
    for v in manual_seeds.values():
        manual_all_genes.update(v["features"])

    # Try to match types by name (after normalization)
    def normalize(name):
        return name.lower().replace("_", "").replace("-", "").replace(" ", "")

    auto_norm = {normalize(k): k for k in auto_seeds}
    manual_norm = {normalize(k): k for k in manual_seeds}

    matched_pairs = []
    for norm_name, auto_name in auto_norm.items():
        if norm_name in manual_norm:
            matched_pairs.append((auto_name, manual_norm[norm_name]))

    per_type = {}
    for auto_name, manual_name in matched_pairs:
        auto_genes = set(auto_seeds[auto_name]["features"])
        manual_genes = set(manual_seeds[manual_name]["features"])
        intersection = auto_genes & manual_genes
        union = auto_genes | manual_genes
        jaccard = len(intersection) / len(union) if union else 0
        per_type[auto_name] = {
            "manual_name": manual_name,
            "jaccard": jaccard,
            "overlap_count": len(intersection),
            "auto_count": len(auto_genes),
            "manual_count": len(manual_genes),
        }

    # Unmatched types
    auto_only = set(auto_seeds.keys()) - {p[0] for p in matched_pairs}
    manual_only = set(manual_seeds.keys()) - {p[1] for p in matched_pairs}

    # Gene-level overlap
    gene_jaccard = (
        len(auto_all_genes & manual_all_genes) / len(auto_all_genes | manual_all_genes)
        if (auto_all_genes | manual_all_genes) else 0
    )

    results = {
        "n_auto_types": len(auto_seeds),
        "n_manual_types": len(manual_seeds),
        "n_matched_types": len(matched_pairs),
        "auto_only_types": sorted(auto_only),
        "manual_only_types": sorted(manual_only),
        "gene_level_jaccard": gene_jaccard,
        "total_auto_genes": len(auto_all_genes),
        "total_manual_genes": len(manual_all_genes),
        "gene_overlap": len(auto_all_genes & manual_all_genes),
        "per_type": per_type,
    }
    return results


def run_structural_validation():
    """Compare pipeline output structure to manual seeds (no GPU needed)."""
    import scanpy as sc
    from seededntm.seed_construction import SeedConstructionPipeline, parse_llm_response
    from seededntm.prompt_templates import build_annotation_prompt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The xenium h5ad already has leiden clusters computed; use those directly
    # rather than re-running leiden (which requires igraph on the login node)
    logger.info(f"Loading h5ad: {XENIUM_BC_H5AD}")
    adata = sc.read_h5ad(XENIUM_BC_H5AD)
    logger.info(f"Data: {adata.n_obs} spots x {adata.n_vars} genes")

    # Check what obs columns contain clustering info
    cluster_keys = [k for k in adata.obs.columns if 'leiden' in k.lower() or 'cluster' in k.lower()]
    logger.info(f"Available cluster keys: {cluster_keys}")

    # Use existing cluster assignments for DE
    if "leiden" in adata.obs.columns:
        cluster_key = "leiden"
    elif "Cluster" in adata.obs.columns:
        cluster_key = "Cluster"
    else:
        cluster_key = cluster_keys[0] if cluster_keys else None

    if cluster_key is None:
        logger.error("No clustering found in h5ad. Need igraph for leiden.")
        sys.exit(1)

    n_clusters = adata.obs[cluster_key].nunique()
    logger.info(f"Using existing clusters from '{cluster_key}': {n_clusters} clusters")

    # Compute DE markers on existing clusters
    from scipy.sparse import issparse as _issparse
    X_check = adata.X[:100] if not _issparse(adata.X) else adata.X[:100].toarray()
    max_val = np.max(X_check)
    if max_val > 20:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        logger.info("Data already log-transformed")

    logger.info("Running rank_genes_groups for DE markers...")
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")

    # Extract markers per cluster
    result = adata.uns["rank_genes_groups"]
    group_names = result["names"].dtype.names
    markers_per_cluster = {}
    for group in group_names:
        genes = result["names"][group]
        logfcs = result["logfoldchanges"][group]
        filtered = []
        for gene, lfc in zip(genes, logfcs):
            if lfc >= 0.5:
                filtered.append((gene, float(lfc)))
            if len(filtered) >= 20:
                break
        markers_per_cluster[group] = filtered if filtered else [(g, 0.0) for g in genes[:20]]

    logger.info(f"Computed markers for {len(markers_per_cluster)} clusters")

    # Generate prompt
    prompt = build_annotation_prompt(
        markers_per_cluster,
        tissue_description="human breast cancer tissue (10x Xenium panel, 313 genes)",
        top_n=20,
    )
    prompt_path = OUTPUT_DIR / "prompt_round1.txt"
    prompt_path.write_text(prompt)
    logger.info(f"Prompt saved: {prompt_path}")

    # Structural comparison: check gene overlap between pipeline markers and manual seeds
    manual_seeds = load_seeds(MANUAL_SEEDS_PATH)

    pipeline_genes = set()
    for genes in markers_per_cluster.values():
        for g in genes:
            pipeline_genes.add(g[0] if isinstance(g, (tuple, list)) else g)

    manual_genes = set()
    for v in manual_seeds.values():
        manual_genes.update(v["features"])

    overlap = pipeline_genes & manual_genes
    logger.info(f"\nGene overlap with manual seeds:")
    logger.info(f"  Pipeline DE genes: {len(pipeline_genes)}")
    logger.info(f"  Manual seed genes: {len(manual_genes)}")
    logger.info(f"  Overlap: {len(overlap)} ({100*len(overlap)/len(manual_genes):.1f}% of manual)")

    # Save diagnostics
    diag = {
        "cluster_key": cluster_key,
        "n_clusters": n_clusters,
        "cluster_sizes": adata.obs[cluster_key].value_counts().to_dict(),
    }
    (OUTPUT_DIR / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit" / "diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))

    # Save comparison results
    comparison = {
        "pipeline_n_clusters": len(markers_per_cluster),
        "cluster_key_used": cluster_key,
        "manual_n_types": len(manual_seeds),
        "n_pipeline_de_genes": len(pipeline_genes),
        "n_manual_seed_genes": len(manual_genes),
        "gene_overlap": len(overlap),
        "overlap_pct_of_manual": 100 * len(overlap) / len(manual_genes) if manual_genes else 0,
        "prompt_path": str(prompt_path),
        "note": "Structural comparison only. Run with --full on GPU for ARI comparison.",
    }
    comp_path = OUTPUT_DIR / "structural_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    logger.info(f"\nStructural comparison saved: {comp_path}")

    print(f"\n{'='*60}")
    print("STRUCTURAL VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Pipeline clusters: {len(markers_per_cluster)} (from existing '{cluster_key}')")
    print(f"Manual types: {len(manual_seeds)}")
    print(f"DE gene overlap with manual seeds: {len(overlap)}/{len(manual_genes)} "
          f"({100*len(overlap)/len(manual_genes):.1f}%)")
    print(f"\nPrompt ready at: {prompt_path}")
    print(f"To complete validation, paste prompt into LLM and run finalize.")
    print(f"{'='*60}")


def run_full_validation():
    """Full validation including SeedTopic run (requires GPU)."""
    import time
    import pandas as pd
    import scanpy as sc

    from seededntm.seed_construction import SeedConstructionPipeline
    from experiments.evaluate import compute_metrics, load_ground_truth

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we already have seeds from the pipeline
    final_seeds_path = OUTPUT_DIR / "final_seeds.txt"
    if not final_seeds_path.exists():
        logger.error(f"No final seeds found at {final_seeds_path}")
        logger.error("Run structural validation first, then complete the LLM step.")
        sys.exit(1)

    auto_seeds = load_seeds(str(final_seeds_path))
    manual_seeds = load_seeds(MANUAL_SEEDS_PATH)

    # Structural comparison
    comparison = compare_seed_sets(auto_seeds, manual_seeds)
    logger.info(f"Structural comparison:")
    logger.info(f"  Matched types: {comparison['n_matched_types']}/{comparison['n_auto_types']}")
    logger.info(f"  Gene Jaccard: {comparison['gene_level_jaccard']:.3f}")

    # TODO: Run SeedTopic with auto seeds and compare ARI
    # This requires GPU and the full model training pipeline
    logger.info("Full ARI comparison requires GPU. Use run_on_compute.sh")

    comp_path = OUTPUT_DIR / "full_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2, default=str))
    logger.info(f"Full comparison saved: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate seed construction pipeline")
    parser.add_argument("--structural-only", action="store_true", default=True,
                        help="Only structural comparison (no GPU needed)")
    parser.add_argument("--full", action="store_true",
                        help="Full comparison including SeedTopic ARI (needs GPU)")
    args = parser.parse_args()

    if args.full:
        run_full_validation()
    else:
        run_structural_validation()


if __name__ == "__main__":
    main()
