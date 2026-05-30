#!/usr/bin/env python
"""Ref-free experiment v2: improved Leiden-LLM seeds + SeedTopic deconvolution.

Two-phase design:
  Phase 1 (seeds): Run SeedConstructionPipeline on each dataset (requires LLM/internet, no GPU)
  Phase 2 (eval):  Run SeedTopic deconvolution with improved seeds (requires GPU, no internet)

Usage:
    # Phase 1 only (login node with internet):
    python experiments/run_reffree_v2.py --phase seeds --all

    # Phase 2 only (compute node with GPU, uses saved seeds):
    python experiments/run_reffree_v2.py --phase eval --all

    # Both phases:
    python experiments/run_reffree_v2.py --phase both --all

    # Single dataset:
    python experiments/run_reffree_v2.py --phase both --dataset visiumHD_CRC_I
"""

import argparse
import gc
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("AZURE_OPENAI_API_KEY", "978e40b462a14c53adb242077ca5d362")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://rnd-artificialintelligenceopenai-dev.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-2")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "benchmark_fair"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "experiments" / "benchmark_fair" / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "marker_agent_eval" / "reffree_v2"

DATASET_CONTEXTS = {
    "visium_NPC": {
        "target_types": ["B", "Myeloid", "T", "Treg", "fibroblast", "normal", "tumor"],
        "tissue_desc": "Nasopharyngeal carcinoma tumor microenvironment, nasopharynx tissue",
        "organ": "Nasopharynx",
        "condition": "EBV+ squamous cell carcinoma",
        "K": 7,
    },
    "xenium_BC": {
        "target_types": [
            "B_Cells", "CD4+_T_Cells", "CD8+_T_Cells", "DCIS_1", "DCIS_2",
            "Endothelial", "IRF7+_DCs", "Invasive_Tumor", "LAMP3+_DCs",
            "Macrophages_1", "Macrophages_2", "Mast_Cells", "Myoepi_ACTA2+",
            "Myoepi_KRT15+", "Perivascular-Like", "Prolif_Invasive_Tumor",
            "Stromal", "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid",
        ],
        "tissue_desc": "Breast cancer DCIS and invasive carcinoma, breast tissue",
        "organ": "Breast",
        "condition": "DCIS and invasive carcinoma",
        "K": 19,
    },
    "visiumHD_CRC_I": {
        "target_types": ["CAF", "Endothelial", "Macrophage", "Pericytes", "Tumor III"],
        "tissue_desc": "Colorectal cancer tumor microenvironment, colon tissue, ROI 1",
        "organ": "Colon",
        "condition": "Colorectal adenocarcinoma",
        "K": 5,
    },
    "visiumHD_CRC_II": {
        "target_types": ["CAF", "Endothelial", "Macrophage", "Neutrophil", "Pericytes", "Tumor III"],
        "tissue_desc": "Colorectal cancer tumor microenvironment, colon tissue, ROI 2",
        "organ": "Colon",
        "condition": "Colorectal adenocarcinoma",
        "K": 6,
    },
}

ALL_METRICS = ["F1", "Precision", "Recall", "AMI", "ARI", "AUPRC", "Pearson", "Spearman", "RMSE", "JSD"]


# ---------------------------------------------------------------------------
# Phase 1: Seed Generation
# ---------------------------------------------------------------------------

def generate_seeds(dataset_name: str) -> dict:
    """Run SeedConstructionPipeline for a single dataset."""
    import scanpy as sc
    from seededntm.seed_construction import SeedConstructionPipeline

    ctx = DATASET_CONTEXTS[dataset_name]
    adata_path = PROCESSED_DIR / dataset_name / "spatial_adata.h5ad"

    logger.info(f"[SEEDS] {dataset_name}: loading {adata_path}")
    adata = sc.read_h5ad(str(adata_path))

    pipeline = SeedConstructionPipeline(
        adata=adata,
        tissue_description=ctx["tissue_desc"],
        expected_cell_types=ctx["target_types"],
        top_n_genes=20,
        species="Human",
        organ=ctx["organ"],
        condition=ctx["condition"],
    )

    seeds = pipeline.run()

    seeds_dir = OUTPUT_DIR / "seeds"
    os.makedirs(seeds_dir, exist_ok=True)

    seeds_path = seeds_dir / f"{dataset_name}_seeds_improved.json"
    with open(seeds_path, "w") as f:
        json.dump(seeds, f, indent=2)
    logger.info(f"[SEEDS] {dataset_name}: saved {seeds_path}")

    prov_path = seeds_dir / f"{dataset_name}_provenance.json"
    provenance = pipeline.get_provenance()
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    for ct, info in seeds.items():
        src = info.get("source", "leiden_de")
        logger.info(f"  {ct}: {len(info['features'])} genes (topic_index={info['topic_index']}, source={src})")

    return seeds


def run_phase_seeds(datasets: list, n_workers: int = 4):
    """Phase 1: Generate improved seeds for all specified datasets (parallel)."""
    logger.info("=" * 70)
    logger.info("PHASE 1: SEED GENERATION (%d workers)", n_workers)
    logger.info("=" * 70)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_seeds, ds): ds for ds in datasets}
        for future in as_completed(futures):
            ds = futures[future]
            try:
                future.result()
                logger.info(f"[SEEDS] {ds}: completed")
            except Exception as e:
                logger.error(f"[SEEDS] {ds}: FAILED - {e}")
                import traceback
                traceback.print_exc()
    logger.info(f"[SEEDS] All datasets done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Phase 2: Evaluation
# ---------------------------------------------------------------------------

def _load_trainer():
    """Import run_perspot_dirichlet from the ablation trainer module."""
    trainer_path = str(
        PROJECT_ROOT / "experiments" / "v3_reffree_leverage" / "run_ablation_v21d_universal_prior.py"
    )
    spec = importlib.util.spec_from_file_location("trainer_module", trainer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_perspot_dirichlet


def run_seedtopic_seeded(dataset_name, seeds_path, random_seed, device="cuda"):
    """Run SeedTopic with explicit random seed control."""
    import scanpy as sc
    from scipy.stats import entropy as scipy_entropy
    from scipy.sparse import issparse

    from run_fair_benchmark import DATASETS, SEEDTOPIC_CONFIG
    from seededntm.util import compute_topic_prior, compute_topic_prior_idf, spatial_smooth_prior
    from seededntm.experiment import preprocess_ST
    from experiments.run_seedtopic import compute_representation

    run_perspot_dirichlet = _load_trainer()

    cfg = DATASETS[dataset_name]
    processed_dir = cfg["processed_dir"]
    K = cfg["K"]

    adata_path = processed_dir / "spatial_adata.h5ad"
    adata = sc.read_h5ad(str(adata_path))
    adata.var_names_make_unique()

    with open(seeds_path) as f:
        seeds = json.load(f)
    ct_names = [None] * K
    for name, info in seeds.items():
        idx = info["topic_index"]
        if idx < K:
            ct_names[idx] = name

    missing_genes = {}
    for ct_name, info in seeds.items():
        genes = info.get("features", [])
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            missing_genes[ct_name] = missing
    if missing_genes:
        logger.warning(f"Missing genes in panel: {missing_genes}")
        for ct_name, mg in missing_genes.items():
            seeds[ct_name]["features"] = [g for g in seeds[ct_name]["features"] if g in adata.var_names]

    X_counts = adata.X
    var_names = np.array(adata.var_names)

    rep_mode = SEEDTOPIC_CONFIG["representation"]
    rep = compute_representation(X_counts, var_names, seeds, rep_mode, K)

    adata.obsm["seedtopic_input_rep"] = rep

    if "rna_count" not in adata.obsm:
        if issparse(X_counts):
            adata.obsm["rna_count"] = np.asarray(X_counts.todense(), dtype=np.float32)
        else:
            adata.obsm["rna_count"] = np.asarray(X_counts, dtype=np.float32)

    prior_mode = SEEDTOPIC_CONFIG["prior_mode"]
    if "idf" in prior_mode:
        raw_prior = compute_topic_prior_idf(adata, seeds, temperature=SEEDTOPIC_CONFIG["temperature"])
    else:
        raw_prior = compute_topic_prior(adata, seeds, temperature=SEEDTOPIC_CONFIG["temperature"])

    if "spatial" in prior_mode and "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=SEEDTOPIC_CONFIG["k_neighbors"])
    else:
        topic_prior = raw_prior

    alpha = SEEDTOPIC_CONFIG["alpha"]
    floor = SEEDTOPIC_CONFIG["floor"]
    per_spot_ent = scipy_entropy(topic_prior, axis=1)
    max_ent = np.log(K)
    confidence = np.clip(1.0 - per_spot_ent / max_ent, 0.0, 1.0)
    alpha_np = (alpha * (floor + (1.0 - floor) * confidence)).astype(np.float32)

    if issparse(X_counts):
        out_counts = np.asarray(X_counts.todense(), dtype=np.float32)
    else:
        out_counts = np.asarray(X_counts, dtype=np.float32)

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

    early_stop = (dataset_name == "xenium_BC")

    with tempfile.TemporaryDirectory(prefix="seedtopic_reffree_") as tmp_dir:
        df_topic = run_perspot_dirichlet(
            exp_data, K, tmp_dir, alpha_np, device,
            early_stop=early_stop,
            n_epochs=SEEDTOPIC_CONFIG["n_epochs"],
            batch_size=SEEDTOPIC_CONFIG["batch_size"],
            wt_fusion_top_seed=SEEDTOPIC_CONFIG["wt_fusion_top_seed"],
            use_nb_obs=SEEDTOPIC_CONFIG["use_nb_obs"],
            seed=random_seed,
        )

    proportions = df_topic.values
    return proportions, ct_names


def _eval_single_run(args_tuple):
    """Evaluate a single (dataset, seed) pair. Top-level function for pickling."""
    dataset_name, random_seed, device = args_tuple
    import gc as _gc
    from run_fair_benchmark import DATASETS, compute_metrics

    cfg = DATASETS[dataset_name]
    processed_dir = cfg["processed_dir"]
    K = cfg["K"]

    seeds_path = OUTPUT_DIR / "seeds" / f"{dataset_name}_seeds_improved.json"
    if not seeds_path.exists():
        return dataset_name, random_seed, {"seed": random_seed, "error": "seeds not found"}

    gt_csv = processed_dir / "ground_truth.csv"
    gt_df = pd.read_csv(str(gt_csv))
    gt_labels = gt_df["ground_truth"].values.astype(str)

    t0 = time.time()
    try:
        proportions, ct_names = run_seedtopic_seeded(dataset_name, seeds_path, random_seed, device)
        elapsed = time.time() - t0
        metrics = compute_metrics(proportions, ct_names, gt_labels)
        run_result = {"seed": random_seed, "runtime_s": round(elapsed, 1)}
        run_result.update(metrics)
        logger.info(f"[EVAL] {dataset_name} seed={random_seed}: F1={metrics['F1']:.4f} "
                    f"AMI={metrics['AMI']:.4f} AUPRC={metrics['AUPRC']:.4f} time={elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[EVAL] {dataset_name} seed={random_seed}: FAILED ({e})")
        import traceback
        traceback.print_exc()
        run_result = {"seed": random_seed, "runtime_s": round(elapsed, 1), "error": str(e)}

    _gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return dataset_name, random_seed, run_result


def run_phase_eval(datasets: list, random_seeds: list, device: str, n_workers: int = 4,
                   max_xenium_workers: int = 2):
    """Phase 2: Run SeedTopic deconvolution with improved seeds (parallel)."""
    logger.info("=" * 70)
    logger.info("PHASE 2: EVALUATION (%d workers, xenium max %d)", n_workers, max_xenium_workers)
    logger.info("=" * 70)

    results_dir = OUTPUT_DIR / "results"
    os.makedirs(results_dir, exist_ok=True)

    non_xenium_tasks = []
    xenium_tasks = []
    for ds in datasets:
        for rs in random_seeds:
            task = (ds, rs, device)
            if "xenium" in ds:
                xenium_tasks.append(task)
            else:
                non_xenium_tasks.append(task)

    all_run_results = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        pending_futures = {}

        for task in non_xenium_tasks:
            f = executor.submit(_eval_single_run, task)
            pending_futures[f] = task

        xenium_pending = list(xenium_tasks)
        xenium_active = []
        while xenium_pending and len(xenium_active) < max_xenium_workers:
            task = xenium_pending.pop(0)
            f = executor.submit(_eval_single_run, task)
            pending_futures[f] = task
            xenium_active.append(f)

        while pending_futures:
            done_futures = [f for f in pending_futures if f.done()]
            if not done_futures:
                time.sleep(0.5)
                continue
            for future in done_futures:
                ds_name, rs, run_result = future.result()
                all_run_results.setdefault(ds_name, []).append(run_result)
                del pending_futures[future]

                if future in xenium_active:
                    xenium_active.remove(future)
                    if xenium_pending and len(xenium_active) < max_xenium_workers:
                        task = xenium_pending.pop(0)
                        f = executor.submit(_eval_single_run, task)
                        pending_futures[f] = task
                        xenium_active.append(f)

    all_results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_type": "reffree_v2",
        "random_seeds": random_seeds,
        "datasets": {},
    }

    for ds in datasets:
        runs = all_run_results.get(ds, [])
        if not runs:
            continue
        from run_fair_benchmark import DATASETS
        K = DATASETS[ds]["K"]
        result = {"runs": runs, "n_types": K}
        successful = [r for r in runs if "error" not in r]
        if successful:
            mean_m = {}
            std_m = {}
            for m in ALL_METRICS:
                vals = [r[m] for r in successful if m in r and not np.isnan(r[m])]
                if vals:
                    mean_m[m] = round(float(np.mean(vals)), 4)
                    std_m[m] = round(float(np.std(vals)), 4)
            result["mean"] = mean_m
            result["std"] = std_m
        all_results["datasets"][ds] = result

        per_ds_path = results_dir / f"{ds}_results.json"
        with open(per_ds_path, "w") as f:
            json.dump({ds: result}, f, indent=2, default=str)

    results_path = results_dir / "reffree_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_path}")

    print_comparison(all_results)
    return all_results


def print_comparison(results: dict):
    """Print comparison table including previous heavy test results."""
    print(f"\n{'=' * 100}")
    print("COMPARISON: improved_leiden vs previous seed sources")
    print(f"{'=' * 100}")

    prev_path = PROJECT_ROOT / "experiments" / "marker_agent_eval" / "heavy" / "heavy_results.json"
    prev = {}
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)

    header = f"{'Dataset':<18} {'Source':<20} {'F1':>6} {'AMI':>6} {'ARI':>6} {'AUPRC':>6} {'Pearson':>8} {'JSD':>6}"
    print(header)
    print("-" * 100)

    for ds_name in results.get("datasets", {}):
        # Previous sources
        if ds_name in prev.get("datasets", {}):
            for source in ["reference", "leiden_chatgpt", "agent"]:
                src_data = prev["datasets"][ds_name].get("seed_sources", {}).get(source, {})
                mean = src_data.get("mean", {})
                if mean:
                    print(f"{ds_name:<18} {source:<20} "
                          f"{mean.get('F1', 0):.3f}  {mean.get('AMI', 0):.3f}  "
                          f"{mean.get('ARI', 0):.3f}  {mean.get('AUPRC', 0):.3f}  "
                          f"{mean.get('Pearson', 0):.4f}  {mean.get('JSD', 0):.3f}")

        # New improved leiden
        ds_data = results["datasets"][ds_name]
        mean = ds_data.get("mean", {})
        if mean:
            print(f"{ds_name:<18} {'improved_leiden':<20} "
                  f"{mean.get('F1', 0):.3f}  {mean.get('AMI', 0):.3f}  "
                  f"{mean.get('ARI', 0):.3f}  {mean.get('AUPRC', 0):.3f}  "
                  f"{mean.get('Pearson', 0):.4f}  {mean.get('JSD', 0):.3f}")
        print()

    print(f"{'=' * 100}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ref-free experiment v2: improved Leiden-LLM seeds + SeedTopic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", choices=["seeds", "eval", "both"], default="both",
                        help="Which phase to run (default: both)")
    parser.add_argument("--all", action="store_true", help="Run all 4 datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset: " + ", ".join(DATASET_CONTEXTS.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated random seeds for evaluation (default: 0,1,2)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for seed generation (default: 4)")
    args = parser.parse_args()

    if not args.all and args.dataset is None:
        parser.error("Specify --all or --dataset DATASET")

    datasets = list(DATASET_CONTEXTS.keys()) if args.all else [args.dataset]
    random_seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dataset and args.dataset not in DATASET_CONTEXTS:
        parser.error(f"Unknown dataset: {args.dataset}. Choices: {list(DATASET_CONTEXTS.keys())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Ref-free v2 experiment")
    logger.info(f"  Phase: {args.phase}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Random seeds: {random_seeds}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Output: {OUTPUT_DIR}")

    total_t0 = time.time()

    if args.phase in ("seeds", "both"):
        run_phase_seeds(datasets, n_workers=args.workers)

    if args.phase in ("eval", "both"):
        run_phase_eval(datasets, random_seeds, args.device, n_workers=args.workers)

    logger.info(f"\nTotal runtime: {time.time() - total_t0:.0f}s")


if __name__ == "__main__":
    main()
