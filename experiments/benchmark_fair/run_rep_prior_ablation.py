#!/usr/bin/env python
"""Representation x Prior Mode Ablation for SeedTopic.

Tests all 9 (rep x prior) configs across 4 datasets = 36 experiments.
Selects the universal best config by average rank across all metrics.

Usage:
  python run_rep_prior_ablation.py --idx N  # run single experiment
  python run_rep_prior_ablation.py --analyze  # compute ranks and select winner
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import issparse
from scipy.stats import entropy as scipy_entropy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from seededntm.util import (
    compute_topic_prior,
    compute_topic_prior_idf,
    spatial_smooth_prior,
    compute_reffree_leverage,
)
from experiments.run_seedtopic import compute_metrics_hungarian, compute_representation
from experiments.evaluate import load_ground_truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "experiments" / "benchmark_fair" / "data" / "processed"
RESULTS_PATH = Path(__file__).resolve().parent / "ablation_results.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "raw_outputs_ablation"

DATASETS = ["visium_NPC", "xenium_BC", "visiumHD_CRC_I", "visiumHD_CRC_II"]
DATASET_K = {"visium_NPC": 7, "xenium_BC": 19, "visiumHD_CRC_I": 5, "visiumHD_CRC_II": 6}

REPRESENTATIONS = ["idf_pca", "idf_lev_seed_spec", "lev_pseudo_sig"]
PRIOR_MODES = ["standard", "spatial", "idf_spatial"]

CANONICAL_PARAMS = {
    "use_nb_obs": True,
    "enc_hid_dim": 64,
    "n_epochs": 800,
    "batch_size": 8192,
    "lr": 0.01,
    "wt_fusion_top_seed": 1.0,
    "pos_scale": 0.5,
    "dropout": 0.2,
    "is_group_mode": False,
    "seed": 0,
    "clip_norm": 5.0,
    "base_alpha": 199,
    "floor": 0.8,
    "temperature": 0.8,
    "k_neighbors": 6,
}


def build_experiment_matrix():
    """Build 36 experiments interleaved by dataset.

    Pattern: iterate over (rep, prior) combos, within each iterate over datasets.
    This ensures consecutive indices run different datasets for GPU memory safety.
    """
    configs = []
    for rep in REPRESENTATIONS:
        for prior_mode in PRIOR_MODES:
            for ds in DATASETS:
                configs.append({
                    "dataset": ds,
                    "representation": rep,
                    "prior_mode": prior_mode,
                })
    return configs


EXPERIMENT_MATRIX = build_experiment_matrix()
assert len(EXPERIMENT_MATRIX) == 36


def compute_per_spot_alpha(topic_prior, base_alpha, floor=0.1):
    """Per-spot Dirichlet concentration: alpha_i = base * (floor + (1-floor)*confidence)."""
    N, K = topic_prior.shape
    per_spot_ent = scipy_entropy(topic_prior, axis=1)
    max_ent = np.log(K)
    confidence = np.clip(1.0 - per_spot_ent / max_ent, 0.0, 1.0)
    alpha = base_alpha * (floor + (1.0 - floor) * confidence)
    return alpha.astype(np.float32), confidence.astype(np.float32)


def make_perspot_train_step(alpha_np, device):
    """Patched train_step with per-spot Dirichlet weighting."""
    import pyro

    alpha_tensor = torch.FloatTensor(alpha_np).to(device)

    def perspot_train_step(dataloader, model, elbo, adam, clip_norm, params,
                           reg_topic_prior, epoch):
        losses = []
        elbo_losses = []
        prior_losses = []

        for ith_batch, batch_data in enumerate(dataloader):
            bs = batch_data['input'].shape[0]
            elbo_loss = elbo.differentiable_loss(
                model.model, model.guide,
                **{key: value.to(device) for key, value in batch_data.items()
                   if key in {'input', 'out_counts', 'out_normal', 'batch_labels'}}
            )
            elbo_loss = elbo_loss / bs

            if 'topic_prior' in batch_data and 'indices' in batch_data:
                idx = batch_data['indices']
                batch_alpha = alpha_tensor[idx]
                topic_prior_batch = batch_data['topic_prior'].to(device)
                logtheta_loc = model.encode(
                    **{key: value.to(device) for key, value in batch_data.items()
                       if key in {'input', 'batch_labels'}})
                theta = F.softmax(logtheta_loc, dim=-1)
                valid_sel = (topic_prior_batch.sum(dim=1) > 0.5)

                if valid_sel.sum() > 0:
                    ce_per_spot = -(topic_prior_batch[valid_sel] *
                                    torch.log(theta[valid_sel] + 1e-8)).sum(dim=-1)
                    weighted_ce = (batch_alpha[valid_sel] * ce_per_spot).mean()
                else:
                    weighted_ce = torch.zeros(1, device=device)
                loss = elbo_loss + weighted_ce
                elbo_losses.append(elbo_loss.item())
                prior_losses.append(weighted_ce.item())
            else:
                loss = elbo_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
            adam.step()
            adam.zero_grad()
            losses.append(loss.item())

        if prior_losses:
            return {"total": np.mean(losses),
                    "elbo": np.mean(elbo_losses),
                    "topReg": np.mean(prior_losses)}
        return {"total": np.mean(losses)}

    return perspot_train_step


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item['indices'] = idx
        return item


import pyro
from pyro.infer import TraceMeanField_ELBO
from torch.utils.data import DataLoader
from tqdm import tqdm
import seededntm.experiment as _exp_module
from seededntm.model import SeededNTM
from seededntm.data import MyDataset
from seededntm.util import EarlyStopper


def run_perspot_dirichlet(exp_data, K, exp_outdir, alpha_np, device,
                          early_stop=False, n_epochs=800, batch_size=8192, seed=0):
    """Run per-spot Dirichlet experiment with canonical hyperparameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(exp_outdir, exist_ok=True)
    pyro.clear_param_store()
    _exp_module.seed_everything(seed)
    pyro.set_rng_seed(seed)

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
        enc_hid_dim=CANONICAL_PARAMS["enc_hid_dim"],
        clamp_logvar_max=None,
        scale_normal_feat=-1,
        wt_fusion_top_seed=CANONICAL_PARAMS["wt_fusion_top_seed"],
        is_group_mode=CANONICAL_PARAMS["is_group_mode"],
        pos_scale=CANONICAL_PARAMS["pos_scale"],
        device=device,
        n_batches=n_batches,
        use_nb_obs=CANONICAL_PARAMS["use_nb_obs"],
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
    for ith_batch, batch_data in enumerate(init_loader):
        elbo.differentiable_loss(
            model.model, model.guide,
            **{key: val.to(device) for key, val in batch_data.items()
               if key in {'input', 'out_counts', 'out_normal', 'batch_labels'}}
        )
        params = [value for name, value in pyro.get_param_store().named_parameters()]
        adam = torch.optim.AdamW(params, lr=CANONICAL_PARAMS["lr"],
                                betas=(0.90, 0.999))
        break

    patched_train_step = make_perspot_train_step(alpha_np, device)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        dataset=indexed_dataset, batch_size=batch_size, generator=g, shuffle=True
    )

    losses = []
    p_bar = tqdm(range(n_epochs))
    es = EarlyStopper(20)

    for epoch in p_bar:
        loss_dict = patched_train_step(
            train_loader, model, elbo, adam, CANONICAL_PARAMS["clip_norm"],
            params, 0.0, epoch
        )
        loss_info = "  ".join([f"{k}({v:.4f})" for k, v in loss_dict.items()])
        p_bar.set_description(f"Epoch: {loss_info}")
        losses.append(loss_dict['total'])

        if early_stop and epoch > 100:
            if es.early_stop(loss_dict['total']):
                logger.info("Early Stopping")
                break

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    topics = []
    for ith_batch, batch_data in enumerate(loader):
        model.eval()
        with torch.no_grad():
            theta = model.infer_topic(
                **{key: val.to(device) for key, val in batch_data.items()
                   if key in {'input', 'batch_labels'}})
            topics.append(theta.cpu())
    topics = torch.concat(topics, dim=0)

    top_names = [f'topic_{i}' for i in range(K)]
    df_topic = pd.DataFrame(topics.numpy(), columns=top_names)
    if exp_data.obs_names is not None:
        df_topic.index = pd.Series(exp_data.obs_names, name='obs_names')

    df_topic.to_csv(os.path.join(exp_outdir, 'df_topic.csv'))
    torch.save(model.cpu().state_dict(), os.path.join(exp_outdir, 'model.ckpt'))

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(exp_outdir, 'loss.png'))
    plt.close()

    return df_topic


def load_exp_data(adata, input_key, seeds_path, K, topic_prior):
    """Load ExpData for the experiment."""
    from seededntm.experiment import preprocess_ST

    X_counts = adata.X
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
        condition_feat_path=str(seeds_path),
        topic_prior=topic_prior.astype(np.float32),
    )
    return exp_data


def is_done(key):
    if not RESULTS_PATH.exists():
        return False
    try:
        with open(RESULTS_PATH) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        if not isinstance(data, dict):
            return False
        return key in data and isinstance(data[key], dict) and data[key].get("status") == "ok"
    except (json.JSONDecodeError, IOError, TypeError, AttributeError):
        return False


def save_result(key, result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = RESULTS_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if RESULTS_PATH.exists():
            try:
                with open(RESULTS_PATH) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
        data[key] = result
        with open(RESULTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def log_provenance(ds_name, rep, prior_mode):
    """Log provenance information at startup."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"

    logger.info(f"{'='*60}")
    logger.info(f"PROVENANCE")
    logger.info(f"  git_hash: {git_hash}")
    logger.info(f"  timestamp: {datetime.now().isoformat()}")
    logger.info(f"  dataset: {ds_name}")
    logger.info(f"  representation: {rep}")
    logger.info(f"  prior_mode: {prior_mode}")
    logger.info(f"  canonical_params: {CANONICAL_PARAMS}")
    logger.info(f"  torch: {torch.__version__}")
    logger.info(f"  numpy: {np.__version__}")
    logger.info(f"  scanpy: {sc.__version__}")
    logger.info(f"{'='*60}")
    return git_hash


def run_single(exp_cfg):
    """Run a single experiment configuration."""
    ds_name = exp_cfg["dataset"]
    rep = exp_cfg["representation"]
    prior_mode = exp_cfg["prior_mode"]
    K = DATASET_K[ds_name]

    key = f"ablation_{ds_name}_{rep}_{prior_mode}"
    if is_done(key):
        logger.info(f"SKIP: {key}")
        return

    git_hash = log_provenance(ds_name, rep, prior_mode)

    logger.info(f"RUN: {key}")
    logger.info(f"  dataset={ds_name}, K={K}, rep={rep}, prior={prior_mode}")

    # Load spatial adata from processed/ directory
    adata_path = PROCESSED_DIR / ds_name / "spatial_adata.h5ad"
    seeds_path = PROCESSED_DIR / ds_name / "seeds.json"

    assert adata_path.exists(), f"Missing: {adata_path}"
    assert seeds_path.exists(), f"Missing: {seeds_path}"

    adata = sc.read_h5ad(str(adata_path))
    adata.var_names_make_unique()
    logger.info(f"  adata shape: {adata.shape}")

    with open(seeds_path) as f:
        seeds = json.load(f)
    logger.info(f"  Loaded {len(seeds)} seed types: {list(seeds.keys())}")

    # Validate all seed genes exist in spatial adata
    missing_genes = []
    for ct_name, ct_info in seeds.items():
        for gene in ct_info.get("features", ct_info.get("genes", [])):
            if gene not in adata.var_names:
                missing_genes.append((ct_name, gene))
    assert len(missing_genes) == 0, (
        f"Seed genes missing from spatial_adata.var_names: {missing_genes[:10]}"
    )

    gt_labels = load_ground_truth(ds_name)
    gt_types = sorted(set(gt_labels.tolist()))
    logger.info(f"  Ground truth: {len(gt_labels)} spots, {len(gt_types)} types")

    X_counts = adata.X
    var_names = np.array(adata.var_names)

    if "rna_count" not in adata.obsm:
        if issparse(X_counts):
            adata.obsm["rna_count"] = np.asarray(X_counts.todense(), dtype=np.float32)
        else:
            adata.obsm["rna_count"] = np.asarray(X_counts, dtype=np.float32)

    # Compute representation
    logger.info(f"  Computing representation: {rep}")
    input_rep = compute_representation(X_counts, var_names, seeds, rep, K)
    input_key = "seedtopic_input_rep"
    adata.obsm[input_key] = input_rep
    logger.info(f"  Representation shape: {input_rep.shape}")

    # Compute topic prior
    logger.info(f"  Computing prior: {prior_mode}")
    temperature = CANONICAL_PARAMS["temperature"]
    k_neighbors = CANONICAL_PARAMS["k_neighbors"]

    if "idf" in prior_mode:
        raw_prior = compute_topic_prior_idf(adata, seeds, temperature=temperature)
    else:
        raw_prior = compute_topic_prior(adata, seeds, temperature=temperature)

    if "spatial" in prior_mode and "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=k_neighbors)
    else:
        topic_prior = raw_prior

    adata.obsm["topic_prior"] = topic_prior.astype(np.float32)
    logger.info(f"  Prior shape: {topic_prior.shape}, sum check: {topic_prior.sum(axis=1).mean():.4f}")

    # Per-spot alpha
    base_alpha = CANONICAL_PARAMS["base_alpha"]
    floor_val = CANONICAL_PARAMS["floor"]
    alpha_np, confidence = compute_per_spot_alpha(topic_prior, base_alpha, floor=floor_val)
    logger.info(f"  Alpha: mean={alpha_np.mean():.1f}, range=[{alpha_np.min():.1f}, {alpha_np.max():.1f}]")

    # Setup output directory
    exp_outdir = str((OUTPUT_DIR / ds_name / key).resolve())
    os.makedirs(exp_outdir, exist_ok=True)

    # Write seeds to output for preprocess_ST
    seeds_out_path = Path(exp_outdir) / "seeds.json"
    with open(seeds_out_path, "w") as f:
        json.dump(seeds, f)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    early_stop = (ds_name == "xenium_BC")

    t0 = time.time()
    try:
        exp_data = load_exp_data(adata, input_key, seeds_out_path, K, topic_prior)
        df_topic = run_perspot_dirichlet(
            exp_data, K, exp_outdir, alpha_np, device,
            early_stop=early_stop,
            n_epochs=CANONICAL_PARAMS["n_epochs"],
            batch_size=CANONICAL_PARAMS["batch_size"],
            seed=CANONICAL_PARAMS["seed"],
        )
    except Exception as e:
        logger.error(f"  FAILED: {key}: {e}")
        import traceback
        traceback.print_exc()
        save_result(key, {
            "dataset": ds_name, "K": K,
            "representation": rep, "prior_mode": prior_mode,
            "status": "failed", "error": str(e),
            "timestamp": datetime.now().isoformat(),
        })
        return

    runtime = time.time() - t0

    # Evaluate
    try:
        proportions = df_topic.values.astype(np.float64)
        n = min(len(gt_labels), proportions.shape[0])
        metrics = compute_metrics_hungarian(proportions[:n], K, gt_labels[:n], gt_types)
    except Exception as e:
        logger.error(f"  METRIC FAILED: {key}: {e}")
        save_result(key, {
            "dataset": ds_name, "K": K,
            "representation": rep, "prior_mode": prior_mode,
            "status": "failed", "error": f"metrics: {e}",
            "timestamp": datetime.now().isoformat(),
        })
        return

    logger.info(
        f"  DONE: {key} F1={metrics['f1']:.4f} AMI={metrics['AMI']:.4f} "
        f"ARI={metrics['ARI']:.4f} AUPRC={metrics['AUPRC']:.4f} "
        f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} ({runtime:.0f}s)"
    )

    result = {
        "dataset": ds_name,
        "K": K,
        "representation": rep,
        "prior_mode": prior_mode,
        "metrics": metrics,
        "runtime_s": round(runtime, 1),
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "hyperparameters": CANONICAL_PARAMS,
        "data_shapes": {
            "adata": list(adata.shape),
            "input_rep": list(input_rep.shape),
            "topic_prior": list(topic_prior.shape),
        },
        "alpha_stats": {
            "mean": float(alpha_np.mean()),
            "min": float(alpha_np.min()),
            "max": float(alpha_np.max()),
        },
    }
    save_result(key, result)


def analyze_results():
    """Compute average rank across all metrics and datasets. Print winner."""
    if not RESULTS_PATH.exists():
        logger.error(f"No results file: {RESULTS_PATH}")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    METRIC_NAMES = ["f1", "precision", "recall", "AMI", "ARI", "AUPRC"]

    # Collect successful results
    rows = []
    for key, r in data.items():
        if not isinstance(r, dict) or r.get("status") != "ok":
            continue
        row = {
            "key": key,
            "dataset": r["dataset"],
            "representation": r["representation"],
            "prior_mode": r["prior_mode"],
        }
        for m in METRIC_NAMES:
            row[m] = r["metrics"].get(m, float("nan"))
        rows.append(row)

    if not rows:
        logger.error("No successful results found.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    config_col = "config"
    df[config_col] = df["representation"] + " + " + df["prior_mode"]

    # Per-dataset results table
    print(f"\n{'='*90}")
    print("REPRESENTATION x PRIOR MODE ABLATION - FULL RESULTS")
    print(f"{'='*90}")

    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds].copy()
        if ds_df.empty:
            continue
        print(f"\n  {ds} (K={DATASET_K[ds]}):")
        print(f"  {'Config':<35s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} "
              f"{'AMI':>7s} {'ARI':>7s} {'AUPRC':>7s}")
        print(f"  {'-'*85}")
        ds_df_sorted = ds_df.sort_values("f1", ascending=False)
        for _, row in ds_df_sorted.iterrows():
            print(f"  {row[config_col]:<35s} "
                  f"{row['f1']:>7.4f} {row['precision']:>7.4f} {row['recall']:>7.4f} "
                  f"{row['AMI']:>7.4f} {row['ARI']:>7.4f} {row['AUPRC']:>7.4f}")

    # Compute ranks per dataset per metric (lower rank = better)
    print(f"\n{'='*90}")
    print("AVERAGE RANK ANALYSIS (lower = better)")
    print(f"{'='*90}")

    all_ranks = []
    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds].copy()
        if ds_df.empty:
            continue
        for m in METRIC_NAMES:
            ds_df[f"rank_{m}"] = ds_df[m].rank(ascending=False, method="average")
            for _, row in ds_df.iterrows():
                all_ranks.append({
                    "config": row[config_col],
                    "dataset": ds,
                    "metric": m,
                    "rank": row[f"rank_{m}"],
                    "value": row[m],
                })

    rank_df = pd.DataFrame(all_ranks)
    avg_rank = rank_df.groupby("config")["rank"].mean().sort_values()

    print(f"\n  {'Config':<35s} {'Avg Rank':>10s}")
    print(f"  {'-'*50}")
    for config_name, avg_r in avg_rank.items():
        marker = " <-- WINNER" if config_name == avg_rank.index[0] else ""
        print(f"  {config_name:<35s} {avg_r:>10.3f}{marker}")

    # Per-metric breakdown for top 3
    print(f"\n  Top 3 configs - per-metric average rank:")
    top3 = avg_rank.index[:3].tolist()
    print(f"  {'Config':<35s}", end="")
    for m in METRIC_NAMES:
        print(f" {m:>7s}", end="")
    print(f" {'OVERALL':>9s}")
    print(f"  {'-'*95}")
    for cfg_name in top3:
        cfg_ranks = rank_df[rank_df["config"] == cfg_name]
        print(f"  {cfg_name:<35s}", end="")
        for m in METRIC_NAMES:
            mr = cfg_ranks[cfg_ranks["metric"] == m]["rank"].mean()
            print(f" {mr:>7.2f}", end="")
        print(f" {avg_rank[cfg_name]:>9.3f}")

    # Winner announcement
    winner = avg_rank.index[0]
    winner_rep, winner_prior = winner.split(" + ")
    print(f"\n{'='*90}")
    print(f"WINNER: {winner}")
    print(f"  representation = {winner_rep}")
    print(f"  prior_mode     = {winner_prior}")
    print(f"  average_rank   = {avg_rank.iloc[0]:.3f}")
    print(f"{'='*90}")

    return winner_rep.strip(), winner_prior.strip()


def print_matrix_overview():
    """Print the experiment matrix for inspection."""
    print(f"\nExperiment Matrix ({len(EXPERIMENT_MATRIX)} total):")
    print(f"  {'Idx':<4s} {'Dataset':<18s} {'Representation':<20s} {'Prior Mode':<15s} {'Status'}")
    print(f"  {'-'*75}")
    for i, cfg in enumerate(EXPERIMENT_MATRIX):
        key = f"ablation_{cfg['dataset']}_{cfg['representation']}_{cfg['prior_mode']}"
        done = "DONE" if is_done(key) else "pending"
        print(f"  {i:<4d} {cfg['dataset']:<18s} {cfg['representation']:<20s} "
              f"{cfg['prior_mode']:<15s} {done}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Representation x Prior Mode Ablation for SeedTopic"
    )
    parser.add_argument("--idx", type=int, default=None,
                        help=f"Run single experiment by index (0-{len(EXPERIMENT_MATRIX)-1})")
    parser.add_argument("--analyze", action="store_true",
                        help="Compute average ranks and select winner")
    parser.add_argument("--show", action="store_true",
                        help="Print experiment matrix overview")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    elif args.show:
        print_matrix_overview()
    elif args.idx is not None:
        if args.idx < 0 or args.idx >= len(EXPERIMENT_MATRIX):
            logger.error(f"Invalid idx={args.idx}, valid range: 0-{len(EXPERIMENT_MATRIX)-1}")
            sys.exit(1)
        run_single(EXPERIMENT_MATRIX[args.idx])
    else:
        print_matrix_overview()
        print("\nUsage:")
        print("  python run_rep_prior_ablation.py --idx N    # run single experiment")
        print("  python run_rep_prior_ablation.py --analyze  # compute ranks")
        print("  python run_rep_prior_ablation.py --show     # show matrix")
