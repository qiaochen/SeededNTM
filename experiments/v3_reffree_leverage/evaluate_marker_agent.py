#!/usr/bin/env python
"""Evaluate marker agent seeds against existing seed sets across 4 datasets.

Compares agent-derived seeds with:
  - seeds.json (reference-based seeds)
  - seeds_leiden_chatgpt.json (leiden+ChatGPT seeds)

Metrics: Jaccard, Precision, Recall, F1, Precision@5, Precision@10, novel genes count.

Usage:
    python experiments/v3_reffree_leverage/evaluate_marker_agent.py --all
    python experiments/v3_reffree_leverage/evaluate_marker_agent.py --dataset visium_NPC
    python experiments/v3_reffree_leverage/evaluate_marker_agent.py --all --skip-agent
    python experiments/v3_reffree_leverage/evaluate_marker_agent.py --all --output results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASETS = {
    "visium_NPC": {
        "tissue": "Nasopharyngeal carcinoma",
        "organ": "Nasopharynx",
        "condition": "tumor",
    },
    "xenium_BC": {
        "tissue": "Breast cancer",
        "organ": "Breast",
        "condition": "DCIS/invasive",
    },
    "visiumHD_CRC_I": {
        "tissue": "Colorectal cancer",
        "organ": "Colon",
        "condition": "tumor microenvironment",
    },
    "visiumHD_CRC_II": {
        "tissue": "Colorectal cancer",
        "organ": "Colon",
        "condition": "tumor microenvironment",
    },
}

DATA_DIR = Path("experiments/benchmark_fair/data/processed")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def precision(predicted, reference):
    if not predicted:
        return 0.0
    return len(predicted & reference) / len(predicted)


def recall(predicted, reference):
    if not reference:
        return 0.0
    return len(predicted & reference) / len(reference)


def f1_score(predicted, reference):
    p = precision(predicted, reference)
    r = recall(predicted, reference)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def precision_at_k(predicted_list, reference_set, k):
    top_k = set(predicted_list[:k])
    if not top_k:
        return 0.0
    return len(top_k & reference_set) / len(top_k)


def novel_genes(predicted, reference):
    return predicted - reference


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def load_seeds(path):
    """Load seeds JSON file. Format: {cell_type: {"features": [...], "topic_index": N}}"""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def get_gene_set(seeds_dict, cell_type):
    """Extract gene set for a cell type from seeds dict."""
    if cell_type not in seeds_dict:
        return set(), []
    entry = seeds_dict[cell_type]
    if isinstance(entry, dict) and "features" in entry:
        genes = entry["features"]
    elif isinstance(entry, list):
        genes = entry
    else:
        genes = []
    return set(genes), list(genes)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_name, config, skip_agent=False):
    """Evaluate seeds for a single dataset."""
    data_path = DATA_DIR / dataset_name
    seeds_ref_path = data_path / "seeds.json"
    seeds_leiden_path = data_path / "seeds_leiden_chatgpt.json"
    seeds_agent_path = data_path / "seeds_agent.json"

    seeds_ref = load_seeds(seeds_ref_path)
    seeds_leiden = load_seeds(seeds_leiden_path)
    seeds_agent = load_seeds(seeds_agent_path) if not skip_agent else None

    if seeds_ref is None:
        print(f"  [WARN] No seeds.json found for {dataset_name}")
        return None

    results = {
        "dataset": dataset_name,
        "config": config,
        "cell_types": {},
        "summary": {},
    }

    all_cell_types = set(seeds_ref.keys())
    if seeds_leiden:
        all_cell_types |= set(seeds_leiden.keys())
    if seeds_agent:
        all_cell_types |= set(seeds_agent.keys())

    metrics_leiden = []
    metrics_agent = []

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} | Tissue: {config['tissue']} | Organ: {config['organ']}")
    print(f"{'='*80}")
    print(
        f"  {'Cell Type':<25} {'Method':<15} {'Jaccard':<8} {'Prec':<7} "
        f"{'Rec':<7} {'F1':<7} {'P@5':<6} {'P@10':<6} {'Novel':<6} {'N_genes'}"
    )
    print(f"  {'-'*100}")

    for ct in sorted(all_cell_types):
        ref_set, ref_list = get_gene_set(seeds_ref, ct)

        if seeds_leiden and ct in seeds_leiden:
            leiden_set, leiden_list = get_gene_set(seeds_leiden, ct)
            m = {
                "jaccard": jaccard(leiden_set, ref_set),
                "precision": precision(leiden_set, ref_set),
                "recall": recall(leiden_set, ref_set),
                "f1": f1_score(leiden_set, ref_set),
                "p_at_5": precision_at_k(leiden_list, ref_set, 5),
                "p_at_10": precision_at_k(leiden_list, ref_set, 10),
                "novel": len(novel_genes(leiden_set, ref_set)),
                "n_genes": len(leiden_set),
            }
            metrics_leiden.append(m)
            print(
                f"  {ct:<25} {'leiden_gpt':<15} {m['jaccard']:<8.3f} {m['precision']:<7.3f} "
                f"{m['recall']:<7.3f} {m['f1']:<7.3f} {m['p_at_5']:<6.3f} "
                f"{m['p_at_10']:<6.3f} {m['novel']:<6} {m['n_genes']}"
            )

        if seeds_agent and ct in seeds_agent:
            agent_set, agent_list = get_gene_set(seeds_agent, ct)
            m = {
                "jaccard": jaccard(agent_set, ref_set),
                "precision": precision(agent_set, ref_set),
                "recall": recall(agent_set, ref_set),
                "f1": f1_score(agent_set, ref_set),
                "p_at_5": precision_at_k(agent_list, ref_set, 5),
                "p_at_10": precision_at_k(agent_list, ref_set, 10),
                "novel": len(novel_genes(agent_set, ref_set)),
                "n_genes": len(agent_set),
            }
            metrics_agent.append(m)
            print(
                f"  {ct:<25} {'agent':<15} {m['jaccard']:<8.3f} {m['precision']:<7.3f} "
                f"{m['recall']:<7.3f} {m['f1']:<7.3f} {m['p_at_5']:<6.3f} "
                f"{m['p_at_10']:<6.3f} {m['novel']:<6} {m['n_genes']}"
            )

        results["cell_types"][ct] = {
            "ref_genes": sorted(ref_set),
        }
        if seeds_leiden and ct in seeds_leiden:
            results["cell_types"][ct]["leiden_metrics"] = metrics_leiden[-1] if metrics_leiden else None
        if seeds_agent and ct in seeds_agent:
            results["cell_types"][ct]["agent_metrics"] = metrics_agent[-1] if metrics_agent else None

    def avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        keys = ["jaccard", "precision", "recall", "f1", "p_at_5", "p_at_10", "novel", "n_genes"]
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

    results["summary"]["leiden_avg"] = avg_metrics(metrics_leiden)
    results["summary"]["agent_avg"] = avg_metrics(metrics_agent)

    print(f"\n  --- Summary for {dataset_name} ---")
    if metrics_leiden:
        avg_l = results["summary"]["leiden_avg"]
        print(
            f"  leiden_gpt  (avg): Jaccard={avg_l['jaccard']:.3f} Prec={avg_l['precision']:.3f} "
            f"Rec={avg_l['recall']:.3f} F1={avg_l['f1']:.3f} Novel={avg_l['novel']:.1f}"
        )
    if metrics_agent:
        avg_a = results["summary"]["agent_avg"]
        print(
            f"  agent      (avg): Jaccard={avg_a['jaccard']:.3f} Prec={avg_a['precision']:.3f} "
            f"Rec={avg_a['recall']:.3f} F1={avg_a['f1']:.3f} Novel={avg_a['novel']:.1f}"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate marker agent seeds vs existing seed sets"
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all 4 datasets")
    parser.add_argument("--dataset", type=str, help="Evaluate a specific dataset")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip agent seeds (compare leiden vs ref only)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        datasets_to_run = {args.dataset: DATASETS[args.dataset]}
    elif args.all:
        datasets_to_run = DATASETS
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    t0 = time.time()

    for name, config in datasets_to_run.items():
        result = evaluate_dataset(name, config, skip_agent=args.skip_agent)
        if result:
            all_results[name] = result

    elapsed = time.time() - t0

    # Cross-dataset summary
    print(f"\n{'='*80}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Dataset':<20} {'Method':<15} {'Jaccard':<8} {'F1':<7} {'Novel':<7}")
    print(f"  {'-'*60}")
    for name, res in all_results.items():
        for method, key in [("leiden_gpt", "leiden_avg"), ("agent", "agent_avg")]:
            avg = res["summary"].get(key, {})
            if avg:
                print(
                    f"  {name:<20} {method:<15} {avg['jaccard']:<8.3f} "
                    f"{avg['f1']:<7.3f} {avg['novel']:<7.1f}"
                )
    print(f"\n  Total evaluation time: {elapsed:.1f}s")

    # Save report
    output_path = args.output or "experiments/v3_reffree_leverage/marker_agent_evaluation.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": all_results,
        "elapsed_seconds": elapsed,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
