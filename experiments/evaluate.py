"""Reusable evaluation module for SeedTopic experiments.

Extracted from SeededNTM/experiments/benchmark_deconv/run_benchmark.py.
Computes macro and per-class deconvolution metrics given predicted proportions
and ground-truth labels.

Usage:
    from experiments.evaluate import compute_metrics, compute_per_class_metrics
    metrics = compute_metrics(proportions, ct_names, gt_labels)
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    proportions: np.ndarray,
    cell_type_names: list,
    ground_truth_labels: np.ndarray,
) -> dict:
    """Compute macro-averaged evaluation metrics.

    Args:
        proportions: (N, K) array of predicted cell type proportions per spot.
        cell_type_names: list of K cell type names matching columns of proportions.
        ground_truth_labels: (N,) array of ground truth cell type labels.

    Returns:
        dict with keys: f1, precision, recall, AUROC, AUPRC, ARI, AMI.
    """
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
    gt_onehot_filtered = gt_onehot[:, gt_classes_present]
    prop_filtered = prop_aligned[:, gt_classes_present]

    try:
        auprc_val = average_precision_score(
            gt_onehot_filtered, prop_filtered, average="macro"
        )
    except ValueError:
        auprc_val = float("nan")
    try:
        auroc_val = roc_auc_score(
            gt_onehot_filtered, prop_filtered, average="macro", multi_class="ovr"
        )
    except ValueError:
        auroc_val = float("nan")

    return {
        "f1": f1_score(gt_ids, pred_ids, average="macro", zero_division=0),
        "precision": precision_score(gt_ids, pred_ids, average="macro", zero_division=0),
        "recall": recall_score(gt_ids, pred_ids, average="macro", zero_division=0),
        "AUROC": auroc_val,
        "AUPRC": auprc_val,
        "ARI": adjusted_rand_score(gt_ids, pred_ids),
        "AMI": adjusted_mutual_info_score(gt_ids, pred_ids),
    }


def compute_per_class_metrics(
    proportions: np.ndarray,
    cell_type_names: list,
    ground_truth_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute per-class F1, precision, recall, and AUPRC.

    Args:
        proportions: (N, K) array of predicted cell type proportions.
        cell_type_names: list of K cell type names.
        ground_truth_labels: (N,) array of ground truth labels.

    Returns:
        DataFrame with columns: cell_type, count, pct, f1, precision, recall, auprc.
    """
    predicted_labels = np.array(cell_type_names)[np.argmax(proportions, axis=1)]
    gt_classes = sorted(set(ground_truth_labels.tolist()))
    report = classification_report(
        ground_truth_labels,
        predicted_labels,
        labels=gt_classes,
        output_dict=True,
        zero_division=0,
    )

    ct_to_idx = {ct: i for i, ct in enumerate(cell_type_names)}

    rows = []
    dist = Counter(ground_truth_labels.tolist())
    for ct in gt_classes:
        r = report.get(ct, {})
        ap = 0.0
        if ct in ct_to_idx:
            idx = ct_to_idx[ct]
            y_true_bin = (np.array(ground_truth_labels) == ct).astype(int)
            y_score = proportions[:, idx]
            if y_true_bin.sum() > 0:
                ap = average_precision_score(y_true_bin, y_score)
        rows.append(
            {
                "cell_type": ct,
                "count": dist.get(ct, 0),
                "pct": 100 * dist.get(ct, 0) / len(ground_truth_labels),
                "f1": r.get("f1-score", 0),
                "precision": r.get("precision", 0),
                "recall": r.get("recall", 0),
                "auprc": ap,
            }
        )
    return pd.DataFrame(rows)


def load_baseline_proportions(dataset_name: str, method: str) -> tuple:
    """Load pre-computed baseline proportions from the benchmark directory.

    Args:
        dataset_name: one of 'visium_NPC', 'xenium_BC', 'visiumHD_CRC_I', 'visiumHD_CRC_II'
        method: one of 'flashdeconv', 'rctd', 'marker_scoring', 'seedtopic', 'transdeconv'

    Returns:
        (proportions, cell_type_names) tuple. proportions is (N, K) ndarray.
    """
    from pathlib import Path

    baseline_dir = Path(
        "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM"
        "/experiments/benchmark_deconv/raw_outputs"
    ) / dataset_name

    df = pd.read_csv(baseline_dir / f"{method}_proportions.csv")
    return df.values, df.columns.tolist()


def load_ground_truth(dataset_name: str) -> np.ndarray:
    """Load ground truth labels from the benchmark directory.

    The ground_truth.csv has columns: ground_truth, coord_x, coord_y.
    Labels are in the 'ground_truth' column.

    Args:
        dataset_name: one of 'visium_NPC', 'xenium_BC', 'visiumHD_CRC_I', 'visiumHD_CRC_II'

    Returns:
        (N,) array of ground truth cell type labels (strings).
    """
    from pathlib import Path

    baseline_dir = Path(
        "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM"
        "/experiments/benchmark_deconv/raw_outputs"
    ) / dataset_name

    df = pd.read_csv(baseline_dir / "ground_truth.csv")
    return df["ground_truth"].values.astype(str)


def load_rctd_pixel_mask(dataset_name: str) -> np.ndarray:
    """Load RCTD pixel mask for fair subset comparison.

    Returns:
        Boolean array (N,) -- True for spots retained by RCTD.
    """
    from pathlib import Path

    baseline_dir = Path(
        "/illumina-sdcolo-02/scratch/deep_learning/cqiao/projects/SeededNTM"
        "/experiments/benchmark_deconv/raw_outputs"
    ) / dataset_name

    mask_path = baseline_dir / "rctd_pixel_mask.npy"
    if mask_path.exists():
        return np.load(mask_path)
    return None


def format_metrics_table(results: dict) -> str:
    """Format a metrics dict as a markdown table row."""
    return (
        f"| {results.get('version', '?')} | {results.get('dataset', '?')} "
        f"| {results['f1']:.3f} | {results['AUPRC']:.3f} "
        f"| {results['ARI']:.3f} | {results['AMI']:.3f} "
        f"| {results.get('runtime_s', '?')} | {results.get('notes', '')} |"
    )
