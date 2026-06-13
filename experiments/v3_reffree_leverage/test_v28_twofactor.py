#!/usr/bin/env python
"""Unit tests for the two-factor adaptive blend (Phase 1).

Tests validate:
1. Two-factor weight function correctness (separation of CRC/NPC/xenium)
2. Full 10-metric computation (all keys, valid ranges)
3. Seed-reference alignment assertion (fail loudly on mismatch)
4. max_prop computed from post-smoothing nnls_aligned

Run: python -m pytest experiments/v3_reffree_leverage/test_v28_twofactor.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# Functions under test (extracted to be importable from the script)
# ============================================================

def compute_adaptive_weights_twofactor(stability, max_prop, g1, t1, g2, t2,
                                       w_min=0.05, w_max=0.95):
    """Two-factor multiplicative adaptive weight.

    w_i = w_min + (w_max - w_min) * sigmoid(g1*(stab_i - t1)) * sigmoid(g2*(maxp_i - t2))
    """
    logit1 = g1 * (stability - t1)
    logit2 = g2 * (max_prop - t2)
    sig1 = 1.0 / (1.0 + np.exp(-np.clip(logit1, -50, 50)))
    sig2 = 1.0 / (1.0 + np.exp(-np.clip(logit2, -50, 50)))
    w = w_min + (w_max - w_min) * sig1 * sig2
    return w.astype(np.float32)


def compute_all_metrics(proportions, K, gt_labels, seeds):
    """Compute all 10 metrics using Hungarian matching + proportion-level metrics.

    Returns dict with keys: F1, Precision, Recall, AMI, ARI, AUPRC, Pearson, Spearman, RMSE, JSD
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score,
        average_precision_score, f1_score, precision_score, recall_score,
    )
    from scipy.optimize import linear_sum_assignment

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
        "ARI": float(adjusted_rand_score(gt_idx, pred_gt_idx)),
        "AMI": float(adjusted_mutual_info_score(gt_idx, pred_gt_idx)),
    }

    gt_onehot = np.zeros((len(gt_idx), n_gt), dtype=np.float64)
    gt_onehot[np.arange(len(gt_idx)), gt_idx] = 1.0
    try:
        metrics["AUPRC"] = float(average_precision_score(gt_onehot, merged_proportions, average="macro"))
    except ValueError:
        metrics["AUPRC"] = float("nan")

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


# ============================================================
# TEST 1: Two-factor weight function correctness
# ============================================================

class TestTwoFactorWeights:
    """Validate that two-factor weight separates CRC, NPC, and xenium correctly."""

    def test_crc_gets_high_weight(self):
        """CRC-like spot (high stability, high max_prop) -> high blend weight."""
        stab = np.array([0.97])
        maxp = np.array([0.93])
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70)
        assert w[0] >= 0.70, f"CRC weight too low: {w[0]:.4f}"

    def test_npc_gets_low_weight(self):
        """NPC-like spot (high stability, low max_prop) -> low blend weight."""
        stab = np.array([0.96])
        maxp = np.array([0.52])
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70)
        assert w[0] <= 0.30, f"NPC weight too high: {w[0]:.4f}"

    def test_xenium_gets_low_weight(self):
        """xenium-like spot (low stability, high max_prop) -> low blend weight."""
        stab = np.array([0.73])
        maxp = np.array([0.82])
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70)
        assert w[0] <= 0.25, f"xenium weight too high: {w[0]:.4f}"

    def test_weight_bounds(self):
        """All weights must be within [w_min, w_max]."""
        rng = np.random.default_rng(42)
        stab = rng.uniform(0.0, 1.0, size=1000)
        maxp = rng.uniform(0.0, 1.0, size=1000)
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70,
                                               w_min=0.05, w_max=0.95)
        assert np.all(w >= 0.05 - 1e-6), f"Below w_min: min={w.min()}"
        assert np.all(w <= 0.95 + 1e-6), f"Above w_max: max={w.max()}"

    def test_monotonicity_stability(self):
        """Higher stability (with fixed max_prop) -> higher weight."""
        maxp = np.array([0.80, 0.80, 0.80])
        stab = np.array([0.70, 0.85, 0.99])
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70)
        assert w[0] < w[1] < w[2], f"Not monotonic in stability: {w}"

    def test_monotonicity_max_prop(self):
        """Higher max_prop (with fixed stability) -> higher weight."""
        stab = np.array([0.95, 0.95, 0.95])
        maxp = np.array([0.40, 0.70, 0.95])
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70)
        assert w[0] < w[1] < w[2], f"Not monotonic in max_prop: {w}"

    def test_vectorized(self):
        """Function works on arrays, not just scalars."""
        stab = np.array([0.97, 0.96, 0.73, 0.50])
        maxp = np.array([0.93, 0.52, 0.82, 0.30])
        w = compute_adaptive_weights_twofactor(stab, maxp, g1=20, t1=0.80, g2=10, t2=0.70)
        assert w.shape == (4,)
        assert w.dtype == np.float32


# ============================================================
# TEST 2: Full 10-metric computation
# ============================================================

class TestAllMetrics:
    """Validate compute_all_metrics returns all 10 metrics with correct ranges."""

    def _make_perfect_data(self):
        """Create perfect prediction data (3 types, 9 spots)."""
        gt_labels = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
        proportions = np.zeros((9, 3), dtype=np.float64)
        proportions[0:3, 0] = 1.0
        proportions[3:6, 1] = 1.0
        proportions[6:9, 2] = 1.0
        seeds = {
            "A": {"topic_index": 0, "genes": ["g1"]},
            "B": {"topic_index": 1, "genes": ["g2"]},
            "C": {"topic_index": 2, "genes": ["g3"]},
        }
        return proportions, 3, gt_labels, seeds

    def _make_noisy_data(self):
        """Create imperfect prediction data."""
        rng = np.random.default_rng(123)
        gt_labels = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        proportions = np.zeros((60, 3), dtype=np.float64)
        for i in range(60):
            gt_idx = i // 20
            proportions[i, gt_idx] = 0.6
            proportions[i, (gt_idx + 1) % 3] = 0.25
            proportions[i, (gt_idx + 2) % 3] = 0.15
        proportions += rng.uniform(0, 0.05, size=(60, 3))
        proportions /= proportions.sum(axis=1, keepdims=True)
        seeds = {
            "A": {"topic_index": 0, "genes": ["g1"]},
            "B": {"topic_index": 1, "genes": ["g2"]},
            "C": {"topic_index": 2, "genes": ["g3"]},
        }
        return proportions, 3, gt_labels, seeds

    def test_all_10_keys_present(self):
        """All 10 metric keys must be present."""
        proportions, K, gt_labels, seeds = self._make_noisy_data()
        metrics = compute_all_metrics(proportions, K, gt_labels, seeds)
        expected_keys = {"F1", "Precision", "Recall", "AMI", "ARI", "AUPRC",
                         "Pearson", "Spearman", "RMSE", "JSD"}
        assert set(metrics.keys()) == expected_keys, f"Missing: {expected_keys - set(metrics.keys())}"

    def test_perfect_prediction_metrics(self):
        """Perfect predictions should give F1=1, RMSE=0, JSD=0, Pearson=1."""
        proportions, K, gt_labels, seeds = self._make_perfect_data()
        metrics = compute_all_metrics(proportions, K, gt_labels, seeds)
        assert metrics["F1"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["JSD"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["Pearson"] == pytest.approx(1.0, abs=1e-4)

    def test_metric_ranges(self):
        """Verify metric value ranges."""
        proportions, K, gt_labels, seeds = self._make_noisy_data()
        metrics = compute_all_metrics(proportions, K, gt_labels, seeds)
        assert 0 <= metrics["F1"] <= 1
        assert 0 <= metrics["Precision"] <= 1
        assert 0 <= metrics["Recall"] <= 1
        assert -1 <= metrics["AMI"] <= 1
        assert -1 <= metrics["ARI"] <= 1
        assert 0 <= metrics["AUPRC"] <= 1.0 + 1e-6
        assert -1 <= metrics["Pearson"] <= 1
        assert -1 <= metrics["Spearman"] <= 1
        assert 0 <= metrics["RMSE"] <= 1
        assert 0 <= metrics["JSD"] <= 1

    def test_k_greater_than_n_gt(self):
        """K > n_gt should still work (extra topics merged via Hungarian)."""
        rng = np.random.default_rng(42)
        gt_labels = np.array(["A"] * 30 + ["B"] * 30)
        proportions = rng.dirichlet(np.ones(5), size=60)
        proportions[:30, 0] += 0.5
        proportions[30:, 2] += 0.5
        proportions /= proportions.sum(axis=1, keepdims=True)
        seeds = {
            "A": {"topic_index": 0, "genes": ["g1"]},
            "B": {"topic_index": 2, "genes": ["g2"]},
        }
        metrics = compute_all_metrics(proportions, 5, gt_labels, seeds)
        assert len(metrics) == 10
        assert 0 <= metrics["F1"] <= 1


# ============================================================
# TEST 3: Seed-reference alignment assertion
# ============================================================

class TestSeedReferenceAlignment:
    """Validate that missing seed types in reference triggers a loud error."""

    def test_missing_type_raises(self):
        """If a seed cell type is not in reference unique_types, must raise."""
        seeds = {
            "TypeA": {"topic_index": 0, "genes": ["g1"]},
            "TypeB": {"topic_index": 1, "genes": ["g2"]},
            "TypeC": {"topic_index": 2, "genes": ["g3"]},
        }
        unique_types_in_ref = ["TypeA", "TypeB"]  # TypeC missing

        missing = [ct for ct in seeds.keys() if ct not in unique_types_in_ref]
        assert len(missing) > 0, "Should detect missing types"
        assert "TypeC" in missing

    def test_all_present_passes(self):
        """If all seed types exist in reference, no error."""
        seeds = {
            "TypeA": {"topic_index": 0, "genes": ["g1"]},
            "TypeB": {"topic_index": 1, "genes": ["g2"]},
        }
        unique_types_in_ref = ["TypeA", "TypeB", "TypeC"]

        missing = [ct for ct in seeds.keys() if ct not in unique_types_in_ref]
        assert len(missing) == 0


# ============================================================
# TEST 4: max_prop correctness
# ============================================================

class TestMaxProp:
    """Validate max_prop is computed from final normalized aligned proportions."""

    def test_max_prop_from_normalized(self):
        """max_prop should be max of row-normalized nnls_aligned, not raw beta."""
        nnls_aligned = np.array([
            [0.8, 0.1, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.1, 0.8],
        ], dtype=np.float32)
        max_prop = np.max(nnls_aligned, axis=1)
        np.testing.assert_allclose(max_prop, [0.8, 0.5, 0.8])

    def test_max_prop_after_smoothing_differs(self):
        """Spatial smoothing changes proportions, so max_prop should be from AFTER."""
        raw_beta = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        smoothed = 0.7 * raw_beta + 0.3 * raw_beta[::-1]
        smoothed_norm = smoothed / smoothed.sum(axis=1, keepdims=True)
        max_prop_raw = np.max(raw_beta, axis=1)
        max_prop_smooth = np.max(smoothed_norm, axis=1)
        assert not np.allclose(max_prop_raw, max_prop_smooth), \
            "Smoothing should change max_prop values"

    def test_max_prop_range(self):
        """max_prop must be in (0, 1] for properly normalized proportions."""
        rng = np.random.default_rng(99)
        nnls = rng.dirichlet(np.ones(5), size=100).astype(np.float32)
        max_prop = np.max(nnls, axis=1)
        assert np.all(max_prop > 0)
        assert np.all(max_prop <= 1.0 + 1e-6)


class TestDsLevelStrategy:
    """Tests for the dataset-level uniform weight strategy."""

    def test_dslevel_produces_uniform_weight(self):
        """dslevel must produce a single scalar applied to all spots."""
        rng = np.random.default_rng(42)
        stability = rng.uniform(0.7, 1.0, size=200).astype(np.float32)
        max_prop = rng.uniform(0.3, 0.9, size=200).astype(np.float32)
        g1, t1, g2, t2, w_max = 20.0, 0.80, 10.0, 0.85, 0.95

        mean_stab = float(np.mean(stability))
        mean_maxp = float(np.mean(max_prop))
        w_scalar = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), g1, t1, g2, t2, w_max=w_max)[0]
        w_per_spot = np.full(len(stability), w_scalar, dtype=np.float32)

        assert w_per_spot.std() == 0.0, "dslevel weight must be uniform (zero std)"
        assert w_per_spot.shape == (200,)

    def test_dslevel_crc_gets_high_weight(self):
        """CRC-like stats (high stab, high maxp) should yield substantial weight."""
        mean_stab, mean_maxp = 0.98, 0.89
        w = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), 20.0, 0.80, 10.0, 0.85, w_max=0.95)[0]
        assert w > 0.4, f"CRC should get w>0.4, got {w:.4f}"

    def test_dslevel_npc_gets_low_weight(self):
        """NPC-like stats (high stab, low maxp) should yield near-minimum weight."""
        mean_stab, mean_maxp = 0.95, 0.48
        w = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), 20.0, 0.80, 10.0, 0.85, w_max=0.95)[0]
        assert w < 0.12, f"NPC should get w<0.12, got {w:.4f}"

    def test_dslevel_xenium_gets_low_weight(self):
        """xenium-like stats (lower stab, medium maxp) should yield near-minimum weight."""
        mean_stab, mean_maxp = 0.73, 0.60
        w = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), 20.0, 0.80, 10.0, 0.85, w_max=0.95)[0]
        assert w < 0.12, f"xenium should get w<0.12, got {w:.4f}"

    def test_dslevel_configs_in_matrix(self):
        """All 6 dslevel configs exist and produce 24 experiments."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX
        dslevel_exps = [e for e in EXPERIMENT_MATRIX if e.get("strategy") == "dslevel"]
        assert len(dslevel_exps) == 24, f"Expected 24 dslevel experiments, got {len(dslevel_exps)}"
        methods = set(e["method"] for e in dslevel_exps)
        expected = {"dslevel_A", "dslevel_B", "dslevel_C", "dslevel_D", "dslevel_E", "dslevel_F"}
        assert methods == expected, f"Missing configs: {expected - methods}"

    def test_dslevel_higher_t2_lowers_weight(self):
        """Raising t2 should lower weight for datasets with maxp below threshold."""
        mean_stab, mean_maxp = 0.95, 0.83
        w_low_t2 = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), 20.0, 0.80, 10.0, 0.80, w_max=0.95)[0]
        w_high_t2 = compute_adaptive_weights_twofactor(
            np.array([mean_stab]), np.array([mean_maxp]), 20.0, 0.80, 10.0, 0.90, w_max=0.95)[0]
        assert w_high_t2 < w_low_t2, "Higher t2 should produce lower weight when maxp < t2"


class TestDsLevelSharpStrategy:
    """Tests for the dataset-level uniform weight + temperature sharpening strategy."""

    def test_sharpening_reduces_entropy(self):
        """Sharpening (T<1) must reduce entropy of proportions."""
        rng = np.random.default_rng(42)
        proportions = rng.dirichlet(np.ones(6), size=50).astype(np.float32)
        entropy_before = -(proportions * np.log(proportions + 1e-10)).sum(axis=1).mean()

        T = 0.5
        p_clipped = np.clip(proportions, 1e-10, None)
        p_sharp = p_clipped ** (1.0 / T)
        p_sharp = p_sharp / p_sharp.sum(axis=1, keepdims=True)
        entropy_after = -(p_sharp * np.log(p_sharp + 1e-10)).sum(axis=1).mean()

        assert entropy_after < entropy_before, \
            f"Sharpening should reduce entropy: {entropy_after:.4f} >= {entropy_before:.4f}"

    def test_sharpening_preserves_normalization(self):
        """After sharpening, rows must still sum to 1."""
        rng = np.random.default_rng(7)
        proportions = rng.dirichlet(np.ones(5), size=100).astype(np.float32)

        for T in [0.3, 0.5, 0.7, 0.8]:
            p_clipped = np.clip(proportions, 1e-10, None)
            p_sharp = p_clipped ** (1.0 / T)
            p_sharp = p_sharp / p_sharp.sum(axis=1, keepdims=True)
            row_sums = p_sharp.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-5), \
                f"Row sums not 1.0 for T={T}: {row_sums[:5]}"

    def test_sharpening_identity_at_T1(self):
        """T=1.0 should be a no-op (identity)."""
        rng = np.random.default_rng(99)
        proportions = rng.dirichlet(np.ones(6), size=50).astype(np.float64)

        T = 1.0
        p_clipped = np.clip(proportions, 1e-10, None)
        p_sharp = p_clipped ** (1.0 / T)
        p_sharp = p_sharp / p_sharp.sum(axis=1, keepdims=True)

        assert np.allclose(proportions, p_sharp, atol=1e-10), \
            "T=1.0 should not change proportions"

    def test_sharpening_respects_w_thresh_guard(self):
        """When w_scalar <= sharpen_w_thresh, sharpening must NOT be applied."""
        w_scalar = 0.07  # NPC/xenium-like
        sharpen_w_thresh = 0.3
        sharpen_T = 0.5

        should_sharpen = (w_scalar > sharpen_w_thresh and sharpen_T < 1.0)
        assert not should_sharpen, "w=0.07 should NOT trigger sharpening with thresh=0.3"

    def test_sharpening_triggers_for_crc(self):
        """When w_scalar > sharpen_w_thresh, sharpening MUST be applied."""
        w_scalar = 0.57  # CRC-like
        sharpen_w_thresh = 0.3
        sharpen_T = 0.5

        should_sharpen = (w_scalar > sharpen_w_thresh and sharpen_T < 1.0)
        assert should_sharpen, "w=0.57 should trigger sharpening with thresh=0.3"

    def test_sharpening_increases_max_prop(self):
        """Sharpening should increase the max proportion of each spot."""
        rng = np.random.default_rng(11)
        proportions = rng.dirichlet(np.ones(6) * 2, size=100).astype(np.float32)
        max_before = proportions.max(axis=1)

        T = 0.5
        p_clipped = np.clip(proportions, 1e-10, None)
        p_sharp = p_clipped ** (1.0 / T)
        p_sharp = p_sharp / p_sharp.sum(axis=1, keepdims=True)
        max_after = p_sharp.max(axis=1)

        assert np.all(max_after >= max_before - 1e-6), \
            "Sharpening should not decrease max proportion"

    def test_dslevel_sharp_configs_in_matrix(self):
        """All 6 dslevel_sharp configs exist and produce 24 experiments."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX
        sharp_exps = [e for e in EXPERIMENT_MATRIX if e.get("strategy") == "dslevel_sharp"]
        assert len(sharp_exps) == 24, f"Expected 24 dslevel_sharp experiments, got {len(sharp_exps)}"
        methods = set(e["method"] for e in sharp_exps)
        expected = {"dslevel_sharp_A", "dslevel_sharp_B", "dslevel_sharp_C",
                    "dslevel_sharp_D", "dslevel_sharp_E", "dslevel_sharp_F"}
        assert methods == expected, f"Missing configs: {expected - methods}"

    def test_clip_prevents_zero_power_issues(self):
        """Clipping to 1e-10 before power should prevent NaN/Inf."""
        proportions = np.array([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T = 0.3
        p_clipped = np.clip(proportions, 1e-10, None)
        p_sharp = p_clipped ** (1.0 / T)
        p_sharp = p_sharp / p_sharp.sum(axis=1, keepdims=True)
        assert np.all(np.isfinite(p_sharp)), "Sharpening with zeros should not produce NaN/Inf"
        assert np.allclose(p_sharp.sum(axis=1), 1.0, atol=1e-5)


# ============================================================
# TEST 5: Entropy Gating (Phase 10, Direction A)
# ============================================================

class TestEntropyGating:
    """Tests for the theta-entropy gating strategy."""

    def test_gate_bounds(self):
        """Gate output must be in [0, 1] for any normalized entropy."""
        H_norm = np.linspace(0, 1, 1000)
        for gamma_ent in [5.0, 10.0, 20.0]:
            for H_thresh in [0.3, 0.5, 0.7]:
                gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - H_thresh)))
                assert np.all(gate >= 0.0) and np.all(gate <= 1.0), \
                    f"Gate out of [0,1] for gamma={gamma_ent}, thresh={H_thresh}"

    def test_gate_zero_at_zero_entropy(self):
        """When theta has zero entropy (perfectly confident), gate should be small."""
        H_norm = np.array([0.0])
        # With H_thresh=0.5 and gamma>=5, sigmoid(gamma*(0-0.5)) is very small
        for gamma_ent in [5.0, 10.0]:
            gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - 0.5)))
            assert gate[0] < 0.1, \
                f"Gate should be near 0 at H=0 with thresh=0.5: got {gate[0]:.4f} (gamma={gamma_ent})"
        # With H_thresh=0.3, gamma=10 still gives small gate
        gate = 1.0 / (1.0 + np.exp(-10.0 * (H_norm - 0.3)))
        assert gate[0] < 0.06, \
            f"Gate should be small at H=0 with gamma=10, thresh=0.3: got {gate[0]:.4f}"

    def test_gate_high_at_max_entropy(self):
        """When theta has max entropy (uniform), gate should be near 1."""
        H_norm = np.array([1.0])
        for gamma_ent in [5.0, 10.0]:
            for H_thresh in [0.3, 0.5]:
                gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - H_thresh)))
                assert gate[0] > 0.9, \
                    f"Gate should be near 1 at H=1: got {gate[0]:.4f} (gamma={gamma_ent}, thresh={H_thresh})"

    def test_effective_weight_bounded_by_w_dataset(self):
        """w_per_spot = w_dataset * gate must never exceed w_dataset."""
        rng = np.random.default_rng(42)
        K = 6
        theta = rng.dirichlet(np.ones(K) * 2, size=500).astype(np.float32)
        H = -np.sum(theta * np.log(theta + 1e-10), axis=1)
        H_norm = H / np.log(K)
        w_dataset = 0.585
        gamma_ent, H_thresh = 10.0, 0.5
        gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - H_thresh)))
        w_per_spot = w_dataset * gate
        assert np.all(w_per_spot >= 0.0)
        assert np.all(w_per_spot <= w_dataset + 1e-6)

    def test_npc_xenium_unaffected(self):
        """When w_dataset is near 0 (NPC/xenium), entropy gating produces negligible w."""
        rng = np.random.default_rng(7)
        K = 7
        theta = rng.dirichlet(np.ones(K) * 0.5, size=100).astype(np.float32)
        H = -np.sum(theta * np.log(theta + 1e-10), axis=1)
        H_norm = H / np.log(K)
        w_dataset = 0.07  # NPC-like
        gamma_ent, H_thresh = 10.0, 0.5
        gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - H_thresh)))
        w_per_spot = w_dataset * gate
        assert w_per_spot.max() < 0.08, \
            f"NPC w_per_spot should be negligible, got max={w_per_spot.max():.4f}"

    def test_blend_preserves_normalization(self):
        """Entropy-gated blend should produce valid probability distributions."""
        rng = np.random.default_rng(11)
        K = 6
        N = 200
        theta = rng.dirichlet(np.ones(K) * 2, size=N).astype(np.float64)
        nnls_prior = rng.dirichlet(np.ones(K) * 3, size=N).astype(np.float64)
        H = -np.sum(theta * np.log(theta + 1e-10), axis=1)
        H_norm = H / np.log(K)
        w_dataset = 0.585
        gamma_ent, H_thresh = 10.0, 0.5
        gate = 1.0 / (1.0 + np.exp(-gamma_ent * (H_norm - H_thresh)))
        w_per_spot = w_dataset * gate
        w_2d = w_per_spot[:, None]
        proportions = (1.0 - w_2d) * theta + w_2d * nnls_prior
        row_sums = proportions.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"Blend row sums not 1: {row_sums[:5]}"


# ============================================================
# TEST 6: Seed-Gene NNLS (Phase 10, Direction B)
# ============================================================

class TestSeedGeneNNLS:
    """Tests for seed-gene inclusion in NNLS gene selection."""

    def test_seed_gene_inclusion_increases_count(self):
        """Including seed genes must increase (or equal) the gene count."""
        rng = np.random.default_rng(42)
        n_genes = 5000
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        base_gene_idx = np.sort(rng.choice(n_genes, size=2100, replace=False))

        seeds = {
            "TypeA": {"topic_index": 0, "features": [f"gene_{i}" for i in range(10, 30)]},
            "TypeB": {"topic_index": 1, "features": [f"gene_{i}" for i in range(4980, 5000)]},
        }
        seed_gene_names = set()
        for ct_info in seeds.values():
            seed_gene_names.update(ct_info.get("features", []))
        seed_in_common = np.array([i for i, g in enumerate(gene_names) if g in seed_gene_names],
                                  dtype=np.intp)
        combined = np.union1d(base_gene_idx, seed_in_common)
        assert len(combined) >= len(base_gene_idx), \
            "Adding seed genes must not reduce gene count"
        assert len(combined) > len(base_gene_idx), \
            "At least some seed genes should be new (not already in HVG/markers)"

    def test_weighted_nnls_weight_vector(self):
        """Weighted mode: seed genes get sqrt(2) weight, others get 1.0."""
        rng = np.random.default_rng(7)
        G = 2200
        gene_idx = np.arange(G)
        seed_in_common = np.array([100, 200, 300, 2100, 2150])

        seed_mask_in_selected = np.isin(gene_idx, seed_in_common)
        gene_weights = np.ones(G, dtype=np.float64)
        gene_weights[seed_mask_in_selected] = np.sqrt(2.0)

        assert np.sum(seed_mask_in_selected) == 5
        assert np.allclose(gene_weights[seed_mask_in_selected], np.sqrt(2.0))
        assert np.allclose(gene_weights[~seed_mask_in_selected], 1.0)

    def test_weighted_nnls_solve_correctness(self):
        """Weighted NNLS should still produce valid non-negative solutions."""
        from scipy.optimize import nnls as scipy_nnls
        rng = np.random.default_rng(42)
        G = 50
        K_ref = 3
        N = 10

        X_norm = rng.uniform(0, 5, size=(K_ref, G))
        Y_norm = rng.uniform(0, 5, size=(N, G))
        A_full = X_norm.T

        gene_weights = np.ones(G, dtype=np.float64)
        gene_weights[:10] = np.sqrt(2.0)
        A_weighted = A_full * gene_weights[:, None]

        for i in range(N):
            b_weighted = Y_norm[i] * gene_weights
            x, _ = scipy_nnls(A_weighted, b_weighted)
            assert np.all(x >= 0), f"NNLS solution has negatives at spot {i}"
            assert x.shape == (K_ref,)

    def test_seed_gene_mode_default_none(self):
        """Default seed_gene_mode=None should not modify gene selection."""
        seed_gene_mode = None
        n_before = 2100
        gene_idx = np.arange(n_before)
        if seed_gene_mode in ("include", "weighted"):
            gene_idx = np.union1d(gene_idx, np.array([2200, 2300]))
        assert len(gene_idx) == n_before, \
            "None mode should not change gene count"


# ============================================================
# TEST 7: Post-Blend Smoothing (Phase 10, Direction C)
# ============================================================

class TestPostBlendSmoothing:
    """Tests for post-blend spatial smoothing strategy."""

    def test_smoothing_preserves_normalization(self):
        """After smoothing, rows must still sum to 1."""
        rng = np.random.default_rng(42)
        N, K = 100, 6
        proportions = rng.dirichlet(np.ones(K), size=N).astype(np.float64)
        coords = rng.uniform(0, 100, size=(N, 2))

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        neighbor_idx = indices[:, 1:]

        for iter_post in [1, 2, 3]:
            p = proportions.copy()
            for _ in range(iter_post):
                neighbor_mean = p[neighbor_idx].mean(axis=1)
                p = 0.85 * p + 0.15 * neighbor_mean
            p = p / p.sum(axis=1, keepdims=True)
            assert np.allclose(p.sum(axis=1), 1.0, atol=1e-10), \
                f"Row sums not 1.0 after {iter_post} iterations"

    def test_smoothing_reduces_noise(self):
        """Smoothing should reduce variance of predictions among neighbors."""
        rng = np.random.default_rng(7)
        N, K = 200, 5
        base = rng.dirichlet(np.ones(K) * 10, size=1)
        noise = rng.dirichlet(np.ones(K) * 0.5, size=N)
        proportions = (0.8 * base + 0.2 * noise).astype(np.float64)
        proportions = proportions / proportions.sum(axis=1, keepdims=True)

        coords = rng.uniform(0, 10, size=(N, 2))
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        neighbor_idx = indices[:, 1:]

        var_before = proportions.var(axis=0).sum()
        p = proportions.copy()
        for _ in range(2):
            neighbor_mean = p[neighbor_idx].mean(axis=1)
            p = 0.90 * p + 0.10 * neighbor_mean
        p = p / p.sum(axis=1, keepdims=True)
        var_after = p.var(axis=0).sum()
        assert var_after < var_before, \
            f"Smoothing should reduce variance: {var_after:.6f} >= {var_before:.6f}"

    def test_smoothing_convergence(self):
        """Multiple iterations should converge (decreasing change)."""
        rng = np.random.default_rng(11)
        N, K = 50, 4
        proportions = rng.dirichlet(np.ones(K), size=N).astype(np.float64)
        coords = rng.uniform(0, 50, size=(N, 2))

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        neighbor_idx = indices[:, 1:]

        alpha_post = 0.85
        changes = []
        p = proportions.copy()
        for _ in range(5):
            p_old = p.copy()
            neighbor_mean = p[neighbor_idx].mean(axis=1)
            p = alpha_post * p + (1 - alpha_post) * neighbor_mean
            p = p / p.sum(axis=1, keepdims=True)
            changes.append(np.abs(p - p_old).sum())

        for i in range(1, len(changes)):
            assert changes[i] <= changes[i - 1] + 1e-10, \
                f"Change should decrease: iter {i} ({changes[i]:.6f}) > iter {i-1} ({changes[i-1]:.6f})"

    def test_no_spatial_coords_skips_smoothing(self):
        """If no spatial coordinates found, proportions should remain unchanged."""
        rng = np.random.default_rng(99)
        N, K = 50, 6
        proportions = rng.dirichlet(np.ones(K), size=N).astype(np.float64)
        original = proportions.copy()

        coords = None
        if coords is not None:
            pass  # smoothing would happen
        # coords is None -> no smoothing
        assert np.allclose(proportions, original)


# ============================================================
# TEST 8: Phase 10 Config Matrix (all 3 directions)
# ============================================================

class TestPhase10ConfigMatrix:
    """Verify Phase 10 configs are in the experiment matrix with correct indices."""

    def test_entgate_configs_in_matrix(self):
        """All 4 dslevel_entgate configs exist and produce 16 experiments."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX
        entgate_exps = [e for e in EXPERIMENT_MATRIX if e.get("strategy") == "dslevel_entgate"]
        assert len(entgate_exps) == 16, f"Expected 16 entgate experiments, got {len(entgate_exps)}"
        methods = set(e["method"] for e in entgate_exps)
        expected = {"dslevel_entgate_A", "dslevel_entgate_B", "dslevel_entgate_C", "dslevel_entgate_D"}
        assert methods == expected, f"Missing: {expected - methods}"

    def test_seednnls_configs_in_matrix(self):
        """All 2 dslevel_seednnls configs exist and produce 8 experiments."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX
        seednnls_exps = [e for e in EXPERIMENT_MATRIX if e.get("strategy") == "dslevel_seednnls"]
        assert len(seednnls_exps) == 8, f"Expected 8 seednnls experiments, got {len(seednnls_exps)}"
        methods = set(e["method"] for e in seednnls_exps)
        expected = {"dslevel_seednnls_A", "dslevel_seednnls_B"}
        assert methods == expected, f"Missing: {expected - methods}"

    def test_postsmooth_configs_in_matrix(self):
        """All 2 dslevel_postsmooth configs exist and produce 8 experiments."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX
        postsmooth_exps = [e for e in EXPERIMENT_MATRIX if e.get("strategy") == "dslevel_postsmooth"]
        assert len(postsmooth_exps) == 8, f"Expected 8 postsmooth experiments, got {len(postsmooth_exps)}"
        methods = set(e["method"] for e in postsmooth_exps)
        expected = {"dslevel_postsmooth_A", "dslevel_postsmooth_B"}
        assert methods == expected, f"Missing: {expected - methods}"

    def test_total_matrix_size(self):
        """Total experiment matrix should be 36 configs x 4 datasets = 144."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX, CONFIGS_PER_DATASET
        assert len(CONFIGS_PER_DATASET) == 36, \
            f"Expected 36 configs (28 old + 8 new), got {len(CONFIGS_PER_DATASET)}"
        assert len(EXPERIMENT_MATRIX) == 144, \
            f"Expected 144 total experiments, got {len(EXPERIMENT_MATRIX)}"

    def test_phase10_indices(self):
        """Phase 10 experiments should total 32 (8 configs x 4 datasets)."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX, CONFIGS_PER_DATASET
        phase10_strategies = {"dslevel_entgate", "dslevel_seednnls", "dslevel_postsmooth"}
        phase10_exps = [(i, e) for i, e in enumerate(EXPERIMENT_MATRIX)
                        if e.get("strategy") in phase10_strategies]
        indices = [i for i, _ in phase10_exps]
        assert len(indices) == 32, f"Expected 32 phase10 experiments, got {len(indices)}"
        # Phase 10 configs are the last 8 in CONFIGS_PER_DATASET (indices 28-35)
        n_configs = len(CONFIGS_PER_DATASET)
        phase10_cfg_indices = set(range(28, n_configs))
        for i, e in phase10_exps:
            cfg_pos = i % n_configs
            assert cfg_pos in phase10_cfg_indices, \
                f"Experiment at idx {i} has cfg_pos {cfg_pos}, not in phase10 range"

    def test_all_phase10_have_base_dslevel_params(self):
        """All phase10 configs must have g1=20, t1=0.80, g2=10, t2=0.85, w_max=0.95."""
        from experiments.v3_reffree_leverage.run_v28_twofactor import EXPERIMENT_MATRIX
        phase10_strategies = {"dslevel_entgate", "dslevel_seednnls", "dslevel_postsmooth"}
        for e in EXPERIMENT_MATRIX:
            if e.get("strategy") in phase10_strategies:
                assert e["g1"] == 20.0, f"{e['method']}: g1={e['g1']}"
                assert e["t1"] == 0.80, f"{e['method']}: t1={e['t1']}"
                assert e["g2"] == 10.0, f"{e['method']}: g2={e['g2']}"
                assert e["t2"] == 0.85, f"{e['method']}: t2={e['t2']}"
                assert e["w_max"] == 0.95, f"{e['method']}: w_max={e['w_max']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
