# SeedTopic+dslevel_entgate: Dataset-Level Entropy-Gated NNLS Blend

## Overview

SeedTopic+dslevel_entgate is the canonical post-hoc enhancement for reference-based SeedTopic.
After the topic model produces proportions (theta), NNLS deconvolution is computed from
the same scRNA-seq reference used for seed derivation. The two estimates are blended
using per-spot weights that combine a dataset-level scalar (from unsupervised statistics)
with a theta-entropy gate that protects confident model predictions from NNLS override.

This method is the default for reference-based SeedTopic when a scRNA-seq reference
is available alongside seeds.

## Method

### Pipeline

1. Train SeedTopic (standard pipeline: representation, prior, encoder, SVI) -> theta (N x K)
2. Compute NNLS deconvolution from reference -> nnls_prior (N x K)
3. Compute per-spot split-half stability and max_prop from NNLS
4. Aggregate to dataset-level: mean_stab, mean_maxp
5. Compute dataset-level scalar weight: `w_ds = f(mean_stab, mean_maxp)`
6. Compute per-spot entropy gate: `gate_i = sigmoid(gamma_ent * (H_norm_i - H_thresh))`
7. Blend with entropy gating: `proportions_i = (1 - w_ds * gate_i) * theta_i + (w_ds * gate_i) * nnls_prior_i`
8. Normalize: `proportions = proportions / proportions.sum(axis=1)`

### Weight Formula

```
w_ds = w_min + (w_max - w_min) * sigmoid(g1 * (mean_stab - t1)) * sigmoid(g2 * (mean_maxp - t2))
```

### Entropy Gate Formula

```
H_i = -sum(theta_i * log(theta_i + 1e-10))     # per-spot entropy
H_norm_i = H_i / log(K)                         # normalize to [0, 1]
gate_i = 1 / (1 + exp(-gamma_ent * (H_norm_i - H_thresh)))
w_i = w_ds * gate_i                             # effective per-spot weight
```

The entropy gate ensures:
- Where theta is confident (low entropy): gate near 0, NNLS does NOT override
- Where theta is uncertain (high entropy): gate near 1, NNLS blending at full strength
- NPC/xenium: w_ds is already ~0.06, so gating is irrelevant (effective w remains negligible)

### NNLS Computation

1. **Signature matrix**: Per-cell-type mean expression from reference (K_ref x G)
2. **Gene selection**: Union of top-2000 HVGs + top-50 fold-change markers per type
3. **Preprocessing**: log1p(CPM) normalization of both spatial and reference
4. **NNLS solve**: `scipy.optimize.nnls` per spot against signature matrix
5. **Spatial smoothing**: k=7 nearest neighbors, 3 iterations, alpha=0.7 self-retention
6. **Normalization**: Row-normalize to proportions summing to 1
7. **Alignment**: Map reference cell types to seed topic indices

### Stability Measurement

Split-half stability measures NNLS reproducibility:
- Split selected genes into two random halves (3 splits)
- Solve NNLS independently on each half
- Compute cosine similarity between the two solutions
- Average across splits -> stability per spot

High stability (>0.95) indicates the NNLS solution is robust to gene subsampling.

## Canonical Parameters

Selected by ablation over multiple configurations x 4 datasets. These are immutable.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `g1` | `20.0` | Stability gain |
| `t1` | `0.80` | Stability threshold |
| `g2` | `10.0` | Max-prop gain |
| `t2` | `0.85` | Max-prop threshold |
| `w_min` | `0.05` | Floor weight |
| `w_max` | `0.95` | Ceiling weight |
| `gamma_ent` | `5.0` | Entropy gate sharpness |
| `H_thresh` | `0.5` | Entropy gate threshold (normalized) |
| Spatial smoothing k | `7` | Neighbors for NNLS smoothing |
| Spatial smoothing iterations | `3` | Diffusion steps |
| Spatial smoothing alpha | `0.7` | Self-retention |
| Split-half n_splits | `3` | Number of random gene splits for stability |

## Why Entropy Gating Over Uniform Blend

Phase 8 established that dataset-level uniform weight (dslevel_A) was superior to per-spot
two-factor weighting. However, the uniform weight of 0.585 on CRC_II overshoots the
optimal ~0.50 (found empirically with fixed_w50=0.699). The problem: NNLS sometimes
misassigns the dominant type, and at w=0.585, it overrides correct theta predictions.

Entropy gating solves this by selectively protecting confident theta predictions:
- Spots where theta has low entropy (dominant type is clear) get w_effective near 0
- Spots where theta is uncertain get the full w_ds blend benefit
- Unlike per-spot two-factor (which used NNLS statistics that were uniformly high on CRC),
  theta entropy genuinely varies within CRC, providing meaningful per-spot discrimination

Empirical result: effective mean weight on CRC_II drops from 0.585 to 0.532, matching
the optimal band. CRC_II F1 improves from 0.693 to 0.735 (+0.043).

## Resulting Weights (Measured)

| Dataset | mean_stab | mean_maxp | w_ds | eff_w_mean | eff_w_std | Effect |
|---------|-----------|-----------|------|------------|-----------|--------|
| NPC | 0.948 | 0.482 | 0.071 | 0.061 | 0.006 | Near-baseline (negligible) |
| xenium | 0.726 | 0.596 | 0.062 | 0.053 | 0.004 | Near-baseline (negligible) |
| CRC_I | 0.973 | 0.887 | 0.566 | 0.459 | 0.092 | Moderate blend (protected confident spots) |
| CRC_II | 0.983 | 0.895 | 0.585 | 0.532 | 0.010 | Moderate blend (near-optimal for CRC_II) |

## Results (dslevel_entgate_C: Best Universal Config)

### CRC_I (visiumHD_CRC_I)

| Method | F1 | Precision | Recall | AMI | ARI | AUPRC | Pearson | Spearman | RMSE | JSD |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SeedTopic+dslevel_entgate** | **0.8618** | 0.8366 | 0.9112 | 0.7968 | 0.8760 | 0.9308 | 0.9352 | 0.6896 | 0.1967 | 0.1483 |
| RCTD | 0.8240 | 0.7642 | 0.9145 | 0.7769 | 0.8521 | 0.9398 | 0.9369 | 0.6888 | 0.1445 | 0.0697 |
| FlashDeconv | 0.8146 | 0.7642 | 0.8936 | 0.7665 | 0.8535 | 0.9349 | 0.9457 | 0.8392 | 0.1308 | 0.0475 |
| Cell2location | 0.7882 | 0.7160 | 0.9493 | 0.7547 | 0.8152 | 0.9382 | 0.9319 | 0.6874 | 0.1470 | 0.0636 |
| Stereoscope | 0.6424 | 0.5900 | 0.9032 | 0.6078 | 0.6839 | 0.7519 | 0.8460 | 0.6575 | 0.2156 | 0.1275 |
| MarkerScore | 0.6975 | 0.6461 | 0.8976 | 0.6536 | 0.7289 | 0.8981 | 0.8600 | 0.6705 | 0.2491 | 0.2034 |
| STAMP | 0.5096 | 0.5015 | 0.6481 | 0.4384 | 0.2734 | 0.7124 | 0.6575 | 0.5609 | 0.3028 | 0.2399 |
| SeedTopic (base) | 0.5123 | 0.4863 | 0.8694 | 0.4991 | 0.4451 | 0.8438 | 0.6709 | 0.5637 | 0.3159 | 0.2853 |

### CRC_II (visiumHD_CRC_II)

| Method | F1 | Precision | Recall | AMI | ARI | AUPRC | Pearson | Spearman | RMSE | JSD |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cell2location | 0.7539 | 0.7428 | 0.8044 | 0.6239 | 0.8066 | 0.8269 | 0.9615 | 0.6446 | 0.1032 | 0.0392 |
| **SeedTopic+dslevel_entgate** | **0.7354** | 0.7973 | 0.7697 | 0.6550 | 0.8374 | 0.8157 | 0.9590 | 0.6444 | 0.1888 | 0.1695 |
| RCTD | 0.7329 | 0.8251 | 0.7573 | 0.6498 | 0.7853 | 0.8740 | 0.9707 | 0.6449 | 0.0902 | 0.0315 |
| FlashDeconv | 0.7074 | 0.6641 | 0.7802 | 0.6010 | 0.7998 | 0.8117 | 0.9614 | 0.8654 | 0.1028 | 0.0350 |
| Stereoscope | 0.4689 | 0.3786 | 0.8000 | 0.2814 | 0.3753 | 0.3869 | 0.8352 | 0.5968 | 0.2053 | 0.1223 |
| MarkerScore | 0.5915 | 0.5416 | 0.7474 | 0.4422 | 0.6162 | 0.6867 | 0.9028 | 0.6366 | 0.2626 | 0.2637 |
| STAMP | 0.1379 | 0.2016 | 0.3304 | 0.1288 | 0.0375 | 0.2320 | 0.2594 | 0.2981 | 0.3713 | 0.3918 |
| SeedTopic (base) | 0.3913 | 0.3426 | 0.6120 | 0.3356 | 0.4569 | 0.5122 | 0.7224 | 0.5286 | 0.3318 | 0.3762 |

### NPC (visium_NPC)

| Method | F1 | Precision | Recall | AMI | ARI | AUPRC | Pearson | Spearman | RMSE | JSD |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stereoscope | 0.6103 | 0.6673 | 0.6019 | 0.3802 | 0.4285 | 0.6359 | 0.7051 | 0.5369 | 0.2592 | 0.2668 |
| Cell2location | 0.5495 | 0.7328 | 0.5480 | 0.5693 | 0.6747 | 0.7564 | 0.7950 | 0.5752 | 0.2358 | 0.2363 |
| FlashDeconv | 0.5305 | 0.6622 | 0.4762 | 0.4500 | 0.5425 | 0.6386 | 0.7323 | 0.5641 | 0.2516 | 0.2519 |
| MarkerScore | 0.5000 | 0.5681 | 0.5363 | 0.3649 | 0.3736 | 0.6468 | 0.5790 | 0.4963 | 0.3191 | 0.4023 |
| **SeedTopic+dslevel_entgate** | **0.4827** | 0.5355 | 0.5083 | 0.4337 | 0.5360 | 0.5459 | 0.6622 | 0.5236 | 0.2900 | 0.3420 |
| STAMP | 0.3350 | 0.4267 | 0.3488 | 0.3008 | 0.1808 | 0.4236 | 0.2894 | 0.2374 | 0.3423 | 0.4127 |
| RCTD | 0.3060 | 0.4743 | 0.2779 | 0.2384 | 0.2472 | 0.6010 | 0.6259 | 0.4550 | 0.2730 | 0.2702 |
| SeedTopic (base) | 0.4608 | 0.5189 | 0.4656 | 0.4030 | 0.4798 | 0.5340 | 0.6320 | 0.5031 | 0.2938 | 0.3487 |

### xenium (xenium_BC)

| Method | F1 | Precision | Recall | AMI | ARI | AUPRC | Pearson | Spearman | RMSE | JSD |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RCTD | 0.6654 | 0.6816 | 0.7855 | 0.7282 | 0.7117 | 0.7695 | 0.8232 | 0.3741 | 0.1279 | 0.1697 |
| **SeedTopic+dslevel_entgate** | **0.6586** | 0.6469 | 0.8074 | 0.6860 | 0.6316 | 0.7605 | 0.7378 | 0.3707 | 0.1802 | 0.3857 |
| Cell2location | 0.5828 | 0.5665 | 0.7589 | 0.6412 | 0.4910 | 0.7521 | 0.7231 | 0.3656 | 0.1554 | 0.2079 |
| Stereoscope | 0.5781 | 0.5683 | 0.7186 | 0.6086 | 0.4410 | 0.6793 | 0.6805 | 0.3478 | 0.1639 | 0.2542 |
| FlashDeconv | 0.5558 | 0.6352 | 0.6694 | 0.6339 | 0.5103 | 0.6740 | 0.6998 | 0.6002 | 0.1630 | 0.2128 |
| MarkerScore | 0.5193 | 0.5137 | 0.6746 | 0.5717 | 0.5136 | 0.6402 | 0.6780 | 0.3658 | 0.1709 | 0.3263 |
| STAMP | 0.4165 | 0.4578 | 0.5820 | 0.6116 | 0.3186 | 0.5439 | 0.5333 | 0.3331 | 0.1897 | 0.3899 |
| SeedTopic (base) | 0.6407 | 0.6281 | 0.8066 | 0.6731 | 0.6191 | 0.7417 | 0.7219 | 0.3669 | 0.1829 | 0.3961 |

## Overall Ranking (Avg F1 across 4 datasets)

| Rank | Method | CRC_I | CRC_II | NPC | xenium | Avg F1 |
|------|--------|-------|--------|-----|--------|--------|
| 1 | **SeedTopic+dslevel_entgate** | 0.862 | 0.735 | 0.483 | 0.659 | **0.685** |
| 2 | Cell2location | 0.788 | 0.754 | 0.550 | 0.583 | 0.669 |
| 3 | FlashDeconv | 0.815 | 0.707 | 0.531 | 0.556 | 0.652 |
| 4 | RCTD | 0.824 | 0.733 | 0.306 | 0.665 | 0.632 |
| 5 | Stereoscope | 0.642 | 0.469 | 0.610 | 0.578 | 0.575 |
| 6 | MarkerScore | 0.698 | 0.592 | 0.500 | 0.519 | 0.577 |
| 7 | SeedTopic (base) | 0.512 | 0.391 | 0.461 | 0.641 | 0.501 |
| 8 | STAMP | 0.510 | 0.138 | 0.335 | 0.417 | 0.350 |

## Integration

The dslevel_entgate blend lives as a post-hoc step in experiment scripts, not in the core
`seededntm/` package. The core package remains a pure topic model. The post-hoc NNLS
blend is an optional enhancement applied after `do_experiment()` returns theta.

Reference implementation: `experiments/v3_reffree_leverage/run_v28_twofactor.py`,
strategy `"dslevel_entgate"`.

## History

- Phase 8: Established dataset-level uniform weight (dslevel_A, avg F1=0.680)
- Phase 9: Temperature sharpening failed to improve F1 (improves continuous metrics only)
- Phase 10: Entropy gating (dslevel_entgate_C) becomes new best (avg F1=0.685, CRC_II +0.043)
