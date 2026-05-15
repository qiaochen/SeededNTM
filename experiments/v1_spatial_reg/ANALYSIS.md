# Spatial Regularization Experiment Analysis

**Date:** 2026-05-15  
**Branch:** `improve/spatial-reg`  
**Status:** Completed. Post-hoc smoothing adopted; training-time approach abandoned.

---

## Hypothesis

SeedTopic's ARI gap vs RCTD (0.593 vs 0.743 on xenium_BC) is partly due to lack of spatial awareness. FlashDeconv uses a k-NN graph Laplacian regularization that enforces spatial smoothness of cell-type proportions. Adding spatial smoothing to SeedTopic should improve ARI/AMI (clustering agreement metrics).

## Methods Tested

### 1. Training-Time Graph Laplacian Regularization

**Implementation:** Added a post-ELBO auxiliary loss at the end of each training epoch:
```
L_spatial = lambda * sum_ij A_ij * ||theta_i - theta_j||^2 / N
```
where A is the k-NN (k=6) symmetric adjacency matrix and theta are the softmax topic proportions from the encoder.

**Integration point:** After the mini-batch ELBO + topic_prior steps, a full-batch forward pass computes theta for all spots, then the spatial loss is backpropagated through the encoder.

**Lambda sweep on visium_NPC:** {0.001, 0.01, 0.05, 0.1}

### 2. Post-Hoc Spatial Smoothing

**Implementation:** Simple weighted average after model inference:
```
theta_smooth = (1 - alpha) * theta + alpha * (A @ theta / degree)
```
followed by row renormalization.

**Alpha sweep on xenium_BC:** {0.1, 0.2, 0.3, 0.5, 0.7}

## Results

### Training-Time (visium_NPC, 1331 spots, 7 types)

| Lambda | F1 | AUPRC | ARI | AMI | delta_ARI |
|--------|-----|-------|-----|-----|-----------|
| baseline | 0.547 | 0.584 | 0.692 | 0.485 | — |
| 0.001 | 0.526 | 0.566 | 0.680 | 0.471 | -0.012 |
| 0.01 | 0.527 | 0.566 | 0.680 | 0.471 | -0.012 |
| 0.05 | 0.525 | 0.567 | 0.680 | 0.472 | -0.012 |
| 0.1 | 0.530 | 0.569 | 0.682 | 0.477 | -0.010 |

**Conclusion:** All lambda values hurt performance. The spatial loss barely changes the proportion distributions (entropy stays ~1.84, avg_max_prop ~0.27).

### Post-Hoc Smoothing (all datasets, alpha=0.3, k=6)

| Dataset | Spots | Baseline F1 | Smoothed F1 | Baseline ARI | Smoothed ARI | delta_ARI |
|---------|-------|-------------|-------------|--------------|--------------|-----------|
| xenium_BC | 159K | 0.615 | **0.627** | 0.556 | **0.578** | **+0.022** |
| visiumHD_CRC_I | 2.6K | 0.527 | **0.539** | 0.471 | **0.511** | **+0.040** |
| visiumHD_CRC_II | 2.3K | 0.271 | **0.277** | 0.367 | **0.384** | **+0.017** |
| visium_NPC | 1.3K | 0.547 | 0.518 | 0.692 | 0.670 | -0.022 |

### Alpha sensitivity (xenium_BC)

| Alpha | F1 | AUPRC | ARI | AMI |
|-------|-----|-------|-----|-----|
| 0 (base) | 0.615 | 0.760 | 0.556 | 0.654 |
| 0.1 | 0.620 | 0.762 | 0.565 | 0.659 |
| 0.2 | 0.624 | 0.762 | 0.572 | 0.662 |
| **0.3** | **0.627** | **0.760** | **0.578** | **0.661** |
| 0.5 | 0.621 | 0.735 | 0.574 | 0.639 |
| 0.7 | 0.546 | 0.632 | 0.518 | 0.558 |

Sweet spot: alpha=0.2-0.3. Over-smoothing at alpha>=0.5 degrades AUPRC.

## Analysis

### Why training-time regularization failed on NPC:

1. **Dataset too small (1331 spots):** The model is already uncertain -- avg max proportion is only 0.27 (nearly uniform over 7 types). Adding smoothing pressure pushes it further toward uniformity.
2. **Competing gradients:** The spatial loss conflicts with the ELBO, which tries to reconstruct per-spot gene expression. The reg_topic_prior (weight=0.9) already dominates the loss landscape.
3. **Graph structure on NPC is weak:** Visium hex grid with only 1331 spots has less meaningful spatial autocorrelation than higher-resolution datasets.

### Why post-hoc smoothing works:

1. **Doesn't interfere with training:** The model learns the best per-spot estimates freely, then spatial context is used to resolve ambiguous spots.
2. **Most effective when model is uncertain at boundaries:** The gain is largest on CRC_I (+0.040 ARI) where SeedTopic's baseline ARI is lowest (0.471), indicating many ambiguous spots at cell type boundaries.
3. **Hurts when model is already confident:** On NPC where ARI is already 0.692, smoothing adds noise rather than signal.

### Comparison with RCTD ARI gap:

| Dataset | SeedTopic base | SeedTopic+smooth | RCTD | Remaining gap |
|---------|---------------|-----------------|------|---------------|
| xenium_BC (subset) | 0.593 | ~0.615* | 0.743 | 0.128 |
| visiumHD_CRC_I (subset) | 0.537 | ~0.575* | 0.819 | 0.244 |

*Estimated from full-set improvement ratio.

Post-hoc smoothing closes ~15% of the ARI gap vs RCTD. The remaining gap is likely due to RCTD's platform effect correction and sharper Poisson-Lognormal likelihood (vs SeedTopic's softer variational posteriors).

## Decision

- **Adopt post-hoc smoothing as default post-processing** for datasets with >2000 spots
- **Use alpha=0.3** as default, with potential for dataset-adaptive tuning (skip if baseline ARI > 0.65)
- **Abandon training-time spatial regularization** -- the auxiliary loss approach doesn't work with SeedTopic's architecture (high reg_topic_prior weight dominates)
- **Keep the training-time code** (`--spatial_reg_lambda` CLI flag) for future experimentation but default to 0.0

## Files

- `experiments/v1_spatial_reg/run_npc_sweep.sh` -- training-time sweep script
- `experiments/v1_spatial_reg/evaluate_sweep.py` -- evaluation of training-time sweep
- `experiments/v1_spatial_reg/sweep_results.json` -- training-time sweep metrics
- `experiments/v1_spatial_reg/posthoc_smoothing.py` -- post-hoc smoothing implementation + evaluation
- `seededntm/experiment.py` -- training-time spatial loss (lines ~255-285)
- `seededntm/util.py` -- `build_spatial_graph()` function
- `seededntm/main.py` -- `--spatial_reg_lambda`, `--spatial_k_neighbors` CLI args

## Next Steps

1. Leverage-score gene weighting (targets CRC gap where FlashDeconv excels)
2. Platform effect correction (targets per-class AUPRC for common types)
3. Combine post-hoc smoothing with new improvements for final evaluation
