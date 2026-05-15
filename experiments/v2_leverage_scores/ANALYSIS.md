# V2: Leverage-Score Gene Weighting - Experiment Analysis

## Hypothesis

FlashDeconv uses leverage scores from the reference SVD to identify genes that distinguish
cell types independently of expression variance. Standard PCA-based approaches (including
SeedTopic's TF-IDF+PCA encoder input) are dominated by highly-variable genes, which may
not be the most informative for deconvolution—especially for rare cell types.

By computing per-gene leverage scores from the reference signature matrix and using them
to weight the TF-IDF matrix before PCA projection, we can amplify the encoder's ability
to distinguish rare types that variance-based methods miss.

## Method

### Leverage Score Computation
1. Load reference scRNA-seq, compute per-type mean expression (signature matrix, K×G)
2. Center columns, compute thin SVD: X = U·diag(s)·V^T
3. Leverage score per gene: `l_g = Σ_j (V_gj² · s_j² / Σ s²)`
4. Normalize so mean(l) = 1

### Integration with SeedTopic
- Apply as multiplicative weight to TF-IDF before PCA: `tfidf_weighted = tfidf * leverage^power`
- Power parameter softens the dynamic range (raw leverage ranges 0–46x for NPC)
- Sweep: power ∈ {0.25, 0.5, 1.0}

### Reference Data Sources
| Dataset | Reference | N_cells | N_types | Gene Overlap |
|---------|-----------|---------|---------|--------------|
| visium_NPC | NPC scRNA-seq h5ad | 5,508 | 7 | 4,828/5,000 |
| xenium_BC | 10x h5 + annotation xlsx | 27,472 | 19 | 313/313 |
| visiumHD_CRC_I | CRC reference h5ad | 279,691 | 32 | ~5,000/5,000 |
| visiumHD_CRC_II | Same CRC reference | 279,691 | 32 | ~5,000/5,000 |

## Results

### Full Metrics Table

| Dataset | Power | F1 | AUPRC | ARI | AMI | dF1 | dAUPRC | dARI |
|---------|-------|------|-------|------|------|--------|--------|--------|
| visium_NPC | baseline | 0.547 | 0.584 | 0.692 | 0.485 | — | — | — |
| visium_NPC | 0.25 | 0.577 | 0.624 | 0.660 | 0.472 | +0.030 | +0.040 | -0.032 |
| visium_NPC | 0.5 | 0.586 | 0.631 | 0.616 | 0.460 | +0.038 | +0.047 | -0.076 |
| visium_NPC | 1.0 | 0.571 | 0.612 | 0.598 | 0.437 | +0.023 | +0.028 | -0.095 |
| xenium_BC | baseline | 0.615 | 0.760 | 0.556 | 0.654 | — | — | — |
| xenium_BC | 0.25 | 0.619 | 0.761 | 0.571 | 0.659 | +0.004 | +0.001 | +0.015 |
| xenium_BC | 0.5 | 0.624 | 0.763 | 0.565 | 0.658 | +0.009 | +0.003 | +0.009 |
| xenium_BC | 1.0 | 0.628 | 0.756 | 0.559 | 0.652 | +0.013 | -0.004 | +0.003 |
| visiumHD_CRC_I | baseline | 0.527 | 0.821 | 0.471 | 0.538 | — | — | — |
| visiumHD_CRC_I | 0.25 | 0.542 | 0.868 | 0.473 | 0.515 | +0.016 | +0.047 | +0.002 |
| visiumHD_CRC_I | 0.5 | 0.534 | 0.849 | 0.455 | 0.504 | +0.007 | +0.028 | -0.016 |
| visiumHD_CRC_I | 1.0 | 0.542 | 0.849 | 0.482 | 0.511 | +0.015 | +0.029 | +0.011 |
| visiumHD_CRC_II | baseline | 0.271 | 0.557 | 0.367 | 0.249 | — | — | — |
| visiumHD_CRC_II | 0.25 | 0.354 | 0.595 | 0.428 | 0.291 | +0.083 | +0.038 | +0.062 |
| visiumHD_CRC_II | 0.5 | 0.402 | 0.622 | 0.458 | 0.319 | +0.131 | +0.064 | +0.091 |
| visiumHD_CRC_II | 1.0 | **0.411** | 0.592 | **0.495** | 0.344 | **+0.140** | +0.035 | **+0.128** |

### Best Configuration Per Dataset

| Dataset | Best Power | Best ARI | dARI | dF1 | Notes |
|---------|-----------|---------|------|------|-------|
| visium_NPC | 0.25 (or skip) | 0.660 | -0.032 | +0.030 | F1 up but ARI down |
| xenium_BC | 0.25 | 0.571 | +0.015 | +0.004 | Small gains (limited gene panel) |
| visiumHD_CRC_I | 1.0 | 0.482 | +0.011 | +0.015 | Modest improvement |
| visiumHD_CRC_II | 1.0 | 0.495 | +0.128 | +0.140 | Major improvement |

### Top Leverage Genes (examples)
- **NPC** (power=0.5, max weight=11.3x): COL1A1, SPARC, COL1A2, CST3, COL3A1 — fibroblast/myeloid markers
- **CRC** (32 types): Strong weighting of tumor subtype and immune markers

## Analysis

### Why CRC_II Benefits Most
- CRC_II has 6 ground-truth types mapped from a 32-type reference — many "rare" types in the reference
  context that standard PCA would underweight
- Baseline ARI (0.367) was the worst across all datasets, indicating the encoder input was
  suboptimal for this dataset
- The 32-type reference gives leverage scores with high discriminative power
- Power=1.0 (raw leverage) works best: the extreme dynamic range is appropriate when
  the reference has many types to distinguish

### Why NPC Shows F1/AUPRC Gain but ARI Loss
- NPC has only 7 well-separated types where standard PCA already captures the relevant structure
- Leverage weighting amplifies cell-type-specific genes, improving per-class accuracy (F1, AUPRC)
- But it distorts the overall geometry, making the argmax cluster boundaries less clean (lower ARI)
- This is a precision-vs-clustering trade-off: leverage makes predictions more confident per-type
  but introduces noise into the overall partition

### Xenium_BC Limited by Gene Panel
- Only 313 genes (targeted Xenium panel) — all were pre-selected for biological relevance
- Leverage re-weighting has minimal room to operate vs. 5000-gene Visium panels
- Still slightly positive at gentle power=0.25

### Power Parameter Interpretation
- **power=0.25**: gentle emphasis (max ~3x weight), safest across datasets
- **power=0.5**: moderate (sqrt of leverage, max ~11x), good for medium-complexity references
- **power=1.0**: raw leverage (max ~46x for NPC, varies by ref), best for complex references with many types

## Decisions

1. **Adopt leverage weighting as a configurable option** — not always beneficial, but dramatic
   gains on datasets with complex references (CRC_II: +0.128 ARI)
2. **Recommend power=1.0 for references with >10 types, power=0.25 for ≤10 types**
3. **Make power=0 (off) the default to avoid NPC-like regressions** — user enables when reference is available
4. **Next: combine with post-hoc spatial smoothing** — leverage (best power) + smoothing (alpha=0.3)
   could stack for further gains

## Combined Potential (Leverage + Spatial Smoothing)

From v1 experiment, post-hoc smoothing at alpha=0.3 gave:
- xenium_BC: ARI +0.022
- CRC_I: ARI +0.040
- CRC_II: ARI +0.017

If gains are additive (to be validated):
- **CRC_II: 0.367 → 0.495 (leverage) → ~0.512 (+smoothing)** vs baseline 0.367
- **CRC_I: 0.471 → 0.482 (leverage) → ~0.522 (+smoothing)** vs baseline 0.471
- **xenium_BC: 0.556 → 0.571 (leverage) → ~0.593 (+smoothing)** vs baseline 0.556

## Files

- `seededntm/util.py`: `compute_leverage_scores()`, updated `compute_tfidf_rep(gene_weights=...)`
- `experiments/v2_leverage_scores/run_leverage_test.py`: full experiment pipeline
- `experiments/v2_leverage_scores/run_stage1.sh`: NPC + CRC (~30 min)
- `experiments/v2_leverage_scores/run_stage2_xenium.sh`: xenium_BC (~60 min)
- `experiments/v2_leverage_scores/raw_outputs/`: proportions CSVs, leverage score .npy files
- `experiments/configs/datasets.json`: updated with xenium_BC reference paths
