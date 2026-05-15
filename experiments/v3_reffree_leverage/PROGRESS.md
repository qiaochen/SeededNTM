# Reference-Free Leverage Scores — Progress Log

## Status: Experiment Complete

### Implementation
- [x] `compute_reffree_leverage()` in `seededntm/util.py` — 3 approaches
- [x] Experiment script: `run_reffree_test.py`
- [x] Seed construction pipeline: `seededntm/seed_construction.py`
- [x] CLI: `scripts/construct_seeds.py`
- [x] Structural validation: 88.7% gene overlap with manual seeds

### Results

#### Xenium BC (leiden seeds) — Primary Reference-Free Test Case

| Method | F1 | AUPRC | ARI | AMI | dARI |
|--------|------|-------|------|------|------|
| none (baseline) | 0.402 | 0.574 | 0.379 | 0.598 | — |
| pseudo_sig | 0.343 | 0.441 | 0.338 | 0.567 | -0.041 |
| self_leverage_k | 0.400 | 0.551 | 0.368 | 0.590 | -0.011 |
| **seed_specificity** | **0.403** | **0.576** | **0.402** | **0.606** | **+0.023** |

#### CRC_I (5 types, 2580 spots)

| Method | F1 | AUPRC | ARI | AMI | dARI |
|--------|------|-------|------|------|------|
| none (baseline) | 0.518 | 0.816 | 0.454 | 0.524 | — |
| **pseudo_sig** | **0.556** | **0.813** | **0.578** | **0.586** | **+0.124** |
| self_leverage_k | 0.538 | 0.837 | 0.486 | 0.530 | +0.033 |
| seed_specificity | 0.530 | 0.824 | 0.459 | 0.513 | +0.005 |

#### CRC_II (6 types, 2275 spots)

| Method | F1 | AUPRC | ARI | AMI | dARI |
|--------|------|-------|------|------|------|
| none (baseline) | 0.274 | 0.577 | 0.355 | 0.241 | — |
| pseudo_sig | 0.320 | 0.607 | 0.393 | 0.278 | +0.037 |
| self_leverage_k | 0.372 | 0.622 | 0.432 | 0.304 | +0.077 |
| **seed_specificity** | **0.395** | **0.652** | **0.449** | **0.317** | **+0.094** |

#### NPC (7 types, 1331 spots)

| Method | F1 | AUPRC | ARI | AMI | dARI |
|--------|------|-------|------|------|------|
| none (baseline) | 0.452 | 0.456 | 0.471 | 0.378 | — |
| **pseudo_sig** | **0.476** | **0.462** | **0.512** | **0.395** | **+0.041** |
| self_leverage_k | 0.459 | 0.474 | 0.507 | 0.399 | +0.036 |
| seed_specificity | 0.468 | 0.453 | 0.474 | 0.388 | +0.003 |
| ref_based | 0.461 | 0.466 | 0.446 | 0.361 | -0.025 |

#### CountSketch vs PCA (NPC)

| Method | DimRed | ARI | dARI vs PCA |
|--------|--------|------|-------------|
| pseudo_sig | pca | 0.512 | — |
| pseudo_sig | sketch | 0.438 | -0.074 |
| self_leverage_k | pca | 0.507 | — |
| self_leverage_k | sketch | 0.425 | -0.082 |

### Key Findings

1. **seed_specificity is the safest ref-free method**: Never hurts, gives +0.023 ARI on xenium (primary case) and +0.094 on CRC_II. Recommended as default when no reference is available.

2. **pseudo_sig is high-variance**: Big win on CRC_I (+0.124 ARI) but hurts xenium (-0.041). The circularity issue (seed genes dominating pseudo-signatures) is worse with many types (19) and small gene panels (313 genes).

3. **self_leverage_k is moderate**: Consistent small gains on CRC_II (+0.077) and NPC (+0.036), slight loss on xenium (-0.011).

4. **Reference-based leverage hurts NPC** (-0.025 ARI): Confirms the finding from the v2 experiment. The ref-free pseudo_sig actually outperforms ref-based on NPC.

5. **PCA > CountSketch**: Sketch loses ~0.08 ARI vs PCA on NPC. PCA's variance optimization is more beneficial than sketch's geometry preservation for this model.

6. **Recommendation**: Use `seed_specificity` as default for reference-free scenarios. Consider `pseudo_sig` for datasets with fewer types (5-7) and larger gene panels (5000+).
