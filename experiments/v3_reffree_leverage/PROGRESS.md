# Reference-Free Leverage Scores — Progress Log

## Status: Implementation Complete, Awaiting Compute Node Execution

### Completed
- [x] `compute_reffree_leverage()` implemented in `seededntm/util.py` with 3 approaches
  - Approach 1: `pseudo_sig` — Seed-guided pseudo-signatures with circularity mitigation
  - Approach 2a: `self_leverage_k` — K-dim self-leverage from TF-IDF SVD
  - Approach 3: `seed_specificity` — Binary specificity weighting
  - All return FlashDeconv-style weights clipped to [0.1, 10.0]
- [x] Smoke test passed (synthetic data, all 3 methods)
- [x] Experiment script: `run_reffree_test.py` (PCA + CountSketch conditions)
- [x] Compute node script: `run_on_compute.sh` (3 stages: NPC, xenium_BC, sketch comparison)
- [x] Seed construction pipeline: `seedtopic/seed_construction.py`
- [x] Prompt templates: `seedtopic/prompt_templates.py`
- [x] CLI entry point: `scripts/construct_seeds.py`

### Awaiting Compute Node
- [ ] Run `experiments/v3_reffree_leverage/run_on_compute.sh` on a GPU node
- [ ] Collect results in `reffree_results.json`
- [ ] Document comparison in this file

### Results (to be filled after compute run)

| Dataset | Method | DimRed | F1 | AUPRC | ARI | dARI vs baseline |
|---------|--------|--------|------|-------|------|-----------------|
| visium_NPC | none | pca | — | — | — | 0 |
| visium_NPC | pseudo_sig | pca | — | — | — | — |
| visium_NPC | self_leverage_k | pca | — | — | — | — |
| visium_NPC | seed_specificity | pca | — | — | — | — |
| visium_NPC | ref_based | pca | — | — | — | — |
| xenium_BC_leiden | none | pca | — | — | — | 0 |
| xenium_BC_leiden | pseudo_sig | pca | — | — | — | — |
| xenium_BC_leiden | self_leverage_k | pca | — | — | — | — |
| xenium_BC_leiden | seed_specificity | pca | — | — | — | — |
