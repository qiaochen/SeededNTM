# SeedTopic Experiment Results Log

Append-only log of all experiment results. Never overwrite existing rows.

## Baseline Reference (from SeededNTM benchmark_deconv)

| Version | Dataset | F1 | Prec | Recall | AUROC | AUPRC | ARI | AMI | Runtime | Notes |
|---------|---------|-----|------|--------|-------|-------|-----|-----|---------|-------|
| benchmark | visium_NPC | 0.549 | 0.561 | 0.564 | 0.913 | 0.570 | 0.694 | 0.485 | 119.2s | SeededNTM benchmark run |
| benchmark | xenium_BC | 0.615 | 0.586 | 0.810 | 0.982 | 0.760 | 0.556 | 0.654 | 1458.8s | SeededNTM benchmark run |
| benchmark | xenium_BC (RCTD subset) | 0.607 | — | — | 0.983 | 0.752 | 0.593 | 0.669 | — | 120661 matched spots |
| benchmark | visiumHD_CRC_I | 0.527 | 0.499 | 0.868 | 0.990 | 0.821 | 0.471 | 0.539 | 138.6s | SeededNTM benchmark run |
| benchmark | visiumHD_CRC_I (RCTD subset) | 0.567 | — | — | 0.992 | 0.847 | 0.537 | 0.586 | — | 2400 matched spots |
| benchmark | visiumHD_CRC_II | 0.271 | 0.253 | 0.508 | 0.959 | 0.557 | 0.367 | 0.257 | 128.3s | SeededNTM benchmark run |
| benchmark | visiumHD_CRC_II (RCTD subset) | 0.280 | — | — | 0.970 | 0.624 | 0.438 | 0.303 | — | 2177 matched spots |

## Experiment Results

| Version | Dataset | F1 | Prec | Recall | AUROC | AUPRC | ARI | AMI | Runtime | Notes |
|---------|---------|-----|------|--------|-------|-------|-----|-----|---------|-------|
| v1_spatial_train | visium_NPC | 0.530 | — | — | — | 0.569 | 0.682 | 0.477 | — | best of lambda sweep (0.1), -0.010 ARI |
| v1_posthoc_a0.3 | xenium_BC | 0.627 | — | — | — | 0.760 | 0.578 | 0.661 | 0s | post-hoc alpha=0.3, +0.022 ARI, +0.013 F1 |
| v1_posthoc_a0.3 | visiumHD_CRC_I | 0.539 | — | — | — | — | 0.511 | — | 0s | post-hoc alpha=0.3, +0.040 ARI, +0.013 F1 |
| v1_posthoc_a0.3 | visiumHD_CRC_II | 0.277 | — | — | — | — | 0.384 | — | 0s | post-hoc alpha=0.3, +0.017 ARI, +0.006 F1 |
| v1_posthoc_a0.3 | visium_NPC | 0.518 | — | — | — | — | 0.670 | — | 0s | post-hoc alpha=0.3, -0.022 ARI (hurts small dataset) |
| v2_leverage_p0.25 | visium_NPC | 0.577 | 0.576 | 0.601 | 0.920 | 0.624 | 0.660 | 0.472 | 96s | +0.030 F1, +0.040 AUPRC, -0.032 ARI |
| v2_leverage_p0.25 | xenium_BC | 0.619 | — | — | — | 0.761 | 0.571 | 0.659 | — | +0.004 F1, +0.015 ARI (gentle, best for panel data) |
| v2_leverage_p1.0 | visiumHD_CRC_I | 0.542 | 0.505 | 0.879 | 0.991 | 0.849 | 0.482 | 0.511 | 145s | +0.015 F1, +0.029 AUPRC, +0.011 ARI |
| v2_leverage_p1.0 | visiumHD_CRC_II | 0.411 | 0.356 | 0.688 | 0.966 | 0.592 | 0.495 | 0.344 | 135s | **+0.140 F1, +0.128 ARI** (major gain) |
