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
