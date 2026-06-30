# Optional Sheaf/GNN Cycle Diagnostic Report

## Integration Status

The official Neural Sheaf Diffusion code was inspected but not run.  The local TwistedMerge venv lacks `torch_geometric`, `torch_sparse`, and `torch_scatter`, so official NSD/WebKB runs would require a separate PyG environment.  This optional run is a self-contained PyTorch-only synthetic smoke test inspired by the NSD bundle-sheaf construction.

No external code was vendored or imported.

## Run Configuration

- Synthetic graph family: two-class heterophilic stochastic-block graphs with explicit triangle floor.
- Target heterophily levels: `0.25,0.55,0.85`.
- Seeds: `0,1,2`.
- Epochs: `120` with validation-loss early stopping.
- Methods: dense GCN, rotation-sheaf GNN, rotation-sheaf GNN with cycle regularizer.
- Git commit: `9535ed3`; dirty tree during run: `True`.

## Method Summary

| method                   | rows | mean_test_accuracy | mean_validation_accuracy | mean_cycle_inconsistency | mean_hidden_feature_variance | mean_dirichlet_energy |
| ------------------------ | ---- | ------------------ | ------------------------ | ------------------------ | ---------------------------- | --------------------- |
| gcn                      | 9    | 0.8673             | 0.8889                   |                          | 0.1835                       | 0.3400                |
| rotation_sheaf           | 9    | 0.9012             | 0.9213                   | 1.2220                   | 0.9195                       | 1.7713                |
| rotation_sheaf_cycle_reg | 9    | 0.9136             | 0.9259                   | 0.4597                   | 1.1629                       | 1.7531                |

## Heterophily Slices

| target_heterophily | method                   | mean_observed_heterophily | mean_test_accuracy | mean_cycle_inconsistency | mean_triangle_count |
| ------------------ | ------------------------ | ------------------------- | ------------------ | ------------------------ | ------------------- |
| 0.2500             | gcn                      | 0.2348                    | 0.9583             |                          | 80.0000             |
| 0.2500             | rotation_sheaf           | 0.2348                    | 0.9028             | 1.3662                   | 80.0000             |
| 0.2500             | rotation_sheaf_cycle_reg | 0.2348                    | 0.9120             | 0.3852                   | 80.0000             |
| 0.5500             | gcn                      | 0.5630                    | 0.6713             |                          | 77.6667             |
| 0.5500             | rotation_sheaf           | 0.5630                    | 0.8704             | 1.2176                   | 77.6667             |
| 0.5500             | rotation_sheaf_cycle_reg | 0.5630                    | 0.8935             | 0.4409                   | 77.6667             |
| 0.8500             | gcn                      | 0.8508                    | 0.9722             |                          | 51.6667             |
| 0.8500             | rotation_sheaf           | 0.8508                    | 0.9306             | 1.0822                   | 51.6667             |
| 0.8500             | rotation_sheaf_cycle_reg | 0.8508                    | 0.9352             | 0.5529                   | 51.6667             |

## Correlations

| scope      | cycle_vs_test_accuracy_pearson | heterophily_vs_cycle_pearson | heterophily_vs_test_accuracy_pearson |
| ---------- | ------------------------------ | ---------------------------- | ------------------------------------ |
| sheaf_rows | -0.1139                        | -0.0501                      | 0.3526                               |

## Interpretation

- The cycle score is a diagnostic over learned sheaf transports around observed triangles, not a proof of a cohomology class.
- The regularized sheaf row is included only as a small ablation.  A win here would not support a general GNN regularization claim.
- Because this is synthetic and small, the supported claim is limited to: cycle inconsistency can be measured and may help diagnose learned sheaf behavior on heterophilic graphs.
- The unsupported boundary remains: twisted sheaf regularization improves GNNs in general.

## Artifacts

- CSV: `reports/csv/sheaf_gnn_cycle_diagnostics.csv`
- Plot: `reports/plots/sheaf_gnn_cycle_vs_accuracy.pdf`
