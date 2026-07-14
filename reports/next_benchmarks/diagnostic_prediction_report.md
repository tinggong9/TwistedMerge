# Held-Out Diagnostic Prediction Report

Natural-data diagnostic hypothesis: **unsupported** under the preregistered gate.

## Preregistration and exact command

- Primary target: `weight_average_degradation`
- Primary predictor: `cycle_residual`
- Harmful-merge threshold: `0.01` absolute accuracy
- Evaluation: leave one complete `(n_models,width)` setting out; seeds never cross from a held-out setting into its training folds.

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/heldout_diagnostic_prediction.py --bootstrap-samples 2000
```

- Git commit at execution: `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf`
- Instances: `120` natural trained-checkpoint collections from `reports/csv/improved_validated_ladder_merge_benchmark.csv`
- Planted labels: `False`

## Correlation and held-out prediction

| predictor | n_instances | pearson | pearson_ci_low | pearson_ci_high | spearman | spearman_ci_low | spearman_ci_high | heldout_r2 | harmful_merge_auc | calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairwise_alignment_loss | 120 | 0.642277 | 0.540006 | 0.738856 | 0.640551 | 0.515463 | 0.740381 | 0.387198 | nan | 0.595416 |
| inverse_consistency_residual | 120 | 0.106965 | -0.0645995 | 0.27227 | 0.0881007 | -0.0862922 | 0.256251 | -0.487773 | nan | 0.397054 |
| cycle_residual | 120 | -0.182838 | -0.35117 | -0.0179478 | -0.262557 | -0.41557 | -0.0872215 | -0.205534 | nan | 0.710968 |
| centrality_residual | 120 | -0.0381159 | -0.225082 | 0.104576 | -0.0298403 | -0.222279 | 0.154913 | -0.321115 | nan | 0.73552 |
| synchronization_disagreement | 120 | 0.106965 | -0.0615902 | 0.272298 | 0.0881007 | -0.0916185 | 0.263126 | -0.487773 | nan | 0.397054 |
| log_scale_variance | 120 | 0.207513 | 0.0162256 | 0.382715 | 0.271074 | 0.10577 | 0.435803 | -0.212384 | nan | 0.783878 |
| individual_model_accuracy_variance | 120 | 0.299144 | 0.181269 | 0.514178 | 0.478572 | 0.317649 | 0.611359 | -0.138018 | nan | 0.904713 |
| validation_loss | 120 | 0.757286 | 0.684643 | 0.820726 | 0.781422 | 0.687421 | 0.845049 | 0.510763 | nan | 0.448974 |
| validation_delta | 120 | -0.986719 | -0.990301 | -0.981868 | -0.978332 | -0.984953 | -0.962658 | 0.972287 | nan | 0.66342 |

## Baseline comparison

| model | heldout_r2 | harmful_merge_auc |
| --- | --- | --- |
| all_diagnostics_plus_validation | 0.975289 | nan |
| validation_only_baseline | 0.9757 | nan |
| pairwise_alignment_only_baseline | 0.387198 | nan |

The harmful-merge AUC is undefined when the fixed `0.01` indicator has only one class. In this grid every weight-average instance crossed the threshold, so the report retains `NaN` rather than changing the preregistered threshold after observing outcomes.

## Decision

The diagnostic is promoted only if the preregistered primary correlation interval clears zero, leave-one-setting-out `R^2` is positive, and the full diagnostic model adds held-out value beyond ordinary validation metrics. Result: **unsupported**. Missing cocycle-closure and certified distance-to-coboundary values are left unavailable rather than imputed as evidence.
