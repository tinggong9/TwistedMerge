# Application A: Holonomy-Aware Multiview Fusion

Decision: **negative application result**.

## Commands

Smoke:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_A.py --mode smoke --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data
```

Confirmatory execution:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_A.py --mode confirmatory --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data
```

## Frozen corpus and leakage boundary

This phase loaded the exact shared feature cache and adapter checkpoints and verified every recorded SHA-256. No chart adapter was retrained. Pair transitions used `overlap_fit`; estimator choice used held-out `overlap_validation`; fusion components used only `adapter_train`; the best individual adapter used `validation`. All sixteen candidate test-logit tensors were saved and hashed before test labels were accessed.

## Structural result

- Maximum selected loop distance from identity: `0.000002`.
- Mean rotation/reflection loop-commutator distance: `0.000001`.
- Nonidentity holonomy threshold passed: `False`.
- Noncommuting-holonomy threshold passed: `False`.
- Held-out selection counts by transition estimator: `{'weight_based': 5}`.
- Activation-Procrustes mean held-out residual / bootstrap instability / per-seed maximum loop distance: `0.844680` / `0.490482` / `0.683275`.
- These are feature-space loop operators. They are not central, projective, or Brauer-class claims.

The held-out residual selected the weight-derived map in every seed. Because that map is constructed as `A_j pinv(A_i)`, its near-coboundary loop closure is expected and is not evidence that activation-derived natural holonomy vanished. Activation-derived maps did have nonidentity loops, but their worse held-out fit and high bootstrap instability prevent a stable nonabelian application claim.

## Fusion result

- Independent model-training seeds: `5`.
- Structured branch method beat every preregistered matched control: `False`.
- Worst-view benefit without material mean loss: `False`.
- Overall Application A gate: `False`.

| method | mean_accuracy | mean_average_view_accuracy | mean_worst_view_accuracy | standard_deviation | mean_ece | mean_consistency |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_chart_aware_fusion | 0.6886 | 0.606375 | 0.5572 | 0.00350714 | 0.0667881 | 0.75225 |
| prediction_ensemble_upper_bound | 0.6886 | 0.6886 | 0.6886 | 0.00350714 | 0.0667881 | 1 |
| parameter_matched_generic_concat_head | 0.6462 | 0.58215 | 0.5448 | 0.00327109 | 0.0462469 | 0.772575 |
| global_c2m3_synchronization | 0.6436 | 0.584025 | 0.5462 | 0.0030496 | 0.0493952 | 0.773525 |
| learned_router | 0.6296 | 0.576075 | 0.5336 | 0.00207364 | 0.0467241 | 0.77385 |
| generic_mixture_of_experts | 0.629 | 0.571725 | 0.5298 | 0.000707098 | 0.0244875 | 0.768175 |
| best_individual_adapter | 0.6234 | 0.543825 | 0.481 | 0.00336155 | 0.0565303 | 0.727775 |
| d4_test_time_augmentation | 0.6234 | 0.6234 | 0.6234 | 0.00336155 | 0.0565304 | 1 |
| random_branch_count_matched_control | 0.6192 | 0.617325 | 0.5922 | 0.00311447 | 0.0577675 | 0.927225 |
| regular_d4_branch_invariant_pooling | 0.6192 | 0.6062 | 0.5618 | 0.00311447 | 0.0577675 | 0.874275 |

## Answers to the application questions

1. Loop holonomy is numerically distinct from pairwise fitting, but 5 seeds are inadequate for a held-out claim that it adds predictive information beyond pairwise residuals.
2. Noncommuting loop operators did not pass the threshold, but a causal prediction claim is not made.
3. The worst-view invariant-pooling gate did not pass.
4. The all-controls structured-pooling gate did not pass.
5. Capacity-matched random, generic-routing, and generic-concatenation controls are retained in the paired and capacity tables; extra branches alone are not credited as group-structure evidence.

## Stopping rule

No additional dataset or chart family is opened by a negative result. Application B proceeds only as the required conservative certificate on these exact saved natural transitions.
