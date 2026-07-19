# Application A: Holonomy-Aware Multiview Fusion

Decision: **bounded smoke completed**.

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

- Maximum selected loop distance from identity: `0.000001`.
- Mean rotation/reflection loop-commutator distance: `0.000001`.
- Nonidentity holonomy threshold passed: `False`.
- Noncommuting-holonomy threshold passed: `False`.
- These are feature-space loop operators. They are not central, projective, or Brauer-class claims.

## Fusion result

- Independent model-training seeds: `1`.
- Structured branch method beat every preregistered matched control: `False`.
- Worst-view benefit without material mean loss: `False`.
- Overall Application A gate: `False`.

| method | mean_accuracy | mean_average_view_accuracy | mean_worst_view_accuracy | standard_deviation | mean_ece | mean_consistency |
| --- | --- | --- | --- | --- | --- | --- |
| global_c2m3_synchronization | 0.460938 | 0.420898 | 0.375 | nan | 0.0603684 | 0.710938 |
| parameter_matched_generic_concat_head | 0.40625 | 0.373047 | 0.320312 | nan | 0.406791 | 0.672852 |
| oracle_chart_aware_fusion | 0.21875 | 0.125977 | 0.0703125 | nan | 0.0853187 | 0.3125 |
| prediction_ensemble_upper_bound | 0.21875 | 0.21875 | 0.21875 | nan | 0.0853187 | 1 |
| random_branch_count_matched_control | 0.210938 | 0.175781 | 0.140625 | nan | 0.0845642 | 0.570312 |
| raw_parameter_average | 0.210938 | 0.172852 | 0.148438 | nan | 0.0847477 | 0.560547 |
| regular_d4_branch_invariant_pooling | 0.210938 | 0.1875 | 0.132812 | nan | 0.0845642 | 0.586914 |
| wrong_group_action_control | 0.210938 | 0.181641 | 0.140625 | nan | 0.0845642 | 0.574219 |
| wrong_multiplication_order_control | 0.210938 | 0.170898 | 0.109375 | nan | 0.0845642 | 0.5625 |
| graph_synchronized_adapter_merge | 0.203125 | 0.172852 | 0.15625 | nan | 0.0768491 | 0.563477 |

## Answers to the application questions

1. Loop holonomy is numerically distinct from pairwise fitting, but 1 seeds are inadequate for a held-out claim that it adds predictive information beyond pairwise residuals.
2. Noncommuting loop operators did not pass the threshold, but a causal prediction claim is not made.
3. The worst-view invariant-pooling gate did not pass.
4. The all-controls structured-pooling gate did not pass.
5. Capacity-matched random, generic-routing, and generic-concatenation controls are retained in the paired and capacity tables; extra branches alone are not credited as group-structure evidence.

## Stopping rule

No additional dataset or chart family is opened by a negative result. Application B proceeds only as the required conservative certificate on these exact saved natural transitions.
